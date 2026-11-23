"""Builders for TorchTitan training configuration and runtime components.

Helpers map TitanInfra configs to model, parallelism, optimizer, LR scheduler,
checkpoint, metric, and data-loader settings while preserving explicit
precision and algorithm semantics."""

import logging
import os
import warnings

logger = logging.getLogger(__name__)


def validate_initial_checkpoint(ckpt_setup) -> None:
    """Validate the supplied configuration."""
    if not ckpt_setup.initial_load_path:
        return

    if not os.path.exists(ckpt_setup.initial_load_path):
        raise FileNotFoundError(
            f"initial_load_path 不存在: {ckpt_setup.initial_load_path}\n"
            f"请检查 hf_ckpt_path 配置是否正确。"
        )
    logger.info(f"initial_load_path 路径验证通过: {ckpt_setup.initial_load_path}")

    if not ckpt_setup.initial_load_in_hf:
        logger.warning(
            "initial_load_in_hf=False，TorchTitan 不会以 HF 格式加载权重，"
            "这可能导致使用随机初始化权重而非预训练权重"
        )
        return

    _validate_hf_checkpoint_files(ckpt_setup.initial_load_path)


def _validate_hf_checkpoint_files(hf_path: str) -> None:
    """Validate the supplied configuration."""
    has_config = os.path.exists(os.path.join(hf_path, "config.json"))
    has_index = os.path.exists(os.path.join(hf_path, "model.safetensors.index.json"))
    has_model = os.path.exists(os.path.join(hf_path, "model.safetensors"))
    has_shards = (
        os.path.isdir(hf_path) and any(
            f.startswith("model-") and f.endswith(".safetensors")
            for f in os.listdir(hf_path)
        )
    )

    is_valid = has_config and (has_model or has_index or has_shards)

    if has_shards:
        logger.info(
            f"HF checkpoint 文件检查（分片模型）：\n"
            f"  - config.json: {'OK' if has_config else 'MISSING'}\n"
            f"  - model.safetensors.index.json: {'OK' if has_index else 'MISSING'}\n"
            f"  - 分片权重文件: OK"
        )
    else:
        logger.info(
            f"HF checkpoint 文件检查（单一文件）：\n"
            f"  - config.json: {'OK' if has_config else 'MISSING'}\n"
            f"  - model.safetensors: {'OK' if has_model else 'MISSING'}"
        )

    if not is_valid:
        raise FileNotFoundError(
            f"无效的 HuggingFace checkpoint 目录: {hf_path}\n"
            f"缺少必要的模型文件（需要 config.json + model.safetensors 或分片文件）"
        )


def build_dataloader(
    streams: list,
    train_config,
    *,
    hf_ckpt_path: str | None = None,
    needs_packing: bool = False,
    model_config=None,
):
    """Build and return the configured objects."""
    from torchtitan.components.validator import Validator

    from titaninfra.data.streaming_dataset import StreamingTextDataLoader

    gas = 1
    calibration_steps = 0
    if model_config is not None and train_config.global_batch_size > 0:
        dp_shard_est = model_config.dp_shard if model_config.dp_shard > 0 else 1
        dp_replicate_est = model_config.dp_replicate if model_config.dp_replicate > 0 else 1
        dp_world_est = dp_shard_est * dp_replicate_est
        gas = max(1, train_config.global_batch_size // (train_config.local_batch_size * dp_world_est))

    if train_config.moe_bias_schedule and train_config.moe_bias_schedule.calibration_steps:
        calibration_steps = train_config.moe_bias_schedule.calibration_steps

    has_messages = any(s.data_type == "messages" for s in streams)
    hf_tokenizer_path = ""
    if has_messages:
        hf_tokenizer_path = train_config.tokenizer_path or hf_ckpt_path or "./assets"

    dataloader_cfg = StreamingTextDataLoader.Config(
        streams=streams,
        hf_tokenizer_path=hf_tokenizer_path,
        shuffle=train_config.shuffle,
        shuffle_seed=train_config.shuffle_seed,
        batch_size_per_device=None,
        training_steps=train_config.steps,
        gas=gas,
        calibration_steps=calibration_steps,
        num_workers=train_config.num_workers,
        persistent_workers=train_config.persistent_workers,
        pin_memory=train_config.pin_memory,
        prefetch_factor=train_config.prefetch_factor,
        image_decode_workers=train_config.image_decode_workers,
        image_decode_prefetch=train_config.image_decode_prefetch,
        predownload=train_config.predownload,
        shuffle_block_size=train_config.shuffle_block_size,
        shuffle_algo=train_config.shuffle_algo,
        skip_corrupted_samples=train_config.skip_corrupted_samples,
        max_consecutive_errors=train_config.max_consecutive_errors,
        pp=model_config.pp if model_config else 1,
    )

    if needs_packing:
        logger.info("启用 BlockMask packing（attn_backend='flex', attn_mask_type='block_causal'）")

    logger.info(f"使用 StreamingTextDataLoader，{len(streams)} 个数据流")

    validator_cfg = None
    val_streams = train_config.val_streams
    if val_streams is not None:
        eval_steps = train_config.eval_steps
        if eval_steps == -1:
            raise ValueError(
                "Streaming 验证集必须设置 eval_steps > 0（不能用 -1）。\n"
                "原因：MDS dataloader 默认 repeat=True，eval_steps=-1 会无限循环。\n"
                "建议：eval_steps=10~50"
            )
        eval_freq = train_config.eval_freq
        val_dataloader_cfg = StreamingTextDataLoader.Config(
            streams=val_streams,
            hf_tokenizer_path=hf_tokenizer_path,
            shuffle=False,
            training_steps=train_config.steps,
            gas=gas,
            calibration_steps=calibration_steps,
            num_workers=train_config.num_workers,
            persistent_workers=train_config.persistent_workers,
            pin_memory=train_config.pin_memory,
            prefetch_factor=train_config.prefetch_factor,
            image_decode_workers=train_config.image_decode_workers,
            image_decode_prefetch=train_config.image_decode_prefetch,
            predownload=train_config.predownload,
            shuffle_block_size=None,
            shuffle_algo=train_config.shuffle_algo,
        )
        validator_cfg = Validator.Config(
            freq=eval_freq,
            steps=eval_steps,
            dataloader=val_dataloader_cfg,
        )
        logger.info(
            f"  Validation 已启用：eval_freq={eval_freq}, "
            f"eval_steps={eval_steps}, "
            f"val_streams={len(val_streams)} 个数据源"
        )

    return dataloader_cfg, validator_cfg


def build_and_inject_data_metrics(
    trainer,
    train_config,
    streams: list | None = None,
    model_config=None,
) -> None:
    """Build and return the configured objects."""
    from titaninfra.data.streaming_dataset import _read_single_stream_stats
    from titaninfra.utils.metrics_plugin import DataMetricsPlugin, inject_metrics_plugins

    tokens_per_step = train_config.global_batch_size * train_config.seq_len

    stream_total_tokens: dict[str, int] = {}
    stream_proportions: dict[str, float] = {}

    if streams:
        for s in streams:
            sid = s.resolve_stream_id()
            stats = _read_single_stream_stats(s.local, tokenizer_name=s.tokenizer_name)
            if stats is not None:
                stream_total_tokens[sid] = stats["total_tokens"]
            if s.proportion is not None and s.proportion > 0:
                stream_proportions[sid] = s.proportion

        if len(streams) == 1 and not stream_proportions:
            sid = streams[0].resolve_stream_id()
            stream_proportions[sid] = 1.0

    plugins = [
        DataMetricsPlugin(
            tokens_per_step=tokens_per_step,
            stream_total_tokens=stream_total_tokens,
            stream_proportions=stream_proportions,
        )
    ]

    if train_config.enable_moe_health_monitor and model_config is not None:
        from titaninfra.monitoring.moe_health import MoEHealthPlugin

        has_ac = model_config.ac_mode != "none"
        moe_plugin = MoEHealthPlugin(
            model_parts=trainer.engine.model_parts,
            has_ac=has_ac,
            ep_reduce_interval=train_config.moe_health_ep_reduce_interval,
            parallel_dims=trainer.engine.parallelism_context,
        )
        if moe_plugin.has_moe:
            plugins.append(moe_plugin)

    if model_config is not None:
        from titaninfra.models.moe_balance import resolve_balance

        family, _ = model_config.parse_name()
        balance_spec = resolve_balance(family, model_config)
    else:
        balance_spec = None

    if balance_spec is not None and balance_spec.aux_coeff is not None:
        from titaninfra.data.loss import WeightedCELoss
        from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

        has_ac = model_config.ac_mode != "none"
        aux_plugin = MoEAuxLossPlugin(
            model_parts=trainer.engine.model_parts,
            coeff=balance_spec.aux_coeff,
            has_ac=has_ac,
            parallel_dims=trainer.engine.parallelism_context,
            aux_scope=balance_spec.aux_scope,
        )
        if aux_plugin.has_moe:
            # loss_fn must support set_aux_loss_source — wrap plain CE if needed
            if not hasattr(trainer.engine.loss_fn, "set_aux_loss_source"):
                trainer.engine.loss_fn = WeightedCELoss(trainer.engine.loss_fn)
            trainer.engine.loss_fn.set_aux_loss_source(aux_plugin)
            trainer.engine._moe_aux_plugin = aux_plugin
            plugins.append(aux_plugin)            # for collect() → moe/aux_loss logging
            logger.info(
                f"MoE aux-loss 启用：coeff={balance_spec.aux_coeff}, "
                f"scope={balance_spec.aux_scope}"
            )

    if train_config.enable_sv_entropy_monitor:
        from titaninfra.monitoring.sv_entropy import SVEntropyPlugin

        sv_plugin = SVEntropyPlugin(
            model_parts=trainer.engine.model_parts,
            interval=train_config.sv_entropy_interval,
            max_layers=train_config.sv_entropy_max_layers,
            min_ndim_size=train_config.muon_min_ndim_size,
        )
        if sv_plugin._params:
            plugins.append(sv_plugin)

    if train_config.muon_qk_clip_tau is not None:
        from titaninfra.utils.metrics_plugin import QKClipMetricsPlugin

        qk_clip_plugin = QKClipMetricsPlugin(trainer.engine.optim.optimizers)
        plugins.append(qk_clip_plugin)

    inject_metrics_plugins(trainer, plugins)


def setup_training_hooks(trainer, train_config, model_config=None) -> None:
    """Setup training hooks."""
    if train_config.neftune_noise_alpha is not None:
        from titaninfra.utils.neftune import setup_neftune_hook
        for model_part in trainer.engine.model_parts:
            if hasattr(model_part, "tok_embeddings"):
                setup_neftune_hook(
                    model_part,
                    alpha=train_config.neftune_noise_alpha,
                    seq_len=train_config.seq_len,
                )

    if model_config is not None:
        _setup_moe_bias_schedule(trainer, train_config, model_config)

        from titaninfra.kernels.wiring import wire_custom_kernels
        wire_custom_kernels(trainer.engine.model_parts, model_config)

    if (
        train_config.muon_qk_clip_tau is not None
        and train_config.optimizer_name == "MuonAdamW"
    ):
        from titaninfra.optimizers.qk_clip import register_qk_clip_hook

        register_qk_clip_hook(
            trainer.engine.optim.optimizers,
            trainer.engine.model_parts,
            tau=train_config.muon_qk_clip_tau,
            head_dim=128,
        )


def _setup_moe_bias_schedule(trainer, train_config, model_config) -> None:
    """Setup moe bias schedule."""
    from titaninfra.models.moe_balance import resolve_gamma_at

    sched = train_config.moe_bias_schedule
    if not sched:
        return

    base_gamma = model_config.moe_load_balance_coeff
    if base_gamma is None:
        logger.warning(
            "moe_bias_schedule 已配置但 moe_load_balance_coeff=None，"
            "expert_bias 未启用，γ 调度无效。请在 ModelConfig 设置 moe_load_balance_coeff=1e-3。"
        )
        return

    def _set_gamma(gamma):
        for model_part in trainer.engine.model_parts:
            for module in model_part.modules():
                if getattr(module, "load_balance_coeff", None) is not None:
                    module.load_balance_coeff = gamma

    def bias_schedule_hook(optimizer, args, kwargs):
        gamma = resolve_gamma_at(
            trainer.engine.num_completed_steps + 1,
            base_gamma,
            calibration_steps=sched.calibration_steps,
            calibration_multiplier=sched.calibration_multiplier,
            freeze_after=sched.freeze_after,
        )
        if gamma is not None:
            _set_gamma(gamma)

    trainer.engine.optim.optimizers.register_step_pre_hook(bias_schedule_hook)
    logger.info(
        f"MoE bias γ-schedule hook 已注册：calibration_steps={sched.calibration_steps} "
        f"(γ×{sched.calibration_multiplier}), freeze_after={sched.freeze_after}, base_gamma={base_gamma}"
    )


def apply_calibration_schedule(trainer, train_config) -> None:
    """Apply the requested transformation."""
    cfg = train_config.moe_bias_schedule
    if not cfg or not cfg.calibration_steps:
        return

    cal_steps = cfg.calibration_steps

    trainer.config.training.steps += cal_steps

    for scheduler in trainer.engine.optim.lr_schedulers.schedulers:
        original_lambdas = list(scheduler.lr_lambdas)
        new_lambdas = []
        for fn in original_lambdas:
            def make_wrapped(orig_fn, offset):
                def wrapped(step):
                    if step < offset:
                        return 0.0
                    return orig_fn(step - offset)
                return wrapped
            new_lambdas.append(make_wrapped(fn, cal_steps))
        scheduler.lr_lambdas = new_lambdas

    for scheduler in trainer.engine.optim.lr_schedulers.schedulers:
        for i, (group, lambda_fn) in enumerate(
            zip(scheduler.optimizer.param_groups, scheduler.lr_lambdas, strict=False)
        ):
            group["lr"] = scheduler.base_lrs[i] * lambda_fn(scheduler.last_epoch)
        scheduler._last_lr = [g["lr"] for g in scheduler.optimizer.param_groups]

    logger.info(
        f"MoE calibration LR schedule: 前 {cal_steps} 步 lr=0（平坦校准），"
        f"之后正常 warmup→decay，total_steps 延伸到 {trainer.config.training.steps}"
    )



def apply_repeat_scheduling(
    trainer,
    *,
    streams: list,
    local_batch_size: int,
    seq_len: int,
    warmup_steps: int = 200,
    warmup_ratio: float | None = None,
) -> None:
    """Apply the requested transformation."""
    from titaninfra.data.streaming_dataset import _read_single_stream_stats
    from titaninfra.data.token_estimator import estimate_total_steps_from_repeat

    try:
        dp_world_size = trainer.engine.parallelism_context.dp_replicate * trainer.engine.parallelism_context.dp_shard
        gas = getattr(trainer, "gradient_accumulation_steps", 1) or 1

        all_stats = []
        for s in streams:
            stats = _read_single_stream_stats(s.local, tokenizer_name=s.tokenizer_name)
            if stats is None and getattr(s, "remote", None):
                stats = _read_single_stream_stats(s.remote, tokenizer_name=s.tokenizer_name)
            all_stats.append(stats)

        estimate = estimate_total_steps_from_repeat(
            stream_stats=all_stats,
            stream_configs=streams,
            local_batch_size=local_batch_size,
            gas=gas,
            seq_len=seq_len,
            dp_world_size=dp_world_size,
        )

        if estimate is None:
            logger.error("所有流的 stats 为空或非全部 repeat 模式，无法计算 total_steps。使用固定 steps。")
            return

        total_steps = estimate.total_steps

        if total_steps < 10:
            logger.warning(f"total_steps={total_steps} 过小，可能是数据集太小或配置问题。")

        trainer.config.training.steps = total_steps

        resolved_warmup = (
            int(warmup_ratio * total_steps) if warmup_ratio is not None else warmup_steps
        )
        trainer.config.optim.lr_scheduler.warmup_steps = resolved_warmup
        trainer.engine.optim.lr_schedulers = trainer.config.optim.lr_scheduler.build(
            optimizers=trainer.engine.optim.optimizers,
            training_steps=total_steps,
        )

        if hasattr(trainer.engine, 'checkpointer') and hasattr(trainer.engine.checkpointer, 'states'):
            trainer.engine.checkpointer.states["lr_scheduler"] = trainer.engine.optim.lr_schedulers

        exclude = trainer.config.checkpointer.exclude_from_loading
        if isinstance(exclude, (list, tuple)) and "lr_scheduler" not in exclude:
            trainer.config.checkpointer.exclude_from_loading = list(exclude) + ["lr_scheduler"]
            logger.info("  repeat 模式：lr_scheduler 已排除从 checkpoint 恢复")

        if resolved_warmup >= total_steps:
            warnings.warn(
                f"warmup_steps={resolved_warmup} >= repeat total_steps={total_steps}，"
                "学习率不会进入 cosine decay。",
                UserWarning, stacklevel=3,
            )

        per_stream_info = []
        for est in estimate.per_stream:
            pe_tag = f", PE={est.packing_efficiency:.3f}" if est.packing_efficiency < 1.0 else ""
            per_stream_info.append(
                f"    - {est.stream_id}: steps_per_epoch={est.steps_per_epoch}, "
                f"data_passes={est.data_passes:.1f}{pe_tag}"
            )

        stream_detail = "\n".join(per_stream_info)
        logger.info(
            f"Repeat 调度（token_estimator）：\n"
            f"  - 数据流数量: {len(streams)}\n"
            f"  - gas: {gas}\n"
            f"  - total_steps: {total_steps:,}\n"
            f"  - warmup_steps: {resolved_warmup:,}\n"
            f"  - 各流详情：\n{stream_detail}"
        )

    except AttributeError as e:
        logger.error(f"无法获取数据集统计信息: {e}")
    except Exception as e:
        logger.error(f"Repeat 调度计算失败: {e}")
        raise


def _all_streams_have_repeat(streams: list) -> bool:
    """Check if every stream has ``repeat`` set."""
    return bool(streams) and all(getattr(s, "repeat", None) is not None for s in streams)


def _all_streams_have_choose(streams: list) -> bool:
    """Check if every stream has ``choose`` set."""
    return bool(streams) and all(getattr(s, "choose", None) is not None for s in streams)


def apply_choose_scheduling(
    trainer,
    *,
    streams: list,
    local_batch_size: int,
    seq_len: int,
    warmup_steps: int = 200,
    warmup_ratio: float | None = None,
) -> None:
    """Apply the requested transformation."""
    from titaninfra.data.streaming_dataset import _read_single_stream_stats
    from titaninfra.data.token_estimator import estimate_total_steps_from_choose

    try:
        dp_world_size = trainer.engine.parallelism_context.dp_replicate * trainer.engine.parallelism_context.dp_shard
        gas = getattr(trainer, "gradient_accumulation_steps", 1) or 1

        all_stats = []
        for s in streams:
            stats = _read_single_stream_stats(s.local, tokenizer_name=s.tokenizer_name)
            if stats is None and getattr(s, "remote", None):
                stats = _read_single_stream_stats(s.remote, tokenizer_name=s.tokenizer_name)
            all_stats.append(stats)

        estimate = estimate_total_steps_from_choose(
            stream_stats=all_stats,
            stream_configs=streams,
            local_batch_size=local_batch_size,
            gas=gas,
            seq_len=seq_len,
            dp_world_size=dp_world_size,
        )

        if estimate is None:
            logger.error("所有流的 stats 为空或非全部 choose 模式，无法计算 total_steps。使用固定 steps。")
            return

        total_steps = estimate.total_steps

        if total_steps < 10:
            logger.warning(f"total_steps={total_steps} 过小，可能是数据集太小或配置问题。")

        trainer.config.training.steps = total_steps

        resolved_warmup = (
            int(warmup_ratio * total_steps) if warmup_ratio is not None else warmup_steps
        )
        trainer.config.optim.lr_scheduler.warmup_steps = resolved_warmup
        trainer.engine.optim.lr_schedulers = trainer.config.optim.lr_scheduler.build(
            optimizers=trainer.engine.optim.optimizers,
            training_steps=total_steps,
        )

        if hasattr(trainer.engine, 'checkpointer') and hasattr(trainer.engine.checkpointer, 'states'):
            trainer.engine.checkpointer.states["lr_scheduler"] = trainer.engine.optim.lr_schedulers

        exclude = trainer.config.checkpointer.exclude_from_loading
        if isinstance(exclude, (list, tuple)) and "lr_scheduler" not in exclude:
            trainer.config.checkpointer.exclude_from_loading = list(exclude) + ["lr_scheduler"]
            logger.info("  choose 模式：lr_scheduler 已排除从 checkpoint 恢复")

        if resolved_warmup >= total_steps:
            warnings.warn(
                f"warmup_steps={resolved_warmup} >= choose total_steps={total_steps}，"
                "学习率不会进入 cosine decay。",
                UserWarning, stacklevel=3,
            )

        per_stream_info = []
        for est in estimate.per_stream:
            pe_tag = f", PE={est.packing_efficiency:.3f}" if est.packing_efficiency < 1.0 else ""
            per_stream_info.append(
                f"    - {est.stream_id}: steps={est.steps_per_epoch}, "
                f"data_passes={est.data_passes:.1f}{pe_tag}"
            )

        stream_detail = "\n".join(per_stream_info)
        logger.info(
            f"Choose 调度（token_estimator）：\n"
            f"  - 数据流数量: {len(streams)}\n"
            f"  - gas: {gas}\n"
            f"  - total_steps: {total_steps:,}\n"
            f"  - warmup_steps: {resolved_warmup:,}\n"
            f"  - 各流详情：\n{stream_detail}"
        )

    except AttributeError as e:
        logger.error(f"无法获取数据集统计信息: {e}")
    except Exception as e:
        logger.error(f"Choose 调度计算失败: {e}")
        raise


def log_multi_stream_epoch_estimates(
    *,
    streams: list,
    steps: int,
    seq_len: int,
    global_batch_size: int,
    local_batch_size: int,
) -> None:
    """Log multi stream epoch estimates."""
    if not streams or len(streams) <= 1:
        return

    effective_gbs = global_batch_size if global_batch_size > 0 else (local_batch_size * 4)
    total_tokens = steps * effective_gbs * seq_len

    lines = [f"多流 epoch 预估（固定 steps={steps}，仅供参考）："]
    for i, s in enumerate(streams):
        if s.proportion is not None:
            stream_tokens = total_tokens * s.proportion
            lines.append(
                f"  - [{i}] {s.local}: proportion={s.proportion:.1%}, "
                f"~{stream_tokens:,.0f} tokens"
            )
        else:
            lines.append(f"  - [{i}] {s.local}: 无 proportion（按样本数比例）")

    logger.info("\n".join(lines))


def patch_dataloader_resilient_resume(trainer) -> None:
    """Patch dataloader resilient resume."""
    dataloader = trainer.dataloader
    original_load = dataloader.load_state_dict

    def _resilient_load_state_dict(state_dict):
        if not state_dict:
            return original_load(state_dict)

        saved_ws = state_dict.get("world_size")
        current_ws = dataloader.dp_world_size

        if saved_ws is not None and saved_ws != current_ws:
            logger.warning(
                f"Dataloader world_size mismatch: checkpoint={saved_ws}, current={current_ws}. "
                f"Dataloader state skipped (data restarts from beginning). "
                f"ntokens_seen is still restored from train_state."
            )
            return

        return original_load(state_dict)

    dataloader.load_state_dict = _resilient_load_state_dict


def _build_loss_config(train_config, chunked_loss_cls, cross_entropy_cls, *, global_vocab_size=None):
    """Build and return the configured objects."""
    if train_config.chunked_loss:
        return chunked_loss_cls.Config(
            num_chunks=train_config.loss_num_chunks,
            loss_fn=cross_entropy_cls.Config(global_vocab_size=global_vocab_size),
        )
    return cross_entropy_cls.Config(global_vocab_size=global_vocab_size)


def _adamw_shared_kwargs(train_config) -> dict:
    """Adamw shared kwargs."""
    if train_config.optimizer_state_dtype == "bf16":
        fused, moment_dtype = True, "bfloat16"
    else:
        fused = train_config.optimizer_implementation in ("fused", "fused_opt_states_bf16")
        moment_dtype = "parameter"
    foreach = train_config.optimizer_implementation == "foreach" and not fused
    return {
        "betas": (train_config.beta1, train_config.beta2),
        "eps": train_config.eps,
        "weight_decay": train_config.weight_decay,
        "fused": fused,
        "foreach": foreach,
        "moment_dtype": moment_dtype,
    }


def _build_optimizer_config(train_config, param_groups: list):
    """Build optimizer config: MixedOptimizersContainer for MuonAdamW, else OptimizersContainer.

    ``param_groups`` is a list of ``AdamW.Config`` entries for specific
    patterns (``layer_lr_overrides``/``freeze_params``, built by
    ``build_torchtitan_trainer_config`` using the same ``_adamw_shared_kwargs``
    so every entry -- specific or catch-all -- shares one consistent
    betas/eps/weight_decay/fused/foreach/moment_dtype recipe). This function
    appends the catch-all entry (always last -- upstream's
    ``OptimizersContainer._build_param_group`` claims parameters in pattern
    order, first-match-wins, so specific overrides must be listed before the
    ``.*`` catch-all; the caller already builds ``param_groups`` in that
    order and this function only appends to the end).
    """
    from torchtitan.components.optim import AdamW, OptimizersContainer

    _catch_all = AdamW.Config(
        pattern=r".*",
        lr=train_config.lr,
        **_adamw_shared_kwargs(train_config),
    )
    _optimizers = list(param_groups) + [_catch_all] if param_groups else [_catch_all]

    if train_config.optimizer_name == "MuonAdamW":
        from titaninfra.optimizers.mixed_optimizer import MixedOptimizersContainer

        return MixedOptimizersContainer.Config(
            optimizers=_optimizers,
            muon_lr=train_config.muon_lr,
            muon_momentum=train_config.muon_momentum,
            muon_nesterov=train_config.muon_nesterov,
            muon_ns_steps=train_config.muon_ns_steps,
            muon_weight_decay=train_config.muon_weight_decay,
            muon_min_ndim_size=train_config.muon_min_ndim_size,
            muon_ns_dtype=train_config.muon_ns_dtype,
            muon_scale_factor=train_config.muon_scale_factor,
            muon_include_router=train_config.muon_include_router,
            muon_gather_mode=train_config.muon_gather_mode,
        )

    return OptimizersContainer.Config(
        optimizers=_optimizers,
    )


def build_torchtitan_trainer_config(
    *,
    model_spec,
    hf_assets_path: str,
    dump_folder: str,
    dataloader_cfg,
    validator_cfg,
    train_config,
    model_config,
    ckpt_setup,
    ckpt_dir: str,  # ckpt_subdir from caller, named ckpt_dir for TorchTitan compat
    build_quantization_config_fn,
):
    """Build and return the configured objects."""
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
    from torchtitan.components.optim import AdamW
    from torchtitan.components.optim.lr_scheduler import LRSchedulersContainer
    from torchtitan.components.optim.optim import Optim
    from torchtitan.config.configs import CommConfig, DebugConfig, PrecisionConfig, TrainingConfig
    from torchtitan.trainer import Trainer as TorchTitanTrainer

    from titaninfra.config.train import resolve_warmup_steps
    from titaninfra.profiling import build_profiler_config
    from titaninfra.utils.wandb import build_metrics_config

    tt_dtype = model_config.torchtitan_dtype()
    #
    master_dtype = model_config.torchtitan_param_dtype()
    compute_dtype = tt_dtype

    _adamw_shared = _adamw_shared_kwargs(train_config)
    param_groups: list[AdamW.Config] = []
    for pattern, multiplier in (train_config.layer_lr_overrides or {}).items():
        param_groups.append(AdamW.Config(
            pattern=f".*{pattern}.*", lr=train_config.lr * multiplier, **_adamw_shared,
        ))
    for pattern in (train_config.freeze_params or []):
        param_groups.append(AdamW.Config(
            pattern=f".*{pattern}.*", lr=0.0, **_adamw_shared,
        ))
    if param_groups:
        frozen = [p for p in (train_config.freeze_params or [])]
        scaled = list((train_config.layer_lr_overrides or {}).keys())
        logger.info(
            f"参数组配置：lr_scale={scaled or '无'}, freeze={frozen or '无'}"
        )

    return TorchTitanTrainer.Config(
        model=model_spec,
        hf_assets_path=hf_assets_path,
        dump_folder=dump_folder,
        dataloader=dataloader_cfg,
        validator=validator_cfg,

        #                          （model_config.to_torchtitan_parallelism_config()
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=train_config.local_batch_size * train_config.seq_len,
            num_tokens_per_train_step=(
                train_config.global_batch_size * train_config.seq_len
                if train_config.global_batch_size > 0
                else -1
            ),
            steps=train_config.steps,
            disable_accelerator_graphs=(model_config.ep > 1),
        ),

        precision=PrecisionConfig(
            dtype=master_dtype,
            param_dtype=compute_dtype,
        ),

        parallelism=model_config.to_torchtitan_parallelism_config(),

        loss=_build_loss_config(
            train_config, ChunkedLossWrapper, CrossEntropyLoss,
            global_vocab_size=model_spec.vocab_size,
        ),

        # （optim.optimizer / optim.lr_scheduler / optim.max_norm / optim.ema /
        optim=Optim.Config(
            optimizer=_build_optimizer_config(train_config, param_groups),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=resolve_warmup_steps(train_config, train_config.steps),
                decay_type="cosine" if train_config.lr_decay_type == "constant" else train_config.lr_decay_type,
                min_lr_factor=1.0 if train_config.lr_decay_type == "constant" else train_config.min_lr_ratio,
                decay_ratio=train_config.decay_ratio,
            ),
            max_norm=train_config.max_grad_norm,
        ),

        checkpointer=CheckpointManager.Config(
            folder=ckpt_dir,
            interval=train_config.ckpt_interval,
            keep_latest_k=train_config.keep_latest_k,
            initial_load_path=ckpt_setup.initial_load_path,
            initial_load_in_hf=ckpt_setup.initial_load_in_hf,
            initial_load_model_only=ckpt_setup.initial_load_in_hf,
            load_step=-1,
            exclude_from_loading=train_config.exclude_from_loading,
            async_mode=train_config.async_checkpoint,
            last_save_model_only=train_config.last_save_model_only,
            last_save_in_hf=train_config.last_save_in_hf,
            export_dtype=train_config.export_dtype,
            enable_first_step_checkpoint=train_config.enable_first_step_checkpoint,
        ),

        debug=DebugConfig(seed=train_config.seed),

        activation_remat=model_config.to_torchtitan_activation_checkpoint_config(),

        metrics=build_metrics_config(train_config),

        profiler=build_profiler_config(train_config),


        comm=CommConfig(
            init_timeout_seconds=train_config.comm_init_timeout_seconds,
            train_timeout_seconds=train_config.comm_train_timeout_seconds,
        ),

    )
