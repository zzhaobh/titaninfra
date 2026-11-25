"""Unified TorchTitan training entry point for Pretrain, SFT, and CPT.

run_train resolves model configuration, data loaders, optimizer/scheduler,
checkpoint setup, metrics, and optional hooks, then executes the training loop.
Text streams select Pretrain, messages select SFT, and mixed streams select CPT."""

import dataclasses
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from titaninfra.engine._pretrain_helpers import (
    _all_streams_have_choose,
    _all_streams_have_repeat,
    apply_calibration_schedule,
    apply_choose_scheduling,
    apply_repeat_scheduling,
    build_and_inject_data_metrics,
    build_dataloader,
    build_torchtitan_trainer_config,
    log_multi_stream_epoch_estimates,
    setup_training_hooks,
    validate_initial_checkpoint,
)
from titaninfra.utils.resume import resolve_checkpoint_config
from titaninfra.utils.wandb import save_wandb_run_id, setup_wandb_env

if TYPE_CHECKING:
    from titaninfra.trainer import TrainResult

logger = logging.getLogger(__name__)

# FlexInnerAttention's Triton flash template stages Q/K/V tiles of width head_dim in
# per-SM shared memory. At head_dim > 256 (bf16) none of inductor's default
# autotune block sizes fit SRAM, so every candidate kernel is pruned ->
# NoValidChoicesError at compile time. For such layers we constrain autotune to
# small blocks that DO fit, while keeping max_autotune on so inductor still picks
# the best within that space. Layers at/below the threshold keep default autotune.
# Gemma4 26B-A4B triggers this: full_attention layers use head_dim=512.
_LARGE_HEAD_DIM_THRESHOLD = 256
_LARGE_HEAD_DIM_KERNEL_OPTIONS = {"BLOCK_M": 32, "BLOCK_N": 32}


def _infer_stage(streams) -> str:
    """Infer pretrain, sft, or cpt from the stream data types."""
    data_types = {stream.data_type for stream in streams}
    if data_types == {"text", "messages"}:
        return "cpt"
    if "messages" in data_types:
        return "sft"
    return "pretrain"


def run_train(
    model_config,
    train_config,
    streams: list,
    hf_ckpt_path: str | None = None,
    task_name: str | None = None,
    _batch_validator=None,
) -> "TrainResult":
    """Run train."""
    #
    # name ∈ {locals, barrier, next_epoch, cache_usage, shard_states,
    #          shard_access_times, resume, epoch_shape, epoch_data}
    import os as _os
    import re as _re
    _SHM_PATTERN = _re.compile(
        r"^\d{6}_(locals|barrier|next_epoch|cache_usage|shard_states"
        r"|shard_access_times|resume|epoch_shape|epoch_data)$"
    )
    if int(_os.environ.get("LOCAL_RANK", "0")) == 0:
        _shm_dir = "/dev/shm"
        try:
            _entries = _os.listdir(_shm_dir)
        except OSError:
            _entries = []
        for _name in _entries:
            if _SHM_PATTERN.match(_name):
                try:
                    _os.remove(_os.path.join(_shm_dir, _name))
                except OSError:
                    pass

    from torchtitan.observability.logging import init_logger
    init_logger()

    try:
        import torchtitan.trainer  # noqa: F401  # gate: verify torchtitan importable

        from titaninfra.engine.train_step import (
            TitanInfraTrainer,
            install_torch_dist_set_timeout_compat,
        )

        install_torch_dist_set_timeout_compat()
    except ImportError as e:
        raise ImportError(
            f"训练需要 torchtitan 包：{e}\n"
            "请参考 torchtitan 文档安装：https://github.com/pytorch/torchtitan"
        ) from e

    family, flavor = model_config.parse_name()
    _parallelism_for_build = model_config.to_torchtitan_parallelism_config()
    model_spec = _load_model_spec(family, flavor, parallelism=_parallelism_for_build)
    logger.info(f"加载模型 spec：{family}/{flavor}")

    # local_batch_size = num_tokens_per_microbatch_per_dp_rank //
    model_spec.max_context_length = train_config.seq_len

    _transforms = _build_quantization_config(model_config)
    if _transforms:
        model_spec = _apply_converters(model_spec, _transforms)
        logger.info(f"已应用 {len(_transforms)} 个量化 transform 到 model config")

    model_spec = _apply_attn_backend(model_spec, model_config.attn_backend)
    logger.info(
        f"已覆盖 attn_backend={model_config.attn_backend!r}，"
        f"attn_mask_type={model_config.attn_mask_type!r}（后者仅影响 packing 判定，"
        f"不改变 inner_attention 构造）"
    )

    # "layers.0.attention.inner_attention must use CPInnerAttention, such as
    # AllGatherKVFlexInnerAttention, when the context parallel degree is larger
    if model_config.cp > 1:
        model_spec = _apply_context_parallel_transform(model_spec)
        logger.info(
            f"已应用上游 ContextParallelTransform（cp={model_config.cp}）："
            f"inner_attention -> CPInnerAttention 子类"
        )

    from titaninfra.models.moe_balance import resolve_balance

    balance_spec = resolve_balance(family, model_config)
    if balance_spec.bias_gamma is not None or model_config.moe_top_k is not None:
        model_spec = _apply_moe_config(
            model_spec,
            balance_spec.bias_gamma,
            model_config.moe_top_k,
        )

    if model_config.ep > 1:
        model_spec = _apply_moe_comm_backend(
            model_spec,
            model_config.ep_backend,
            local_batch_size=train_config.local_batch_size,
            seq_len=train_config.seq_len,
            cp=model_config.cp,
            tp=model_config.tp,
        )
        logger.info(f"已覆盖 MoE token_dispatcher → ep_backend={model_config.ep_backend!r}")


    if model_config.fsdp_explicit_prefetch:
        if model_config.pp > 1:
            logger.warning(
                "fsdp_explicit_prefetch=True 但 pp>1 → PP 路径 prefetch 时序复杂，"
                "本版强制关闭（仅 pp=1 生效）。"
            )
        elif family == "deepseek_v3":
            raise NotImplementedError(
                "fsdp_explicit_prefetch 在 torchtitan upstream-tip 迁移后"
                "暂不支持（deepseek_v3 尚未迁移到新 ModelSpec-less 架构，"
                "新架构下 explicit_prefetch 的正确实现方式待 deepseek_v3 "
                "自身迁移时一并确定）。"
            )

    tokenizer_path = train_config.tokenizer_path or hf_ckpt_path or ""
    hf_assets_path = tokenizer_path or "./assets"

    stage = _infer_stage(streams)
    ckpt_setup = resolve_checkpoint_config(
        resume_from=train_config.resume_from,
        ckpt_subdir=train_config.ckpt_subdir,
        dump_folder=train_config.dump_folder,
        hf_ckpt_path=hf_ckpt_path,
        task_name=task_name,
        stage=stage,
        is_hf_path_fn=_is_hf_path,
    )
    ckpt_subdir = ckpt_setup.ckpt_subdir
    dump_folder = ckpt_setup.dump_folder

    logger.info(
        f"Checkpoint 配置详情：\n"
        f"  - hf_ckpt_path (输入): {hf_ckpt_path or 'None'}\n"
        f"  - initial_load_path: {ckpt_setup.initial_load_path or 'None'}\n"
        f"  - initial_load_in_hf: {ckpt_setup.initial_load_in_hf}\n"
        f"  - checkpoint 目录: {ckpt_subdir}\n"
        f"  - dump_folder: {dump_folder}"
    )
    validate_initial_checkpoint(ckpt_setup)

    is_resume = ckpt_setup.initial_load_path is not None and not ckpt_setup.initial_load_in_hf
    setup_wandb_env(train_config, dump_folder=dump_folder, is_resume=is_resume)

    dataloader_cfg, validator_cfg = build_dataloader(
        streams,
        train_config,
        hf_ckpt_path=hf_ckpt_path,
        needs_packing=model_config.needs_packing,
        model_config=model_config,
    )

    trainer_cfg = build_torchtitan_trainer_config(
        model_spec=model_spec,
        hf_assets_path=hf_assets_path,
        dump_folder=dump_folder,
        dataloader_cfg=dataloader_cfg,
        validator_cfg=validator_cfg,
        train_config=train_config,
        model_config=model_config,
        ckpt_setup=ckpt_setup,
        ckpt_dir=ckpt_subdir,
        build_quantization_config_fn=_build_quantization_config,
    )

    logger.info(
        f"开始训练：模型={model_config.name}，步数={train_config.steps}，"
        f"local_batch_size={train_config.local_batch_size}，seq_len={train_config.seq_len}"
    )

    trainer = TitanInfraTrainer(trainer_cfg)

    if _batch_validator is not None:
        trainer._batch_validator = _batch_validator

    if train_config.fsdp_gas_no_sync:
        logger.warning(
            "fsdp_gas_no_sync=True 但当前 torchtitan upstream-tip 升级阶段暂未"
            "重新接回新 TrainingEngine（见 docs/torchtitan_upgrade_plan_2026-11.md "
            "§42）——训练数值结果不受影响，只是缺少这项 GAS 跨机 all-reduce "
            "skip 通信优化，仍会按『每 microbatch 都 all-reduce』的默认路径跑。"
        )

    # NOTE: MoE aux-loss (moe_aux_loss_coeff) wires set_aux_loss_source onto
    # trainer.engine.loss_fn later, inside build_and_inject_data_metrics — after this wrap.
    has_thinking_scale = any(
        getattr(s, "thinking_scale", None) is not None for s in streams
    )

    if has_thinking_scale:
        from titaninfra.data.loss import WeightedCELoss

        # Idempotence guard (unchanged semantics, updated mechanism): avoid
        # double-wrapping if this function somehow runs twice on the same
        # trainer. Previously checked hasattr(loss_fn, "set_loss_weights")
        # (the old mutable-state setter, now removed -- see WeightedCELoss's
        # module-level design-change docstring in data/loss.py); isinstance
        # is the direct, equivalent check against the wrapper class itself.
        if not isinstance(trainer.engine.loss_fn, WeightedCELoss):
            if train_config.chunked_loss:
                raise NotImplementedError(
                    "chunked_loss 与 thinking_scale 暂不支持同开：per-token 权重需经 "
                    "ChunkedLossWrapper 的 per-chunk loss_inputs 接线（M-beta）。当前请关闭其一。"
                )
            trainer.engine.loss_fn = WeightedCELoss(trainer.engine.loss_fn)
            logger.info("已包装 loss_fn → WeightedCELoss（支持 thinking_scale per-token 权重）")

    if train_config.chunked_loss:
        logger.info(
            f"已启用分块 CE loss（upstream ChunkedLossWrapper，num_chunks={train_config.loss_num_chunks}）"
        )

    from titaninfra.engine._pretrain_helpers import patch_dataloader_resilient_resume
    patch_dataloader_resilient_resume(trainer)

    setup_training_hooks(trainer, train_config, model_config=model_config)

    if _all_streams_have_choose(streams):
        apply_choose_scheduling(
            trainer,
            streams=streams,
            local_batch_size=train_config.local_batch_size,
            seq_len=train_config.seq_len,
            warmup_steps=train_config.warmup_steps,
            warmup_ratio=train_config.warmup_ratio,
        )
    elif _all_streams_have_repeat(streams):
        apply_repeat_scheduling(
            trainer,
            streams=streams,
            local_batch_size=train_config.local_batch_size,
            seq_len=train_config.seq_len,
            warmup_steps=train_config.warmup_steps,
            warmup_ratio=train_config.warmup_ratio,
        )
    elif len(streams) > 1:
        log_multi_stream_epoch_estimates(
            streams=streams,
            steps=train_config.steps,
            seq_len=train_config.seq_len,
            global_batch_size=train_config.global_batch_size,
            local_batch_size=train_config.local_batch_size,
        )

    apply_calibration_schedule(trainer, train_config)

    build_and_inject_data_metrics(trainer, train_config, streams, model_config=model_config)

    save_wandb_run_id(dump_folder)

    from titaninfra.profiling import install_op_region_hooks
    if train_config.enable_profiling:
        install_op_region_hooks(trainer.engine.model_parts)

    import torch.distributed
    torch.distributed.barrier()
    logger.info("Phase 1 barrier: all ranks finished model/dataloader construction")

    _original_ckpt_load = trainer.engine.checkpointer.load

    def _load_then_barrier(*args, **kwargs):
        result = _original_ckpt_load(*args, **kwargs)
        torch.distributed.barrier()
        logger.info("Phase 2 barrier: all ranks finished checkpoint load")
        return result

    trainer.engine.checkpointer.load = _load_then_barrier

    trainer.train()

    save_wandb_run_id(dump_folder)

    trainer.close()

    from titaninfra.trainer import TrainResult
    return TrainResult(
        stage=stage,
        ckpt_dir=ckpt_subdir,
        steps=trainer.config.training.steps,
        model_name=model_config.name,
    )


def _build_quantization_config(model_config) -> list:
    """Build and return the configured objects."""
    transforms = []

    if model_config.use_mxfp8:
        try:
            from torchtitan.config.transform.quantization import MXFP8LinearTransform
            transform = MXFP8LinearTransform()
            logger.info("MXFP8 量化已启用（需要 Hopper+ GPU）")
            transforms.append(transform)
        except ImportError:
            logger.warning("MXFP8LinearTransform 不可用，跳过 MXFP8 配置")

    return transforms


def _apply_converters(model_spec, transforms: list):
    """Apply the requested transformation."""
    if not transforms:
        return model_spec
    from torchtitan.config.transform.apply import transform_model_config_
    return transform_model_config_(model_spec, transforms)


def _load_model_spec(family: str, flavor: str, *, parallelism=None):
    """Load and return the requested data."""
    try:
        from titaninfra.models import is_supported as _titaninfra_is_supported

        _in_titaninfra = _titaninfra_is_supported(family)
    except ImportError:
        _in_titaninfra = False
    _in_torchtitan = importlib.util.find_spec(f"torchtitan.models.{family}") is not None

    if not (_in_titaninfra or _in_torchtitan):
        raise ValueError(
            f"不支持的模型系列 '{family}'（titaninfra.models 未注册且 torchtitan.models 无此模块）\n"
            f"model name 格式：'<family>/<flavor>'，如 'qwen3/8B'"
        )

    try:
        from titaninfra.models import is_supported, load_model_spec

        if is_supported(family):
            logger.info(f"从 titaninfra.models.{family} 加载模型：{flavor}")
            return load_model_spec(family, flavor, parallelism=parallelism)
    except ImportError:
        pass

    try:
        module = importlib.import_module(f"torchtitan.models.{family}")
    except ImportError as e:
        raise ImportError(
            f"无法加载模型模块 torchtitan.models.{family}：{e}"
        ) from e

    try:
        model_spec = module.model_registry(flavor)
        logger.info(f"从 torchtitan.models.{family} 加载模型：{flavor}")
    except KeyError as e:
        configs_dict = getattr(module, f"{family}_configs", {})
        available = sorted(configs_dict.keys())

        titaninfra_hint = ""
        if flavor.endswith("_titaninfra"):
            titaninfra_hint = (
                f"\n提示：'{flavor}' 是 TitanInfra 维护版本，"
                f"请使用 'qwen3/{flavor.replace('_titaninfra', '')}' 加载 TorchTitan 版本。"
            )

        raise KeyError(
            f"'{family}' 系列不支持 flavor '{flavor}'，"
            f"可用的 flavor：{available}"
            f"{titaninfra_hint}"
        ) from e

    return model_spec


def _apply_attn_backend(model_spec, attn_backend: str):
    """Apply the requested transformation."""

    from torchtitan.models.common import FlexInnerAttention
    from torchtitan.models.common.attention import (
        MLAAttention,
        SlidingWindowFlexInnerAttention,
        SlidingWindowVarlenInnerAttention,
    )

    if attn_backend != "flex":
        raise ValueError(
            f"Unknown attn_backend: {attn_backend!r} (only 'flex' is "
            "supported — sdpa was removed with the torchtitan upstream-tip "
            "migration, see config/model.py)"
        )

    def _build_plain_inner_attention(current, layer):
        """Pick a replacement for a directly-swappable backend config.

        Sliding-window specialization is architecture, not backend choice,
        and must survive this override under BOTH cases:
        1. Already a FlexInnerAttention SUBCLASS (not the bare class) —
           left untouched entirely (e.g. gemma4-style hybrid sliding+full
           attention already on flex).
        2. A SlidingWindowVarlenInnerAttention (e.g. gpt_oss's own
           internal default attn_backend='varlen' constructs sliding
           layers this way) — CONVERTED to the equivalent
           SlidingWindowFlexInnerAttention, carrying sliding_window_size
           forward. Without this, switching a varlen-default family to
           flex would silently turn every sliding-window layer into a
           full-attention layer (caught by testing gpt_oss end-to-end,
           not a hypothetical).

        Layers with head_dim > _LARGE_HEAD_DIM_THRESHOLD get a small-block
        kernel_options hint so inductor can find a SRAM-fitting Triton kernel;
        others keep default autotune.
        """
        if isinstance(current, FlexInnerAttention.Config) and type(current) is not FlexInnerAttention.Config:
            return current
        if isinstance(current, SlidingWindowVarlenInnerAttention.Config):
            return SlidingWindowFlexInnerAttention.Config(
                sliding_window_size=current.sliding_window_size
            )
        head_dim = getattr(layer.attention, "head_dim", None) or 0
        if head_dim > _LARGE_HEAD_DIM_THRESHOLD:
            return FlexInnerAttention.Config(
                kernel_options=dict(_LARGE_HEAD_DIM_KERNEL_OPTIONS)
            )
        return FlexInnerAttention.Config()

    def _build_inner_attention(layer):
        """Per-layer inner_attention config.

        MLA families (deepseek_v3 and architecturally alike families) wrap
        the real backend one level deeper: attention.inner_attention is an
        MLAAttention.Config carrying wkv_b/packed_kv_head_dim plus its OWN
        nested .inner_attention field holding the actual swappable backend.
        Rebuild the wrapper with only its nested backend swapped — replacing
        the wrapper itself would silently drop wkv_b and build an
        architecturally wrong attention module (DeepSeekMLAAttention.Config
        .__post_init__ raises ValueError if its inner_attention isn't an
        MLAAttention.Config, which is how this was caught — not assumed).
        """
        current = layer.attention.inner_attention
        if isinstance(current, MLAAttention.Config):
            return dataclasses.replace(
                current,
                inner_attention=_build_plain_inner_attention(
                    current.inner_attention, layer
                ),
            )
        return _build_plain_inner_attention(current, layer)

    def _replace_layer_attention(layer):
        """Replace layer attention."""
        from torchtitan.models.common.attention.gdn import GDN

        if isinstance(layer.attention, GDN.Config):
            return layer
        return dataclasses.replace(
            layer,
            attention=dataclasses.replace(
                layer.attention,
                inner_attention=_build_inner_attention(layer),
            ),
        )

    new_layers = [_replace_layer_attention(layer) for layer in model_spec.layers]
    return dataclasses.replace(model_spec, layers=new_layers)


def _apply_context_parallel_transform(model_spec):
    """Apply the requested transformation."""
    from torchtitan.config.transform.context_parallel import ContextParallelTransform
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        AllGatherKVFlexInnerAttention,
    )
    from torchtitan.models.common.attention.cp_mla import AllGatherKVMLAFlexAttention
    from torchtitan.models.common.attention.mla import MLAAttention

    transform = ContextParallelTransform(
        inner_attention_map={
            MLAAttention: AllGatherKVMLAFlexAttention,
            FlexInnerAttention: AllGatherKVFlexInnerAttention,
        }
    )
    return transform.transform(model_spec)


def _apply_moe_config(model_spec, load_balance_coeff: float | None, top_k: int | None):
    """Apply the requested transformation."""

    from torchtitan.models.common.moe import MoE

    first_layer = model_spec.layers[0]
    if not isinstance(first_layer.feed_forward, MoE.Config):
        return model_spec

    if load_balance_coeff is None and top_k is None:
        return model_spec

    new_layers = []
    for layer in model_spec.layers:
        if not isinstance(layer.feed_forward, MoE.Config):
            new_layers.append(layer)
            continue

        original_moe = layer.feed_forward
        new_router = original_moe.router
        if top_k is not None:
            new_router = dataclasses.replace(original_moe.router, top_k=top_k)

        moe_kwargs = {}
        if load_balance_coeff is not None:
            moe_kwargs["load_balance_coeff"] = load_balance_coeff
        if top_k is not None:
            moe_kwargs["router"] = new_router

        new_moe = dataclasses.replace(original_moe, **moe_kwargs)
        new_layers.append(dataclasses.replace(layer, feed_forward=new_moe))

    return dataclasses.replace(model_spec, layers=new_layers)


def _apply_moe_comm_backend(
    model_spec,
    comm_backend: str,
    *,
    local_batch_size: int,
    seq_len: int,
    cp: int,
    tp: int,
):
    """Apply the requested transformation."""

    from torchtitan.models.common.moe import MoE

    from titaninfra.models._tt_compat import make_token_dispatcher_config

    has_moe = any(
        isinstance(layer.feed_forward, MoE.Config) for layer in model_spec.layers
    )
    if not has_moe:
        return model_spec

    num_max_tokens_per_rank = (local_batch_size * seq_len) // (cp * tp)

    new_layers = []
    for layer in model_spec.layers:
        if not isinstance(layer.feed_forward, MoE.Config):
            new_layers.append(layer)
            continue

        old_dispatcher = layer.feed_forward.routed_experts.token_dispatcher
        new_dispatcher = make_token_dispatcher_config(
            num_experts=old_dispatcher.num_experts,
            top_k=old_dispatcher.top_k,
            comm_backend=comm_backend,
            hidden_dim=layer.feed_forward.routed_experts.w13.in_features,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
        )
        new_experts = dataclasses.replace(
            layer.feed_forward.routed_experts, token_dispatcher=new_dispatcher
        )
        new_moe = dataclasses.replace(layer.feed_forward, routed_experts=new_experts)
        new_layers.append(dataclasses.replace(layer, feed_forward=new_moe))

    return dataclasses.replace(model_spec, layers=new_layers)


def _is_hf_path(path: str | None) -> bool:
    """Is hf path."""
    if not path:
        return False

    p = Path(path)
    if not p.exists():
        return not path.endswith(".distcp")

    if p.is_dir():
        files = list(p.iterdir())
        file_names = {f.name for f in files}
        if "model.safetensors" in file_names:
            return True
        if "model.safetensors.index.json" in file_names:
            return True
        if "config.json" in file_names:
            return True
        if any(f.suffix == ".distcp" for f in files):
            return False

    return False
