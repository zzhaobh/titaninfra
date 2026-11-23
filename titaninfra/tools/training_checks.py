"""Training-environment, configuration, and checkpoint validation helpers.

These checks report actionable setup problems before long distributed runs."""

import json
import logging
import math
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


# =========================================================================
# =========================================================================
def check_train_result(result, *, expected_stage: str, min_steps: int | None = None):
    """Check train result."""
    assert result is not None, "TrainResult 不应为 None"
    assert hasattr(result, "stage"), f"TrainResult 缺少 stage 属性: {result}"
    assert hasattr(result, "ckpt_dir"), f"TrainResult 缺少 ckpt_dir 属性: {result}"
    assert hasattr(result, "steps"), f"TrainResult 缺少 steps 属性: {result}"
    assert hasattr(result, "model_name"), f"TrainResult 缺少 model_name 属性: {result}"

    assert result.stage == expected_stage, (
        f"stage 不匹配: 期望 '{expected_stage}', 实际 '{result.stage}'"
    )
    assert result.steps > 0, f"steps 应 > 0, 实际 {result.steps}"

    if min_steps is not None:
        assert result.steps >= min_steps, (
            f"steps 不足: 期望 >= {min_steps}, 实际 {result.steps}"
        )

    logger.info(f"[check_train_result] PASS: stage={result.stage}, steps={result.steps}")


# =========================================================================
# =========================================================================
def check_checkpoint_dir(ckpt_dir: str, *, expect_dataloader_state: bool = True):
    """Check checkpoint dir."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return

    ckpt_path = Path(ckpt_dir)
    assert ckpt_path.exists(), f"checkpoint 目录不存在: {ckpt_dir}"

    step_dirs = sorted(
        [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    assert len(step_dirs) > 0, f"checkpoint 目录下无 step-* 子目录: {ckpt_dir}"

    target = None
    shard_files = []
    for d in reversed(step_dirs):
        if not (d / ".metadata").exists():
            continue
        shard_files = (
            list(d.glob("__*_*.distcp"))
            or list(d.glob("*.pt"))
            or list(d.glob("*.safetensors"))
        )
        if shard_files:
            target = d
            break

    assert target is not None, (
        f"checkpoint 目录下无包含有效 shard 文件的 step 子目录: {ckpt_dir}"
    )

    logger.info(
        f"[check_checkpoint_dir] PASS: {target.name}, "
        f"{len(shard_files)} shards"
    )


# =========================================================================
# =========================================================================
def check_checkpoint_resume_continuity(result1, result2):
    """Check checkpoint resume continuity."""
    assert result2.steps > result1.steps, (
        f"Phase 2 steps ({result2.steps}) 应 > Phase 1 steps ({result1.steps})"
    )
    assert result2.stage == result1.stage, (
        f"两段训练 stage 不一致: '{result1.stage}' vs '{result2.stage}'"
    )
    logger.info(
        f"[check_checkpoint_resume_continuity] PASS: "
        f"phase1={result1.steps} → phase2={result2.steps}"
    )


# =========================================================================
# =========================================================================
def check_repeat_scheduling(result, *, expected_max_steps: int | None = None):
    """Check repeat scheduling."""
    assert result.steps > 0, "repeat 调度后 steps 应 > 0"

    if expected_max_steps is not None:
        assert result.steps <= expected_max_steps, (
            f"steps ({result.steps}) 超过预期上限 ({expected_max_steps})"
        )

    msg = f"[check_repeat_scheduling] PASS: steps={result.steps}"
    if expected_max_steps:
        msg += f" (上限 {expected_max_steps})"
    logger.info(msg)


# =========================================================================
# =========================================================================
def check_wandb_offline_metrics(
    dump_folder: str, *, required_keys: list[str] | None = None
):
    """Check wandb offline metrics."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return

    wandb_dir = Path(dump_folder) / "wandb"
    if not wandb_dir.exists():
        logger.warning(f"[check_wandb_offline_metrics] wandb 目录不存在: {wandb_dir}")
        return

    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda d: d.name)
    if not run_dirs:
        run_dirs = sorted(wandb_dir.glob("offline-run-*"), key=lambda d: d.name)
    if not run_dirs:
        logger.warning(f"[check_wandb_offline_metrics] 无 run 目录: {wandb_dir}")
        return

    summary_file = run_dirs[-1] / "files" / "wandb-summary.json"
    if not summary_file.exists():
        logger.warning(f"[check_wandb_offline_metrics] summary 文件不存在: {summary_file}")
        return

    summary = json.loads(summary_file.read_text())

    if required_keys:
        for key in required_keys:
            assert key in summary, (
                f"WandB summary 缺少 metric '{key}', "
                f"可用 keys: {sorted(summary.keys())}"
            )
            logger.info(f"  {key} = {summary[key]}")

    logger.info(f"[check_wandb_offline_metrics] PASS: {len(summary)} metrics")


# =========================================================================
# =========================================================================
def check_per_stream_metrics(dump_folder: str, stream_ids: list[str]):
    """Check per stream metrics."""
    required_keys = ["data/total_tokens_seen"]
    for sid in stream_ids:
        required_keys.append(f"data/{sid}/tokens")

    check_wandb_offline_metrics(dump_folder, required_keys=required_keys)
    logger.info(
        f"[check_per_stream_metrics] PASS: "
        f"streams={stream_ids}"
    )


# =========================================================================
# =========================================================================
def _read_wandb_summary(dump_folder: str) -> dict | None:
    """Read wandb summary."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return None

    wandb_dir = Path(dump_folder) / "wandb"
    if not wandb_dir.exists():
        return None

    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda d: d.name)
    if not run_dirs:
        run_dirs = sorted(wandb_dir.glob("offline-run-*"), key=lambda d: d.name)
    if not run_dirs:
        return None

    summary_file = run_dirs[-1] / "files" / "wandb-summary.json"
    if not summary_file.exists():
        return None

    return json.loads(summary_file.read_text())


def check_training_health(dump_folder: str):
    """Check training health."""
    summary = _read_wandb_summary(dump_folder)
    if summary is None:
        logger.warning(
            f"[check_training_health] SKIP: wandb summary 不可用 ({dump_folder})"
        )
        return

    issues: list[str] = []

    avg_loss = summary.get("loss_metrics/global_avg_loss")
    if avg_loss is not None:
        if not math.isfinite(avg_loss):
            issues.append(
                f"loss 异常：global_avg_loss={avg_loss}（NaN/Inf 表示梯度爆炸或数据问题）"
            )
        else:
            logger.info(f"  loss_metrics/global_avg_loss = {avg_loss:.6f}")

    max_loss = summary.get("loss_metrics/global_max_loss")
    if max_loss is not None and not math.isfinite(max_loss):
        issues.append(f"max_loss 异常：global_max_loss={max_loss}")

    # 2. Throughput > 0
    tps = summary.get("throughput(tps)")
    if tps is not None:
        if tps <= 0:
            issues.append(
                f"吞吐为 0：throughput(tps)={tps}（数据加载挂起或 GPU 空转）"
            )
        else:
            logger.info(f"  throughput(tps) = {tps:,.0f}")

    num_ooms = summary.get("memory/num_ooms", 0)
    if num_ooms > 0:
        issues.append(
            f"发生 OOM：memory/num_ooms={num_ooms}（需减小 batch_size 或 seq_len）"
        )

    total_tokens = summary.get("data/total_tokens_seen")
    if total_tokens is not None:
        if total_tokens <= 0:
            issues.append(
                f"data/total_tokens_seen={total_tokens}，"
                f"DataMetricsPlugin 未正常追踪 token 消耗"
            )
        else:
            logger.info(f"  data/total_tokens_seen = {total_tokens:,.0f}")

    assert not issues, (
        f"训练健康检查发现 {len(issues)} 个问题:\n"
        + "\n".join(f"  - {issue}" for issue in issues)
    )

    logger.info("[check_training_health] PASS: metrics healthy")


def check_resume_tokens_continuity(
    dump_folder_phase1: str,
    dump_folder_phase2: str,
):
    """Check resume tokens continuity."""
    s1 = _read_wandb_summary(dump_folder_phase1)
    s2 = _read_wandb_summary(dump_folder_phase2)
    if s1 is None or s2 is None:
        return

    t1 = s1.get("data/total_tokens_seen", 0)
    t2 = s2.get("data/total_tokens_seen", 0)

    assert t2 > t1, (
        f"Resume 后 token 计数未连续递增：Phase1={t1:,}, Phase2={t2:,}。"
        f"可能 DataMetricsPlugin 在 resume 后重置为 0"
    )

    logger.info(
        f"[check_resume_tokens_continuity] PASS: "
        f"Phase1={t1:,} → Phase2={t2:,} (+{t2 - t1:,})"
    )


# =========================================================================
# SFT Batch Validator — per-batch data pipeline invariant checks
# =========================================================================
IGNORE_INDEX = -100


def validate_sft_batch(
    step: int,
    input_dict: dict,
    labels,
    *,
    seq_len: int,
    thinking_scale: float | None = None,
    expected_markers: list[int] | None = None,
) -> None:
    """Per-batch invariant checks for SFT data pipeline.

    Called from titaninfra_train_step via trainer._batch_validator.
    All tensors are CPU at this point (before .to(device)).
    Raises AssertionError on any invariant violation.
    """
    inputs = input_dict["input"]
    B = inputs.shape[0]

    assert inputs.shape == (B, seq_len), (
        f"step {step}: input shape {inputs.shape} != ({B}, {seq_len})"
    )
    assert labels.shape == (B, seq_len), (
        f"step {step}: labels shape {labels.shape} != ({B}, {seq_len})"
    )

    has_loss = (labels != IGNORE_INDEX).any()
    if not has_loss:
        positions_early = input_dict.get("positions")
        has_real_tokens = (
            positions_early is not None and (positions_early >= 0).any()
        )
        n_docs = 0
        if positions_early is not None:
            n_docs = (positions_early == 0).sum().item()
        if has_real_tokens and step == 1:
            raise AssertionError(
                f"step {step}: real tokens present ({n_docs} docs) but all labels are "
                f"IGNORE_INDEX — masking bug"
            )
        if not has_real_tokens:
            logger.warning(
                f"step {step}: all-padding batch (data exhaustion from shard imbalance)"
            )
        else:
            logger.warning(
                f"step {step}: all labels IGNORE_INDEX with {n_docs} docs — "
                f"packing boundary or epoch edge"
            )
        return

    positions = input_dict.get("positions")
    loss_weights = input_dict.get("loss_weights")

    if positions is not None:
        assert positions.shape == (B, seq_len), (
            f"step {step}: positions shape {positions.shape} != ({B}, {seq_len})"
        )

        valid_mask = positions >= 0
        if valid_mask.any():
            max_pos = positions[valid_mask].max().item()
            assert max_pos < seq_len, (
                f"step {step}: position overflow {max_pos} >= seq_len {seq_len} — RoPE OOB"
            )

        pad_mask = positions == -1

        if pad_mask.any():
            pad_labels = labels[pad_mask]
            assert (pad_labels == IGNORE_INDEX).all(), (
                f"step {step}: labels != -100 in padding region — "
                f"found {(pad_labels != IGNORE_INDEX).sum().item()} non-masked"
            )

        for b in range(B):
            row_pos = positions[b]
            valid_end = (row_pos >= 0).sum().item()
            if valid_end == 0:
                continue

            # Contiguous padding: all -1 at end, no interleaving
            if valid_end < seq_len:
                assert (row_pos[valid_end:] == -1).all(), (
                    f"step {step}, row {b}: non-contiguous padding"
                )

            doc_region = row_pos[:valid_end]
            for i in range(1, len(doc_region)):
                curr = doc_region[i].item()
                prev = doc_region[i - 1].item()
                ok = (curr == prev + 1) or (curr == 0)
                assert ok, (
                    f"step {step}, row {b}, pos {i}: "
                    f"position jump {prev}→{curr} (expected {prev + 1} or 0)"
                )

    if loss_weights is not None:
        assert loss_weights.shape == (B, seq_len), (
            f"step {step}: loss_weights shape {loss_weights.shape} != ({B}, {seq_len})"
        )

        assert (loss_weights >= 0.0).all() and (loss_weights <= 1.0).all(), (
            f"step {step}: loss_weights out of [0, 1] — "
            f"min={loss_weights.min().item():.4f}, max={loss_weights.max().item():.4f}"
        )

        if positions is not None:
            pad_mask = positions == -1
            if pad_mask.any():
                pad_weights = loss_weights[pad_mask]
                assert (pad_weights == 0.0).all(), (
                    f"step {step}: loss_weights != 0 in padding — "
                    f"found {(pad_weights != 0).sum().item()} non-zero"
                )

        if thinking_scale is not None:
            fractional = (loss_weights > 0.0) & (loss_weights < 1.0)
            if fractional.any():
                frac_vals = loss_weights[fractional]
                unique_frac = frac_vals.unique()
                for uv in unique_frac:
                    diff = (uv - thinking_scale).abs()
                    assert diff <= 0.01, (
                        f"step {step}: fractional weights don't match thinking_scale={thinking_scale} — "
                        f"found values: {unique_frac.tolist()}"
                    )

        # Labels-weights consistency: IGNORE_INDEX in valid (non-padding) region → weight must be 0
        if positions is not None:
            valid_mask = positions >= 0
            ignore_in_valid = valid_mask & (labels == IGNORE_INDEX)
            if ignore_in_valid.any():
                weights_at_ignore = loss_weights[ignore_in_valid]
                assert (weights_at_ignore == 0.0).all(), (
                    f"step {step}: loss_weights != 0 where labels==IGNORE_INDEX in valid region — "
                    f"found {(weights_at_ignore != 0).sum().item()} non-zero"
                )

    if expected_markers is not None:
        for b in range(B):
            row = inputs[b].tolist()
            row_pos = positions[b] if positions is not None else None

            valid_end = len(row)
            if row_pos is not None:
                valid_end = (row_pos >= 0).sum().item()
            if valid_end == 0:
                continue

            if row_pos is not None:
                doc_starts = (row_pos[:valid_end] == 0).nonzero(as_tuple=True)[0].tolist()
            else:
                doc_starts = [0]

            for doc_idx, start in enumerate(doc_starts):
                end = doc_starts[doc_idx + 1] if doc_idx + 1 < len(doc_starts) else valid_end
                is_last_doc = (doc_idx + 1 >= len(doc_starts))
                is_first_doc = (doc_idx == 0)
                # Skip HARD_TRUNCATE fragments: very short segments, the last
                # document touching the packing boundary (tail fragment truncated
                # before its closing marker), or the first document in a multi-doc
                # pack (head fragment from _renumber_positions() after previous
                # flush — lacks opening markers).
                if end - start < 4:
                    continue
                if is_last_doc and end == valid_end and len(doc_starts) > 1:
                    continue
                if is_first_doc and len(doc_starts) > 1:
                    continue
                doc_tokens = row[start:end]
                for marker in expected_markers:
                    assert marker in doc_tokens, (
                        f"step {step}, row {b}, doc {doc_idx} [{start}:{end}]: "
                        f"expected marker token {marker} not found (len={end - start})"
                    )


def make_batch_validator(
    seq_len: int,
    *,
    thinking_scale: float | None = None,
    expected_markers: list[int] | None = None,
    every_n_steps: int = 1,
) -> Callable:
    """Factory: returns a _batch_validator closure for Trainer(_batch_validator=...).

    Args:
        seq_len: expected sequence length of each batch
        thinking_scale: if set, validate fractional loss_weights match this value
        expected_markers: if set, verify these token IDs appear in every document
        every_n_steps: validate every N steps (always validates step 1)
    """
    def _validator(step: int, input_dict: dict, labels) -> None:
        if step != 1 and every_n_steps > 1 and step % every_n_steps != 0:
            return
        validate_sft_batch(
            step, input_dict, labels,
            seq_len=seq_len, thinking_scale=thinking_scale,
            expected_markers=expected_markers,
        )

    return _validator


# =========================================================================
# =========================================================================
def check_sft_data_health(
    dump_folder: str,
    *,
    vocab_size: int = 151936,
) -> None:
    """Post-training SFT health check: loss finite + reasonable range.

    Extends check_training_health with SFT-specific checks:
    - loss < ln(vocab_size)  (random-init upper bound)
    """
    check_training_health(dump_folder)

    summary = _read_wandb_summary(dump_folder)
    if summary is None:
        return

    avg_loss = summary.get("loss_metrics/global_avg_loss")
    if avg_loss is not None and math.isfinite(avg_loss):
        upper_bound = math.log(vocab_size)
        assert avg_loss < upper_bound, (
            f"SFT loss {avg_loss:.4f} >= ln(vocab_size)={upper_bound:.2f} — "
            f"model may not be learning"
        )
        logger.info(
            f"[check_sft_data_health] PASS: loss={avg_loss:.4f} "
            f"< ln({vocab_size})={upper_bound:.2f}"
        )


def check_moe_health_metrics(dump_folder: str) -> None:
    """Verify MoE health monitor produced metrics in WandB summary.

    When enable_moe_health_monitor=True, MoEHealthPlugin should log
    moe_health/* keys (e.g. moe_health/z_loss, moe_health/max_util).
    """
    summary = _read_wandb_summary(dump_folder)
    if summary is None:
        logger.warning(
            f"[check_moe_health_metrics] SKIP: wandb summary not available ({dump_folder})"
        )
        return

    moe_keys = [k for k in summary if k.startswith("moe_health/")]
    assert len(moe_keys) > 0, (
        "No moe_health/* keys found in WandB summary — "
        "MoEHealthPlugin may not be injected or collect() never fired"
    )
    logger.info(
        f"[check_moe_health_metrics] PASS: found {len(moe_keys)} moe_health keys: "
        f"{moe_keys[:5]}"
    )


def check_eval_metrics(
    dump_folder: str,
    *,
    vocab_size: int = 151936,
) -> None:
    """Post-training eval metrics check: validation loss exists, finite, reasonable.

    Verifies that val_streams produced meaningful eval metrics via torchtitan Validator:
    - validation_metrics/loss exists in WandB summary
    - loss is finite (not NaN/Inf)
    - loss < ln(vocab_size) (sanity upper bound)
    """
    summary = _read_wandb_summary(dump_folder)
    if summary is None:
        logger.warning(
            f"[check_eval_metrics] SKIP: wandb summary not available ({dump_folder})"
        )
        return

    val_loss = summary.get("validation_metrics/loss")
    assert val_loss is not None, (
        "validation_metrics/loss not found in WandB summary — "
        "eval may not have run (check eval_freq/eval_steps config)"
    )
    assert math.isfinite(val_loss), (
        f"validation loss is not finite: {val_loss}"
    )

    upper_bound = math.log(vocab_size)
    assert val_loss < upper_bound, (
        f"val loss {val_loss:.4f} >= ln(vocab_size)={upper_bound:.2f} — "
        f"eval pipeline may be broken"
    )
    logger.info(
        f"[check_eval_metrics] PASS: val_loss={val_loss:.4f} "
        f"< ln({vocab_size})={upper_bound:.2f}"
    )
