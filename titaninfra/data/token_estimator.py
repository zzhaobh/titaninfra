"""
Token / epoch-size estimation — pure functions, no side effects.

All I/O (reading stream_stats.json, logging, modifying trainer state) is the
caller's responsibility.  Functions here take numbers in, return numbers out.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from titaninfra.config.data import PackStrategy


# ============================================================================
# Return types
# ============================================================================


@dataclasses.dataclass(frozen=True)
class StreamTokenEstimate:
    stream_id: str
    epoch_size: int | None
    steps_per_epoch: int | None
    estimated_tokens_per_rank: float
    data_passes: float
    packing_efficiency: float = 1.0


@dataclasses.dataclass(frozen=True)
class TrainingEstimate:
    total_steps: int
    steps_per_epoch: int
    per_stream: list[StreamTokenEstimate]


# ============================================================================
# Helpers
# ============================================================================

_SAMPLE_BASED: frozenset[str] = frozenset({"no_pack"})


def _is_sample_based(pack_strategy: PackStrategy | str) -> bool:
    val = pack_strategy.value if hasattr(pack_strategy, "value") else str(pack_strategy)
    return val in _SAMPLE_BASED


def _compute_weights(
    stream_configs: list,
    stream_stats: list[dict | None] | None = None,
    pe_adjustments: list[float] | None = None,
    seq_len: int | None = None,
) -> list[float]:
    """Compute normalized per-stream mixing weights (pure — no file I/O).

    Mirrors ``streaming_dataset._compute_per_stream_weights`` but avoids
    importing the heavy ``streaming_dataset`` module (which pulls torchtitan).
    For ``repeat`` mode, uses ``stream_stats`` instead of reading from disk.

    When ``pe_adjustments`` is provided, repeat/choose raw weights are divided
    by each stream's packing efficiency so that all streams exhaust their
    target sample count at the same training step.

    Raw weights are computed in **batch-slot units** so that packed and
    sample-based (no_pack) streams are in the same space:
    - packed: ``choose × avg / (seq_len × PE)``  (tokens → batch slots)
    - no_pack: ``choose``  (1 sample = 1 batch slot)
    When all streams share the same packing type, ``seq_len`` cancels in
    normalization and can be omitted.
    """
    raw: list[float] = []
    for i, s in enumerate(stream_configs):
        if s.proportion is not None:
            raw.append(s.proportion)
        elif getattr(s, "repeat", None) is not None:
            n = 1000
            avg = 1.0
            if stream_stats and i < len(stream_stats) and stream_stats[i]:
                n = stream_stats[i].get("num_samples", 1000)
                avg = stream_stats[i].get("avg_tokens_per_sample", 1.0) or 1.0
            sample_based = _is_sample_based(s.pack_strategy)
            w = s.repeat * n * (1.0 if sample_based else avg)
            if not sample_based and pe_adjustments and i < len(pe_adjustments) and pe_adjustments[i] > 0:
                w /= pe_adjustments[i]
            if not sample_based and seq_len and seq_len > 0:
                w /= seq_len
            raw.append(w)
        elif getattr(s, "choose", None) is not None:
            avg = 1.0
            if stream_stats and i < len(stream_stats) and stream_stats[i]:
                avg = stream_stats[i].get("avg_tokens_per_sample", 1.0) or 1.0
            sample_based = _is_sample_based(s.pack_strategy)
            w = float(s.choose) * (1.0 if sample_based else avg)
            if not sample_based and pe_adjustments and i < len(pe_adjustments) and pe_adjustments[i] > 0:
                w /= pe_adjustments[i]
            if not sample_based and seq_len and seq_len > 0:
                w /= seq_len
            raw.append(w)
        else:
            raw.append(1.0)
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [w / total for w in raw]


# ============================================================================
# Packing efficiency estimation
# ============================================================================

_SENTENCE_BOUNDARY_MAX_ROLLBACK = 100


def estimate_packing_efficiency(
    fine_token_length_dist: dict[str, int] | None,
    seq_len: int,
    pack_strategy: PackStrategy | str = "hard_truncate",
) -> float:
    """Estimate packing efficiency from a fine-grained token length histogram.

    Returns a float in (0, 1] representing the ratio of useful tokens to total
    token slots (num_sequences × seq_len).  Used to correct epoch_size estimates.

    - HARD_TRUNCATE: continuous buffer, no waste → 1.0
    - SENTENCE_BOUNDARY: up to 100-token rollback per sequence → fixed correction
    - FULL_SAMPLE: bin-packing complete documents → greedy simulation from histogram
    - NO_PACK: sample-based path, efficiency not applicable → 1.0
    """
    val = pack_strategy.value if hasattr(pack_strategy, "value") else str(pack_strategy)

    if val in ("hard_truncate", "no_pack"):
        return 1.0

    if val == "sentence_boundary":
        rollback = min(_SENTENCE_BOUNDARY_MAX_ROLLBACK, seq_len * 0.05)
        return 1.0 - rollback / seq_len

    if val != "full_sample":
        return 1.0

    if not fine_token_length_dist:
        return 1.0

    buckets = _parse_fine_hist(fine_token_length_dist)
    if not buckets:
        return 1.0

    return _simulate_full_sample_packing(buckets, seq_len)


def _parse_fine_hist(hist: dict[str, int]) -> list[tuple[int, int]]:
    """Parse fine histogram into (midpoint, count) pairs, sorted by midpoint."""
    result = []
    for key, count in hist.items():
        if count <= 0:
            continue
        if key.endswith("+"):
            lo = int(key[:-1])
            result.append((lo + 32, count))
        else:
            parts = key.split("-")
            lo, hi = int(parts[0]), int(parts[1])
            result.append(((lo + hi) // 2, count))
    result.sort(key=lambda x: x[0])
    return result


def _simulate_full_sample_packing(
    buckets: list[tuple[int, int]],
    seq_len: int,
    num_trials: int = 5,
    max_samples: int = 10000,
) -> float:
    """Streaming bin-packing simulation matching runtime _flush_sft() behavior.

    Unlike offline FFD (which can backfill gaps with later small docs), the
    runtime packer processes documents sequentially: when a new doc causes
    overflow, the accumulated prefix is emitted and the new doc starts the
    next bin.  This means a bin with 500 tokens of padding will NOT be filled
    by a future 400-token doc — that space is wasted.

    Multiple trials with different random orderings are averaged to account
    for shuffle variance.  Deterministic seeds ensure all ranks get identical
    results.
    """
    import random

    target = seq_len + 1

    docs: list[int] = []
    for mid, count in buckets:
        doc_len = max(mid, 1)
        if doc_len <= target:
            docs.extend([doc_len] * count)

    if not docs:
        return 1.0

    if len(docs) > max_samples:
        rng = random.Random(42)
        docs = rng.sample(docs, max_samples)

    total_pe = 0.0

    for trial in range(num_trials):
        rng = random.Random(trial * 7919)
        trial_docs = docs.copy()
        rng.shuffle(trial_docs)

        total_useful = 0
        num_bins = 0
        current_fill = 0

        for doc_len in trial_docs:
            if current_fill + doc_len <= target:
                current_fill += doc_len
            else:
                if current_fill > 0:
                    total_useful += current_fill
                    num_bins += 1
                current_fill = doc_len

        if current_fill > 0:
            total_useful += current_fill
            num_bins += 1

        if num_bins == 0:
            total_pe += 1.0
        else:
            total_pe += total_useful / (num_bins * seq_len)

    efficiency = total_pe / num_trials
    return max(0.1, min(1.0, efficiency))


# ============================================================================
# Core: per-stream epoch_size
# ============================================================================


def estimate_epoch_size(
    *,
    num_samples: int,
    avg_tokens_per_sample: float,
    pack_strategy: PackStrategy | str,
    normalized_weight: float,
    training_steps: int | None,
    seq_len: int,
    local_batch_size: int,
    gas: int,
    dp_world_size: int,
    mosaicml_world_size: int,
    margin: float,
    calibration_steps: int = 0,
    prefetch_batches: int = 0,
    packing_efficiency: float = 1.0,
) -> int | None:
    """Estimate ``epoch_size`` for a single stream's ``StreamingDataset``.

    *prefetch_batches* = ``num_workers × prefetch_factor`` from the
    DataLoader.  When ``avg_tokens >> seq_len``, one raw sample produces
    many packed batches, so the training-only estimate rounds to very few
    samples.  The prefetch buffer consumes packed batches ahead of training;
    we add the equivalent raw-sample demand so the epoch never resets
    mid-prefetch (which stalls MosaicML's shard coordinator → NCCL timeout).

    Returns ``None`` when *training_steps* is not given
    (caller should let MosaicML decide).

    Raises ``ValueError`` on invalid inputs.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if avg_tokens_per_sample <= 0:
        raise ValueError(f"avg_tokens_per_sample must be > 0, got {avg_tokens_per_sample}")
    if gas < 1:
        raise ValueError(f"gas must be >= 1, got {gas}")
    if dp_world_size < 1:
        raise ValueError(f"dp_world_size must be >= 1, got {dp_world_size}")
    if mosaicml_world_size < 1:
        raise ValueError(f"mosaicml_world_size must be >= 1, got {mosaicml_world_size}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if local_batch_size < 1:
        raise ValueError(f"local_batch_size must be >= 1, got {local_batch_size}")

    weight = max(normalized_weight, 1e-6)
    sample_based = _is_sample_based(pack_strategy)

    # ------------------------------------------------------------------
    # Determine effective_steps (including calibration)
    # ------------------------------------------------------------------
    effective_steps: int | None = None

    if training_steps is not None:
        effective_steps = training_steps
    else:
        return None

    effective_steps += calibration_steps

    # ------------------------------------------------------------------
    # Per-rank token demand → global epoch_size
    # ------------------------------------------------------------------
    if sample_based:
        samples_per_rank = effective_steps * local_batch_size * gas * weight
        if prefetch_batches > 0:
            samples_per_rank += prefetch_batches
        epoch_size = math.ceil(samples_per_rank * mosaicml_world_size * (1.0 + margin))
    else:
        tokens_per_rank = effective_steps * local_batch_size * gas * seq_len * weight
        samples_per_rank = tokens_per_rank / (avg_tokens_per_sample * packing_efficiency)
        if prefetch_batches > 0:
            packing_ratio = max(avg_tokens_per_sample / seq_len, 1.0)
            prefetch_samples = math.ceil(prefetch_batches / packing_ratio)
            samples_per_rank += prefetch_samples
        epoch_size = math.ceil(samples_per_rank * mosaicml_world_size * (1.0 + margin))

    return epoch_size


# ============================================================================


def estimate_total_steps_from_repeat(
    *,
    stream_stats: list[dict | None],
    stream_configs: list,
    local_batch_size: int,
    gas: int,
    seq_len: int,
    dp_world_size: int,
) -> TrainingEstimate | None:
    """Estimate ``total_steps`` when every stream uses ``repeat``.

    Each stream's contribution is ``steps_per_epoch × repeat``.  The
    bottleneck stream (highest per-stream total steps) determines
    ``total_steps``.

    Returns ``None`` when no usable stats exist or not all streams
    have ``repeat`` set.
    """
    if not stream_configs or not stream_stats:
        return None

    for cfg in stream_configs:
        if getattr(cfg, "repeat", None) is None:
            return None

    # Pass 1: compute per-stream PE for weight adjustment
    pe_list: list[float] = []
    for cfg, stats in zip(stream_configs, stream_stats, strict=False):
        if stats is None or stats.get("num_samples", 0) <= 0:
            pe_list.append(1.0)
            continue
        avg_tokens = stats.get("avg_tokens_per_sample", 0.0)
        if avg_tokens <= 0:
            pe_list.append(1.0)
            continue
        if _is_sample_based(cfg.pack_strategy):
            pe_list.append(1.0)
        else:
            pe_list.append(estimate_packing_efficiency(
                stats.get("fine_token_length_dist"), seq_len, cfg.pack_strategy,
            ))

    # Pass 2: PE-adjusted weights ensure all streams exhaust simultaneously
    weights = _compute_weights(stream_configs, stream_stats, pe_adjustments=pe_list, seq_len=seq_len)
    per_stream: list[StreamTokenEstimate] = []
    bottleneck_total = 0

    for cfg, stats, w, pe in zip(stream_configs, stream_stats, weights, pe_list, strict=False):
        sid = cfg.resolve_stream_id() if hasattr(cfg, "resolve_stream_id") else str(cfg.local)
        if stats is None or stats.get("num_samples", 0) <= 0:
            per_stream.append(StreamTokenEstimate(
                stream_id=sid, epoch_size=None, steps_per_epoch=None,
                estimated_tokens_per_rank=0.0, data_passes=0.0,
            ))
            continue

        num_samples = stats["num_samples"]
        avg_tokens = stats.get("avg_tokens_per_sample", 0.0)
        if avg_tokens <= 0:
            per_stream.append(StreamTokenEstimate(
                stream_id=sid, epoch_size=None, steps_per_epoch=None,
                estimated_tokens_per_rank=0.0, data_passes=0.0,
            ))
            continue

        sample_based = _is_sample_based(cfg.pack_strategy)
        if sample_based:
            samples_per_rank = num_samples / max(dp_world_size, 1)
            samples_per_step = local_batch_size * gas * max(w, 1e-6)
            steps_per_epoch_exact = samples_per_rank / max(samples_per_step, 1e-9)
        else:
            tokens_per_rank = num_samples * avg_tokens / max(dp_world_size, 1)
            tokens_per_step = local_batch_size * gas * seq_len * max(w, 1e-6) * pe
            steps_per_epoch_exact = tokens_per_rank / max(tokens_per_step, 1e-9)

        steps_per_epoch = max(1, math.ceil(steps_per_epoch_exact))
        stream_total_steps = max(1, math.ceil(steps_per_epoch * cfg.repeat))
        estimated_tokens = stream_total_steps * local_batch_size * gas * seq_len * max(w, 1e-6)
        data_passes = float(cfg.repeat)

        per_stream.append(StreamTokenEstimate(
            stream_id=sid,
            epoch_size=None,
            steps_per_epoch=steps_per_epoch,
            estimated_tokens_per_rank=estimated_tokens,
            data_passes=data_passes,
            packing_efficiency=pe,
        ))

        if stream_total_steps > bottleneck_total:
            bottleneck_total = stream_total_steps

    if bottleneck_total == 0:
        return None

    return TrainingEstimate(
        total_steps=bottleneck_total,
        steps_per_epoch=bottleneck_total,
        per_stream=per_stream,
    )


# ============================================================================
# Aggregate: multi-stream total_steps (for all-choose mode)
# ============================================================================


def estimate_total_steps_from_choose(
    *,
    stream_stats: list[dict | None],
    stream_configs: list,
    local_batch_size: int,
    gas: int,
    seq_len: int,
    dp_world_size: int,
) -> TrainingEstimate | None:
    """Estimate ``total_steps`` when every stream uses ``choose``.

    Each stream specifies an absolute sample count via ``choose=K``.
    Unlike ``repeat``, ``choose`` already defines the total samples
    (no epoch multiplier).  The bottleneck stream (highest steps)
    determines ``total_steps``.

    ``avg_tokens_per_sample`` from stream_stats is still needed to
    convert sample counts into training steps (for packing modes).

    Returns ``None`` when no usable stats exist or not all streams
    have ``choose`` set.
    """
    if not stream_configs or not stream_stats:
        return None

    for cfg in stream_configs:
        if getattr(cfg, "choose", None) is None:
            return None

    # Pass 1: compute per-stream PE for weight adjustment
    pe_list: list[float] = []
    for cfg, stats in zip(stream_configs, stream_stats, strict=False):
        if stats is None or stats.get("avg_tokens_per_sample", 0.0) <= 0:
            pe_list.append(1.0)
            continue
        if _is_sample_based(cfg.pack_strategy):
            pe_list.append(1.0)
        else:
            pe_list.append(estimate_packing_efficiency(
                stats.get("fine_token_length_dist"), seq_len, cfg.pack_strategy,
            ))

    # Pass 2: PE-adjusted weights ensure all streams exhaust simultaneously
    weights = _compute_weights(stream_configs, stream_stats, pe_adjustments=pe_list, seq_len=seq_len)
    per_stream: list[StreamTokenEstimate] = []
    bottleneck_total = 0

    for cfg, stats, w, pe in zip(stream_configs, stream_stats, weights, pe_list, strict=False):
        sid = cfg.resolve_stream_id() if hasattr(cfg, "resolve_stream_id") else str(cfg.local)
        if stats is None or stats.get("avg_tokens_per_sample", 0.0) <= 0:
            per_stream.append(StreamTokenEstimate(
                stream_id=sid, epoch_size=None, steps_per_epoch=None,
                estimated_tokens_per_rank=0.0, data_passes=0.0,
            ))
            continue

        choose_samples = cfg.choose
        avg_tokens = stats["avg_tokens_per_sample"]
        num_samples = stats.get("num_samples", choose_samples)

        sample_based = _is_sample_based(cfg.pack_strategy)
        if sample_based:
            samples_per_rank = choose_samples / max(dp_world_size, 1)
            samples_per_step = local_batch_size * gas * max(w, 1e-6)
            steps = max(1, math.ceil(samples_per_rank / max(samples_per_step, 1e-9)))
        else:
            total_tokens = choose_samples * avg_tokens
            tokens_per_rank = total_tokens / max(dp_world_size, 1)
            tokens_per_step = local_batch_size * gas * seq_len * max(w, 1e-6) * pe
            steps = max(1, math.ceil(tokens_per_rank / max(tokens_per_step, 1e-9)))

        estimated_tokens = steps * local_batch_size * gas * seq_len * max(w, 1e-6)
        data_passes = cfg.choose / num_samples if num_samples > 0 else 0.0

        per_stream.append(StreamTokenEstimate(
            stream_id=sid,
            epoch_size=None,
            steps_per_epoch=steps,
            estimated_tokens_per_rank=estimated_tokens,
            data_passes=data_passes,
            packing_efficiency=pe,
        ))

        if steps > bottleneck_total:
            bottleneck_total = steps

    if bottleneck_total == 0:
        return None

    return TrainingEstimate(
        total_steps=bottleneck_total,
        steps_per_epoch=bottleneck_total,
        per_stream=per_stream,
    )
