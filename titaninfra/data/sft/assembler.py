"""Layer-3 Assembler: Segment list -> EncodedSample.

The assembler is the only place in the new pipeline that talks to the
tokenizer. It walks a flat ``list[Segment]`` produced by a
``FamilyEncoder.encode_segments`` call and emits one ``EncodedSample``,
threading per-token (label, loss_weight, position) emission.

Three modes are supported:

    "sft"
        Full sequence. labels filled per loss_scale; loss_weights only
        emitted when any segment has a non-binary scale.
    "rl_prompt"
        Stops at the first ``is_generation=True`` segment. Returns the
        prompt-side EncodedSample with labels = all IGNORE_INDEX.
    "rl_full"
        Full sequence with completion mask: labels masked on prompt,
        token_id on the response side, regardless of declared scale.

Truncation contract: ``max_length`` is segment-aligned. If a segment
straddles the boundary, ``assemble`` raises ``ValueError`` -- the caller
must rebudget or pre-trim. Silently emitting a half-segment whose
role-marker / content split is broken is worse than failing loud.

Phase 4 (encoder-layer-refactor §4): a merge pass collapses runs of
adjacent text segments sharing the same (kind, role, loss_scale,
token_class, is_generation) tuple before tokenization. This eliminates
the "encoder remembers to plant a linker \\n" contract — the qwen3
fixture 09 bug class (3-vs-2 \\n drift between body and tool_call)
becomes structurally impossible: encoders emit complete logical units;
if their merge keys differ they stay separate by construction;
concatenating their text fields and tokenizing once per merged run
prevents BPE-boundary realignment.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Literal

from titaninfra.data.sft.segment import SegKind, Segment
from titaninfra.data.sft.types import IGNORE_INDEX, EncodedSample

AssemblyMode = Literal["sft", "rl_prompt", "rl_full"]

_VALID_MODES: frozenset[str] = frozenset({"sft", "rl_prompt", "rl_full"})


class TruncatedToEmptyError(ValueError):
    """truncate_at_last_segment=True kept 0 segments — sample is empty.

    Raised when the very first segment exceeds max_length. Parent
    callers (streaming_dataset._make_messages_doc_iter) should treat
    this like EmptyAssistantError: warn + skip the sample, do not
    abort the pipeline.
    """


def _segment_tokens(seg: Segment, tokenizer) -> list[int]:
    """Return token ids for one segment. token_ids path skips tokenize."""
    if seg.token_ids is not None:
        return list(seg.token_ids)
    return list(tokenizer.encode(seg.text, add_special_tokens=False))


def _expand_image_segment(pixel_values, grid_thw, tokenizer, spatial_merge_size):
    """Expand predecoded image tensors into vision marker and pad IDs.

    Decode is already complete upstream. N = grid.prod() // merge^2 uses the
    same invariant as offline token estimation, and all image tokens are masked.
    """
    from titaninfra.data.multimodal.mm_doc_iter import (
        _grid_to_n_tokens,
        _resolve_vision_token_ids,
    )

    merge = spatial_merge_size if spatial_merge_size is not None else 2
    grid_tuple = tuple(int(x) for x in grid_thw.tolist())
    vision_start_id, image_pad_id, vision_end_id = _resolve_vision_token_ids(tokenizer)
    n = _grid_to_n_tokens(grid_tuple, merge)
    token_ids = [vision_start_id] + [image_pad_id] * n + [vision_end_id]
    return token_ids, pixel_values, grid_tuple


def _merge_adjacent_segments(segments: list[Segment]) -> list[Segment]:
    """Collapse runs of same-(kind, role, loss_scale, token_class, is_generation) text segments.

    Inspired by ms-swift's ``_simplify_context_list`` — see spec §4 + memory
    ``reference_industry_sft_pipeline_survey``. Two adjacent segments merge
    iff both:

    * carry ``text`` (not pre-tokenized ``token_ids`` — those skip BPE so
      merging them would defeat the point);
    * share ``(kind, role, loss_scale, token_class, is_generation)``;
    * have the same ``msg_index`` (so per-message diagnostics stay clean
      — different turns never collapse into one boundary).

    Merging concatenates their text. Tokenization then runs once over
    the merged run, so any BPE merge across the seam is naturally
    captured. Encoder authors no longer need to manually plant a linker
    ``\\n`` between adjacent same-kind segments — the segment list itself
    is the source of truth.
    """
    if not segments:
        return segments
    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        can_merge = (
            seg.text is not None
            and prev.text is not None
            and seg.kind == prev.kind
            and seg.role == prev.role
            and seg.loss_scale == prev.loss_scale
            and seg.token_class == prev.token_class
            and seg.is_generation == prev.is_generation
            and seg.msg_index == prev.msg_index
        )
        if can_merge:
            merged[-1] = replace(prev, text=(prev.text or "") + (seg.text or ""))
        else:
            merged.append(seg)
    return merged


def assemble(
    segments: list[Segment],
    tokenizer,
    *,
    mode: AssemblyMode = "sft",
    max_length: int | None = None,
    add_generation_prompt: bool = False,
    truncate_at_last_segment: bool = False,
    spatial_merge_size: int | None = None,
    decoded_images: list | None = None,
) -> EncodedSample:
    """Walk segments -> tokenize -> produce one EncodedSample.

    ``truncate_at_last_segment`` (opt-in, default False) — when True AND
    ``max_length`` would fall mid-segment, keep the largest
    segment-aligned prefix (effective[:kept]) and emit a ``warnings.warn``
    instead of raising. This is the only sanctioned way to do "SFT-style
    truncation": template integrity at every kept boundary, BPE merges
    intact (relies on Phase 4 merge pass), loss-mask invariants preserved
    by construction (whole segments only, no half role markers).
    Default False keeps the legacy fail-loud contract — recommended for
    most workflows so over-long samples surface as data bugs.

    ``add_generation_prompt`` mirrors ``tokenizer.apply_chat_template``'s
    flag but bypasses jinja entirely:

    * ``False`` (default) — assemble exactly what the encoder emitted.
    * ``True`` in ``rl_prompt`` mode — include the FIRST segment marked
      ``is_generation=True`` (the assistant role marker) so the returned
      token stream is byte-identical to vLLM/SGLang's prompt construction
      via jinja with ``add_generation_prompt=True``. The included marker
      keeps ``loss_scale=0`` since the prompt side never carries loss.
    * ``True`` in ``sft`` / ``rl_full`` modes — disallowed; those modes
      already emit the full sequence including the assistant turn, so
      ``add_generation_prompt`` would be a no-op at best and double-emit
      at worst. Raise rather than silently no-op.

    The encoder is the single source of truth for what the generation
    prompt looks like. We deliberately do NOT probe the tokenizer (the
    rollout prompt pattern) because that would couple us to jinja's behavior --
    the whole point of the segment pipeline is that jinja never decides
    what enters training or inference.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"assemble: mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
        )
    if not segments:
        raise ValueError("assemble: segments list must be non-empty")
    if add_generation_prompt and mode != "rl_prompt":
        raise ValueError(
            f"assemble: add_generation_prompt=True requires mode='rl_prompt', "
            f"got mode={mode!r}. The other modes already emit the full "
            "sequence including the assistant turn, so the flag would "
            "double-emit the role marker."
        )

    # rl_prompt truncates at the first generation segment, BEFORE tokenize.
    effective = segments
    if mode == "rl_prompt":
        cut = None
        for i, seg in enumerate(segments):
            if seg.is_generation:
                cut = i
                break
        if cut is None:
            raise ValueError(
                "assemble(mode='rl_prompt'): no segment with is_generation=True; "
                "encoder must mark at least one generation segment."
            )
        if cut == 0:
            raise ValueError(
                "assemble(mode='rl_prompt'): first segment is generation -- "
                "nothing to encode on the prompt side."
            )
        if add_generation_prompt:
            # vLLM-aligned: include the assistant role marker so the
            # returned token IDs match what an inference engine feeds
            # the model. The marker stays loss_scale=0 (prompt side
            # never trains) -- we coerce here in case the encoder
            # author accidentally set a non-zero scale on a marker
            # they also flagged as is_generation.
            marker = segments[cut]
            if marker.loss_scale != 0.0:
                marker = Segment(
                    text=marker.text,
                    token_ids=marker.token_ids,
                    kind=marker.kind,
                    role=marker.role,
                    scale_factor=0.0,
                    is_generation=marker.is_generation,
                    token_class=marker.token_class,
                    msg_index=marker.msg_index,
                )
            effective = list(segments[:cut]) + [marker]
        else:
            effective = list(segments[:cut])

    # Phase 4 merge pass — see _merge_adjacent_segments docstring + spec §4.
    # Runs AFTER the rl_prompt cut so prompt and response don't collapse.
    # Seam BPE stability across cross-class boundaries (e.g. CONTENT|STOP_MARKER)
    # is guaranteed by encoders landing every segment boundary on a special
    # token (BPE never merges across one) plus this merge pass collapsing
    # same-metadata runs; there is no separate seam-repair step. The
    # test_encoder_segment_seam suite pins this per family.
    effective = _merge_adjacent_segments(list(effective))

    # First pass: tokenize each segment, decide whether non-binary weights are needed.
    seg_tokens: list[list[int]] = []
    non_binary = False
    _pixel_parts: list = []
    _grid_all: list[tuple[int, int, int]] = []
    _img_cursor = 0
    _n_image_segs = sum(1 for s in effective if s.kind is SegKind.IMAGE)
    if _n_image_segs > 0 and decoded_images is None:
        raise ValueError(
            f"assemble: {_n_image_segs} IMAGE segment(s) present but decoded_images "
            "is None. Image messages streams must pre-decode via ImageDecodePool "
            "(spec 2026-07-29); assembler no longer decodes inline."
        )
    for seg in effective:
        if seg.kind is SegKind.IMAGE:
            if _img_cursor >= len(decoded_images):
                raise ValueError(
                    f"assemble: more IMAGE segments than decoded_images "
                    f"({_img_cursor + 1} > {len(decoded_images)}) — segment/decode "
                    "order desync; image-text interleaving would be wrong."
                )
            pv, gthw = decoded_images[_img_cursor]
            _img_cursor += 1
            ids, pv, gthw_tuple = _expand_image_segment(pv, gthw, tokenizer, spatial_merge_size)
            _pixel_parts.append(pv)
            _grid_all.append(gthw_tuple)
        else:
            ids = _segment_tokens(seg, tokenizer)
        if not ids:
            # Empty segments would create role/marker drift; reject so the
            # encoder author notices instead of silently dropping a turn.
            raise ValueError(
                f"assemble: segment produced 0 tokens (kind={seg.kind}, "
                f"role={seg.role}, text={seg.text!r}, token_ids={seg.token_ids!r})"
            )
        seg_tokens.append(ids)
        if seg.loss_scale not in (0.0, 1.0):
            non_binary = True
    if decoded_images is not None and _img_cursor != len(decoded_images):
        raise ValueError(
            f"assemble: consumed {_img_cursor} images but decoded_images has "
            f"{len(decoded_images)} — segment/decode count mismatch."
        )


    # max_length: segment-aligned truncation; mid-segment is fail-loud.
    if max_length is not None:
        total = sum(len(t) for t in seg_tokens)
        if total > max_length:
            # Find cutoff: largest prefix whose cumulative length <= max_length.
            cum = 0
            kept = 0
            for ids in seg_tokens:
                if cum + len(ids) > max_length:
                    break
                cum += len(ids)
                kept += 1
            if cum != max_length:
                # The next segment would have straddled the boundary.
                # Two paths: legacy fail-loud (default) vs. opt-in
                # truncate_at_last_segment (keep effective[:kept]).
                if not truncate_at_last_segment:
                    raise ValueError(
                        f"assemble: max_length={max_length} falls mid-segment "
                        f"(cumulative {cum} after {kept} segments, next segment "
                        f"adds {len(seg_tokens[kept]) if kept < len(seg_tokens) else 0}). "
                        "Caller must pre-trim, raise max_length, or set "
                        "truncate_at_last_segment=True (opt-in SFT-style truncation)."
                    )
                # Opt-in SFT-style truncation. RL modes are disallowed:
                # truncating a prompt destroys prompt-response correspondence
                # (rollout vs training would diverge → KL misalignment).
                if mode != "sft":
                    raise ValueError(
                        f"assemble: truncate_at_last_segment=True is only valid "
                        f"for mode='sft', got mode={mode!r}. RL modes need exact "
                        "prompt boundaries; pre-trim or raise max_length instead."
                    )
                # kept==0 → even the first segment exceeds max_length. We
                # raise a sentinel that callers like streaming_dataset
                # treat as "skip this sample" (parallel to EmptyAssistantError).
                if kept == 0:
                    first_seg = effective[0]
                    raise TruncatedToEmptyError(
                        f"assemble: first segment alone is {len(seg_tokens[0])} "
                        f"tokens > max_length={max_length} "
                        f"(kind={first_seg.kind} role={first_seg.role}). "
                        "Sample dropped under truncate_at_last_segment=True."
                    )
                dropped_seg = effective[kept]
                pct = round(100 * cum / max_length)
                warnings.warn(
                    f"assemble: truncated SFT sample at segment boundary "
                    f"(mode={mode!r}, {cum}/{max_length} tokens kept ≈ {pct}%, "
                    f"dropped {len(seg_tokens) - kept}/{len(seg_tokens)} segments, "
                    f"first dropped: kind={dropped_seg.kind} role={dropped_seg.role}). "
                    "Set truncate_at_last_segment=False to fail loud instead.",
                    stacklevel=2,
                )
                # fall through to "effective = effective[:kept]" below
            effective = effective[:kept]
            seg_tokens = seg_tokens[:kept]
            # If truncation dropped non-binary segments, recompute the flag.
            non_binary = any(
                seg.loss_scale not in (0.0, 1.0) for seg in effective
            )
            # Audit #4 (2026-06-06): truncation may land AFTER the prompt but
            # BEFORE the first loss-bearing segment, leaving every retained
            # segment at loss_scale==0.0 -> an all-IGNORE_INDEX sample with
            # zero training signal. Previously this only warned and fell
            # through, silently emitting a dead sample (streaming_dataset's
            # guard only checks for empty input_ids, not zero loss). Raise the
            # same sentinel the kept==0 path uses so callers skip it.
            if mode == "sft" and not any(s.loss_scale > 0.0 for s in effective):
                raise TruncatedToEmptyError(
                    f"assemble: truncation to max_length={max_length} retained "
                    f"{kept} segment(s) but none carry loss (all loss_scale==0.0) "
                    "-> the sample would train on nothing. Dropped."
                )

    # Identify the prompt/response cut for rl_full (where to start emitting labels).
    rl_full_gen_seg_idx: int | None = None
    if mode == "rl_full":
        for i, seg in enumerate(effective):
            if seg.is_generation:
                rl_full_gen_seg_idx = i
                break
        if rl_full_gen_seg_idx is None:
            raise ValueError(
                "assemble(mode='rl_full'): no segment with is_generation=True."
            )

    # Walk and emit.
    input_ids: list[int] = []
    labels: list[int] = []
    loss_weights: list[float] = [] if non_binary else []

    for seg_idx, (seg, ids) in enumerate(zip(effective, seg_tokens, strict=False)):
        input_ids.extend(ids)

        # Decide effective scale for this segment under the current mode.
        if mode == "rl_prompt":
            effective_scale = 0.0  # all prompt
        elif mode == "rl_full":
            effective_scale = seg.loss_scale if seg_idx >= rl_full_gen_seg_idx else 0.0  # type: ignore[operator]
        else:  # sft
            effective_scale = seg.loss_scale

        if effective_scale == 0.0:
            labels.extend([IGNORE_INDEX] * len(ids))
            if non_binary:
                loss_weights.extend([0.0] * len(ids))
        else:
            labels.extend(ids)
            if non_binary:
                loss_weights.extend([effective_scale] * len(ids))

    n = len(input_ids)
    positions = tuple(range(n))
    weights_tuple: tuple[float, ...] | None = (
        tuple(loss_weights) if non_binary else None
    )

    if _pixel_parts:
        import torch
        _pv = torch.cat(_pixel_parts, dim=0)
        _gt = _grid_all
    else:
        _pv = None
        _gt = None
    return EncodedSample(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        positions=positions,
        loss_weights=weights_tuple,
        pixel_values=_pv,
        grid_thw=_gt,
    )


def split_prompt_response(
    segments: list[Segment],
) -> tuple[list[Segment], list[Segment]]:
    """Partition segments at the first ``is_generation=True`` boundary."""
    for i, seg in enumerate(segments):
        if seg.is_generation:
            return list(segments[:i]), list(segments[i:])
    raise ValueError(
        "split_prompt_response: no segment with is_generation=True; "
        "encoder must mark at least one generation segment for RL modes."
    )


__all__ = ["AssemblyMode", "assemble", "split_prompt_response"]
