"""Family-dispatched L2 divergence classification for renderers-oracle tests.

``_jinja_oracle.diff.compare_encoded`` gives a raw L1 (``input_ids``)/L2
(``labels``) diff. Once a caller has established identical L1 token sequences,
this module classifies *every* L2-divergent position using the selected
family's declared, documented ``renderers`` deviation patterns. A position
that matches no pattern is ``"UNEXPLAINED"`` and must fail the calling test;
it is a real signal, not noise to filter away.

Adding a family means registering its pattern detectors in
``_FAMILY_PATTERNS`` below. Detectors receive the complete token sequence and
return all positions belonging to that pattern, so multi-token spans are found
by a full forward scan rather than a sampling or fixed lookback window.
Families with no known L2 deviations (currently llama3 and deepseek_v3) still
register explicitly with an empty pattern tuple: every divergence is then
unexplained.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Pattern = str  # family-specific pattern name | "UNEXPLAINED"
PatternDetector = Callable[[list[int], Any], set[int]]


def _think_envelope_positions(ids: list[int], tokenizer) -> set[int]:
    """Every ``<think>``, ``\\n\\n``, ``</think>`` 3-token empty-envelope span.

    A full forward scan (not a backward-only lookback window) is required:
    the opener token's own classification depends on a *later* token (the
    closer), so a single-pass per-position lookback would misclassify the
    opener and the inner newline. See the 2026-11 feasibility investigation
    for the bug this produces if done wrong.
    """
    think_open = tokenizer.convert_tokens_to_ids("<think>")
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    newline_double = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    positions: set[int] = set()
    for i in range(len(ids) - 2):
        if ids[i] == think_open and ids[i + 1] == newline_double and ids[i + 2] == think_close:
            positions.update((i, i + 1, i + 2))
    return positions


def _trailing_newline_after_im_end_positions(ids: list[int], tokenizer) -> set[int]:
    """Every ``\\n`` immediately following an ``<|im_end|>`` token."""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    newline_single = tokenizer.encode("\n", add_special_tokens=False)[0]
    return {i for i in range(1, len(ids)) if ids[i] == newline_single and ids[i - 1] == im_end}


def _assistant_analysis_or_final_end_positions(ids: list[int], tokenizer) -> set[int]:
    """Every ``<|end|>`` closing an assistant analysis/final Harmony message.

    Scan complete messages in the forward direction. The closer's category
    depends on its *earlier* channel header, while finding message boundaries
    also requires looking ahead to the closer; neither a fixed lookback window
    nor a per-token regular expression is sufficient. ``<|return|>`` and tool
    ``<|call|>`` are deliberately excluded, as are user/system/tool ``<|end|>``
    markers.
    """
    start = tokenizer.convert_tokens_to_ids("<|start|>")
    message = tokenizer.convert_tokens_to_ids("<|message|>")
    end = tokenizer.convert_tokens_to_ids("<|end|>")
    return_id = tokenizer.convert_tokens_to_ids("<|return|>")
    call = tokenizer.convert_tokens_to_ids("<|call|>")
    stop_ids = {end, return_id, call}
    positions: set[int] = set()
    cursor = 0
    while cursor < len(ids):
        if ids[cursor] != start:
            cursor += 1
            continue
        close = next(
            (i for i in range(cursor + 1, len(ids)) if ids[i] in stop_ids),
            None,
        )
        if close is None:
            break
        try:
            message_pos = ids.index(message, cursor + 1, close)
        except ValueError:
            cursor = close + 1
            continue
        header = tokenizer.decode(
            ids[cursor:message_pos + 1], skip_special_tokens=False
        )
        is_assistant_body = header.startswith("<|start|>assistant") and (
            "<|channel|>analysis<|message|>" in header
            or "<|channel|>final<|message|>" in header
        )
        if ids[close] == end and is_assistant_body:
            positions.add(close)
        cursor = close + 1
    return positions


def _gemma4_assistant_turn_newline_positions(ids: list[int], tokenizer) -> set[int]:
    """Every newline immediately after a model turn's ``<turn|>`` marker.

    Track the current wire role through the complete sequence. A continuation
    after a tool response has no fresh ``<|turn>model`` opener, so role state
    changes only at an explicit turn opener, never at a turn close.
    """
    turn_start = tokenizer.convert_tokens_to_ids("<|turn>")
    turn_end = tokenizer.convert_tokens_to_ids("<turn|>")
    model_prefix = tokenizer.encode("model\n", add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline) == 1
    newline = newline[0]
    in_model_turn = False
    positions: set[int] = set()
    for i, token_id in enumerate(ids):
        if token_id == turn_start:
            in_model_turn = ids[i + 1:i + 1 + len(model_prefix)] == model_prefix
        if (
            in_model_turn
            and token_id == turn_end
            and i + 1 < len(ids)
            and ids[i + 1] == newline
        ):
            positions.add(i + 1)
    return positions


def _gemma4_initial_tool_response_open_positions(ids: list[int], tokenizer) -> set[int]:
    """The first response opener in each actual tool-result run, not a stop marker.

    A terminal ``<|tool_response>`` sampled to request a tool call has no
    matching ``<tool_response|>`` and is excluded. Subsequent response blocks
    in one parallel-result run are excluded by the immediately preceding close.
    """
    response_start = tokenizer.convert_tokens_to_ids("<|tool_response>")
    response_end = tokenizer.convert_tokens_to_ids("<tool_response|>")
    turn_end = tokenizer.convert_tokens_to_ids("<turn|>")
    stop_ids = {response_start, response_end, turn_end}
    positions: set[int] = set()
    for i, token_id in enumerate(ids):
        if token_id != response_start:
            continue
        if i > 0 and ids[i - 1] == response_end:
            continue
        next_stop = next(
            (ids[j] for j in range(i + 1, len(ids)) if ids[j] in stop_ids),
            None,
        )
        if next_stop == response_end:
            positions.add(i)
    return positions


def _gemma4_thought_scaffold_positions(ids: list[int], tokenizer) -> set[int]:
    """Opening/closing scaffold of every non-empty ``thought`` channel.

    The detector scans the whole envelope before classifying its opener. It
    deliberately excludes the reasoning body and the empty generation prefill
    (which callers handle as a separately asserted raw-L1 span).
    """
    channel_start = tokenizer.convert_tokens_to_ids("<|channel>")
    channel_end = tokenizer.convert_tokens_to_ids("<channel|>")
    thought_prefix = tokenizer.encode("thought\n", add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline) == 1
    newline = newline[0]
    positions: set[int] = set()
    cursor = 0
    while cursor < len(ids):
        if ids[cursor] != channel_start:
            cursor += 1
            continue
        body_start = cursor + 1 + len(thought_prefix)
        if ids[cursor + 1:body_start] != thought_prefix:
            cursor += 1
            continue
        try:
            close = ids.index(channel_end, body_start)
        except ValueError:
            break
        if close == body_start:  # empty thought prefill, not reasoning scaffold
            cursor = close + 1
            continue
        positions.update(range(cursor, body_start))
        if ids[close - 1] == newline:
            positions.add(close - 1)
        positions.add(close)
        cursor = close + 1
    return positions


_FAMILY_PATTERNS: dict[str, tuple[tuple[str, PatternDetector], ...]] = {
    "qwen3": (
        ("A", _think_envelope_positions),
        ("B", _trailing_newline_after_im_end_positions),
    ),
    # Llama3 has no accepted L2 deviations after the caller canonicalizes its
    # separately-documented L1 metadata spans. Do not add a catch-all here:
    # any native/renderers label difference must remain visible as UNEXPLAINED.
    "llama3": (),
    # DeepSeek-V3 likewise has zero accepted core-L2 deviations. Its two raw-L1
    # tool-wire spans are exactly asserted and removed by the caller before this
    # classifier runs; no label difference is tolerated in the shared core.
    "deepseek_v3": (),
    # Native masks analysis-channel closers and non-terminal final-channel
    # <|end|> markers as protocol; renderers treats every assistant terminator
    # as model-sampled. All other core-L2 differences remain UNEXPLAINED.
    "gpt_oss": (
        (
            "renderers_trains_assistant_end",
            _assistant_analysis_or_final_end_positions,
        ),
    ),
    # Gemma4 has three narrowly-scoped core-L2 design differences. Raw-L1
    # thought/tool serialization spans and the fixture-08 renderer bug are
    # asserted and removed by the caller before these detectors run.
    "gemma4": (
        (
            "renderers_masks_assistant_turn_close_newline",
            _gemma4_assistant_turn_newline_positions,
        ),
        (
            "renderers_trains_initial_tool_response_open",
            _gemma4_initial_tool_response_open_positions,
        ),
        (
            "renderers_trains_thought_channel_scaffold",
            _gemma4_thought_scaffold_positions,
        ),
    ),
}


def classify_label_divergence(
    *,
    ids: tuple[int, ...],
    native_labels: tuple[int, ...],
    renderers_labels: tuple[int, ...],
    tokenizer,
    family: str = "qwen3",
) -> dict[str, Any]:
    """Classify every position where ``native_labels`` != ``renderers_labels``.

    Requires ``ids`` to be the (already-confirmed-identical) shared token
    sequence — callers must check L1 equality themselves first; this
    function does not re-derive or validate it.

    ``family`` defaults to qwen3 for backward compatibility with the original
    prototype call sites. An unregistered family raises ``ValueError`` rather
    than silently reporting all positions as unexplained.

    Returns a dict with:
      ``n_divergent``: total divergent positions.
      ``counts``: per-pattern counts plus ``UNEXPLAINED`` when applicable
      (only present keys).
      ``unexplained_positions``: list of ``(index, token_id, native_label,
      renderers_label)`` tuples for every position matched by no pattern.
    """
    try:
        pattern_specs = _FAMILY_PATTERNS[family]
    except KeyError as exc:
        raise ValueError(f"no renderers divergence patterns registered for {family!r}") from exc

    ids_list = list(ids)
    # Each detector sees the full sequence before any position is classified.
    # This is intentional: qwen3 Pattern A's opener depends on a later closer.
    pattern_positions = {
        name: detector(ids_list, tokenizer) for name, detector in pattern_specs
    }

    counts: dict[str, int] = {}
    unexplained: list[tuple[int, int, int, int]] = []
    for i, (n, r) in enumerate(zip(native_labels, renderers_labels, strict=True)):
        if n == r:
            continue
        cat = next(
            (name for name, positions in pattern_positions.items() if i in positions),
            "UNEXPLAINED",
        )
        if cat == "UNEXPLAINED":
            unexplained.append((i, ids_list[i], n, r))
        counts[cat] = counts.get(cat, 0) + 1

    return {
        "n_divergent": sum(counts.values()),
        "counts": counts,
        "unexplained_positions": unexplained,
    }


__all__ = ["classify_label_divergence"]
