"""
SFTSample (raw on-disk JSON) → NormalizedSample (in-memory contract).

This module is the **only** place that knows about titaninfra's rich
``blocks`` / ``content`` / ``tool_use`` / ``thinking`` shape. Every
encoder downstream sees the canonical ``NormalizedSample`` form and
nothing else. Family-specific quirks (Qwen3.5 ``arguments: dict``
vs HF-standard ``arguments: str``) are deferred to the per-family
encoder; they are NOT re-encoded here.

Phase 2 (encoder-layer-refactor):
  * Populates the new ``blocks`` tuple alongside the legacy text /
    thinking / tool_calls fields. Encoders migrating to the blocks-
    only path read ``nm.blocks`` directly; un-migrated encoders
    continue to read the legacy fields.
  * Applies the effective ``history_policy`` (resolved by dispatch from
    ``StreamConfig.history_policy``, injected into ``sample_json``) to
    mark historical turns with ``turn_role="history"``, drop / summarize
    their blocks per the policy, and set their ``weight=0`` (history v2,
    spec §4.3). The synthesized summary turn (when
    ``insert_summary=True``) is marked ``turn_role="summary"``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

import msgspec

from titaninfra.data.schema import (
    Block,
    HistoryPolicy,
    HistorySummaryBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from titaninfra.data.sft.message_converter import _render_block, normalize_content
from titaninfra.data.sft.types import (
    NormalizedMessage,
    NormalizedSample,
    NormalizedThinkingIntent,
    Role,
)

_logger = logging.getLogger(__name__)

# Role default weight when the sample omits ``weight``. Matches the
# legacy ``normalize_to_hf_messages`` behaviour: assistant turns
# contribute to loss by default, everything else is masked. The
# boundary builder (resolve_weight + sft_mask_mode) overrides this
# upstream when training policy demands.
_DEFAULT_ASSISTANT_WEIGHT: float = 1.0
_DEFAULT_OTHER_WEIGHT: float = 0.0


def _compute_history_boundary(raw_messages: list, keep_last_n: int | None) -> int:
    """Return the index where the current-turn region begins.

    keep_last_n counts backward through assistant turns and includes their
    logical prompt/tool boundaries without splitting a turn. None falls back
    to the last user index; values beyond the sample keep it all current.
    """
    if not keep_last_n or keep_last_n < 1:
        last_user_idx = -1
        for i, raw in enumerate(raw_messages):
            if isinstance(raw, dict) and raw.get("role") == "user":
                last_user_idx = i
        return last_user_idx if last_user_idx > 0 else 0

    asst_positions = [
        i for i, raw in enumerate(raw_messages)
        if isinstance(raw, dict) and raw.get("role") == "assistant"
    ]
    if len(asst_positions) <= keep_last_n:
        return 0  # The entire sample is current.

    nth_asst_idx = asst_positions[-keep_last_n]
    # Walk backward to the logical turn start without splitting assistant and
    # its preceding user or following tool messages.
    start = nth_asst_idx
    j = nth_asst_idx - 1
    while j >= 0:
        role = raw_messages[j].get("role") if isinstance(raw_messages[j], dict) else None
        if role in ("assistant", "tool"):
            break  # The previous turn must not be merged.
        start = j   # Include user/system in the current turn boundary.
        j -= 1
    return start


def from_sft_sample(sample_json: dict) -> NormalizedSample:
    """Normalize a raw titaninfra SFTSample dict into a NormalizedSample.

    Input shape (titaninfra on-disk MDS row, ``messages`` column already
    JSON-decoded into a dict / list)::

        {
          "system": "you are helpful",
          "tools":  [{"name": "search", "parameters": {...}}, ...],
          "history_policy": {"per_type": {...}, "insert_summary": bool},
          "messages": [
              {"role": "user", "content": "..."},
              {"role": "assistant", "blocks": [
                  {"type": "thinking", "text": "..."},
                  {"type": "text", "text": "..."},
              ]},
              {"role": "assistant", "blocks": [
                  {"type": "tool_use", "name": "search",
                   "input": {...}, "id": "call_1"},
              ]},
              {"role": "tool", "content": "...", "tool_call_id": "call_1"},
          ]
        }

    Output: a ``NormalizedSample`` whose messages are aligned 1:1
    with the input (no merging, no splitting) — except when
    ``history_policy.insert_summary`` collapses the historical
    region into a single synthesized summary turn. The dict input
    is not mutated.
    """
    raw_messages = sample_json.get("messages", [])
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(
            "SFTSample.messages must be a non-empty list; "
            f"got {type(raw_messages).__name__}"
        )

    system = sample_json.get("system")
    tools = tuple(sample_json.get("tools") or ())
    thinking_intent = _normalize_thinking_intent(sample_json.get("thinking_intent"))
    history_policy = _coerce_history_policy(sample_json.get("history_policy"))

    keep_last_n = history_policy.keep_last_n if history_policy is not None else None
    current_start = _compute_history_boundary(raw_messages, keep_last_n)

    last_user_idx = -1
    for i, raw in enumerate(raw_messages):
        if isinstance(raw, dict) and raw.get("role") == "user":
            last_user_idx = i

    normalized: list[NormalizedMessage] = []
    for i, raw in enumerate(raw_messages):
        turn_role = "history" if (current_start > 0 and i < current_start) else "current"
        nm = _normalize_one_message(raw, turn_role=turn_role)
        if turn_role == "history" and history_policy is not None and not history_policy.insert_summary:
            after = _apply_history_policy_to_message(nm, history_policy)
            if after is not None:
                nm = after
            nm = dataclasses.replace(nm, weight=0.0)
        normalized.append(nm)

    if history_policy is not None and history_policy.insert_summary and last_user_idx > 0:
        # Replace the entire historical region with a single synthesized
        # summary turn carrying any HistorySummaryBlock authored in the
        # source. If the source carries no HistorySummaryBlock, fall
        # back to dropping history entirely (the upstream pipeline owns
        # generating the summary text, not normalize — spec §3.3).
        summary_blocks = _collect_history_summary_blocks(raw_messages, last_user_idx)
        head: list[NormalizedMessage] = []
        if summary_blocks:
            head.append(
                NormalizedMessage(
                    role="system",
                    weight=0.0,
                    blocks=tuple(summary_blocks),
                    turn_role="summary",
                )
            )
        tail = [nm for nm in normalized if nm.turn_role != "history"]
        normalized = head + tail

    return NormalizedSample(
        messages=tuple(normalized),
        tools=tools,
        system=system if isinstance(system, str) and system else None,
        thinking_intent=thinking_intent,
        history_policy=history_policy,
    )


def _coerce_history_policy(raw: object) -> HistoryPolicy | None:
    """Accept dict / msgspec HistoryPolicy / None."""
    if raw is None:
        return None
    if isinstance(raw, HistoryPolicy):
        return raw
    if isinstance(raw, dict):
        return msgspec.convert(raw, type=HistoryPolicy)
    return None


def _normalize_thinking_intent(raw: object) -> NormalizedThinkingIntent | None:
    """Coerce the raw ``thinking_intent`` field into ``NormalizedThinkingIntent``.

    Accepts three input shapes (mirrors ``dispatch._detect_features``):

    * ``None`` / missing → returns ``None``.
    * ``dict`` with optional ``mode`` / ``effort`` keys → boxed as struct.
    * msgspec-decoded ``ThinkingIntent`` object → attribute-copied.

    Both sub-fields are independently optional; a struct with both ``None``
    is considered "no intent" and returns ``None`` so downstream encoders
    can short-circuit on a single ``is None`` check.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        mode = raw.get("mode")
        effort = raw.get("effort")
    else:
        mode = getattr(raw, "mode", None)
        effort = getattr(raw, "effort", None)
    if mode is None and effort is None:
        return None
    return NormalizedThinkingIntent(
        mode=mode if isinstance(mode, str) else None,
        effort=effort if isinstance(effort, str) else None,
    )


# ----------------------------------------------------------------------
# Per-message normalization
# ----------------------------------------------------------------------
def _normalize_one_message(raw: dict, *, turn_role: str = "current") -> NormalizedMessage:
    role: Role = raw.get("role", "user")
    # ``content`` (string) wins when set; legacy titaninfra rich rows use
    # ``blocks``. Keeping the fallback identical to
    # ``normalize_to_hf_messages`` so M0 mode='jinja' and mode='native'
    # share the same string view.
    content_field: Any = raw.get("content")
    blocks_field: Any = raw.get("blocks")

    weight = _resolve_weight(raw, role)
    # tool_call_id: prefer the message-level field; fall back to the
    # ``tool_use_id`` carried by the first tool_result block (titaninfra
    # rich-format puts the ID on the block). Used by _coerce_blocks to
    # propagate the id onto the synthesized ToolResultBlock.
    tool_call_id = raw.get("tool_call_id") if role == "tool" else None
    if (
        role == "tool"
        and tool_call_id is None
        and isinstance(blocks_field, list)
    ):
        for block in blocks_field:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tcid = block.get("tool_use_id")
                if isinstance(tcid, str) and tcid:
                    tool_call_id = tcid
                    break
    task = raw.get("task")
    structured_blocks = _coerce_blocks(role, content_field, blocks_field, tool_call_id)

    return NormalizedMessage(
        role=role,
        weight=weight,
        task=task if isinstance(task, str) else None,
        blocks=structured_blocks,
        turn_role=turn_role,  # type: ignore[arg-type]
    )


def _coerce_blocks(
    role: str,
    content_field: Any,
    blocks_field: Any,
    tool_call_id: str | None,
) -> tuple[Block, ...]:
    """Build the structured Block tuple Phase 2+ encoders walk directly.

    Three input shapes (all observed in production MDS):

    1. ``blocks`` is a list of dicts with ``type`` discriminators — decode
       each through msgspec into the appropriate Block subclass.
    2. ``content`` is a plain string — wrap as a single TextBlock.
       Tool-role messages additionally carry ``tool_call_id`` →
       ToolResultBlock(tool_use_id=...).
    3. Both absent (rare): empty tuple.

    Unknown block ``type`` values are dropped with a warn-loud log line
    so the user sees their schema typo instead of training silently on
    a stripped block (matches the capability-mismatch protocol).
    """
    out: list[Block] = []
    if isinstance(blocks_field, list):
        for blk in blocks_field:
            if isinstance(blk, Block):
                out.append(blk)
                continue
            if not isinstance(blk, dict):
                continue
            decoded = _decode_block_dict(blk)
            if decoded is not None:
                out.append(decoded)
        if out:
            return tuple(out)
    if isinstance(content_field, str) and content_field:
        if role == "tool":
            out.append(
                ToolResultBlock(
                    tool_use_id=tool_call_id or "",
                    text=content_field,
                )
            )
        else:
            out.append(TextBlock(text=content_field))
        return tuple(out)
    return ()


def _decode_block_dict(blk: dict) -> Block | None:
    """Map a raw dict to the matching Block subclass via the ``type`` tag."""
    btype = blk.get("type", "text")
    try:
        if btype == "text":
            return TextBlock(text=blk.get("text", ""), weight=blk.get("weight"))
        if btype == "thinking":
            return ThinkingBlock(text=blk.get("text", ""), weight=blk.get("weight"))
        if btype == "tool_use":
            return ToolUseBlock(
                id=blk.get("id", ""),
                name=blk.get("name", ""),
                input=blk.get("input", {}) or {},
                weight=blk.get("weight"),
            )
        if btype == "tool_result":
            return ToolResultBlock(
                tool_use_id=blk.get("tool_use_id", ""),
                text=blk.get("text", ""),
                is_error=bool(blk.get("is_error", False)),
                weight=blk.get("weight"),
            )
        if btype == "image":
            return ImageBlock(
                source=blk.get("source", {}) or {},
                weight=blk.get("weight"),
            )
        if btype == "history_summary":
            return HistorySummaryBlock(
                text=blk.get("text", ""),
                compressed_turn_count=int(blk.get("compressed_turn_count", 0)),
                style=blk.get("style", "system_prefix"),
                weight=blk.get("weight"),
            )
    except (TypeError, ValueError) as exc:
        _logger.warning("normalize: failed to decode block type=%s err=%s", btype, exc)
        return None
    _logger.warning("normalize: unknown block type=%r — skipped", btype)
    return None


def _apply_history_policy_to_message(
    nm: NormalizedMessage,
    policy: HistoryPolicy,
) -> NormalizedMessage:
    """Apply per-Block-type policy to one historical message.

    Returns a new NormalizedMessage with policy-filtered blocks. When
    drop empties all blocks, returns an **empty-blocks shell** (NOT None)
    — the message entry is preserved for gpt_oss/gemma4 index protection
    (spec §3.3); the caller sets ``weight=0`` uniformly. Legacy ``text``
    field is recomputed so encoders still reading the legacy path see the
    post-policy view.
    """
    new_blocks: list[Block] = []
    for blk in nm.blocks:
        action = policy.per_type.get(_block_tag(blk))
        if action is None or action.action == "keep":
            new_blocks.append(blk)
            continue
        if action.action == "drop":
            continue
        if action.action == "summarize":
            template = action.summary_template or "[history block summarized]"
            summary_text = _render_summary(template, blk)
            new_blocks.append(TextBlock(text=summary_text))
    return NormalizedMessage(
        role=nm.role,
        weight=nm.weight,
        task=nm.task,
        blocks=tuple(new_blocks),
        turn_role=nm.turn_role,
    )


def _collect_history_summary_blocks(
    raw_messages: list, last_user_idx: int
) -> list[HistorySummaryBlock]:
    """Pull any author-supplied HistorySummaryBlock from the historical region."""
    out: list[HistorySummaryBlock] = []
    for i in range(last_user_idx):
        raw = raw_messages[i]
        if not isinstance(raw, dict):
            continue
        blocks = raw.get("blocks") or []
        for blk in blocks:
            if isinstance(blk, dict) and blk.get("type") == "history_summary":
                decoded = _decode_block_dict(blk)
                if isinstance(decoded, HistorySummaryBlock):
                    out.append(decoded)
    return out


def _block_tag(blk: Block) -> str:
    if isinstance(blk, TextBlock):
        return "text"
    if isinstance(blk, ThinkingBlock):
        return "thinking"
    if isinstance(blk, ToolUseBlock):
        return "tool_use"
    if isinstance(blk, ToolResultBlock):
        return "tool_result"
    if isinstance(blk, HistorySummaryBlock):
        return "history_summary"
    return "unknown"


def _block_to_dict(blk: Block) -> dict:
    if isinstance(blk, TextBlock):
        return {"type": "text", "text": blk.text}
    if isinstance(blk, ThinkingBlock):
        return {"type": "thinking", "text": blk.text}
    if isinstance(blk, ToolUseBlock):
        return {"type": "tool_use", "name": blk.name, "input": blk.input, "id": blk.id}
    if isinstance(blk, ToolResultBlock):
        return {"type": "tool_result", "tool_use_id": blk.tool_use_id, "text": blk.text}
    if isinstance(blk, HistorySummaryBlock):
        return {"type": "text", "text": blk.text}
    return {"type": "text", "text": ""}


def _render_summary(template: str, blk: Block) -> str:
    """Format ``template`` against the block's primary content fields.

    Supported substitutions: ``{content}`` (primary text), ``{name}``
    (tool_use only). Malformed templates fall through to the literal
    template string — a stray ``{`` won't blow up production.
    """
    fields: dict[str, str] = {}
    if isinstance(blk, (TextBlock, ThinkingBlock, ToolResultBlock, HistorySummaryBlock)):
        fields["content"] = getattr(blk, "text", "")
    if isinstance(blk, ToolUseBlock):
        fields["content"] = json.dumps(blk.input, ensure_ascii=False)
        fields["name"] = blk.name
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError):
        return template


def _extract_thinking_from_blocks(blocks: list[Block]) -> str | None:
    parts = [b.text for b in blocks if isinstance(b, ThinkingBlock) and b.text]
    return "\n".join(parts) if parts else None


def _extract_tool_calls_from_blocks(blocks: list[Block]) -> tuple[dict, ...]:
    out: list[dict] = []
    for b in blocks:
        if isinstance(b, ToolUseBlock):
            out.append({"name": b.name, "arguments": b.input, "id": b.id})
    return tuple(out)


def _resolve_weight(raw: dict, role: Role) -> float:
    """Per-message weight in [0, 1]. Boundary builder may override."""
    if "weight" in raw and raw["weight"] is not None:
        return float(raw["weight"])
    return _DEFAULT_ASSISTANT_WEIGHT if role == "assistant" else _DEFAULT_OTHER_WEIGHT


def _render_text_only(raw_content: Any, blocks: Any) -> str:
    """Render text+thinking content for ``nm.text``, excluding ``tool_use``.

    Tool-use blocks are captured separately on ``nm.tool_calls``; including
    them again as inline JSON in ``nm.text`` would force every encoder to
    re-parse and strip them. Returning text-only here gives the per-family
    encoder a clean string view that can be concatenated with
    ``<tool_call>...</tool_call>`` blocks built from ``nm.tool_calls``.
    """
    if isinstance(blocks, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks
    ):
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                parts.append(str(block))
            elif block.get("type") == "tool_use":
                continue
            else:
                parts.append(_render_block(block))
        return "".join(parts)
    return normalize_content(raw_content)


def _extract_thinking(blocks: list) -> str | None:
    """Concatenate ``thinking`` block text, newline-joined; else None."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "thinking":
            t = block.get("text", "")
            if t:
                parts.append(t)
    return "\n".join(parts) if parts else None


def _extract_tool_calls(raw: dict, blocks: Any) -> tuple[dict, ...]:
    """Canonical tool-call shape: ``{name, arguments, id}``.

    Two input shapes are supported:

    * HF-standard ``tool_calls`` already on the message (used by
      data already converted via ``message_converter``). Each entry
      looks like ``{"id", "type": "function", "function": {"name",
      "arguments": str}}``. We unpack ``function`` and JSON-decode
      ``arguments`` so encoders see Python objects, not strings.
    * Rich-format ``tool_use`` blocks. Each block looks like
      ``{"type": "tool_use", "name", "input": dict, "id"}``.

    Per-family argument formatting (dict vs JSON string) is the
    encoder's job, not normalize's.
    """
    canonical: list[dict] = []

    raw_tool_calls = raw.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for entry in raw_tool_calls:
            if not isinstance(entry, dict):
                continue
            fn = entry.get("function", entry)
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    pass  # leave as string; encoder decides
            canonical.append({"name": name, "arguments": args, "id": entry.get("id")})

    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                canonical.append({
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                    "id": block.get("id"),
                })

    return tuple(canonical)


__all__ = ["from_sft_sample"]
