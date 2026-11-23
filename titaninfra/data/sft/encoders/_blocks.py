"""Shared block-extraction helpers for the per-family segment encoders.

These are pure functions that read a ``NormalizedMessage.blocks`` tuple
(or a list of ``NormalizedMessage``) and pull out one block species. Five
of the seven family encoders duplicated byte-identical copies of these
(``_block_text`` / ``_extract_thinking`` / ``_extract_tool_calls`` /
``_extract_tool_result_text`` / ``_last_assistant_index``); this module
is the single definition they now share.

Scope discipline (spec §F3): this module holds ONLY the family-agnostic
block readers. Each family's *wire formatting* (how a tool_call renders
to its protocol bytes, where the stop marker goes, role-marker layout)
stays in that family's encoder — that is the per-family adapter layer and
must not be hoisted here.

``last_assistant_index`` operates on ``NormalizedMessage`` objects (the
``.role`` attribute). Gemma4's ``_last_model_index`` walks HF-shape
``list[dict]`` (``.get("role")``) and is intentionally NOT unified here —
different input type, same concept, two honest names (spec §F3).
"""

from __future__ import annotations

from titaninfra.data.schema import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock


def block_text(blocks) -> str:
    """Concatenate the ``text`` of every ``TextBlock``; ignore other blocks."""
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def extract_thinking(blocks) -> str | None:
    """Join every non-empty ``ThinkingBlock.text`` with ``\\n``; ``None`` if none."""
    parts = [b.text for b in blocks if isinstance(b, ThinkingBlock) and b.text]
    return "\n".join(parts) if parts else None


def extract_tool_calls(blocks) -> list[ToolUseBlock]:
    """Return every ``ToolUseBlock`` in declaration order."""
    return [b for b in blocks if isinstance(b, ToolUseBlock)]


def extract_tool_result_text(blocks) -> str:
    """Concatenate the text of ``ToolResultBlock`` and ``TextBlock`` blocks.

    Tool-role messages carry their payload as a ``ToolResultBlock``; some
    samples also append plain ``TextBlock`` commentary. Both contribute
    the tool's observable output, so both are concatenated in order.
    """
    parts: list[str] = []
    for b in blocks:
        if isinstance(b, (ToolResultBlock, TextBlock)):
            parts.append(b.text)
    return "".join(parts)


def last_assistant_index(messages) -> int:
    """Index of the last assistant ``NormalizedMessage``; ``-1`` if none."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            return i
    return -1


__all__ = [
    "block_text",
    "extract_thinking",
    "extract_tool_calls",
    "extract_tool_result_text",
    "last_assistant_index",
]
