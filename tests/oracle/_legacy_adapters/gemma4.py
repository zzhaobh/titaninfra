"""Legacy Gemma4 ToolUseAdapter — TEST-ONLY oracle helper.

This file was moved out of ``titaninfra/data/sft/tool_use_adapters/`` in PR-C
(2026-05-28). Production runtime uses ``Gemma4Encoder`` (native path); this
adapter is preserved verbatim so the Jinja oracle remains reachable for
tool-use parity fixtures (gemma4/06, gemma4/08).

Do NOT import from production code. The ``gemma4_oracle_adapter_registered``
fixture in ``tests/oracle/conftest.py`` registers this class into the
production registry for the duration of a test, then tears down.

Root cause
----------
Gemma4's chat_template renders tool-using conversations with two structural
peculiarities that the default _ROLE_PATTERNS cannot handle:

1. ``continue_same_model_turn``: when an assistant message follows a tool
   response, the template suppresses the second ``<|turn>model\\n`` marker
   so the post-tool assistant text is rendered *inside* the previous model
   turn. The default pattern matcher therefore finds no marker for that
   message.

2. Tool messages are skipped in the outer loop entirely
   (``{%- if message[\'role\'] != \'tool\' -%}``). They appear as inline
   ``<|tool_response>...<tool_response|>`` blocks within the preceding
   model turn, with no top-level role marker.

Strategy:
* role=tool messages: inherit prev boundary (auto mode → weight=0 →
  fail-loud skipped).
* continue_same_model_turn assistant: ``boundaries[i]`` = start of the
  *last* ``<|tool_response>`` opener in current model segment;
  ``content_starts[i]`` = end of its matching ``<tool_response|>`` closer.
  Guarantees ``boundaries[i] != content_starts[i]`` → fail-loud passes,
  loss range covers assistant text written after the closing tool_response.
"""

from __future__ import annotations

import bisect
import logging
import re

from tests.oracle._legacy.tool_use_adapters.base import ToolUseAdapter

logger = logging.getLogger(__name__)

_GEMMA4_TURN = re.compile(r"<\|turn>(system|user|assistant|tool|model)\n?")
_GEMMA4_TR_OPEN = re.compile(r"<\|tool_response>")
_GEMMA4_TR_CLOSE = re.compile(r"<tool_response\|>")

_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "model": "assistant",
}


class Gemma4OracleAdapter(ToolUseAdapter):
    def compute_boundaries(
        self,
        hf_messages: list[dict],
        tokenizer,
        tools: list[dict] | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        template_kwargs = {"tokenize": False, "add_generation_prompt": False}
        if tools:
            template_kwargs["tools"] = tools

        full_text = tokenizer.apply_chat_template(hf_messages, **template_kwargs)
        encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        full_ids = encoded["input_ids"]
        token_starts = [off[0] for off in encoded["offset_mapping"]]

        def _char_to_token(cp: int) -> int:
            return min(bisect.bisect_left(token_starts, cp), len(full_ids))

        turn_markers: list[tuple[int, int, str]] = []
        for m in _GEMMA4_TURN.finditer(full_text):
            role = _ROLE_MAP.get(m.group(1))
            if role is not None:
                turn_markers.append((m.start(), m.end(), role))

        if not turn_markers:
            logger.warning("Gemma4OracleAdapter: no <|turn> markers found; falling back to default.")
            from tests.oracle._legacy._boundary import _compute_message_boundaries
            return _compute_message_boundaries(hf_messages, tokenizer, tools=tools)

        tr_opens = [(m.start(), m.end()) for m in _GEMMA4_TR_OPEN.finditer(full_text)]
        tr_closes = [(m.start(), m.end()) for m in _GEMMA4_TR_CLOSE.finditer(full_text)]

        header_chars: list[int] = []
        content_chars: list[int] = []
        turn_idx = 0
        # Char-position right after the most recently matched <|turn>model\n
        # marker. This is the start of the model turn's *body* — used as the
        # search lower-bound when a later assistant message lands inside the
        # same turn (continue_same_model_turn).
        last_model_body_start: int | None = None

        for i, msg in enumerate(hf_messages):
            role = msg.get("role", "user")

            if role == "tool":
                cp = header_chars[-1] if header_chars else 0
                header_chars.append(cp)
                content_chars.append(cp)
                continue

            mapped_role = "assistant" if role == "assistant" else role

            if (
                role == "assistant"
                and last_model_body_start is not None
                and (turn_idx >= len(turn_markers) or turn_markers[turn_idx][2] != "assistant")
            ):
                segment_start = last_model_body_start
                segment_end = (
                    turn_markers[turn_idx][0] if turn_idx < len(turn_markers) else len(full_text)
                )
                last_open = None
                last_close = None
                for o_start, o_end in tr_opens:
                    if segment_start <= o_start < segment_end:
                        last_open = (o_start, o_end)
                for c_start, c_end in tr_closes:
                    if segment_start <= c_start < segment_end:
                        last_close = (c_start, c_end)
                if last_open is not None and last_close is not None and last_close[0] > last_open[0]:
                    header_chars.append(last_open[0])
                    content_chars.append(last_close[1])
                    continue
                cp = header_chars[-1] if header_chars else 0
                header_chars.append(cp)
                content_chars.append(cp)
                continue

            found = False
            while turn_idx < len(turn_markers):
                h_start, c_start, m_role = turn_markers[turn_idx]
                if m_role == mapped_role:
                    header_chars.append(h_start)
                    content_chars.append(c_start)
                    turn_idx += 1
                    found = True
                    if mapped_role == "assistant":
                        last_model_body_start = c_start
                    break
                turn_idx += 1

            if not found:
                cp = header_chars[-1] if header_chars else 0
                header_chars.append(cp)
                content_chars.append(cp)

        header_chars.append(len(full_text))
        boundaries = [_char_to_token(cp) for cp in header_chars]
        content_starts = [_char_to_token(cp) for cp in content_chars]
        for i in range(len(content_starts)):
            content_starts[i] = max(content_starts[i], boundaries[i])
            content_starts[i] = min(content_starts[i], len(full_ids))
        return full_ids, boundaries, content_starts
