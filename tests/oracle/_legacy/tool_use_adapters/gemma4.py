"""Gemma4 ToolUseAdapter — self-contained oracle boundary computation.

Root cause
----------
The default ``_compute_message_boundaries`` only recognises the
``<|turn>{role}`` marker for the explicit ``user``, ``assistant``,
``system``, ``tool`` roles, and does not understand Gemma4's
**continue_same_model_turn** rule: an assistant turn that follows a
``role='tool'`` message lives INSIDE the previous ``<|turn>model`` block
without emitting its own ``<|turn>model`` marker. The default detector
sets ``boundaries[i] == content_starts[i]`` for that assistant message
and the fail-loud guard rejects the sample.

Additionally the gemma4 chat_template references ``follow.get('name')``
on tool messages but titaninfra-normalized tool messages carry only
``tool_call_id``. We rewrite tool messages here so the template's
``tool_call_id`` -> tool name lookup resolves; ``content=None`` on a
tool message also crashes the template's response-block macro so we
coerce it to the empty string.

Independence
------------
This adapter renders via ``tokenizer.apply_chat_template`` (the real
Jinja chat template — the genuine oracle) and computes boundaries with
its OWN copy of the turn-marker / continue_same_model_turn offset logic.
It does NOT import from ``titaninfra.data.sft.encoders.gemma4`` — the
production encoder is now a pure-Segment encoder with no flat-string /
offset machinery to borrow, and the whole point of the oracle is to be
an INDEPENDENT reference (render-then-measure, the "slime camp" approach
that is legitimate only here, in the test baseline). Mirrors the
deepseek_v3 adapter's self-contained pattern.
"""

from __future__ import annotations

import bisect
import re

from tests.oracle._legacy.tool_use_adapters.base import ToolUseAdapter

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


class Gemma4Adapter(ToolUseAdapter):
    """Gemma4-aware adapter: handles continue_same_model_turn + tool name."""

    def preprocess_messages(self, hf_messages, tools=None):
        """Coerce ``content=None`` on tool messages to empty string.

        The gemma4 jinja's ``format_tool_response_block`` does
        ``'response:' + tool_name + ...`` and crashes when ``content``
        is None (the inner ``format_argument(response)`` path tries
        string concat on None). Empty string is safe: the template
        renders ``response:<name>{value:}`` which decodes cleanly.
        """
        rewritten = []
        for msg in hf_messages:
            if msg.get("role") == "tool" and msg.get("content") is None:
                new_msg = dict(msg)
                new_msg["content"] = ""
                rewritten.append(new_msg)
            else:
                rewritten.append(msg)
        return rewritten, tools

    def compute_boundaries(self, hf_messages, tokenizer, tools=None):
        """Render via Jinja apply_chat_template, then offset-measure boundaries.

        Self-contained port of the (deleted-from-production) gemma4 offset
        logic. Returns (full_ids, boundaries, content_starts) with
        len(boundaries) == len(hf_messages) + 1.
        """
        import logging
        logger = logging.getLogger(__name__)

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
            logger.warning("Gemma4Adapter(oracle): no <|turn> markers; degenerate sample.")
            boundaries = [0] * (len(hf_messages) + 1)
            content_starts = [0] * len(hf_messages)
            return full_ids, boundaries, content_starts

        tr_opens = [(m.start(), m.end()) for m in _GEMMA4_TR_OPEN.finditer(full_text)]
        tr_closes = [(m.start(), m.end()) for m in _GEMMA4_TR_CLOSE.finditer(full_text)]

        header_chars: list[int] = []
        content_chars: list[int] = []
        turn_idx = 0
        last_model_body_start: int | None = None

        for msg in hf_messages:
            role = msg.get("role", "user")

            if role == "tool":
                cp = header_chars[-1] if header_chars else 0
                header_chars.append(cp)
                content_chars.append(cp)
                continue

            mapped_role = "assistant" if role == "assistant" else role

            # continue_same_model_turn: post-tool assistant text reuses the
            # last model turn's body, anchored to the most recent
            # <|tool_response>...<tool_response|> pair.
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
