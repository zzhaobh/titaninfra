"""DeepSeek-V3 (deepseek_r1 template) adapter.

Root cause
----------
The default _ROLE_PATTERNS only recognises <｜User｜> and <｜Assistant｜>
for DeepSeek.  In a tool-use conversation the rendered text has two additional
structure points that must be treated as role markers:

* <｜tool▁output▁begin｜>  - appears once per role=tool message inside the
  <｜tool▁outputs▁begin｜>...<｜tool▁outputs▁end｜> block.
* <｜tool▁outputs▁end｜>   - appears immediately before the assistant text
  response that follows tool messages (no <｜Assistant｜> is emitted here).

Without these markers the fallback sets boundaries[i] == content_starts[i]
for the post-tool assistant message, triggering the fail-loud ValueError.
"""

from __future__ import annotations

import bisect
import re

from tests.oracle._legacy.tool_use_adapters.base import ToolUseAdapter

_DS_USER_ASSISTANT = re.compile(r"<｜(User|Assistant)｜>")
_DS_TOOL_OUTPUT_BEGIN = re.compile(r"<｜tool▁output▁begin｜>")
_DS_TOOL_OUTPUTS_END = re.compile(r"<｜tool▁outputs▁end｜>")


def _find_markers(full_text: str) -> list[tuple[int, int, str]]:
    markers: list[tuple[int, int, str]] = []
    for m in _DS_USER_ASSISTANT.finditer(full_text):
        role = "user" if m.group(1) == "User" else "assistant"
        markers.append((m.start(), m.end(), role))
    for m in _DS_TOOL_OUTPUT_BEGIN.finditer(full_text):
        markers.append((m.start(), m.end(), "tool"))
    for m in _DS_TOOL_OUTPUTS_END.finditer(full_text):
        markers.append((m.start(), m.end(), "assistant"))
    markers.sort(key=lambda x: x[0])
    return markers


class DeepSeekV3Adapter(ToolUseAdapter):
    def preprocess_messages(self, hf_messages, tools=None):
        # DSV3/R1 Jinja gates the <｜tool▁call▁begin｜> branch on
        # `message['content'] is none`. An empty string falls through to the
        # text branch and silently drops tool_calls (no loss on tool tokens
        # during SFT). Rewrite content="" → None for assistant turns that
        # have tool_calls and no real text.
        #
        # Sequential multi-turn limit: DSV3 Jinja assumes ONE round of
        # (assistant_tool_calls → tool_outputs → assistant_text). When two
        # assistant turns both take the tool_call branch with a tool message
        # between them, the rendered text interleaves wrong:
        # `<｜tool▁outputs▁begin｜>` opens at the first tool, and the second
        # assistant's `<｜Assistant｜><｜tool▁calls▁begin｜>` is emitted INSIDE
        # that unclosed block — no `<｜tool▁outputs▁end｜>` marker is produced
        # for the post-tool assistant turn, breaking boundary computation.
        # Fix: only rewrite to None for the LAST assistant tool_call in the
        # conversation; earlier ones keep "" and fall through the text branch
        # (tool_calls silently dropped, no training signal on those tokens, but
        # the conversation structure stays parseable).
        #
        # Same-turn text + tool_call (DSV3 protocol XOR limit) keeps the text
        # string here; the alignment layer tracks that as xfail separately.
        last_tool_call_idx = -1
        for idx, msg in enumerate(hf_messages):
            if (
                msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and not msg.get("content")
            ):
                last_tool_call_idx = idx
        if last_tool_call_idx == -1:
            return hf_messages, tools
        rewritten = list(hf_messages)
        new_msg = dict(rewritten[last_tool_call_idx])
        new_msg["content"] = None
        rewritten[last_tool_call_idx] = new_msg
        return rewritten, tools

    def compute_boundaries(self, hf_messages, tokenizer, tools=None):
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

        markers = _find_markers(full_text)
        if not markers:
            logger.warning("DeepSeekV3Adapter: no markers found; falling back.")
            return (full_ids, [0] + [len(full_ids)] * len(hf_messages), [0] * len(hf_messages))

        marker_roles = {m[2] for m in markers}
        header_chars: list[int] = []
        content_chars: list[int] = []
        marker_idx = 0

        for i, msg in enumerate(hf_messages):
            role = msg.get("role", "user")
            if role not in marker_roles:
                cp = 0 if i == 0 else header_chars[-1]
                header_chars.append(cp)
                content_chars.append(cp)
                continue
            found = False
            while marker_idx < len(markers):
                h_start, c_start, m_role = markers[marker_idx]
                if m_role == role:
                    header_chars.append(h_start)
                    content_chars.append(c_start)
                    marker_idx += 1
                    found = True
                    break
                marker_idx += 1
            if not found:
                header_chars.append(len(full_text))
                content_chars.append(len(full_text))

        header_chars.append(len(full_text))
        boundaries = [_char_to_token(cp) for cp in header_chars]
        content_starts = [_char_to_token(cp) for cp in content_chars]
        for i in range(len(content_starts)):
            content_starts[i] = max(content_starts[i], boundaries[i])
            content_starts[i] = min(content_starts[i], len(full_ids))
        return full_ids, boundaries, content_starts
