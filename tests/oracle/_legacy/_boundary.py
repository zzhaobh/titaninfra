"""Jinja-path boundary detection for SFT tokenization.

These helpers are used by the legacy Jinja-rendered tokenization path
(``DefaultAdapter.compute_boundaries``, alignment tests, etc.).  Segment
encoders (qwen3, gemma4, gpt_oss, …) do NOT import from here — they own
their boundary logic natively.

Exported symbols (all package-internal by convention):
    _ROLE_PATTERNS, _WRAPPED_ROLE_FALLBACK
    _gpt_oss_role_mapper, _find_role_markers
    _compute_message_boundaries
"""

from __future__ import annotations

import bisect
import logging
import re

logger = logging.getLogger(__name__)

# ============================================================
# Role marker patterns for different model families
# ============================================================
# Each entry: (compiled_regex, role_mapping_dict)
# The regex captures the role name as group(1) and matches the FULL role
# marker line (including any trailing whitespace/separator) so that
# m.end() points to the position where message content begins.

_ROLE_PATTERNS: list[tuple[re.Pattern, dict[str, str]]] = [
    # ChatML: <|im_start|>role\n  (Qwen, Yi, etc.)
    (
        re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n?"),
        {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"},
    ),
    # Llama 3: <|start_header_id|>role<|end_header_id|>\n\n
    (
        re.compile(
            r"<\|start_header_id\|>(system|user|assistant|tool|ipython)<\|end_header_id\|>(?:\n\n)?"
        ),
        {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "ipython": "tool",
        },
    ),
    # Llama 4: <|header_start|>role<|header_end|>\n\n
    (
        re.compile(
            r"<\|header_start\|>(system|user|assistant|tool|ipython)<\|header_end\|>(?:\n\n)?"
        ),
        {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "ipython": "tool",
        },
    ),
    # GPT-OSS: <|start|>role<|message|>
    # Note: `to=[^<\s]+` (not `\S+`). `\S` matches `<`, so a greedy `\S+` would
    # swallow `<|channel|>commentary` from a tool-call header like
    # `<|start|>assistant to=functions.X<|channel|>commentary json<|message|>`,
    # leaving content_start mid-header and labelling ` json` / `<|message|>`
    # as loss tokens.
    (
        re.compile(
            r"<\|start\|>"
            r"(system|developer|user|assistant(?:\s+to=[^<\s]+)?|functions\.\w+\s+to=assistant|tool)"
            r"(?:<\|channel\|>[^<]*)?(?:<\|message\|>)?"
        ),
        None,  # uses _gpt_oss_role_mapper
    ),
    # Kimi: <|im_user|> / <|im_assistant|>
    (
        re.compile(r"<\|im_(system|user|assistant|tool)\|>"),
        {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"},
    ),
    # GLM: <|user|> / <|assistant|> / <|system|>
    (
        re.compile(r"<\|(system|user|assistant|observation)\|>"),
        {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "observation": "tool",
        },
    ),
    # DeepSeek: <｜User｜> / <｜Assistant｜>  (full-width brackets)
    (
        re.compile(r"<｜(User|Assistant)｜>"),
        {"User": "user", "Assistant": "assistant"},
    ),
    # Gemma 3: <start_of_turn>role\n
    (
        re.compile(r"<start_of_turn>(system|user|assistant|tool|model)\n?"),
        {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "model": "assistant",
        },
    ),
    # Gemma 4: <|turn>role\n
    (
        re.compile(r"<\|turn>(system|user|assistant|tool|model)\n?"),
        {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "model": "assistant",
        },
    ),
]

# Roles that the chat-template renderer wraps into another role's marker block.
# When _compute_message_boundaries encounters such a role in hf_messages but
# finds no matching marker in the text, it uses a forward-looking position
# (the next wrapping-role marker) rather than the previous message's header.
_WRAPPED_ROLE_FALLBACK: dict[str, str] = {"tool": "user"}


def _gpt_oss_role_mapper(raw_role: str) -> str | None:
    """Map GPT-OSS raw role captures to standard role names."""
    if raw_role in ("system", "developer"):
        return "system"
    if raw_role == "user":
        return "user"
    if raw_role == "assistant" or raw_role.startswith("assistant "):
        return "assistant"
    if raw_role == "tool" or raw_role.startswith("functions."):
        return "tool"
    return None


def _find_role_markers(full_text: str) -> list[tuple[int, int, str]]:
    """Scan *full_text* for role markers; return (header_start, content_start, role) list."""
    for pattern, role_map in _ROLE_PATTERNS:
        matches = list(pattern.finditer(full_text))
        if matches:
            markers = []
            for m in matches:
                raw = m.group(1)
                if role_map is None:
                    role = _gpt_oss_role_mapper(raw)
                else:
                    role = role_map.get(raw)
                if role:
                    markers.append((m.start(), m.end(), role))
            if markers:
                return markers
    return []


def _compute_message_boundaries(
    hf_messages: list[dict[str, str]],
    tokenizer,
    tools: list[dict] | None = None,
    pre_rendered_text: str | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Compute per-message token boundaries via pattern-based role-marker detection.

    Returns (full_ids, boundaries, content_starts):
    - full_ids: complete token ID sequence
    - boundaries: length N+1; boundaries[i] = start token of message i's role marker
    - content_starts: length N; content_starts[i] = first content token after marker
    """
    if pre_rendered_text is not None:
        full_text = pre_rendered_text
    else:
        template_kwargs: dict = {"tokenize": False, "add_generation_prompt": False}
        if tools:
            template_kwargs["tools"] = tools
        full_text = tokenizer.apply_chat_template(hf_messages, **template_kwargs)

    encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    full_ids = encoded["input_ids"]
    token_starts = [off[0] for off in encoded["offset_mapping"]]

    def _char_to_token(cp: int) -> int:
        return min(bisect.bisect_left(token_starts, cp), len(full_ids))

    markers = _find_role_markers(full_text)

    if not markers:
        logger.warning(
            "No role markers recognised; loss mask applied to the full sequence. "
            "Check the tokenizer's chat_template format."
        )
        return (
            full_ids,
            [0] + [len(full_ids)] * len(hf_messages),
            [0] * len(hf_messages),
        )

    marker_roles = {m[2] for m in markers}

    header_chars: list[int] = []
    content_chars: list[int] = []
    marker_idx = 0

    for i, msg in enumerate(hf_messages):
        role = msg.get("role", "user")

        if role not in marker_roles:
            wrapping = _WRAPPED_ROLE_FALLBACK.get(role)
            if wrapping and wrapping in marker_roles:
                is_chain_start = i == 0 or hf_messages[i - 1].get("role") != role
                if is_chain_start:
                    k = marker_idx
                    while k < len(markers) and markers[k][2] != wrapping:
                        k += 1
                    if k < len(markers):
                        h_start, c_start, _ = markers[k]
                        header_chars.append(h_start)
                        content_chars.append(c_start)
                        marker_idx = k + 1
                        continue
                cp = header_chars[-1] if header_chars else 0
                header_chars.append(cp)
                content_chars.append(cp)
                continue
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


__all__ = [
    "_ROLE_PATTERNS",
    "_WRAPPED_ROLE_FALLBACK",
    "_gpt_oss_role_mapper",
    "_find_role_markers",
    "_compute_message_boundaries",
]
