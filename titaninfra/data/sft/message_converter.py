"""Convert rich TitanInfra messages to Hugging Face message dictionaries.

Text and thinking blocks are rendered into content, tool_use blocks become HF
tool_calls, and message weights are returned alongside the converted messages.
Image blocks are ignored by this text-only compatibility path.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def _render_block(block: dict) -> str:
    """Render one content block to text, returning an empty string if unknown."""
    block_type = block.get("type", "text")

    if block_type == "text":
        return block.get("text", "")

    if block_type == "thinking":
        think_text = block.get("text", "")
        return f"<think>\n{think_text}\n</think>\n\n" if think_text else ""

    if block_type == "tool_use":
        tool_call = {
            "name": block.get("name", ""),
            "arguments": block.get("input", {}),
        }
        if "id" in block:
            tool_call["id"] = block["id"]
        return json.dumps(tool_call, ensure_ascii=False)

    if block_type == "tool_result":
        return block.get("text", "")

    if block_type == "image":
        logger.debug("Skipping image block in text-only conversion mode")
        return ""

    logger.warning(f"未知 content block 类型: {block_type}，跳过")
    return ""


def _build_tool_calls_from_blocks(blocks: list[dict]) -> list[dict]:
    """Extract tool_use blocks and build HF-standard tool_calls list."""
    tool_calls = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        entry = {
            "id": block.get("id", ""),
            "type": "function",
            "function": {
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
            },
        }
        tool_calls.append(entry)
    return tool_calls


def _extract_non_tool_content(blocks: list[dict]) -> tuple[str, str | None]:
    """Render text/thinking blocks (excluding tool_use), return (content, reasoning_content)."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        btype = block.get("type", "text")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_parts.append(block.get("text", ""))
    content = "".join(text_parts)
    reasoning = "\n".join(thinking_parts) if thinking_parts else None
    return content, reasoning


def normalize_content(content) -> str:
    """Normalize a string or list of content blocks into plain text.

    Thinking blocks are wrapped in <think> tags, tool_use blocks become JSON,
    and images are skipped by this text-only conversion.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        logger.warning(f"未知 content 类型: {type(content)}，转为 str")
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
        else:
            parts.append(_render_block(block))

    return "".join(parts)


def normalize_to_hf_messages(
    messages: list[dict],
    system: str | None = None,
) -> tuple[list[dict[str, str]], list[float]]:
    """Convert rich messages and return (hf_messages, weights).

    An optional system prompt is inserted first with weight zero. Other weights
    default to 1 for assistant messages and 0 for non-assistant roles.
    """
    hf_messages: list[dict[str, str]] = []
    weights: list[float] = []

    # Prepend the optional system prompt.
    if system:
        hf_messages.append({"role": "system", "content": system})
        weights.append(0.0)  # System prompts never contribute to loss.

    for msg in messages:
        role = msg.get("role", "user")
        # Rich messages may use either content or blocks keys.
        content = msg.get("content") or msg.get("blocks", "")
        weight = float(msg.get("weight", 1.0 if role == "assistant" else 0.0))

        hf_msg: dict = {"role": role}

        # Synthesize HF tool_calls from tool_use blocks on assistant messages.
        has_blocks = isinstance(content, list)
        has_tool_use_blocks = (
            has_blocks
            and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
        )

        if role == "assistant" and has_tool_use_blocks and "tool_calls" not in msg:
            # Build HF tool_calls while preserving text/reasoning content.
            # Family-specific null-content rewrites belong to their adapters.
            hf_msg["tool_calls"] = _build_tool_calls_from_blocks(content)
            text_content, reasoning = _extract_non_tool_content(content)
            hf_msg["content"] = text_content
            if reasoning:
                hf_msg["reasoning_content"] = reasoning
        else:
            hf_msg["content"] = normalize_content(content)

        # Preserve standard HF fields consumed by family templates.
        if role == "assistant":
            # Reasoning models read thinking content from reasoning_content.
            if "reasoning_content" in msg and "reasoning_content" not in hf_msg:
                hf_msg["reasoning_content"] = msg["reasoning_content"]
            # Function-calling samples may already carry tool_calls.
            if "tool_calls" in msg and "tool_calls" not in hf_msg:
                hf_msg["tool_calls"] = msg["tool_calls"]
        if role == "tool":
            # Extract tool_result content and call IDs from blocks.
            if has_blocks and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        hf_msg["content"] = block.get("text", "")
                        if "tool_use_id" in block:
                            hf_msg["tool_call_id"] = block["tool_use_id"]
                        break
            # Prefer the HF-standard tool_call_id over legacy tool_use_id.
            if "tool_call_id" in msg and "tool_call_id" not in hf_msg:
                hf_msg["tool_call_id"] = msg["tool_call_id"]
            elif "tool_use_id" in msg and "tool_call_id" not in hf_msg:
                hf_msg["tool_call_id"] = msg["tool_use_id"]
            if "name" in msg:
                hf_msg["name"] = msg["name"]

        hf_messages.append(hf_msg)
        weights.append(weight)

    return hf_messages, weights


# ─────────────────────────────────────────────────────────────────────────
# HF / loose-shape to TitanInfra IR (`blocks`)
# ─────────────────────────────────────────────────────────────────────────
#
# Reverse of ``normalize_to_hf_messages``. The converter G1 strict
# validator (``titaninfra.data.validator``) decodes via msgspec against
# ``schema.Message`` which REQUIRES a non-empty ``blocks`` list. Most
# real-world JSONL fixtures (and historical data) ship HF-style
# ``{role, content}`` rows; this function is the single place that
# adapts them up to IR, so converter callers don't need to know the
# shape difference.
#
# Idempotency: a message that already has ``blocks`` (titaninfra IR) is
# passed through with only role/weight/extras preserved — never
# re-wrapped. This makes the converter call site safe for mixed inputs.


_THINK_TAG_RE = re.compile(r"<think>\n?(.*?)\n?</think>(?:\n\n)?", re.DOTALL)


def _text_to_blocks(content: str) -> list[dict]:
    """Split text and <think> envelopes into ordered IR blocks."""
    blocks: list[dict] = []
    cursor = 0
    for match in _THINK_TAG_RE.finditer(content):
        if match.start() > cursor:
            blocks.append({"type": "text", "text": content[cursor:match.start()]})
        blocks.append({"type": "thinking", "text": match.group(1)})
        cursor = match.end()
    if cursor < len(content):
        blocks.append({"type": "text", "text": content[cursor:]})
    return blocks


def _content_to_blocks(content) -> list[dict]:
    """Adapt an HF ``content`` value (str or list) into IR ``blocks``.

    String content preserves text and <think> envelopes in their original order.
    List entries pass through if already typed; plain strings become text blocks.
    Anything else is stringified into a text block without dropping data.
    """
    if isinstance(content, str):
        return _text_to_blocks(content)

    if isinstance(content, list):
        out: list[dict] = []
        for item in content:
            if isinstance(item, dict) and "type" in item:
                out.append(item)
            elif isinstance(item, str):
                if item:
                    out.append({"type": "text", "text": item})
            else:
                out.append({"type": "text", "text": str(item)})
        return out

    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _hf_tool_calls_to_blocks(tool_calls: list) -> list[dict]:
    """HF-standard ``tool_calls`` list → titaninfra ``tool_use`` blocks.

    HF entry shape: ``{"id", "type": "function", "function": {"name", "arguments": str|dict}}``.
    IR ``tool_use`` block: ``{"type": "tool_use", "id", "name", "input": dict}``.
    Arguments are JSON-decoded when they arrive as strings so the schema
    invariant (``input`` is dict) holds.
    """
    blocks: list[dict] = []
    for entry in tool_calls or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function", entry)
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": args}
        blocks.append({
            "type": "tool_use",
            "id": entry.get("id", ""),
            "name": name,
            "input": args,
        })
    return blocks


def _to_ir_one_message(msg: dict) -> dict:
    """Adapt one HF/loose message dict to a strict titaninfra IR message."""
    if not isinstance(msg, dict):
        raise TypeError(f"message must be a dict, got {type(msg).__name__}")

    role = msg.get("role", "user")

    # Preserve messages already in IR form verbatim.
    if "blocks" in msg and msg.get("content") is None:
        return msg

    # Promote tool-role content and tool_call_id into a tool_result block.
    if role == "tool":
        tool_call_id = msg.get("tool_call_id") or msg.get("tool_use_id") or ""
        content = msg.get("content")
        text = content if isinstance(content, str) else ""
        return {
            "role": "tool",
            "blocks": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "text": text,
            }],
        }

    blocks: list[dict] = []
    blocks.extend(_content_to_blocks(msg.get("content")))

    if role == "assistant" and isinstance(msg.get("tool_calls"), list):
        blocks.extend(_hf_tool_calls_to_blocks(msg["tool_calls"]))

    if role == "assistant":
        reasoning = msg.get("reasoning_content")
        has_thinking = any(b.get("type") == "thinking" for b in blocks)
        if isinstance(reasoning, str) and reasoning and not has_thinking:
            blocks.insert(0, {"type": "thinking", "text": reasoning})

    if not blocks:
        # Preserve an empty text block so strict validation reports a useful
        # empty-blocks error with full message context.
        blocks = [{"type": "text", "text": ""}]

    out: dict = {"role": role, "blocks": blocks}
    for passthrough in ("weight", "name", "task"):
        if passthrough in msg and msg[passthrough] is not None:
            out[passthrough] = msg[passthrough]
    return out


def to_ir_messages(messages: list) -> list[dict]:
    """Adapt a list of HF / loose-shape messages to titaninfra IR.

    Single chokepoint called from the converter before strict
    ``validate_or_raise``. Already-IR rows are preserved unchanged so
    mixing IR and HF-style fixtures in one source is safe.
    """
    if not isinstance(messages, list):
        raise TypeError(
            f"messages must be a list, got {type(messages).__name__}"
        )
    return [_to_ir_one_message(m) for m in messages]
