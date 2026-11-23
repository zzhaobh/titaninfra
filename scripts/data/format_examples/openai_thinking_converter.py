"""Convert OpenAI messages with <think> tags to rich TitanInfra blocks.

Assistant thinking content becomes a ThinkingBlock without the tags, and the
remaining answer becomes a TextBlock. The default thinking loss weight is 0.5.

Example:
    convert_messages_to_mds(
        source="/data/distill.jsonl",
        output="/data/distill/mds/train",
        tokenizer_path="/models/Qwen3-8B",
        family="qwen3",
        message_converter_fn=convert_openai_thinking,
        num_shards=16,
    )
"""

import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

THINKING_WEIGHT = 0.5


def convert_openai_thinking(record: dict) -> dict:
    """Split assistant <think> sections into ThinkingBlock values."""
    raw_messages = record.get("messages", [])
    if not raw_messages:
        return {}

    messages = []
    has_thinking = False

    for msg in raw_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            match = _THINK_RE.search(content)
            if match:
                has_thinking = True
                think_text = match.group(1).strip()
                rest_text = content[match.end():].strip()
                blocks = []
                if think_text:
                    blocks.append({"type": "thinking", "text": think_text})
                if rest_text:
                    blocks.append({"type": "text", "text": rest_text})
                if blocks:
                    messages.append({"role": "assistant", "blocks": blocks})
                continue

        if role == "system":
            messages.append({"role": "system", "content": content})
        else:
            messages.append({
                "role": role,
                "blocks": [{"type": "text", "text": content}],
            })

    if len(messages) < 2:
        return {}

    result = {"messages": messages}
    if has_thinking:
        result["loss_config"] = {"type_overrides": {"thinking": THINKING_WEIGHT}}
    return result
