"""Convert ShareGPT conversations to TitanInfra message records.

The standard form uses conversations with from/value entries. The alternate
converter also accepts role/content aliases.

Example:
    convert_messages_to_mds(
        source="/data/sharegpt.jsonl",
        output="/data/sharegpt/mds/train",
        tokenizer_path="/models/Qwen3-8B",
        family="qwen3",
        message_converter_fn=convert_sharegpt,
        num_shards=16,
    )
"""

ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def convert_sharegpt(record: dict) -> dict:
    """Convert standard ShareGPT conversations with from/value fields."""
    conversations = record.get("conversations", [])
    if not conversations:
        return {}

    messages = []
    for turn in conversations:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        messages.append({
            "role": role,
            "blocks": [{"type": "text", "text": turn["value"]}],
        })

    if len(messages) < 2:
        return {}
    return {"messages": messages}


def convert_sharegpt_alt(record: dict) -> dict:
    """Convert ShareGPT variants using role/content aliases."""
    conversations = record.get("conversations", [])
    if not conversations:
        return {}

    messages = []
    for turn in conversations:
        from_field = turn.get("from") or turn.get("role", "")
        value_field = turn.get("value") or turn.get("content", "")
        role = ROLE_MAP.get(from_field, from_field)
        messages.append({
            "role": role,
            "blocks": [{"type": "text", "text": value_field}],
        })

    if len(messages) < 2:
        return {}
    return {"messages": messages}
