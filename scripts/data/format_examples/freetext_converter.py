"""Convert free-text speaker dialogues to TitanInfra message records.

The parser detects "Speaker: text" sections, maps the first two distinct
speakers to user/assistant, and merges consecutive messages with the same role.

Example:
    convert_messages_to_mds(
        source="/data/dialogues.parquet",
        output="/data/dialogues/mds/train",
        tokenizer_path="/models/Qwen3-8B",
        family="qwen3",
        message_converter_fn=convert_freetext,
        num_shards=16,
    )
"""

import re

_SPEAKER_RE = re.compile(
    r"(?:^|\n\n)"
    r"([A-Z][\w]*(?:[^\S\n]+[\w]+){0,3})"
    r":\s",
)


def convert_freetext(record: dict) -> dict:
    """Detect speakers and convert a free-text dialogue to messages."""
    content = record.get("content", "")
    if not content:
        return {}

    matches = list(_SPEAKER_RE.finditer(content))
    if len(matches) < 2:
        return {}

    role_map: dict[str, str] = {}
    role_idx = 0
    roles = ["user", "assistant"]

    messages: list[dict] = []
    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        if speaker not in role_map:
            role_map[speaker] = roles[min(role_idx, 1)]
            role_idx += 1

        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[m.end():text_end].strip()
        if not text:
            continue

        role = role_map[speaker]
        if messages and messages[-1]["role"] == role:
            messages[-1]["blocks"][0]["text"] += "\n\n" + text
        else:
            messages.append({
                "role": role,
                "blocks": [{"type": "text", "text": text}],
            })

    if len(messages) < 2:
        return {}
    return {"messages": messages}
