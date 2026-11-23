"""Convert mathematical dialogues from Parquet to TitanInfra messages.

Input content contains "Speaker: text" sections. The module-level parser is
picklable for multiprocessing and can also be used as message_converter_fn.

Example:
    convert_messages_to_mds(
        source="/data/math.parquet",
        output="/data/math/mds/train",
        tokenizer_path="/models/Qwen3-8B",
        family="qwen3",
        message_converter_fn=parse_math_dialogue,
        num_shards=256,
        num_workers=32,
    )
"""

from __future__ import annotations

import json
import re
import sys

# Detect one to four words followed by a colon and whitespace. Restrict
# whitespace to non-newline characters so matches cannot cross paragraphs.
_SPEAKER_RE = re.compile(
    r"(?:^|\n\n)"
    r"([A-Z][\w]*(?:[^\S\n]+[\w]+){0,3})"
    r":\s",
)


def _parse_turns(content: str) -> list[tuple[str, str]]:
    """Split Speaker-prefixed content into (speaker, text) turns."""
    matches = list(_SPEAKER_RE.finditer(content))
    if not matches:
        return []

    turns: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[text_start:text_end].strip()
        if text:
            turns.append((speaker, text))
    return turns


def _map_roles(turns: list[tuple[str, str]]) -> list[dict]:
    """Map alternating unique speakers to user/assistant and merge runs."""
    if not turns:
        return []

    role_map: dict[str, str] = {}
    roles_cycle = ["user", "assistant"]
    role_idx = 0
    for speaker, _ in turns:
        if speaker not in role_map:
            role_map[speaker] = roles_cycle[role_idx % 2]
            role_idx += 1

    messages: list[dict] = []
    for speaker, text in turns:
        role = role_map[speaker]
        if messages and messages[-1]["role"] == role:
            messages[-1]["blocks"][0]["text"] += "\n\n" + text
        else:
            messages.append({
                "role": role,
                "blocks": [{"type": "text", "text": text}],
            })
    return messages


def parse_math_dialogue(record: dict) -> dict:
    """Convert one math dialogue record, or return {} for fewer than two turns.

    The function is module-level and picklable for multiprocessing converters.
    """
    content = record.get("content", "")
    if not content:
        return {}

    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)

    turns = _parse_turns(content)
    if len(turns) < 2:
        return {}

    messages = _map_roles(turns)
    if len(messages) < 2:
        return {}

    return {"messages": messages}


def _cli_convert(parquet_path: str, output_path: str) -> None:
    """Convert a Parquet file with uid/content columns to JSONL."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["uid", "content"])
    total = len(table)
    converted = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(total):
            record = {
                "uid": table.column("uid")[i].as_py(),
                "content": table.column("content")[i].as_py(),
            }
            result = parse_math_dialogue(record)
            if not result:
                skipped += 1
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Done: {converted} converted, {skipped} skipped, {total} total")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.parquet> <output.jsonl>")
        sys.exit(1)
    _cli_convert(sys.argv[1], sys.argv[2])
