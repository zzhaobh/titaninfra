"""Convert Alpaca-style records to TitanInfra message records.

Typical input: {"instruction": "...", "input": "...", "output": "..."}.
The optional input context is appended to the instruction. Records using prompt
instead of instruction are also accepted.

Example:
    convert_messages_to_mds(
        source="/data/alpaca.jsonl",
        output="/data/alpaca/mds/train",
        tokenizer_path="/models/Qwen3-8B",
        family="qwen3",
        message_converter_fn=convert_alpaca,
        num_shards=16,
    )
"""


def convert_alpaca(record: dict) -> dict:
    """Convert one Alpaca record and append any extra input context."""
    instruction = record.get("instruction") or record.get("prompt", "")
    instruction = instruction.strip()

    extra_input = record.get("input", "").strip()
    if extra_input:
        instruction = f"{instruction}\n\n{extra_input}"

    output = record.get("output", "").strip()

    if not instruction or not output:
        return {}

    return {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": instruction}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": output}]},
        ]
    }
