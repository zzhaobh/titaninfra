"""Convert single-turn Q&A records to TitanInfra messages.

Accepted question fields include question, input, prompt, and instruction.
Accepted answer fields include answer, response, output, and solution.

Example:
    convert_messages_to_mds(
        source="/data/qa.parquet",
        output="/data/qa/mds/train",
        tokenizer_path="/models/Qwen2.5-7B-Instruct",
        family="qwen2_5",
        message_converter_fn=convert_qa,
        num_shards=16,
    )
"""


def convert_qa(record: dict) -> dict:
    """Convert one Q&A record using common field aliases."""
    q = (
        record.get("question")
        or record.get("input")
        or record.get("prompt")
        or record.get("instruction", "")
    )
    a = (
        record.get("answer")
        or record.get("response")
        or record.get("output")
        or record.get("solution", "")
    )

    q = q.strip() if isinstance(q, str) else ""
    a = a.strip() if isinstance(a, str) else ""

    if not q or not a:
        return {}

    return {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": q}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": a}]},
        ]
    }


def convert_qa_with_system(record: dict) -> dict:
    """Convert Q&A and prepend an optional system or context prompt."""
    base = convert_qa(record)
    if not base:
        return {}

    system = record.get("system") or record.get("context") or record.get("system_prompt", "")
    system = system.strip() if isinstance(system, str) else ""

    if system:
        base["messages"].insert(0, {"role": "system", "content": system})

    return base
