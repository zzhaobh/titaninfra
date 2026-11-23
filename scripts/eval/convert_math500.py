"""Convert the MATH-500 fixture into HF JSONL and TitanInfra MDS formats.

The problem field becomes the user prompt and solution becomes the assistant
response. No synthetic thinking block is added.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATH500_RAW = os.path.join(REPO_ROOT, "tests/fixtures/datasets/math500.jsonl")
SFT_JSONL = os.path.join(REPO_ROOT, "data/math500/sft/math500_hf.jsonl")
MDS_JSONL = os.path.join(REPO_ROOT, "data/math500/sft/math500_titaninfra.jsonl")
MDS_OUTPUT = os.path.join(REPO_ROOT, "data/math500/mds_sft/train")


def raw_to_hf_record(raw: dict) -> dict:
    """Convert one MATH-500 record to HF content messages."""
    return {
        "messages": [
            {"role": "user", "content": raw["problem"]},
            {"role": "assistant", "content": raw["solution"]},
        ],
    }


def raw_to_titaninfra_record(raw: dict) -> dict:
    """Convert one MATH-500 record to TitanInfra rich blocks."""
    return {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": raw["problem"]}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": raw["solution"]}]},
        ],
    }


def convert_to_jsonl():
    """Write HF and TitanInfra JSONL versions of the fixture."""
    os.makedirs(os.path.dirname(SFT_JSONL), exist_ok=True)

    count = 0
    with open(MATH500_RAW) as fin, \
         open(SFT_JSONL, "w") as f_hf, \
         open(MDS_JSONL, "w") as f_titaninfra:
        for line in fin:
            raw = json.loads(line)
            f_hf.write(json.dumps(raw_to_hf_record(raw), ensure_ascii=False) + "\n")
            f_titaninfra.write(json.dumps(raw_to_titaninfra_record(raw), ensure_ascii=False) + "\n")
            count += 1

    print(f"HF JSONL:      {SFT_JSONL} ({count} samples)")
    print(f"TitanInfra JSONL: {MDS_JSONL} ({count} samples)")
    return count


def convert_to_mds(tokenizer_path: str):
    """Convert the TitanInfra JSONL fixture to MDS."""
    from titaninfra.data.converter import convert_messages_to_mds

    stats = convert_messages_to_mds(
        source=MDS_JSONL,
        output=MDS_OUTPUT,
        tokenizer_path=tokenizer_path,
        family="qwen3",
        num_shards=8,
        num_workers=4,
        overwrite=True,
        compression="zstd",
    )
    print(f"MDS: {MDS_OUTPUT}")
    print(f"  samples: {stats['num_samples']}")
    print(f"  avg tokens: {stats['avg_tokens_per_sample']:.0f}")
    print(f"  total tokens: {stats['total_tokens']}")


def main():
    parser = argparse.ArgumentParser(description="Convert MATH-500 to SFT formats")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path; when supplied, also generate MDS",
    )
    args = parser.parse_args()

    convert_to_jsonl()

    if args.tokenizer:
        convert_to_mds(args.tokenizer)
    else:
        print("\nNo --tokenizer supplied; skipping MDS conversion.")
        print("Run with: python scripts/eval/convert_math500.py --tokenizer /models/Qwen3-8B")


if __name__ == "__main__":
    main()
