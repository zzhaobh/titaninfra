"""Convert an SFT JSONL file or directory to MDS using a native family encoder."""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Input JSONL file or directory")
    parser.add_argument("--output", required=True, help="MDS output directory")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path")
    parser.add_argument(
        "--family",
        required=True,
        choices=["qwen3", "qwen2_5", "llama3", "deepseek_v3", "deepseek_v4", "gpt_oss", "gemma4"],
        help="Native SFT encoder family",
    )
    parser.add_argument("--name", default="qwen3", help="Tokenizer key stored in stream_stats.json")
    parser.add_argument("--n-shards", type=int, default=16)
    parser.add_argument("--clean", action="store_true", help="Remove an existing output directory first")
    args = parser.parse_args()

    import os
    import shutil

    if args.clean and os.path.isdir(args.output):
        shutil.rmtree(args.output)

    from titaninfra.data.converter import convert_messages_to_mds

    stats = convert_messages_to_mds(
        source=args.source,
        output=args.output,
        tokenizer_path=args.tokenizer,
        family=args.family,
        num_shards=args.n_shards,
        overwrite=True,
        tokenizer_name=args.name,
    )
    logger.info("Converted %s samples to %s", stats["num_samples"], args.output)


if __name__ == "__main__":
    main()
