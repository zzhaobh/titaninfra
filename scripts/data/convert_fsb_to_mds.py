"""Convert fsb_v1_think JSONL to TitanInfra MDS.

User turns become TextBlock messages. Assistant <think> content becomes a
ThinkingBlock without tags, and subsequent answer text becomes a TextBlock.
The native renderer restores the tags when constructing training sequences.

Usage:
    python scripts/data/convert_fsb_to_mds.py \
        --input /path/to/fsb_v1_think.jsonl \
        --tokenizer /path/to/tokenizer [--n-shards 16] [--clean]
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

OUTPUT_JSONL = os.path.join(REPO_ROOT, "data/fsb/sft/train/fsb_v1_think.jsonl")
OUTPUT_MDS_DIR = os.path.join(REPO_ROOT, "data/fsb/mds_sft/train")

# Cross-model special tokens that should never appear in training text content.
# These are BOS/EOS/header tokens from LLaMA3, LLaMA2, Mistral, Gemma, DeepSeek, etc.
# When training data is sourced from models using these templates, their special tokens
# can leak into the text content and poison gradient updates on the target model (Qwen3).
_FOREIGN_SPECIAL_TOKENS = re.compile(
    r"<\|begin_of_text\|>"       # LLaMA3 BOS
    r"|<\|end_of_text\|>"        # LLaMA3 EOS
    r"|<\|eot_id\|>"             # LLaMA3 end-of-turn
    r"|<\|start_header_id\|>.*?<\|end_header_id\|>"  # LLaMA3 role headers
    r"|<\|python_tag\|>"         # LLaMA3 code tag
    r"|\[INST\]|\[/INST\]"       # LLaMA2 / Mistral
    r"|<<SYS>>|<</SYS>>"         # LLaMA2 system block
    r"|<start_of_turn>|<end_of_turn>"   # Gemma
    r"|<｜begin▁of▁sentence｜>|<｜end▁of▁sentence｜>",  # DeepSeek V3
    re.DOTALL,
)

_contamination_count = 0


def _sanitize_text(text: str, label: str = "") -> str:
    """Strip cross-model special tokens from plain text content."""
    global _contamination_count
    cleaned, n = _FOREIGN_SPECIAL_TOKENS.subn("", text)
    if n:
        _contamination_count += 1
        logger.warning(f"Stripped {n} foreign special token(s) from {label or 'text'}")
    return cleaned


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _parse_assistant_blocks(content: str) -> list[dict]:
    """Parse assistant content into ThinkingBlock + TextBlock list.

    Extracts pure text inside <think>...</think> as ThinkingBlock (without tags).
    The template will wrap it with <think>\\n...\\n</think>\\n\\n at runtime.
    """
    blocks = []
    match = THINK_PATTERN.search(content)

    if match:
        think_text = _sanitize_text(match.group(1).strip(), label="thinking block")
        rest_text = _sanitize_text(content[match.end() :].strip(), label="text block")

        if think_text:
            blocks.append({"type": "thinking", "text": think_text})
        if rest_text:
            blocks.append({"type": "text", "text": rest_text})
    else:
        blocks.append({"type": "text", "text": _sanitize_text(content.strip(), label="text block")})

    return blocks


def _convert_row(messages: list[dict]) -> dict | None:
    """Single record → titaninfra messages dict."""
    converted = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            converted.append({"role": "system", "content": _sanitize_text(content.strip(), label="system")})
        elif role == "user":
            user_text = _sanitize_text(content.strip(), label="user")
            converted.append({"role": "user", "blocks": [{"type": "text", "text": user_text}]})
        elif role == "assistant":
            blocks = _parse_assistant_blocks(content)
            if blocks:
                converted.append({"role": "assistant", "blocks": blocks})
        else:
            logger.warning(f"Unknown role: {role}, skipping")

    if len(converted) < 2:
        return None

    return {"messages": converted}


def _convert_line(line: str) -> tuple[str | None, int]:
    """Worker: single raw JSONL line → (titaninfra JSON str | None, foreign-token count).

    Module-level + pickle-safe for multiprocessing. Contamination is detected
    by scanning the raw line directly (avoids the cross-process global counter).
    Returns (None, 0) for blank/invalid/too-short records (caller treats as skipped).
    """
    line = line.strip()
    if not line:
        return None, 0

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None, 0

    messages = record.get("messages")
    if not messages:
        return None, 0

    n_contam = len(_FOREIGN_SPECIAL_TOKENS.findall(line))

    result = _convert_row(messages)
    if result is None:
        return None, n_contam

    return json.dumps(result, ensure_ascii=False), n_contam


def convert_jsonl_to_titaninfra(input_path: str, num_workers: int = 8, chunksize: int = 200) -> int:
    """Source JSONL → titaninfra SFT JSONL (parallel)."""
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    count = 0
    skipped = 0
    contaminated = 0
    processed = 0

    def _line_reader():
        with open(input_path, encoding="utf-8") as fin:
            yield from fin

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        with mp.Pool(processes=num_workers) as pool:
            for out_str, n_contam in pool.imap(_convert_line, _line_reader(), chunksize=chunksize):
                processed += 1
                if n_contam:
                    contaminated += 1
                if out_str is None:
                    skipped += 1
                else:
                    fout.write(out_str + "\n")
                    count += 1

                if processed % 50_000 == 0:
                    logger.info(f"  ... processed {processed} lines ({count} kept, {skipped} skipped)")

    if skipped:
        logger.info(f"Skipped {skipped} invalid/short records")
    if contaminated:
        logger.warning(
            f"Found foreign special tokens in {contaminated} samples — "
            f"stripped during conversion. Re-generate MDS to apply fix."
        )
    return count


def convert_to_mds(
    tokenizer_path: str,
    n_shards: int = 16,
    output_dir: str | None = None,
    num_workers: int = 8,
    num_tokenizer_workers: int = 16,
) -> None:
    """TitanInfra JSONL → MDS."""
    from titaninfra.data.converter import convert_messages_to_mds

    mds_dir = output_dir or OUTPUT_MDS_DIR

    stats = convert_messages_to_mds(
        source=OUTPUT_JSONL,
        output=mds_dir,
        tokenizer_path=tokenizer_path,
        family="qwen3",
        num_shards=n_shards,
        num_workers=num_workers,
        num_tokenizer_workers=num_tokenizer_workers,
        compression=None,
        overwrite=True,
    )
    logger.info(
        f"MDS done: {stats['num_samples']} samples, "
        f"{stats['total_tokens']} tokens, "
        f"avg {stats['avg_tokens_per_sample']:.0f} tokens/sample"
    )


def main():
    parser = argparse.ArgumentParser(description="fsb_v1_think → titaninfra MDS")
    parser.add_argument("--input", required=True, help="Source JSONL path")
    parser.add_argument("--tokenizer", help="HF tokenizer path (required for mds/all)")
    parser.add_argument("--step", choices=["jsonl", "mds", "all"], default="all")
    parser.add_argument("--n-shards", type=int, default=16)
    parser.add_argument("--output-dir", help="MDS output directory (overrides default)")
    parser.add_argument("--clean", action="store_true", help="Remove existing MDS before regeneration")
    parser.add_argument("--num-workers", type=int, default=8, help="Reader processes for MDS conversion")
    parser.add_argument("--num-tokenizer-workers", type=int, default=16, help="Tokenizer threads (Rust, GIL-free)")
    args = parser.parse_args()

    if args.step in ("mds", "all") and not args.tokenizer:
        parser.error("--tokenizer is required for mds/all step")

    if args.step in ("jsonl", "all"):
        logger.info("Step 1: Source JSONL → titaninfra JSONL")
        count = convert_jsonl_to_titaninfra(args.input, num_workers=args.num_workers)
        logger.info(f"  Wrote {count} records → {OUTPUT_JSONL}")

    if args.step in ("mds", "all"):
        mds_dir = args.output_dir or OUTPUT_MDS_DIR
        if os.path.exists(mds_dir) and not args.clean:
            logger.info(f"Already exists, skipping: {mds_dir} (use --clean to regenerate)")
            return
        if os.path.exists(mds_dir) and args.clean:
            import shutil

            shutil.rmtree(mds_dir)

        logger.info("Step 2: titaninfra JSONL → MDS")
        convert_to_mds(args.tokenizer, args.n_shards, args.output_dir, args.num_workers, args.num_tokenizer_workers)


if __name__ == "__main__":
    main()
