"""Recompute token statistics for existing MDS datasets without rewriting shards.

Text uses tokenizer.encode, while messages use the same native
 dispatch.tokenize_sample path as runtime training. Updated statistics and
 tokenizer assets are written back to stream_stats.json.

Example:
    python -m titaninfra.data.recompute_stats \
        --mds-path /data/sft/mds/train \
        --tokenizer /models/Qwen2.5-7B-Instruct \
        --name qwen2_5 --data-type messages --family qwen2_5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def recompute_token_stats(
    mds_path: str,
    tokenizer_path: str,
    data_type: str = "text",
    family: str | None = None,
    num_workers: int | None = None,
) -> tuple[int, int, int, int, dict[str, int], dict[str, int]]:
    """Read MDS samples and recompute token counts and length histograms.

    Tokenization runs concurrently with ThreadPoolExecutor. Messages require
    family so statistics use the same native encoder as training. Samples
    dropped by EmptyAssistantError are excluded from counts and histograms.
    """
    from streaming import StreamingDataset
    from transformers import AutoTokenizer

    from titaninfra.data.converter import _empty_hist, _tc_to_bucket, _tc_to_fine_bucket

    if data_type == "messages" and not family:
        raise ValueError(
            "data_type='messages' requires --family (e.g. 'qwen3', 'llama3'). "
            "Token counts must be computed via the family's native encoder so "
            "stream_stats.json aligns with runtime dispatch.tokenize_sample."
        )

    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    ds = StreamingDataset(local=mds_path, shuffle=False)

    try:
        total_in_mds = len(ds)

        # Stage 1: read samples sequentially before parallel tokenization.
        samples = []
        for i in range(total_in_mds):
            samples.append(ds[i])
            if (i + 1) % 5000 == 0:
                logger.info(f"读取进度: {i + 1}/{total_in_mds}")
    finally:
        import gc

        if hasattr(ds, "_locals_shm") and ds._locals_shm is not None:
            try:
                import numpy as np
                ds._locals_shm.buf[:4] = np.int32(0).tobytes()
            except Exception:
                pass
        del ds
        gc.collect()

    # Stage 2: tokenize concurrently using the GIL-releasing HF tokenizer.
    if data_type == "messages":
        from titaninfra.data.sft import dispatch
        from titaninfra.data.sft.capabilities import EmptyAssistantError

        def _tokenize_one(sample):
            """Returns token count, or 0 if dispatch dropped the sample.

            EmptyAssistantError drops a sample by returning 0 so it cannot
            distort counts or histograms.
            """
            messages_raw = sample.get("messages", "")
            if isinstance(messages_raw, bytes):
                messages_raw = messages_raw.decode("utf-8")
            messages = json.loads(messages_raw)
            system_raw = sample.get("system")
            if isinstance(system_raw, bytes):
                system_raw = system_raw.decode("utf-8")
            tools_raw = sample.get("tools")
            if isinstance(tools_raw, bytes):
                tools_raw = tools_raw.decode("utf-8")
            if isinstance(tools_raw, str) and tools_raw:
                tools_raw = json.loads(tools_raw)
            sample_json = {"messages": messages, "system": system_raw, "tools": tools_raw}
            try:
                encoded = dispatch.tokenize_sample(
                    sample_json,
                    tokenizer,
                    family,
                    max_length=None,
                    loss_config=None,
                )
            except EmptyAssistantError as exc:
                logger.warning("Dropping sample (family=%s): %s", family, exc)
                return 0
            return len(encoded.input_ids)
    else:
        def _tokenize_one(sample):
            text = sample.get("text", "")
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            return len(tokenizer.encode(text, add_special_tokens=False))

    total_tokens = 0
    max_tc = 0
    min_tc = 0
    hist = _empty_hist()
    fine_hist: dict[str, int] = {}
    dropped = 0
    counted_samples = 0

    # Stage 3: aggregate token counts in input order.
    logger.info(f"开始并行 tokenize: {total_in_mds} samples, {num_workers} workers")
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for i, tc in enumerate(pool.map(_tokenize_one, samples, chunksize=64)):
            if tc <= 0:
                dropped += 1
                continue
            counted_samples += 1
            total_tokens += tc
            if tc > max_tc:
                max_tc = tc
            if min_tc == 0 or tc < min_tc:
                min_tc = tc
            hist[_tc_to_bucket(tc)] += 1
            fine_key = _tc_to_fine_bucket(tc)
            fine_hist[fine_key] = fine_hist.get(fine_key, 0) + 1
            if (i + 1) % 1000 == 0:
                logger.info(f"tokenize 进度: {i + 1}/{total_in_mds}, {total_tokens:,} tokens")

    if dropped:
        logger.warning(
            f"{dropped} samples dropped (EmptyAssistantError); "
            f"stats computed over {counted_samples} of {total_in_mds} samples"
        )

    logger.info(f"统计完成: {counted_samples} samples, {total_tokens:,} tokens, "
                f"avg={total_tokens / max(counted_samples, 1):.1f}, max={max_tc}, min={min_tc}")
    return counted_samples, total_tokens, max_tc, min_tc, hist, fine_hist


def add_tokenizer_to_mds(
    mds_path: str,
    tokenizer_path: str,
    tokenizer_name: str,
    data_type: str = "text",
    family: str | None = None,
    set_default: bool = False,
    num_workers: int | None = None,
) -> None:
    """Recompute statistics, update stream_stats.json, and embed assets.

    set_default also makes tokenizer_name the default key used by runtime
    scheduling.
    """
    from datetime import UTC, datetime

    from titaninfra.data.schema import StreamStats

    num_samples, total_tokens, max_tc, min_tc, hist, fine_hist = recompute_token_stats(
        mds_path, tokenizer_path, data_type, family, num_workers=num_workers,
    )

    stream_stats = StreamStats.load(mds_path)
    if stream_stats is None:
        raise FileNotFoundError(f"stream_stats.json 不存在: {mds_path}")

    now_iso = datetime.now(UTC).isoformat()
    stream_stats.add_tokenizer_stats(
        tokenizer_name, total_tokens, now_iso,
        max_token_length=max_tc,
        min_token_length=min_tc,
        token_length_dist=hist,
        fine_token_length_dist=fine_hist,
    )

    if set_default or not stream_stats.default_tokenizer:
        stream_stats.default_tokenizer = tokenizer_name

    stream_stats.save(mds_path)

    logger.info(
        f"已添加 tokenizer '{tokenizer_name}' 到 {mds_path}/stream_stats.json "
        f"({total_tokens:,} tokens, default={'是' if set_default else '否'})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为已有 MDS 数据集添加新 tokenizer 的 token 统计",
    )
    parser.add_argument("--mds-path", required=True, help="MDS 数据目录")
    parser.add_argument("--tokenizer", required=True, help="tokenizer 路径")
    parser.add_argument("--name", required=True, help="tokenizer 短名（key + 目录名）")
    parser.add_argument("--data-type", default="text", choices=["text", "messages"])
    parser.add_argument(
        "--family", default=None,
        help="SFT encoder 家族名（messages 模式必填）。合法值：qwen3 / qwen2_5 / "
             "llama3 / deepseek_v3 / deepseek_v4 / gpt_oss / gemma4。",
    )
    parser.add_argument("--set-default", action="store_true", help="设为默认 tokenizer")
    parser.add_argument("--workers", type=int, default=None, help="并行 tokenize 线程数（默认 min(cpu_count, 8)）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.isdir(args.mds_path):
        print(f"错误: MDS 目录不存在: {args.mds_path}", file=sys.stderr)
        sys.exit(1)

    start = time.time()
    add_tokenizer_to_mds(
        mds_path=args.mds_path,
        tokenizer_path=args.tokenizer,
        tokenizer_name=args.name,
        data_type=args.data_type,
        family=args.family,
        set_default=args.set_default,
        num_workers=args.workers,
    )
    elapsed = time.time() - start
    print(f"完成 ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
