"""Inspect MDS stream statistics and preview a small number of samples.

Usage:
    python scripts/tools/inspect_mds.py /path/to/mds_dir
    python scripts/tools/inspect_mds.py /path/to/mds_dir --n 5 --show-full
    python scripts/tools/inspect_mds.py /path/to/mds_dir --stats-only
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def show_stats(mds_dir: str) -> None:
    stats_path = os.path.join(mds_dir, "stream_stats.json")
    if not os.path.exists(stats_path):
        print(f"  [!] stream_stats.json 不存在: {stats_path}")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    print(f"  stream_id:    {stats.get('stream_id', '?')}")
    print(f"  stream_type:  {stats.get('stream_type', '?')}")
    print(f"  num_samples:  {stats.get('num_samples', 0):,}")
    print(f"  version:      {stats.get('version', '?')}")
    print(f"  created_at:   {stats.get('created_at', '?')}")

    for tok_name, tok_stats in stats.get("tokenizer_stats", {}).items():
        print(f"\n  tokenizer [{tok_name}]:")
        print(f"    total_tokens:          {tok_stats['total_tokens']:,}")
        print(f"    avg_tokens_per_sample: {tok_stats['avg_tokens_per_sample']:.1f}")
        if tok_stats.get("max_token_length"):
            print(f"    max_token_length:      {tok_stats['max_token_length']:,}")
        if tok_stats.get("min_token_length"):
            print(f"    min_token_length:      {tok_stats['min_token_length']:,}")
        if tok_stats.get("token_length_dist"):
            print(f"    length_dist:           {tok_stats['token_length_dist']}")


def show_samples(mds_dir: str, n: int, show_full: bool) -> None:
    try:
        from streaming import StreamingDataset
    except ImportError:
        print("\n  [!] mosaicml-streaming 未安装，无法读取样本")
        return

    ds = StreamingDataset(local=mds_dir, shuffle=False)
    total = len(ds)
    print(f"\n  总样本数 (from index): {total:,}")

    indices = list(range(min(n, total)))
    print(f"  展示前 {len(indices)} 条:\n")

    for idx in indices:
        sample = ds[idx]
        print(f"  {'─' * 60}")
        print(f"  [样本 {idx}]")

        for col_name, col_value in sample.items():
            if isinstance(col_value, bytes):
                text = col_value.decode("utf-8", errors="replace")
            elif isinstance(col_value, str):
                text = col_value
            else:
                text = repr(col_value)

            if col_name == "messages":
                try:
                    msgs = json.loads(text)
                    print(f"    {col_name}: ({len(msgs)} messages)")
                    for i, msg in enumerate(msgs):
                        role = msg.get("role", "?")
                        blocks = msg.get("blocks", [])
                        content = msg.get("content", "")

                        if blocks:
                            block_summary = ", ".join(
                                f"{b['type']}({len(b.get('text', ''))}chars)"
                                for b in blocks
                            )
                            preview = blocks[0].get("text", "")[:120]
                        elif content:
                            block_summary = f"content({len(content)}chars)"
                            preview = content[:120]
                        else:
                            block_summary = "empty"
                            preview = ""

                        print(f"      [{i}] {role}: {block_summary}")
                        if show_full:
                            for b in blocks:
                                txt = b.get("text", "")
                                print(f"          <{b['type']}> {txt[:500]}")
                                if len(txt) > 500:
                                    print(f"          ... ({len(txt)} chars total)")
                        else:
                            if preview:
                                print(f"          {preview}...")
                except (json.JSONDecodeError, TypeError):
                    print(f"    {col_name}: {text[:200]}")
            else:
                display = text if len(text) < 100 else f"{text[:100]}... ({len(text)} chars)"
                print(f"    {col_name}: {display}")

    print(f"\n  {'─' * 60}")


def main():
    parser = argparse.ArgumentParser(description="MDS 数据集检查工具")
    parser.add_argument("mds_dir", help="MDS 数据目录路径")
    parser.add_argument("--n", type=int, default=3, help="抽样条数 (default: 3)")
    parser.add_argument("--show-full", action="store_true", help="展示完整 block 内容")
    parser.add_argument("--stats-only", action="store_true", help="仅展示统计信息")
    args = parser.parse_args()

    if not os.path.isdir(args.mds_dir):
        print(f"错误: 目录不存在 {args.mds_dir}")
        sys.exit(1)

    print(f"\nMDS 目录: {args.mds_dir}")
    print(f"{'=' * 60}")

    files = os.listdir(args.mds_dir)
    shard_files = [f for f in files if f.endswith(".mds")]
    print(f"  shard 文件数: {len(shard_files)}")
    print(f"  其他文件:     {[f for f in files if not f.endswith('.mds')]}")

    print(f"\n{'=' * 60}")
    print("  stream_stats.json:")
    show_stats(args.mds_dir)

    if not args.stats_only:
        show_samples(args.mds_dir, args.n, args.show_full)


if __name__ == "__main__":
    main()
