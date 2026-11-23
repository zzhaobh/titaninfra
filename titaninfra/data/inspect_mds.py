"""Small MDS inspection helpers built on StreamingDataset."""

from typing import Any


def inspect_mds(path: str, n: int = 5, shuffle: bool = False) -> list[dict[str, Any]]:
    """Return up to n samples from an MDS directory.

    Args:
        path: MDS directory containing index.json and shards.
        n: Maximum number of samples to return.
        shuffle: Whether to shuffle dataset iteration.
    """
    from streaming import StreamingDataset

    ds = StreamingDataset(local=path, shuffle=shuffle)

    samples = []
    for i in range(min(n, len(ds))):
        samples.append(dict(ds[i]))

    return samples


def get_mds_info(path: str, shuffle: bool = False) -> dict[str, Any]:
    """Return the sample count, shard count, and path for an MDS directory."""
    from streaming import StreamingDataset

    ds = StreamingDataset(local=path, shuffle=shuffle)

    return {
        "num_samples": len(ds),
        "num_shards": len(ds.shards),
        "path": path,
    }


def print_mds_samples(
    path: str,
    n: int = 5,
    max_len: int = 500,
    text_key: str = "text",
) -> None:
    """Print summary information and up to n samples from an MDS directory."""
    samples = inspect_mds(path, n)
    info = get_mds_info(path)

    print(f"MDS directory: {path}")
    print(f"Samples: {info['num_samples']}")
    print(f"Shards: {info['num_shards']}")
    print("=" * 60)

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i} ---")
        for key, value in sample.items():
            if key == text_key and isinstance(value, str):
                display = value[:max_len] + "..." if len(value) > max_len else value
                print(f"{key}: {display}")
            elif isinstance(value, (list, tuple)) and len(value) > 10:
                print(f"{key}: {type(value).__name__}(len={len(value)}, first={value[:3]}...)")
            elif isinstance(value, bytes):
                print(f"{key}: bytes(len={len(value)})")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m titaninfra.data.inspect_mds <mds_path> [n]")
        print("Example: python -m titaninfra.data.inspect_mds ./streaming_data 5")
        sys.exit(1)

    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print_mds_samples(path, n)
