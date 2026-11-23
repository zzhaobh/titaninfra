"""Process-safe MDS reader for parallel pytest workers.

Problem: MosaicML StreamingDataset uses /dev/shm/ SharedMemory with a
sequential prefix allocator (get_shm_prefix). When multiple pytest-xdist
workers create StreamingDataset concurrently, they race for the same prefix
→ FileExistsError. The xdist_group marker only works with --dist loadgroup
(not the default --dist load), so it doesn't prevent this.

Solution: Serialize ALL StreamingDataset creation/usage behind a single
file lock. Only one process can hold a StreamingDataset at any time.
After reading, we explicitly tear down all SharedMemory before releasing
the lock. This eliminates the race entirely regardless of --dist mode.

Usage:
    from tests._mds_utils import load_mds_samples, load_mds_messages

    samples = load_mds_samples("/path/to/mds")  # returns list[dict]
    messages = load_mds_messages("/path/to/mds")  # returns list[list[dict]]
"""

from __future__ import annotations

import gc
import glob
import json
import os
from multiprocessing.shared_memory import SharedMemory

import filelock

_LOCK_PATH = os.path.join(os.environ.get("TMPDIR", "/tmp"), "pytest_mosaicml_streaming.lock")
_LOCK = filelock.FileLock(_LOCK_PATH, timeout=120)


def _purge_all_mosaicml_shm() -> int:
    """Unlink ALL MosaicML shm entries from /dev/shm/."""
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return 0
    cleaned = 0
    for path in glob.glob(os.path.join(shm_dir, "[0-9][0-9][0-9][0-9][0-9][0-9]_*")):
        name = os.path.basename(path)
        try:
            shm = SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
            cleaned += 1
        except (FileNotFoundError, OSError):
            pass
    return cleaned


def load_mds_samples(
    mds_path: str,
    columns: list[str] | None = None,
) -> list[dict]:
    """Load all samples from an MDS directory, process-safe.

    Holds a file lock for the entire StreamingDataset lifecycle to prevent
    concurrent /dev/shm allocation races across pytest-xdist workers.

    Args:
        mds_path: Path to MDS directory (must contain index.json)
        columns: Columns to extract. Default: messages, stream_id, token_count

    Returns:
        List of dicts with decoded column values.
    """
    from streaming import StreamingDataset

    cols = columns or ["messages", "stream_id", "token_count"]

    with _LOCK:
        _purge_all_mosaicml_shm()
        ds = StreamingDataset(local=mds_path, shuffle=False, batch_size=1)
        try:
            samples = []
            for sample in ds:
                row = {}
                for col in cols:
                    val = sample.get(col)
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    if col == "messages":
                        val = json.loads(val) if val else []
                    elif col == "token_count":
                        val = int(val) if val else 0
                    elif col == "loss_config":
                        val = json.loads(val) if val else {}
                    row[col] = val
                samples.append(row)
        finally:
            del ds
            gc.collect()
            _purge_all_mosaicml_shm()

    return samples


def load_mds_messages(mds_path: str) -> list[list[dict]]:
    """Load just the messages column from MDS, process-safe.

    Convenience wrapper for alignment tests that only need messages.
    """
    samples = load_mds_samples(mds_path, columns=["messages"])
    return [s["messages"] for s in samples]
