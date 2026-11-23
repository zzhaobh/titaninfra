"""Threaded runtime image decoding with per-document prefetch.

A ThreadPoolExecutor decodes images inside dataset iteration without creating
PyTorch DataLoader workers. The main thread submits raw bytes, keeps a bounded
future queue, and consumes results in document order while tokenization and
packing continue. Workers share image_processor.preprocess_image as the single
preprocessing implementation and re-raise failures on result().
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from titaninfra.data.multimodal.image_processor import preprocess_image

if TYPE_CHECKING:
    import torch


def default_workers() -> int:
    """Return min(8, CPU count - 2), with a minimum of one worker."""
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu - 2))


class ImageDecodePool:
    """Manage an image-decode thread pool and bounded prefetch queue.

    The pool is created with a context manager and closed after iteration. Only
    the main thread calls map_docs; workers run stateless preprocessing.
    """

    def __init__(
        self,
        *,
        workers: int | None = None,
        prefetch: int | None = None,
        preprocess_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._workers = workers if workers is not None else default_workers()
        # Default prefetch depth is twice the worker count.
        self._prefetch = prefetch if prefetch is not None else 2 * self._workers
        self._preprocess_kwargs = preprocess_kwargs or {}
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> ImageDecodePool:
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _submit_one(self, raw: bytes) -> Future:
        """Submit one raw image and return its preprocessing future."""
        assert self._executor is not None, "ImageDecodePool 须在 with 块内使用"
        return self._executor.submit(
            preprocess_image, raw, **self._preprocess_kwargs
        )

    def map_docs(
        self,
        doc_source: Iterator[Any],
        images_of: Callable[[Any], list[bytes]],
    ) -> Iterator[tuple[Any, list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Decode each document's images concurrently while preserving order.

        Maintain at most prefetch in-flight documents. When the queue is full
        or the source ends, consume the oldest future results so backpressure
        stays bounded and failures propagate to the caller.
        """
        inflight: deque[tuple[Any, list[Future]]] = deque()

        def _drain_one() -> tuple[Any, list[tuple[torch.Tensor, torch.Tensor]]]:
            sample, futures = inflight.popleft()
            # result() re-raises preprocessing failures in the main thread.
            decoded = [f.result() for f in futures]
            return sample, decoded

        for sample in doc_source:
            raw_list = images_of(sample)
            futures = [self._submit_one(raw) for raw in raw_list]
            inflight.append((sample, futures))
            # Apply backpressure once the prefetch queue is full.
            while len(inflight) > self._prefetch:
                yield _drain_one()

        # Drain remaining futures after the source is exhausted.
        while inflight:
            yield _drain_one()
