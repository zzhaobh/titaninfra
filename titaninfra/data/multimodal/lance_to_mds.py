"""Convert interleaved Lance image-text documents to structured MDS.

Each output record contains text_segments, raw image bytes, nondecreasing
image_positions, decoded image_sizes, and est_tokens. Raw bytes are stored as
pkl rather than JPEGList so decode_pool can decode concurrently at runtime
instead of forcing synchronous PIL decoding in __getitem__.

A failed/empty image or alignment mismatch drops the whole document, preserving
position invariants. Parsing and token estimation are pure Python; Pillow,
pylance, and streaming are required only for their respective layers.
"""

from __future__ import annotations

import io
import json
import re
from typing import TYPE_CHECKING, Any

from titaninfra.data.multimodal.image_processor import estimate_image_tokens

if TYPE_CHECKING:
    from PIL import Image

# Defaults for Qwen3-VL-compatible image processing.
_DEFAULT_PATCH_SIZE = 16
_DEFAULT_SPATIAL_MERGE_SIZE = 2
_DEFAULT_TEMPORAL_PATCH_SIZE = 2
_DEFAULT_MIN_PIXELS = 65536
_DEFAULT_MAX_PIXELS = 16777216

# Each image also has vision_start and vision_end protocol tokens.
_VISION_WRAPPER_TOKENS = 2

_IMG_MARKER_RE = re.compile(r"\[IMG_(\d+)\]")


# Pure parsing without Lance or PIL dependencies.


def parse_mml_text(mml_text: str) -> tuple[list[str], list[int]]:
    """Split [IMG_i] markers into text_segments and image_positions.

    Position p inserts before segment p; position N appends after the last
    segment. Empty fragments do not increment segment count, so adjacent
    markers can share a position and order stays nondecreasing.
    """
    parts = _IMG_MARKER_RE.split(mml_text)
    # Capturing split yields alternating text fragments and marker indices.
    text_segments: list[str] = []
    image_positions: list[int] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Text fragment
            if part:  # Empty fragments produce no segment.
                text_segments.append(part)
        else:  # Marker position is the current segment count.
            image_positions.append(len(text_segments))
    return text_segments, image_positions


def estimate_est_tokens(
    text_segments: list[str],
    image_sizes: list[tuple[int, int]],
    tokenizer: Any,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
    patch_size: int = _DEFAULT_PATCH_SIZE,
    spatial_merge_size: int = _DEFAULT_SPATIAL_MERGE_SIZE,
    temporal_patch_size: int = _DEFAULT_TEMPORAL_PATCH_SIZE,
    min_pixels: int = _DEFAULT_MIN_PIXELS,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
) -> int:
    """Return the exact runtime input_ids length for one document.

    Encode each segment independently, add BOS/EOS when enabled, and add
    vision_start + N image pads + vision_end for each image. Segment-wise
    encoding mirrors runtime image boundaries exactly.
    """
    total = 0
    if add_bos:
        total += 1
    for seg in text_segments:
        total += len(tokenizer.encode(seg, add_special_tokens=False))
    for w, h in image_sizes:
        n = estimate_image_tokens(
            w,
            h,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        total += _VISION_WRAPPER_TOKENS + n
    if add_eos:
        total += 1
    return total


# Decode-to-validate gate (requires Pillow).


def decode_and_size(raw: bytes) -> tuple[Image.Image, int, int] | None:
    """Decode raw JPEG bytes to (image, width, height), or None on failure."""
    from PIL import Image

    if not raw:
        return None
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()  # Force decoding to catch truncated/corrupt images.
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im, int(im.width), int(im.height)
    except Exception:
        return None


def validate_and_prepare_doc(
    mml_text: str,
    media: list[dict],
    image_sizes_col: list[dict],
) -> tuple[list[str], list[bytes], list[int], list[tuple[int, int]]] | None:
    """Validate one document and return its four structured fields.

    Any failed image or marker/media/size-count mismatch returns None. Sizes
    are measured from decoded images rather than trusting source metadata.
    """
    text_segments, image_positions = parse_mml_text(mml_text)
    m = len(image_positions)

    # Marker, media, and source-size counts must match exactly.
    if not (m == len(media) == len(image_sizes_col)):
        return None

    images_bytes: list[bytes] = []
    image_sizes: list[tuple[int, int]] = []
    for md in media:
        raw = md.get("media_bytes") or b""
        decoded = decode_and_size(raw)
        if decoded is None:  # Drop the whole document on decode failure.
            return None
        _, w, h = decoded
        images_bytes.append(raw)
        image_sizes.append((w, h))

    return text_segments, images_bytes, image_positions, image_sizes


# Lance reading and MDS writing (requires pylance and streaming).

_MDS_COLUMNS: dict[str, str] = {
    "text_segments": "pkl",
    "images": "pkl",  # Raw bytes preserve concurrent runtime decoding.
    "image_positions": "pkl",
    "image_sizes": "pkl",
    "est_tokens": "int",
}


def convert_lance_to_mds(
    lance_path: str,
    output_dir: str,
    tokenizer: Any,
    *,
    stream_id: str = "",
    tokenizer_name: str | None = None,
    num_shards: int = 16,
    add_bos: bool = True,
    add_eos: bool = True,
    patch_size: int = _DEFAULT_PATCH_SIZE,
    spatial_merge_size: int = _DEFAULT_SPATIAL_MERGE_SIZE,
    temporal_patch_size: int = _DEFAULT_TEMPORAL_PATCH_SIZE,
    min_pixels: int = _DEFAULT_MIN_PIXELS,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    compression: str | None = None,
) -> dict:
    """Convert Lance documents to balanced MDS shards.

    Assign each document to the currently lightest shard by token count and
    return {samples, dropped, total_tokens}.
    """
    import heapq
    import logging
    import os

    import lance
    from streaming.base import MDSWriter
    from streaming.base.util import merge_index

    from titaninfra.data.converter import (
        _empty_hist,
        _tc_to_bucket,
        _tc_to_fine_bucket,
        _write_stream_stats,
    )

    logger = logging.getLogger(__name__)

    ds = lance.dataset(lance_path)
    rows = ds.to_table().to_pylist()

    shard_dirs = [
        os.path.join(output_dir, f"shard_{i:05d}") if num_shards > 1 else output_dir
        for i in range(num_shards)
    ]
    writers = [
        MDSWriter(out=sd, columns=_MDS_COLUMNS, compression=compression)
        for sd in shard_dirs
    ]
    heap: list[tuple[int, int]] = [(0, i) for i in range(num_shards)]
    heapq.heapify(heap)

    hist = _empty_hist()
    fine_hist: dict[str, int] = {}
    total_tokens = 0
    max_tc = 0
    min_tc = 0
    samples = 0
    dropped = 0

    for row in rows:
        rid = row.get("id", "")
        try:
            meta = json.loads(row.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            logger.warning("drop %s: metadata 非法 JSON", rid)
            dropped += 1
            continue
        mml_text = meta.get("mml_text", "")
        media = row.get("media") or []
        image_sizes_col = row.get("image_sizes") or []

        prepared = validate_and_prepare_doc(mml_text, media, image_sizes_col)
        if prepared is None:
            logger.warning("drop %s: invariant #1/#5 gate（decode 失败/空/尺寸/对齐）", rid)
            dropped += 1
            continue
        text_segments, images_bytes, image_positions, image_sizes = prepared

        est = estimate_est_tokens(
            text_segments,
            image_sizes,
            tokenizer,
            add_bos=add_bos,
            add_eos=add_eos,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        payload: dict[str, Any] = {
            "text_segments": text_segments,
            "images": images_bytes,
            "image_positions": image_positions,
            "image_sizes": image_sizes,
            "est_tokens": est,
        }

        max_tc = max(max_tc, est)
        min_tc = est if min_tc == 0 else min(min_tc, est)
        hist[_tc_to_bucket(est)] += 1
        fk = _tc_to_fine_bucket(est)
        fine_hist[fk] = fine_hist.get(fk, 0) + 1

        tokens, idx = heapq.heappop(heap)
        heapq.heappush(heap, (tokens + est, idx))
        writers[idx].write(payload)
        total_tokens += est
        samples += 1

    for w in writers:
        w.finish()
    if num_shards > 1:
        merge_index(output_dir, keep_local=True)

    if samples > 0:
        _write_stream_stats(
            output_dir,
            num_samples=samples,
            total_tokens=total_tokens,
            stream_id=stream_id,
            stream_type="pretrain",
            tokenizer_name=tokenizer_name,
            max_token_length=max_tc,
            min_token_length=min_tc,
            token_length_dist=hist,
            fine_token_length_dist=fine_hist,
        )

    logger.info(
        "lance→MDS 完成: %d samples, %d dropped, %d tokens", samples, dropped, total_tokens
    )
    return {"samples": samples, "dropped": dropped, "total_tokens": total_tokens}
