"""Single source of truth for multimodal image sizing and patch conversion.

Offline token estimates and runtime image-pad/ViT scatter counts must use the
same smart_resize and calculate_vision_tokens path. This guarantees N agrees
across est_tokens, runtime grid_thw, and preprocess_image.

Defaults target Qwen3-VL: patch_size=16 and spatial_merge_size=2, so factor=32.
Pure sizing functions have no torch dependency; decode and patchify import
torch/torchvision lazily.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# Pure sizing math with no torch dependency.


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Resize height/width to factor multiples within the pixel budget.

    Oversized images use floor to stay below max_pixels; undersized images use
    ceil to reach min_pixels. Aspect ratios above 200 raise ValueError. These
    rounding directions are required to avoid off-by-one token counts.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width):.1f}"
        )

    h_bar = max(round(height / factor) * factor, factor)
    w_bar = max(round(width / factor) * factor, factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(math.floor(height / beta / factor) * factor, factor)
        w_bar = max(math.floor(width / beta / factor) * factor, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def calculate_vision_tokens(
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
) -> tuple[int, int, int]:
    """Return visual token count, tokens per row, and row count after merging.

    A single image always has one temporal patch; temporal_patch_size affects
    only multi-frame video.
    """
    t_patches = math.ceil(num_frames / temporal_patch_size)
    tokens_per_row = width // (patch_size * spatial_merge_size)
    num_rows = height // (patch_size * spatial_merge_size)
    total_tokens = t_patches * tokens_per_row * num_rows
    return total_tokens, tokens_per_row, num_rows


# Shared offline/runtime token-estimation entry point.


def estimate_image_tokens(
    width: int,
    height: int,
    *,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> int:
    """Return the number of image_pad tokens produced by one image.

    Compose smart_resize and calculate_vision_tokens so offline estimates and
    runtime insertion always use identical dimensions and merge semantics.
    """
    factor = patch_size * spatial_merge_size
    h_bar, w_bar = smart_resize(height, width, factor, min_pixels, max_pixels)

    # Both dimensions must be factor multiples; otherwise integer division in
    # calculate_vision_tokens would silently truncate the token count.
    assert h_bar % factor == 0 and w_bar % factor == 0, (
        f"smart_resize 应返回 factor({factor}) 倍数，实得 h_bar={h_bar} w_bar={w_bar}"
    )

    n, _, _ = calculate_vision_tokens(
        num_frames=1,
        height=h_bar,
        width=w_bar,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        temporal_patch_size=temporal_patch_size,
    )
    return n


# Runtime decode and patchify with lazy torch/torchvision imports.


def decode_image(image: bytes | str) -> torch.Tensor:
    """Decode bytes or a file path into a (C, H, W) uint8 RGB tensor."""
    import torch
    import torchvision.io

    if isinstance(image, bytes):
        raw = torch.frombuffer(bytearray(image), dtype=torch.uint8)
        return torchvision.io.decode_image(raw, mode=torchvision.io.ImageReadMode.RGB)
    if isinstance(image, str):
        return torchvision.io.decode_image(image, mode=torchvision.io.ImageReadMode.RGB)
    raise TypeError(f"decode_image 只接受 bytes / str（文件路径），实际: {type(image)}")


def vision_to_patches(
    img: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (T, H, W, C) input to flattened patches and grid_thw.

    Preserve contiguous merge_size×merge_size groups and channel-first patch
    vectors using pure torch reshape/permute instead of einops.
    """
    import torch

    T, H, W, C = img.shape
    ps = patch_size
    ts = temporal_patch_size
    m = merge_size

    # Pad the temporal dimension by repeating the final frame when necessary.
    if T % ts != 0:
        pad_t = ts - (T % ts)
        img = torch.cat([img, img[-1:].expand(pad_t, -1, -1, -1)], dim=0)
        T = img.shape[0]

    t = T // ts
    bh = H // (ps * m)
    bw = W // (ps * m)

    # Source axes: t, pt, bh, m, ph, bw, n, pw, c. Rearrange to sequence axes
    # t, bh, bw, m, n and vector axes c, pt, ph, pw.
    x = img.reshape(t, ts, bh, m, ps, bw, m, ps, C)
    x = x.permute(0, 2, 5, 3, 6, 8, 1, 4, 7).contiguous()
    patches = x.reshape(t * bh * bw * m * m, C * ts * ps * ps)

    grid_thw = torch.tensor([t, H // ps, W // ps])
    return patches, grid_thw


def sft_images_of(sample: dict, messages_key: str = "messages") -> list[bytes]:
    """Extract raw image bytes from ImageBlock values in document order.

    Supports bytes and base64 sources. Unknown source types fail loudly;
    missing or malformed message input returns an empty list. This pure
    JSON/base64 path is shared by offline conversion and runtime streaming.
    """
    import base64
    import json

    raw = sample.get(messages_key, "")
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            return []
    elif isinstance(raw, list):
        messages = raw
    else:
        return []

    out: list[bytes] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        blocks = msg.get("blocks")
        if not isinstance(blocks, list):
            continue
        for blk in blocks:
            if not isinstance(blk, dict) or blk.get("type") != "image":
                continue
            src = blk.get("source") or {}
            src_type = src.get("type")
            data = src.get("data")
            if data is None:
                raise ValueError(f"sft_images_of: image block missing source.data (source={src!r})")
            if src_type == "bytes":
                out.append(data if isinstance(data, bytes) else data.encode("latin1"))
            elif src_type == "base64":
                out.append(base64.b64decode(data))
            else:
                raise ValueError(
                    "sft_images_of: image source.type must be 'bytes' or 'base64', "
                    f"got {src_type!r}"
                )
    return out


def preprocess_image(
    raw: bytes | str,
    *,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode, resize, and patchify one image into pixel_values and grid_thw.

    After patchify, validate that runtime grid token count equals
    estimate_image_tokens for the original dimensions. Any mismatch raises
    instead of corrupting packing or ViT scatter.
    """
    import torchvision.transforms.v2.functional as tvf

    img_chw = decode_image(raw)  # (C, H0, W0) uint8
    _, h0, w0 = img_chw.shape

    factor = patch_size * spatial_merge_size
    h_bar, w_bar = smart_resize(h0, w0, factor, min_pixels, max_pixels)

    # Resize to the same dimensions used by offline token estimation.
    img_chw = tvf.resize(img_chw, [h_bar, w_bar], antialias=True)

    # Convert CHW to THWC with one temporal frame.
    img_thwc = img_chw.permute(1, 2, 0).unsqueeze(0).contiguous()

    pixel_values, grid_thw = vision_to_patches(
        img_thwc,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=spatial_merge_size,
    )

    # Runtime grid token count must equal the offline estimate.
    n_runtime = int(grid_thw.prod().item()) // (spatial_merge_size ** 2)
    n_offline = estimate_image_tokens(
        w0, h0,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        temporal_patch_size=temporal_patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if n_runtime != n_offline:
        raise ValueError(
            f"invariant #2 破裂：runtime grid N={n_runtime} != offline estimate N={n_offline} "
            f"(w={w0} h={h0} → resize {w_bar}x{h_bar}, grid_thw={grid_thw.tolist()})"
        )

    return pixel_values, grid_thw
