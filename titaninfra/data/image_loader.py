"""Unified image loading for file paths, raw bytes, and existing PIL images."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def load_image(
    source: dict,
    base_dir: str | Path | None = None,
) -> Image.Image:
    """Load an image and return it in RGB mode.

    Args:
        source: One of {"type": "path", "data": "relative/path.jpg"},
            {"type": "bytes", "data": b"..."}, or
            {"type": "pil", "data": <PIL.Image>}.
        base_dir: Optional parent directory for relative path inputs.

    Returns:
        An RGB PIL.Image.Image.
    """
    src_type = source.get("type", "path")
    data = source.get("data")

    if src_type == "path":
        return _load_from_path(data, base_dir)

    if src_type == "bytes":
        return _load_from_bytes(data)

    if src_type == "pil":
        if not isinstance(data, Image.Image):
            raise TypeError(f"type='pil' 时 data 必须是 PIL.Image，实际: {type(data)}")
        return _ensure_rgb(data)

    raise ValueError(f"不支持的图像来源类型: {src_type!r}，支持: path, bytes, pil")


def _load_from_path(path: str, base_dir: str | Path | None) -> Image.Image:
    """Load an image from a filesystem path."""
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p

    if not p.exists():
        raise FileNotFoundError(f"图像文件不存在: {p}")

    img = Image.open(p)
    img.load()  # Force decoding now so the file handle does not leak.
    return _ensure_rgb(img)


def _load_from_bytes(data: bytes) -> Image.Image:
    """Decode an image from raw bytes."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"type='bytes' 时 data 必须是 bytes，实际: {type(data)}")
    img = Image.open(io.BytesIO(data))
    img.load()
    return _ensure_rgb(img)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert grayscale, RGBA, and other modes to RGB when necessary."""
    if img.mode == "RGB":
        return img
    return img.convert("RGB")
