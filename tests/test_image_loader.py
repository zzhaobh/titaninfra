"""Unit tests for image loader."""

from pathlib import Path

import pytest
from PIL import Image

from titaninfra.data.image_loader import load_image

IMAGES_DIR = Path(__file__).parent / "fixtures" / "images"


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestLoadFromPath:

    def test_absolute_path_jpg(self):
        img = load_image({"type": "path", "data": str(IMAGES_DIR / "red_square.jpg")})
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_absolute_path_png(self):
        img = load_image({"type": "path", "data": str(IMAGES_DIR / "gray_small.png")})
        assert img.mode == "RGB"
        assert img.size == (50, 50)

    def test_relative_path_with_base_dir(self):
        img = load_image(
            {"type": "path", "data": "red_square.jpg"},
            base_dir=IMAGES_DIR,
        )
        assert img.size == (100, 100)

    def test_landscape_image(self):
        img = load_image({"type": "path", "data": str(IMAGES_DIR / "green_landscape.jpg")})
        assert img.size == (640, 480)

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image({"type": "path", "data": "/no/such/file.jpg"})

    def test_default_type_is_path(self):
        """Verify default type is path."""
        img = load_image({"data": str(IMAGES_DIR / "blue_224.jpg")})
        assert img.size == (224, 224)


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestLoadFromBytes:

    def test_jpeg_bytes(self):
        raw = (IMAGES_DIR / "red_square.jpg").read_bytes()
        img = load_image({"type": "bytes", "data": raw})
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_png_bytes(self):
        raw = (IMAGES_DIR / "gray_small.png").read_bytes()
        img = load_image({"type": "bytes", "data": raw})
        assert img.mode == "RGB"

    def test_invalid_bytes_raises(self):
        with pytest.raises((ValueError, OSError)):
            load_image({"type": "bytes", "data": b"not an image"})

    def test_non_bytes_type_raises(self):
        with pytest.raises(TypeError, match="bytes"):
            load_image({"type": "bytes", "data": "string_not_bytes"})


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestLoadFromPIL:

    def test_rgb_passthrough(self):
        pil_img = Image.new("RGB", (32, 32), color=(1, 2, 3))
        img = load_image({"type": "pil", "data": pil_img})
        assert img is pil_img

    def test_grayscale_converted(self):
        pil_img = Image.new("L", (32, 32), color=128)
        img = load_image({"type": "pil", "data": pil_img})
        assert img.mode == "RGB"

    def test_rgba_converted(self):
        pil_img = Image.new("RGBA", (32, 32), color=(1, 2, 3, 255))
        img = load_image({"type": "pil", "data": pil_img})
        assert img.mode == "RGB"

    def test_non_pil_raises(self):
        with pytest.raises(TypeError, match="PIL.Image"):
            load_image({"type": "pil", "data": "not a pil image"})


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestErrors:

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="不支持"):
            load_image({"type": "ftp", "data": "ftp://img"})


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestAllFixtureImages:

    @pytest.mark.parametrize("filename", [
        "red_square.jpg", "gradient.jpg", "gray_small.png",
        "green_landscape.jpg", "blue_224.jpg",
    ])
    def test_load_fixture(self, filename):
        img = load_image({"type": "path", "data": str(IMAGES_DIR / filename)})
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size[0] > 0 and img.size[1] > 0
