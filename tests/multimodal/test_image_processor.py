"""Unit tests for image processor."""

from __future__ import annotations

import pytest

from titaninfra.data.multimodal.image_processor import (
    calculate_vision_tokens,
    estimate_image_tokens,
    smart_resize,
)

FACTOR = 32  # qwen3_5: patch(16) * merge(2)
MIN_PX = 65536
MAX_PX = 16777216


class TestSmartResize:
    def test_multiple_of_factor(self):
        for w, h in [(640, 480), (500, 375), (333, 500), (95, 97), (1024, 768)]:
            hb, wb = smart_resize(h, w, FACTOR, MIN_PX, MAX_PX)
            assert hb % FACTOR == 0 and wb % FACTOR == 0

    def test_factor_floor_on_tiny(self):
        hb, wb = smart_resize(1, 1, FACTOR, MIN_PX, MAX_PX)
        assert hb >= FACTOR and wb >= FACTOR
        assert hb * wb >= MIN_PX

    def test_max_pixels_never_exceeded(self):
        hb, wb = smart_resize(5000, 5000, FACTOR, MIN_PX, MAX_PX)
        assert hb * wb <= MAX_PX

    def test_min_pixels_never_below(self):
        hb, wb = smart_resize(37, 37, FACTOR, MIN_PX, MAX_PX)
        assert hb * wb >= MIN_PX

    def test_aspect_ratio_over_200_raises(self):
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(17, 4096, FACTOR, MIN_PX, MAX_PX)
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(4096, 17, FACTOR, MIN_PX, MAX_PX)


class TestCalculateVisionTokens:
    def test_single_image_temporal_is_one(self):
        n, tpr, rows = calculate_vision_tokens(1, 480, 640, 16, 2, 2)
        assert n == (640 // 32) * (480 // 32)  # 20 * 15 = 300
        assert n == 300 and tpr == 20 and rows == 15

    def test_temporal_only_affects_video(self):
        n1, _, _ = calculate_vision_tokens(1, 480, 640, 16, 2, 2)
        n4, _, _ = calculate_vision_tokens(4, 480, 640, 16, 2, 2)
        assert n4 == n1 * 2


class TestEstimateImageTokensSeam:
    """Tests for test estimate image tokens seam."""

    SIZES = [
        (640, 480), (500, 375), (333, 500), (1024, 768), (37, 37),
        (31, 200), (2000, 13), (1, 1), (96, 96), (95, 97),
        (63, 65), (129, 129), (5000, 5000), (800, 600), (224, 224),
    ]

    @pytest.mark.parametrize("w,h", SIZES)
    def test_seam_self_consistent(self, w, h):
        n_combined = estimate_image_tokens(w, h)
        hb, wb = smart_resize(h, w, FACTOR, MIN_PX, MAX_PX)
        n_manual, _, _ = calculate_vision_tokens(1, hb, wb, 16, 2, 2)
        assert n_combined == n_manual

    def test_aspect_over_200_propagates(self):
        with pytest.raises(ValueError):
            estimate_image_tokens(4096, 17)

    def test_returns_positive_int(self):
        for w, h in self.SIZES:
            n = estimate_image_tokens(w, h)
            assert isinstance(n, int) and n > 0


class TestVisionToPatches:
    def test_patchify_matches_calculate_vision_tokens(self):
        torch = pytest.importorskip("torch")
        from titaninfra.data.multimodal.image_processor import vision_to_patches

        # T=2 H=64 W=96 C=3, patch=16 merge=2 → grid [1,4,6]
        img = torch.randn(2, 64, 96, 3)
        patches, grid = vision_to_patches(img, patch_size=16, temporal_patch_size=2, merge_size=2)
        assert grid.tolist() == [1, 4, 6]
        n, _, _ = calculate_vision_tokens(2, 64, 96, 16, 2, 2)
        assert n == patches.shape[0] // 4
        assert patches.shape[1] == 3 * 2 * 16 * 16

    def test_temporal_padding(self):
        torch = pytest.importorskip("torch")
        from titaninfra.data.multimodal.image_processor import vision_to_patches

        img = torch.randn(1, 32, 32, 3)
        _, grid = vision_to_patches(img, patch_size=16, temporal_patch_size=2, merge_size=2)
        assert grid[0].item() == 1
