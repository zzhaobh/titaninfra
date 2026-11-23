"""Unit tests for lance to mds."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from titaninfra.data.multimodal.image_processor import estimate_image_tokens
from titaninfra.data.multimodal.lance_to_mds import (
    decode_and_size,
    estimate_est_tokens,
    parse_mml_text,
    validate_and_prepare_doc,
)

QWEN3_5 = str(Path(__file__).parent.parent / "fixtures" / "tokenizers" / "qwen3_5")


def _jpeg(w: int, h: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, "JPEG")
    return buf.getvalue()


class TestParseMmlText:
    def test_caption_leading_image(self):
        # fixture _caption_row: "[IMG_0] {cap}"
        segs, pos = parse_mml_text("[IMG_0] a dog runs")
        assert pos == [0]
        assert segs == [" a dog runs"]

    def test_interleaved_two_images(self):
        # fixture _interleaved_row: "cap0 [IMG_0] cap1 [IMG_1]"
        segs, pos = parse_mml_text("cap0 [IMG_0] cap1 [IMG_1]")
        assert pos == [1, 2]
        assert segs == ["cap0 ", " cap1 "]

    def test_consecutive_images_repeated_position(self):
        segs, pos = parse_mml_text("text [IMG_0][IMG_1] more")
        assert pos == [1, 1]

    def test_trailing_image_at_point_N(self):
        segs, pos = parse_mml_text("all text here [IMG_0]")
        assert pos == [1]

    @pytest.mark.parametrize(
        "mml",
        ["[IMG_0] x", "a [IMG_0] b [IMG_1]", "[IMG_0][IMG_1] z", "y [IMG_0]", "no images here"],
    )
    def test_invariant5_positions(self, mml):
        segs, pos = parse_mml_text(mml)
        n = len(segs)
        assert all(0 <= p <= n for p in pos)
        assert pos == sorted(pos)


class TestInvariant1Gate:
    def test_decode_good_image(self):
        result = decode_and_size(_jpeg(120, 90))
        assert result is not None
        _, w, h = result
        assert (w, h) == (120, 90)

    def test_decode_empty_bytes(self):
        assert decode_and_size(b"") is None

    def test_decode_undecodable(self):
        assert decode_and_size(b"\xff\xd8not_a_real_jpeg") is None

    def test_three_dirty_rows_all_dropped(self):
        good_size = [{"width": 640, "height": 480}]
        d1 = validate_and_prepare_doc(
            "[IMG_0] broken", [{"media_bytes": b"", "position_in_text": 0, "url": ""}], good_size
        )
        d2 = validate_and_prepare_doc(
            "[IMG_0] undecodable",
            [{"media_bytes": b"\xff\xd8not_a_real_jpeg", "position_in_text": 0, "url": ""}],
            [{"width": 0, "height": 0}],
        )
        d3 = validate_and_prepare_doc(
            "[IMG_0] url only",
            [{"media_bytes": b"", "position_in_text": 0, "url": "http://x/y.jpg"}],
            [{"width": 0, "height": 0}],
        )
        assert d1 is None and d2 is None and d3 is None

    def test_good_doc_uses_decoded_size_not_column(self):
        good = _jpeg(120, 90)
        result = validate_and_prepare_doc(
            "cap0 [IMG_0] cap1 [IMG_1]",
            [
                {"media_bytes": good, "position_in_text": 5, "url": ""},
                {"media_bytes": good, "position_in_text": 15, "url": ""},
            ],
            [{"width": 999, "height": 999}, {"width": 999, "height": 999}],
        )
        assert result is not None
        segs, imgs, pos, sizes = result
        assert sizes == [(120, 90), (120, 90)]
        assert len(imgs) == len(pos) == len(sizes) == 2  # invariant #5

    def test_misalignment_dropped(self):
        result = validate_and_prepare_doc(
            "a [IMG_0] b [IMG_1]",
            [{"media_bytes": _jpeg(64, 64), "position_in_text": 2, "url": ""}],
            [{"width": 64, "height": 64}],
        )
        assert result is None


class TestEstimateEstTokens:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(QWEN3_5, trust_remote_code=True)

    def test_matches_manual_reconstruction(self, tokenizer):
        segs = ["cap0 ", " cap1 "]
        sizes = [(640, 480), (500, 375)]
        est = estimate_est_tokens(segs, sizes, tokenizer, add_bos=True, add_eos=True)

        manual = 1  # bos
        for s in segs:
            manual += len(tokenizer.encode(s, add_special_tokens=False))
        for w, h in sizes:
            manual += 2 + estimate_image_tokens(w, h)  # vision_start+end + N
        manual += 1  # eos
        assert est == manual

    def test_no_bos_eos(self, tokenizer):
        with_be = estimate_est_tokens([], [], tokenizer, add_bos=True, add_eos=True)
        without = estimate_est_tokens([], [], tokenizer, add_bos=False, add_eos=False)
        assert with_be - without == 2

    def test_text_only_no_images(self, tokenizer):
        est = estimate_est_tokens(["hello world"], [], tokenizer, add_bos=True, add_eos=True)
        expected = 1 + len(tokenizer.encode("hello world", add_special_tokens=False)) + 1
        assert est == expected


class TestConvertLanceToMds:
    """Tests for test convert lance to mds."""

    def test_fixture_roundtrip(self, tmp_path):
        pytest.importorskip("lance")
        pytest.importorskip("streaming")
        from transformers import AutoTokenizer

        from titaninfra.data.multimodal.lance_to_mds import convert_lance_to_mds

        fixture = (
            Path(__file__).parent.parent / "fixtures" / "multimodal" / "datatrove_sample.lance"
        )
        if not fixture.exists():
            pytest.skip("multimodal lance fixture 缺失")

        tok = AutoTokenizer.from_pretrained(QWEN3_5, trust_remote_code=True)
        result = convert_lance_to_mds(
            str(fixture),
            str(tmp_path / "mds"),
            tok,
            stream_id="mm_test",
            tokenizer_name="qwen3_5",
            num_shards=4,
        )
        assert result["dropped"] >= 3
        assert result["samples"] > 0
        assert result["total_tokens"] > 0


def test_fixture_builder_zero_real_images_never_opens_network(monkeypatch):
    """Synthetic-only fixture construction must not create a Parquet reader."""
    from tests.fixtures.multimodal import build_fixture

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network-backed ParquetFile must not be opened")

    monkeypatch.setattr(build_fixture.pq, "ParquetFile", fail_if_called)
    assert build_fixture._stream_flickr(0) == []
