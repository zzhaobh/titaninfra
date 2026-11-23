"""Unit tests for mm packing."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from titaninfra.data.packer import PackMode, StreamingPacker  # noqa: E402


def _img(n_patches: int, dim: int = 8, fill: float = 1.0):
    """Img."""
    return torch.full((n_patches, dim), fill)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_single_mm_doc_carries_image():
    """Verify single mm doc carries image."""
    packer = StreamingPacker(seq_len=100, mode=PackMode.SFT)
    tokens = [(9, -100), (9, -100), (1, 1), (2, 2), (3, 3)]
    images = (_img(4, fill=7.0), [(1, 2, 2)])
    packer.add_document(tokens, images=images)

    out = list(packer.flush_all())
    assert len(out) == 1
    s = out[0]
    assert s.pixel_values is not None
    assert s.pixel_values.shape == (4, 8)
    assert torch.all(s.pixel_values == 7.0)
    assert s.grid_thw == [(1, 2, 2)]


def test_pure_text_doc_has_no_image_fields():
    """Verify pure text doc has no image fields."""
    packer = StreamingPacker(seq_len=100, mode=PackMode.SFT)
    packer.add_document([(1, 1), (2, 2), (3, 3)])
    out = list(packer.flush_all())
    assert len(out) == 1
    assert out[0].pixel_values is None
    assert out[0].grid_thw is None


def test_multiple_mm_docs_binpacked_concat_order():
    """Verify multiple mm docs binpacked concat order."""
    packer = StreamingPacker(seq_len=100, mode=PackMode.SFT)
    packer.add_document([(1, 1), (2, 2)], images=(_img(2, fill=1.0), [(1, 1, 2)]))
    packer.add_document([(3, 3), (4, 4)], images=(_img(3, fill=2.0), [(1, 1, 3)]))

    out = list(packer.flush_all())
    assert len(out) == 1
    s = out[0]
    assert s.pixel_values.shape == (5, 8)
    assert torch.all(s.pixel_values[:2] == 1.0)
    assert torch.all(s.pixel_values[2:] == 2.0)
    assert s.grid_thw == [(1, 1, 2), (1, 1, 3)]


def test_mm_docs_split_across_samples_keep_image_alignment():
    """Verify mm docs split across samples keep image alignment."""
    packer = StreamingPacker(seq_len=5, mode=PackMode.SFT)
    packer.add_document([(1, 1), (1, 1), (1, 1), (1, 1)], images=(_img(2, fill=1.0), [(1, 1, 2)]))
    out1 = list(packer.flush_ready())
    packer.add_document([(2, 2), (2, 2), (2, 2), (2, 2)], images=(_img(2, fill=2.0), [(1, 1, 2)]))
    out2 = list(packer.flush_ready())
    out3 = list(packer.flush_all())

    all_out = out1 + out2 + out3
    imgs_seen = [(s.pixel_values, s.grid_thw) for s in all_out if s.pixel_values is not None]
    assert len(imgs_seen) == 2
    assert torch.all(imgs_seen[0][0] == 1.0)
    assert torch.all(imgs_seen[1][0] == 2.0)


def test_reset_clears_image_buffer():
    """Verify reset clears image buffer."""
    packer = StreamingPacker(seq_len=100, mode=PackMode.SFT)
    packer.add_document([(1, 1), (2, 2)], images=(_img(2), [(1, 1, 2)]))
    packer.reset()
    assert packer._image_buffer == []
    packer.add_document([(3, 3), (4, 4)])
    out = list(packer.flush_all())
    assert out[0].pixel_values is None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Tests for mock tokenizer."""

    bos_id = 100
    eos_id = 101

    def get_vocab(self):
        return {"<|vision_start|>": 200, "<|image_pad|>": 201, "<|vision_end|>": 202}

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [ord(c) for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids


def test_build_interleaved_doc_caption_before_text():
    """Verify build interleaved doc caption before text."""
    from titaninfra.data.multimodal.mm_doc_iter import _build_interleaved_doc

    tok = _MockTokenizer()
    grid = torch.tensor([1, 2, 2])  # N = 4 // 4 = 1
    decoded = [(_img(4, fill=1.0), grid)]
    doc = _build_interleaved_doc(
        text_segments=["ab"],
        image_positions=[0],
        decoded=decoded,
        tokenizer=tok,
        vision_ids=(200, 201, 202),
        spatial_merge_size=2,
        add_bos_eos=True,
    )
    assert doc is not None
    ids = [t for t, _ in doc.tokens]
    labels = [lbl for _, lbl in doc.tokens]
    assert ids == [100, 200, 201, 202, ord("a"), ord("b"), 101]
    assert labels == [100, -100, -100, -100, ord("a"), ord("b"), 101]
    assert doc.images[1] == [(1, 2, 2)]


def test_build_interleaved_doc_trailing_image():
    """Verify build interleaved doc trailing image."""
    from titaninfra.data.multimodal.mm_doc_iter import _build_interleaved_doc

    tok = _MockTokenizer()
    grid = torch.tensor([1, 2, 2])
    doc = _build_interleaved_doc(
        text_segments=["x"],
        image_positions=[1],
        decoded=[(_img(4), grid)],
        tokenizer=tok,
        vision_ids=(200, 201, 202),
        spatial_merge_size=2,
        add_bos_eos=False,
    )
    ids = [t for t, _ in doc.tokens]
    assert ids == [ord("x"), 200, 201, 202]


def test_build_interleaved_doc_consecutive_images():
    """Verify build interleaved doc consecutive images."""
    from titaninfra.data.multimodal.mm_doc_iter import _build_interleaved_doc

    tok = _MockTokenizer()
    g = torch.tensor([1, 2, 2])
    doc = _build_interleaved_doc(
        text_segments=["z"],
        image_positions=[0, 0],
        decoded=[(_img(4, fill=1.0), g), (_img(4, fill=2.0), g)],
        tokenizer=tok,
        vision_ids=(200, 201, 202),
        spatial_merge_size=2,
        add_bos_eos=False,
    )
    ids = [t for t, _ in doc.tokens]
    assert ids == [200, 201, 202, 200, 201, 202, ord("z")]
    assert doc.images[0].shape == (8, 8)
    assert doc.images[1] == [(1, 2, 2), (1, 2, 2)]


def test_grid_to_n_tokens():
    """grid_thw → N = prod // merge²。"""
    from titaninfra.data.multimodal.mm_doc_iter import _grid_to_n_tokens

    assert _grid_to_n_tokens((1, 2, 2), 2) == 1
    assert _grid_to_n_tokens((1, 4, 4), 2) == 4
    assert _grid_to_n_tokens((1, 8, 4), 2) == 8
