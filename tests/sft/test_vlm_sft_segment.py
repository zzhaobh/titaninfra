import pytest

from titaninfra.data.sft.segment import SegKind, Segment, TokenClass, _base_mask


def test_image_segkind_exists():
    assert SegKind.IMAGE.value == "image"


def test_image_base_mask_always_zero():
    assert _base_mask(SegKind.IMAGE, "assistant") == 0.0
    assert _base_mask(SegKind.IMAGE, "user") == 0.0


def test_image_segment_carries_source_no_text():
    seg = Segment(
        text=None, token_ids=None, kind=SegKind.IMAGE, role="user",
        is_generation=False, token_class=TokenClass.PROTOCOL,
        image_source={"type": "bytes", "data": b"\xff\xd8jpeg"},
    )
    assert seg.image_source["type"] == "bytes"
    assert seg.loss_scale == 0.0


def test_image_segment_requires_source():
    with pytest.raises(ValueError, match="image_source"):
        Segment(
            text=None, token_ids=None, kind=SegKind.IMAGE, role="user",
            is_generation=False, token_class=TokenClass.PROTOCOL,
            image_source=None,
        )


def test_non_image_segment_rejects_source():
    with pytest.raises(ValueError, match="image_source"):
        Segment(
            text="hi", token_ids=None, kind=SegKind.CONTENT, role="assistant",
            is_generation=True, token_class=TokenClass.DECODED,
            image_source={"type": "bytes", "data": b"x"},
        )


def test_text_segment_still_requires_exactly_one():
    with pytest.raises(ValueError, match="exactly one"):
        Segment(
            text=None, token_ids=None, kind=SegKind.CONTENT, role="assistant",
            is_generation=True, token_class=TokenClass.DECODED,
        )


def test_encoded_sample_image_slots_default_none():
    from titaninfra.data.sft.types import EncodedSample
    es = EncodedSample(input_ids=(1, 2), labels=(-100, 2), positions=(0, 1))
    assert es.pixel_values is None
    assert es.grid_thw is None


def test_encoded_sample_image_slots_not_length_checked():
    from titaninfra.data.sft.types import EncodedSample
    es = EncodedSample(
        input_ids=(1, 2, 3), labels=(-100, -100, 3), positions=(0, 1, 2),
        pixel_values=[[0.0]] * 100,          # 100 patch rows ≠ 3 tokens
        grid_thw=[(1, 20, 20)],
    )
    assert es.grid_thw == [(1, 20, 20)]
    assert len(es.pixel_values) == 100
