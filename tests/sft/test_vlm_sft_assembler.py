import pytest
import torch

from titaninfra.data.sft import assembler
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass
from titaninfra.data.sft.types import IGNORE_INDEX


class _StubTok:
    _VOCAB = {"<|vision_start|>": 900, "<|image_pad|>": 901, "<|vision_end|>": 902}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 100 + 10 for c in text] or [10]

    def get_vocab(self):
        return dict(self._VOCAB)


def _img_seg():
    return Segment(
        text=None, token_ids=None, kind=SegKind.IMAGE, role="user",
        is_generation=False, token_class=TokenClass.PROTOCOL,
        image_source={"type": "bytes", "data": b"x"},
    )


def _text_seg(t, scale=1.0):
    return Segment(
        text=t, token_ids=None, kind=SegKind.CONTENT, role="assistant",
        scale_factor=scale, is_generation=True, token_class=TokenClass.DECODED,
    )


def _decoded(grid=(1, 4, 4), rows=16):
    return (torch.zeros(rows, 8), torch.tensor(list(grid)))


def test_image_segment_expands_from_decoded():
    segs = [
        Segment(text="<|im_start|>user\n", token_ids=None, kind=SegKind.MARKER,
                role="user", is_generation=False, token_class=TokenClass.PROTOCOL),
        _img_seg(),
        Segment(text="hi<|im_end|>\n", token_ids=None, kind=SegKind.CONTENT,
                role="user", is_generation=False, token_class=TokenClass.PROTOCOL),
        _text_seg("answer<|im_end|>\n"),
    ]
    es = assembler.assemble(segs, _StubTok(), mode="sft",
                            spatial_merge_size=2, decoded_images=[_decoded()])
    ids = list(es.input_ids)
    vs = ids.index(900)
    assert ids[vs:vs + 6] == [900, 901, 901, 901, 901, 902]
    assert es.labels[vs:vs + 6] == (IGNORE_INDEX,) * 6
    assert es.grid_thw == [(1, 4, 4)]
    assert es.pixel_values.shape[0] == 16


def test_two_images_bind_by_segment_order():
    segs = [
        Segment(text="<|im_start|>user\n", token_ids=None, kind=SegKind.MARKER,
                role="user", is_generation=False, token_class=TokenClass.PROTOCOL),
        _img_seg(),
        Segment(text="q1<|im_end|>\n", token_ids=None, kind=SegKind.CONTENT,
                role="user", is_generation=False, token_class=TokenClass.PROTOCOL),
        _text_seg("a1<|im_end|>\n"),
        Segment(text="<|im_start|>user\n", token_ids=None, kind=SegKind.MARKER,
                role="user", is_generation=False, token_class=TokenClass.PROTOCOL),
        _img_seg(),
        Segment(text="q2<|im_end|>\n", token_ids=None, kind=SegKind.CONTENT,
                role="user", is_generation=False, token_class=TokenClass.PROTOCOL),
        _text_seg("a2<|im_end|>\n"),
    ]
    decoded = [_decoded((1, 4, 4), 16), _decoded((1, 8, 8), 64)]  # N=4, N=16
    es = assembler.assemble(segs, _StubTok(), mode="sft",
                            spatial_merge_size=2, decoded_images=decoded)
    ids = list(es.input_ids)
    starts = [i for i, t in enumerate(ids) if t == 900]
    assert len(starts) == 2
    first_pads = ids[starts[0] + 1: ids.index(902, starts[0])]
    second_pads = ids[starts[1] + 1: ids.index(902, starts[1])]
    assert len(first_pads) == 4 and all(t == 901 for t in first_pads)
    assert len(second_pads) == 16 and all(t == 901 for t in second_pads)
    assert es.grid_thw == [(1, 4, 4), (1, 8, 8)]
    assert es.pixel_values.shape[0] == 16 + 64


def test_image_count_mismatch_fails_loud():
    segs = [_img_seg(), _text_seg("a<|im_end|>\n")]
    with pytest.raises(ValueError, match="decoded_images|IMAGE|image"):
        assembler.assemble(segs, _StubTok(), mode="sft",
                           spatial_merge_size=2, decoded_images=[_decoded(), _decoded()])


def test_image_without_decoded_fails_loud():
    segs = [_img_seg(), _text_seg("a<|im_end|>\n")]
    with pytest.raises(ValueError, match="decoded_images"):
        assembler.assemble(segs, _StubTok(), mode="sft", spatial_merge_size=2)


def test_pure_text_assemble_unchanged():
    segs = [_text_seg("hello<|im_end|>\n")]
    es = assembler.assemble(segs, _StubTok(), mode="sft")
    assert es.pixel_values is None
    assert es.grid_thw is None


def test_dispatch_threads_decoded_images_to_encoder():
    from titaninfra.data.sft.dispatch import tokenize_sample
    sample_json = {
        "messages": [
            {"role": "user", "blocks": [
                {"type": "image", "source": {"type": "bytes", "data": b"x"}},
                {"type": "text", "text": "q"},
            ]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "a"}]},
        ],
    }
    es = tokenize_sample(
        sample_json, _StubTok(), family="qwen3_5",
        spatial_merge_size=2, decoded_images=[_decoded((1, 4, 4), 16)],
    )
    assert es.grid_thw == [(1, 4, 4)]
    assert 901 in es.input_ids


def test_dispatch_pure_text_no_decoded():
    from titaninfra.data.sft.dispatch import tokenize_sample
    sample_json = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "yo"}]},
        ],
    }
    es = tokenize_sample(sample_json, _StubTok(), family="qwen3")
    assert es.pixel_values is None
