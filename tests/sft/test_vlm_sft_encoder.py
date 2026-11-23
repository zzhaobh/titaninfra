from titaninfra.data.schema import ImageBlock
from titaninfra.data.sft import normalize
from titaninfra.data.sft.encoders.qwen3 import Qwen3Encoder
from titaninfra.data.sft.encoders.qwen3_5 import Qwen3_5Encoder
from titaninfra.data.sft.loss_policy import LossPolicy, resolve_weights
from titaninfra.data.sft.segment import SegKind


def test_normalize_preserves_image_block():
    sample = {
        "messages": [
            {"role": "user", "blocks": [
                {"type": "image", "source": {"type": "bytes", "data": "AABB"}},
                {"type": "text", "text": "what is this?"},
            ]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "a cat"}]},
        ],
    }
    ns = normalize.from_sft_sample(sample)
    imgs = [b for b in ns.messages[0].blocks if isinstance(b, ImageBlock)]
    assert len(imgs) == 1
    assert imgs[0].source == {"type": "bytes", "data": "AABB"}


def _encode_segs(sample):
    ns = normalize.from_sft_sample(sample)
    ns = resolve_weights(ns, LossPolicy(sft_mask_mode="auto"))
    enc = Qwen3Encoder(tokenizer=None)
    return enc.encode_segments(ns)


def _encode_segs_q35(sample):
    ns = normalize.from_sft_sample(sample)
    ns = resolve_weights(ns, LossPolicy(sft_mask_mode="auto"))
    enc = Qwen3_5Encoder(tokenizer=None)
    return enc.encode_segments(ns)


def test_qwen3_5_image_in_user_turn_emits_image_segment():
    sample = {
        "messages": [
            {"role": "user", "blocks": [
                {"type": "image", "source": {"type": "bytes", "data": "IMG"}},
                {"type": "text", "text": "describe"},
            ]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "ok"}]},
        ],
    }
    segs = _encode_segs_q35(sample)
    img_segs = [s for s in segs if s.kind is SegKind.IMAGE]
    assert len(img_segs) == 1
    assert img_segs[0].image_source == {"type": "bytes", "data": "IMG"}
    assert img_segs[0].loss_scale == 0.0
    kinds = [s.kind for s in segs]
    img_i = kinds.index(SegKind.IMAGE)
    user_content_i = next(
        i for i, s in enumerate(segs)
        if s.kind is SegKind.CONTENT and s.role == "user"
    )
    assert img_i < user_content_i


def test_qwen3_no_image_user_turn_single_segment():
    sample = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "yo"}]},
        ],
    }
    segs = _encode_segs(sample)
    user_segs = [s for s in segs if s.role == "user"]
    assert len(user_segs) == 1
    assert user_segs[0].kind is SegKind.CONTENT
    assert user_segs[0].text == "<|im_start|>user\nhi<|im_end|>\n"


def test_qwen3_5_supports_image_but_qwen3_does_not():
    from titaninfra.data.schema import ImageBlock
    from titaninfra.data.sft.capabilities import FEATURE_IMAGE
    from titaninfra.data.sft.encoders import qwen3 as q3
    from titaninfra.data.sft.encoders import qwen3_5 as q35
    assert ImageBlock in q35.SUPPORTED_BLOCKS
    assert FEATURE_IMAGE in q35.SUPPORTED_FEATURES
    assert ImageBlock not in q3.SUPPORTED_BLOCKS
    assert FEATURE_IMAGE not in q3.SUPPORTED_FEATURES
