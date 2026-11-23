import base64 as _b64
import json
import json as _json

import pytest

from titaninfra.config.data import ResolutionPolicy, StreamConfig
from titaninfra.data.packer import MMDoc

try:
    import torchtitan  # noqa: F401
    HAS_TORCHTITAN = True
except ImportError:
    HAS_TORCHTITAN = False

pytestmark = pytest.mark.skipif(not HAS_TORCHTITAN, reason="torchtitan 未安装")


class _StubTok:
    _VOCAB = {"<|vision_start|>": 900, "<|image_pad|>": 901, "<|vision_end|>": 902}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 100 + 10 for c in text] or [10]

    def get_vocab(self):
        return dict(self._VOCAB)


def test_messages_doc_iter_yields_mmdoc_via_pool(monkeypatch):
    import torch

    from titaninfra.data.streaming_dataset import _make_messages_doc_iter

    def _fake(raw, **kw):
        return torch.zeros(16, 8), torch.tensor([1, 4, 4])
    monkeypatch.setattr(
        "titaninfra.data.multimodal.decode_pool.preprocess_image", _fake
    )
    sample = {"messages": _json.dumps([
        {"role": "user", "blocks": [
            {"type": "image", "source": {"type": "base64", "data": _b64.b64encode(b"\xff\xd8x").decode("ascii")}},
            {"type": "text", "text": "q"},
        ]},
        {"role": "assistant", "blocks": [{"type": "text", "text": "a"}]},
    ])}
    cfg = StreamConfig(
        tokenizer_name="qwen3_5", family="qwen3_5", local="/tmp/x",
        data_type="messages", pack_strategy="full_sample",
        resolution_policy=ResolutionPolicy(),
    )
    docs = list(_make_messages_doc_iter([sample], _StubTok(), cfg, seq_len=4096,
                                        image_decode_workers=2))
    assert len(docs) == 1
    assert isinstance(docs[0], MMDoc)
    _pv, gthw = docs[0].images
    assert gthw == [(1, 4, 4)]
    assert any(t == 901 for t, _ in docs[0].tokens)


def test_messages_doc_iter_pure_text_no_mmdoc():
    from titaninfra.data.streaming_dataset import _make_messages_doc_iter

    sample = {"messages": json.dumps([
        {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "blocks": [{"type": "text", "text": "yo"}]},
    ])}
    cfg = StreamConfig(
        tokenizer_name="qwen3", family="qwen3", local="/tmp/x",
        data_type="messages", pack_strategy="full_sample",
    )
    docs = list(_make_messages_doc_iter([sample], _StubTok(), cfg, seq_len=4096))
    assert len(docs) == 1
    assert not isinstance(docs[0], MMDoc)


def test_sft_images_of_extracts_bytes_in_doc_order():
    from titaninfra.data.streaming_dataset import _sft_images_of
    img_b = b"\xff\xd8BBB"
    sample = {"messages": _json.dumps([
        {"role": "user", "blocks": [
            {"type": "image", "source": {"type": "bytes", "data": b"\xff\xd8AAA".decode("latin1")}},
            {"type": "text", "text": "q1"},
        ]},
        {"role": "assistant", "blocks": [{"type": "text", "text": "a1"}]},
        {"role": "user", "blocks": [
            {"type": "image", "source": {"type": "base64", "data": _b64.b64encode(img_b).decode("ascii")}},
            {"type": "text", "text": "q2"},
        ]},
    ])}
    out = _sft_images_of(sample)
    assert len(out) == 2
    assert out[1] == img_b


def test_sft_images_of_no_images_returns_empty():
    from titaninfra.data.streaming_dataset import _sft_images_of
    sample = {"messages": _json.dumps([
        {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "blocks": [{"type": "text", "text": "yo"}]},
    ])}
    assert _sft_images_of(sample) == []


def test_sft_images_of_unknown_type_fails_loud():
    from titaninfra.data.streaming_dataset import _sft_images_of
    sample = {"messages": _json.dumps([
        {"role": "user", "blocks": [
            {"type": "image", "source": {"type": "url", "data": "http://x/y.jpg"}},
        ]},
    ])}
    with pytest.raises(ValueError, match="bytes.*base64|base64"):
        _sft_images_of(sample)
