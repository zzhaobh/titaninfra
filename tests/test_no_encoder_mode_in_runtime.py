"""encoder_mode must not appear in the messages-doc-iter call chain."""

from pathlib import Path


def _src(p: str) -> str:
    return Path(p).read_text(encoding="utf-8")


def test_streaming_dataset_no_encoder_mode():
    assert "encoder_mode" not in _src("titaninfra/data/streaming_dataset.py")


def test_pretrain_helpers_no_encoder_mode():
    assert "encoder_mode" not in _src("titaninfra/engine/_pretrain_helpers.py")
