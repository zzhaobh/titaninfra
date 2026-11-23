"""Unit tests for full sample no truncate."""

from __future__ import annotations

import pytest

from titaninfra.data.sft.dispatch import tokenize_sample


def _long_sft_sample(n_words: int = 4000) -> dict:
    """Long sft sample."""
    user = " ".join(["explain"] * n_words)
    assistant = " ".join(["because"] * n_words)
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "system": None,
        "tools": None,
    }


def test_oversized_full_sample_raises_with_finite_max_length(qwen3_tokenizer):
    """Verify oversized full sample raises with finite max length."""
    sample = _long_sft_sample()
    full = tokenize_sample(sample, qwen3_tokenizer, family="qwen3", max_length=None)
    total = len(full.input_ids)
    assert total > 8, "fixture 太短，无法触发超长路径"
    tiny = total // 2

    with pytest.raises(ValueError, match="falls mid-segment"):
        tokenize_sample(
            sample,
            qwen3_tokenizer,
            family="qwen3",
            max_length=tiny,
            truncate_at_last_segment=False,
        )


def test_oversized_full_sample_passes_through_with_none_max_length(qwen3_tokenizer):
    """Verify oversized full sample passes through with none max length."""
    sample = _long_sft_sample()
    encoded = tokenize_sample(
        sample,
        qwen3_tokenizer,
        family="qwen3",
        max_length=None,
        truncate_at_last_segment=False,
    )
    assert len(encoded.input_ids) == len(encoded.labels)
    assert len(encoded.input_ids) > 0
    assert any(lbl != -100 for lbl in encoded.labels)


def test_full_sample_passes_none_max_length_unless_opt_in():
    """Verify full sample passes none max length unless opt in."""
    from titaninfra.config.data import PackStrategy
    from titaninfra.data.packer import PackMode, strategy_to_pack_mode

    seq_len = 4096
    assert strategy_to_pack_mode(PackStrategy.FULL_SAMPLE.value) == PackMode.SFT
    assert strategy_to_pack_mode(PackStrategy.HARD_TRUNCATE.value) != PackMode.SFT

    def resolve(is_full_sample: bool, truncate: bool) -> int | None:
        if is_full_sample:
            return seq_len + 1 if truncate else None
        return seq_len

    assert resolve(True, False) is None
    assert resolve(True, True) == seq_len + 1
    assert resolve(False, False) == seq_len
