"""Unit tests for validate_sft_batch stop_marker_token assertion (Phase C)."""

import pytest
import torch

from titaninfra.eval.pretrain_checks import validate_sft_batch

SEQ_LEN = 32
STOP_TOKEN = 151645  # <|im_end|> for qwen family


def _make_batch(token_ids, positions, loss_weights):
    """Helper: construct input_dict + labels from raw lists."""
    inputs = torch.tensor([token_ids], dtype=torch.long)
    pos = torch.tensor([positions], dtype=torch.long)
    weights = torch.tensor([loss_weights], dtype=torch.float32)
    labels = torch.where(weights > 0, inputs, torch.full_like(inputs, -100))
    input_dict = {"input": inputs, "positions": pos, "loss_weights": weights}
    return input_dict, labels


def test_stop_marker_trained_passes():
    """Stop marker token with loss > 0 → passes."""
    n_content = 10
    tokens = [100] + list(range(200, 200 + n_content)) + [STOP_TOKEN] + [-1] * (SEQ_LEN - n_content - 2)
    positions = list(range(n_content + 2)) + [-1] * (SEQ_LEN - n_content - 2)
    weights = [0.0] + [1.0] * n_content + [1.0] + [0.0] * (SEQ_LEN - n_content - 2)

    input_dict, labels = _make_batch(tokens, positions, weights)
    validate_sft_batch(1, input_dict, labels, seq_len=SEQ_LEN, stop_marker_token=STOP_TOKEN)


def test_stop_marker_all_masked_fails():
    """Stop marker present but weight=0 everywhere → AssertionError (F1 regression)."""
    n_content = 10
    tokens = [100] + list(range(200, 200 + n_content)) + [STOP_TOKEN] + [-1] * (SEQ_LEN - n_content - 2)
    positions = list(range(n_content + 2)) + [-1] * (SEQ_LEN - n_content - 2)
    weights = [0.0] + [1.0] * n_content + [0.0] + [0.0] * (SEQ_LEN - n_content - 2)

    input_dict, labels = _make_batch(tokens, positions, weights)
    with pytest.raises(AssertionError, match="STOP_MARKER loss regression"):
        validate_sft_batch(1, input_dict, labels, seq_len=SEQ_LEN, stop_marker_token=STOP_TOKEN)


def test_user_turn_marker_masked_no_false_positive():
    """Same token in user turn (masked) + assistant turn (trained) → passes.

    Conservative check: >=1 trained marker in the doc is enough.
    """
    tokens = (
        [100, STOP_TOKEN]  # user header + user-turn stop (masked)
        + list(range(200, 210))  # assistant content
        + [STOP_TOKEN]  # assistant stop (trained)
        + [-1] * (SEQ_LEN - 13)
    )
    positions = list(range(13)) + [-1] * (SEQ_LEN - 13)
    weights = (
        [0.0, 0.0]  # user region masked
        + [1.0] * 10  # assistant content
        + [1.0]  # assistant stop trained
        + [0.0] * (SEQ_LEN - 13)
    )

    input_dict, labels = _make_batch(tokens, positions, weights)
    validate_sft_batch(1, input_dict, labels, seq_len=SEQ_LEN, stop_marker_token=STOP_TOKEN)


def test_no_stop_marker_token_skips_check():
    """When stop_marker_token=None, no assertion runs (backward compat)."""
    n_content = 10
    tokens = [100] + list(range(200, 200 + n_content)) + [STOP_TOKEN] + [-1] * (SEQ_LEN - n_content - 2)
    positions = list(range(n_content + 2)) + [-1] * (SEQ_LEN - n_content - 2)
    weights = [0.0] + [1.0] * n_content + [0.0] + [0.0] * (SEQ_LEN - n_content - 2)

    input_dict, labels = _make_batch(tokens, positions, weights)
    validate_sft_batch(1, input_dict, labels, seq_len=SEQ_LEN, stop_marker_token=None)
