"""Regression test for causal shift alignment between labels and loss_weights.

Bug (2026-05-16): streaming_dataset.py used loss_weights[:-1] (input side)
instead of loss_weights[1:] (target side). This caused loss_weights to be
misaligned with labels by one token when thinking_scale was active. The bug
was masked by WeightedCELoss's valid_mask but boundary tokens got wrong
gradient magnitude.

This test constructs PackedSample with known loss_weights (simulating
thinking_scale), applies the same causal shift logic as
StreamingTextDataLoader, and verifies the invariant:
  wherever labels == IGNORE_INDEX in valid (non-padding) region,
  loss_weights must be 0.0.
"""

import pytest

from titaninfra.data.packer import PackedSample

IGNORE_INDEX = -100


def apply_causal_shift(packed_sample: PackedSample, seq_len: int, pad_id: int = 0):
    """Replicate the causal shift logic from StreamingTextDataLoader._iter_single."""
    input_ids = packed_sample.input_ids[:-1]
    labels = packed_sample.labels[1:]
    pos = packed_sample.positions[:-1] if packed_sample.positions is not None else None
    weights = packed_sample.loss_weights[1:] if packed_sample.loss_weights is not None else None

    actual_len = len(input_ids)
    if actual_len < seq_len:
        pad_len = seq_len - actual_len
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len
        if pos is not None:
            pos = pos + [-1] * pad_len
        if weights is not None:
            weights = weights + [0.0] * pad_len

    return input_ids, labels, pos, weights


def make_sft_sample_with_thinking(
    seq_len: int = 16,
    thinking_scale: float = 0.3,
) -> PackedSample:
    """Create a realistic SFT packed sample with thinking_scale weights.

    Layout (seq_len+1 tokens for causal shift margin):
      [system tokens] [user tokens] [assistant thinking] [assistant response] [eos]
    Labels: IGNORE_INDEX for system+user, token_ids for assistant, eos
    Weights: 0.0 for masked, thinking_scale for thinking, 1.0 for response
    """
    n = seq_len + 1
    system_len = 3
    user_len = 4
    thinking_len = 3
    response_len = n - system_len - user_len - thinking_len - 1
    assert response_len > 0

    input_ids = list(range(100, 100 + n))
    labels = []
    weights = []

    for i in range(n):
        if i < system_len + user_len:
            labels.append(IGNORE_INDEX)
            weights.append(0.0)
        elif i < system_len + user_len + thinking_len:
            labels.append(input_ids[i])
            weights.append(thinking_scale)
        else:
            labels.append(input_ids[i])
            weights.append(1.0)

    positions = list(range(n))

    return PackedSample(
        input_ids=input_ids,
        labels=labels,
        positions=positions,
        loss_weights=weights,
    )


class TestCausalShiftAlignment:
    """Verify labels and loss_weights stay aligned after causal shift."""

    def test_ignore_index_implies_zero_weight(self):
        """Core invariant: wherever labels==IGNORE_INDEX, loss_weights must be 0."""
        sample = make_sft_sample_with_thinking(seq_len=16, thinking_scale=0.3)
        _, labels, _, weights = apply_causal_shift(sample, seq_len=16)

        assert weights is not None
        for i, (lab, w) in enumerate(zip(labels, weights, strict=False)):
            if lab == IGNORE_INDEX:
                assert w == 0.0, (
                    f"Position {i}: labels=IGNORE_INDEX but loss_weights={w} "
                    f"(expected 0.0) — causal shift misalignment"
                )

    def test_thinking_scale_preserved(self):
        """Thinking tokens should retain their scaled weight after shift."""
        sample = make_sft_sample_with_thinking(seq_len=16, thinking_scale=0.3)
        _, labels, _, weights = apply_causal_shift(sample, seq_len=16)

        assert weights is not None
        thinking_weights = [w for lab, w in zip(labels, weights, strict=False) if lab != IGNORE_INDEX and w < 1.0 and w > 0.0]
        assert len(thinking_weights) > 0, "No thinking-scale weights found after shift"
        assert all(w == pytest.approx(0.3) for w in thinking_weights)

    def test_response_weight_is_one(self):
        """Normal response tokens should have weight=1.0."""
        sample = make_sft_sample_with_thinking(seq_len=16, thinking_scale=0.3)
        _, labels, _, weights = apply_causal_shift(sample, seq_len=16)

        assert weights is not None
        response_weights = [w for lab, w in zip(labels, weights, strict=False) if lab != IGNORE_INDEX and w == 1.0]
        assert len(response_weights) > 0, "No response tokens with weight=1.0 found"

    def test_none_weights_passthrough(self):
        """When loss_weights is None (no thinking_scale), shift should yield None."""
        sample = PackedSample(
            input_ids=list(range(17)),
            labels=[IGNORE_INDEX] * 5 + list(range(5, 17)),
            positions=list(range(17)),
            loss_weights=None,
        )
        _, _, _, weights = apply_causal_shift(sample, seq_len=16)
        assert weights is None

    def test_padding_region_zeroed(self):
        """Padding tokens (from short samples) must have labels=IGNORE_INDEX and weights=0."""
        sample = make_sft_sample_with_thinking(seq_len=16, thinking_scale=0.5)
        _, labels, _, weights = apply_causal_shift(sample, seq_len=32)

        assert weights is not None
        for i in range(16, 32):
            assert labels[i] == IGNORE_INDEX, f"Padding pos {i}: labels={labels[i]}"
            assert weights[i] == 0.0, f"Padding pos {i}: weights={weights[i]}"

    @pytest.mark.parametrize("thinking_scale", [0.0, 0.1, 0.5, 1.0])
    def test_various_thinking_scales(self, thinking_scale: float):
        """Invariant holds across different thinking_scale values."""
        sample = make_sft_sample_with_thinking(seq_len=32, thinking_scale=thinking_scale)
        _, labels, _, weights = apply_causal_shift(sample, seq_len=32)

        assert weights is not None
        for i, (lab, w) in enumerate(zip(labels, weights, strict=False)):
            if lab == IGNORE_INDEX:
                assert w == 0.0, (
                    f"thinking_scale={thinking_scale}, pos {i}: "
                    f"labels=IGNORE_INDEX but weights={w}"
                )

    def test_wrong_slice_causes_weight_shift(self):
        """Demonstrate the old bug: [:-1] vs [1:] produces different weight vectors.

        Even when the forward invariant (IGNORE→0) doesn't catch the shift,
        the weights are misaligned by 1 token, giving wrong gradient magnitude
        at boundary tokens.
        """
        sample = make_sft_sample_with_thinking(seq_len=16, thinking_scale=0.3)
        correct_weights = sample.loss_weights[1:]
        wrong_weights = sample.loss_weights[:-1]

        assert correct_weights != wrong_weights, (
            "Correct [1:] and wrong [:-1] slices produce identical weights — "
            "test setup has no boundary transition to detect"
        )

    def test_wrong_slice_misassigns_boundary_weight(self):
        """With a tight layout the wrong slice gives non-zero weight to masked tokens.

        Layout: [masked(1)] [thinking(3)] [response(rest)] — the masked→thinking
        boundary at position 1 means [:-1] shifts a thinking weight onto the
        IGNORE_INDEX position.
        """
        n = 17
        input_ids = list(range(100, 100 + n))
        labels = [IGNORE_INDEX] + [input_ids[i] for i in range(1, n)]
        weights = [0.0] + [0.3] * 3 + [1.0] * (n - 4)
        positions = list(range(n))

        sample = PackedSample(input_ids=input_ids, labels=labels, positions=positions, loss_weights=weights)

        shifted_labels = sample.labels[1:]
        wrong_weights = sample.loss_weights[:-1]

        has_violation = any(lab == IGNORE_INDEX and w != 0.0 for lab, w in zip(shifted_labels, wrong_weights, strict=False))
        assert not has_violation, "This layout doesn't trigger forward invariant violation (only 1 masked token)"

        correct_weights = sample.loss_weights[1:]
        assert correct_weights != wrong_weights, "Slices differ — boundary token gets wrong weight"
        assert correct_weights[0] == 0.3, "Correct: first target token should get thinking weight"
        assert wrong_weights[0] == 0.0, "Wrong: first target token gets masked weight instead"
