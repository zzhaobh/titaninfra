"""Tests for titaninfra.engine.train_step: batch validation."""

import pytest
import torch

from titaninfra.engine.train_step import _microbatch_to_legacy_2d


class TestMicrobatchToLegacy2d:
    """Tests for test microbatch to legacy2d."""

    def _make_microbatch(self, *, local_batch_size, seq_len, with_weights=False):
        from torchtitan.components.data.types import TokenizedTrainingMicrobatch

        total = local_batch_size * seq_len
        input_ids = torch.arange(total, dtype=torch.long) % 1000
        labels = torch.arange(total, dtype=torch.long) % 1000
        positions = torch.arange(seq_len, dtype=torch.long).repeat(local_batch_size)
        padding_mask = torch.zeros(total, dtype=torch.bool)
        model_kwargs = {}
        if with_weights:
            weights = torch.ones(total, dtype=torch.float32)
            model_kwargs["loss_kwargs"] = {"loss_weights": weights}
        return TokenizedTrainingMicrobatch(
            input=input_ids, labels=labels, positions=positions,
            padding_mask=padding_mask,
            loss_token_counts=(labels != -100).sum(),
            routing_token_counts=torch.stack([(~padding_mask).sum()]),
            model_kwargs=model_kwargs,
        )

    def test_reshape_is_lossless(self):
        mb = self._make_microbatch(local_batch_size=3, seq_len=8)
        input_dict, labels = _microbatch_to_legacy_2d(mb, local_batch_size=3, seq_len=8)
        assert input_dict["input"].shape == (3, 8)
        assert input_dict["positions"].shape == (3, 8)
        assert labels.shape == (3, 8)
        # Row boundaries land exactly where the flat tensor was 1024-token
        # chunks of the original mb.input / mb.labels, element-for-element.
        assert torch.equal(input_dict["input"].flatten(), mb.input)
        assert torch.equal(labels.flatten(), mb.labels)

    def test_loss_weights_extracted_from_loss_kwargs_dict(self):
        mb = self._make_microbatch(local_batch_size=2, seq_len=4, with_weights=True)
        input_dict, _ = _microbatch_to_legacy_2d(mb, local_batch_size=2, seq_len=4)
        assert "loss_weights" in input_dict
        assert input_dict["loss_weights"].shape == (2, 4)

    def test_no_loss_weights_key_omitted(self):
        mb = self._make_microbatch(local_batch_size=2, seq_len=4, with_weights=False)
        input_dict, _ = _microbatch_to_legacy_2d(mb, local_batch_size=2, seq_len=4)
        assert "loss_weights" not in input_dict


class TestBatchValidator:
    """Unit tests for validate_sft_batch from titaninfra/eval/pretrain_checks.py."""

    IGNORE_INDEX = -100
    SEQ_LEN = 8

    def _make_packed_batch(self, *, thinking_scale=None):
        """Build a valid 2-doc packed batch [B=1, seq_len=8]."""
        input_ids = torch.arange(self.SEQ_LEN, dtype=torch.long).unsqueeze(0)
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, self.IGNORE_INDEX, 20, 21, 22]])
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        loss_weights = torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]])
        if thinking_scale is not None:
            loss_weights[0, 1] = thinking_scale
            loss_weights[0, 5] = thinking_scale
        input_dict = {"input": input_ids, "positions": positions, "loss_weights": loss_weights}
        return input_dict, labels

    def test_valid_packed_batch_passes(self):
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict, labels = self._make_packed_batch()
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_position_overflow_detected(self):
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict, labels = self._make_packed_batch()
        input_dict["positions"][0, 3] = self.SEQ_LEN
        with pytest.raises(AssertionError, match="position overflow"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_non_contiguous_padding_detected(self):
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, -1, 2, 3, -1, -1, -1]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
        }
        labels = torch.tensor([[
            self.IGNORE_INDEX, 10, self.IGNORE_INDEX, 11, 12,
            self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX,
        ]])
        with pytest.raises(AssertionError, match="non-contiguous padding"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_padding_label_mismatch_detected(self):
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        }
        labels = torch.tensor([[
            self.IGNORE_INDEX, 10, 11, 12, 99,
            self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX,
        ]])
        with pytest.raises(AssertionError, match="labels != -100 in padding"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_padding_weight_mismatch_detected(self):
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]]),
        }
        labels = torch.full((1, self.SEQ_LEN), self.IGNORE_INDEX, dtype=torch.long)
        labels[0, 1:4] = torch.tensor([10, 11, 12])
        with pytest.raises(AssertionError, match="loss_weights != 0 in padding"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_wrong_thinking_scale_detected(self):
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict, labels = self._make_packed_batch(thinking_scale=0.3)
        with pytest.raises(AssertionError, match="fractional weights don't match thinking_scale"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN, thinking_scale=0.7)

    def test_no_positions_skips_position_checks(self):
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {"input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long)}
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, 13, 14, 15, 16]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_no_weights_skips_weight_checks(self):
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, self.IGNORE_INDEX, 20, 21, 22]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_all_ignore_with_real_tokens_early_step_raises(self):
        """Real tokens present but all labels IGNORE_INDEX at step 1 → masking bug."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]]),
        }
        labels = torch.full((1, self.SEQ_LEN), self.IGNORE_INDEX, dtype=torch.long)
        with pytest.raises(AssertionError, match=r"1 docs.*masking bug"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_all_ignore_all_padding_early_step_warns(self):
        """All-padding batch (data exhaustion) at step 2 → warning, no raise."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.full((1, self.SEQ_LEN), -1, dtype=torch.long),
        }
        labels = torch.full((1, self.SEQ_LEN), self.IGNORE_INDEX, dtype=torch.long)
        validate_sft_batch(2, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_all_ignore_real_tokens_step2_warns(self):
        """Real tokens + all IGNORE_INDEX at step 2 → packing boundary, no raise."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]]),
        }
        labels = torch.full((1, self.SEQ_LEN), self.IGNORE_INDEX, dtype=torch.long)
        validate_sft_batch(2, input_dict, labels, seq_len=self.SEQ_LEN)


class TestExpectedMarkersPerDoc:
    """expected_markers: per-document (not per-row) for HARD_TRUNCATE compatibility."""

    IGNORE_INDEX = -100
    SEQ_LEN = 12
    MARKER_A = 999
    MARKER_B = 888

    def test_two_complete_docs_both_have_markers(self):
        """2 complete docs in one row, both contain markers → pass."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_ids = torch.tensor([[self.MARKER_A, 1, 2, self.MARKER_B, 3, 4, self.MARKER_A, 5, 6, self.MARKER_B, 7, 8]])
        positions = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]])
        labels = torch.tensor([[self.IGNORE_INDEX, 1, 2, 3, 4, 5, self.IGNORE_INDEX, 5, 6, 7, 8, 9]])
        input_dict = {"input": input_ids, "positions": positions}
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN,
                           expected_markers=[self.MARKER_A, self.MARKER_B])

    def test_short_fragment_skipped(self):
        """A 3-token doc fragment (< 4 tokens) is skipped by marker check."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        # Doc 1: 3 tokens (fragment, skipped), Doc 2: 5 tokens (checked), padding: 4
        input_ids = torch.tensor([[1, 2, 3, self.MARKER_A, 5, 6, self.MARKER_B, 7, 0, 0, 0, 0]])
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4, -1, -1, -1, -1]])
        labels = torch.tensor([[self.IGNORE_INDEX, 2, 3, self.IGNORE_INDEX, 5, 6, 7, 8,
                                self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX]])
        input_dict = {"input": input_ids, "positions": positions}
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN,
                           expected_markers=[self.MARKER_A, self.MARKER_B])

    def test_missing_marker_in_complete_doc_raises(self):
        """A complete doc (>= 4 tokens) missing a marker → raises."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        # Single doc of 8 tokens, has MARKER_A but missing MARKER_B
        input_ids = torch.tensor([[self.MARKER_A, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0]])
        positions = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1]])
        labels = torch.tensor([[self.IGNORE_INDEX, 1, 2, 3, 4, 5, 6, 7,
                                self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX]])
        input_dict = {"input": input_ids, "positions": positions}
        with pytest.raises(AssertionError, match="expected marker token 888"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN,
                               expected_markers=[self.MARKER_A, self.MARKER_B])

    def test_no_positions_checks_whole_row(self):
        """Without positions, the entire row is treated as one doc."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_ids = torch.tensor([[self.MARKER_A, 1, 2, 3, 4, 5, 6, self.MARKER_B, 8, 9, 10, 11]])
        labels = torch.tensor([[self.IGNORE_INDEX, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        input_dict = {"input": input_ids}
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN,
                           expected_markers=[self.MARKER_A, self.MARKER_B])

    def test_all_padding_row_skipped(self):
        """A row that is entirely padding is skipped (no marker check)."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_ids = torch.tensor([
            [self.MARKER_A, 1, 2, self.MARKER_B, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        positions = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ])
        labels = torch.tensor([
            [self.IGNORE_INDEX, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [self.IGNORE_INDEX] * self.SEQ_LEN,
        ])
        input_dict = {"input": input_ids, "positions": positions}
        # Row 0 has markers, row 1 is all padding (valid_end=0) → skipped, should pass
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN,
                           expected_markers=[self.MARKER_A, self.MARKER_B])


class TestLabelsWeightsConsistency:
    """Labels-weights consistency: IGNORE_INDEX in valid region → weight must be 0."""

    IGNORE_INDEX = -100
    SEQ_LEN = 8

    def test_consistent_labels_weights_passes(self):
        """IGNORE_INDEX labels have weight=0 in valid region → pass."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, self.IGNORE_INDEX, 20, 21, 22]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_nonzero_weight_at_ignore_label_raises(self):
        """IGNORE_INDEX label with non-zero weight in valid region → raises."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "loss_weights": torch.tensor([[0.5, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]),
        }
        # Position 0 has IGNORE_INDEX but weight=0.5 → should fail
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, self.IGNORE_INDEX, 20, 21, 22]])
        with pytest.raises(AssertionError, match="loss_weights != 0 where labels==IGNORE_INDEX"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_padding_region_not_checked_by_this_rule(self):
        """IGNORE_INDEX in padding (positions==-1) doesn't trigger this check."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12,
                                self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_thinking_scale_zero_weight_at_ignore_ok(self):
        """thinking_scale with weight=0 at IGNORE_INDEX positions → pass."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "loss_weights": torch.tensor([[0.0, 0.3, 1.0, 1.0, 0.0, 0.3, 1.0, 1.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, self.IGNORE_INDEX, 20, 21, 22]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN, thinking_scale=0.3)


class TestThinkingScaleMultiValue:
    """thinking_scale validation handles unique fractional values correctly."""

    IGNORE_INDEX = -100
    SEQ_LEN = 8

    def test_single_fractional_value_matches(self):
        """All fractional weights == thinking_scale → pass."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "loss_weights": torch.tensor([[0.0, 0.3, 1.0, 1.0, 0.0, 0.3, 1.0, 1.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, self.IGNORE_INDEX, 20, 21, 22]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN, thinking_scale=0.3)

    def test_wrong_fractional_value_raises(self):
        """Fractional weights != thinking_scale → raises."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "loss_weights": torch.tensor([[0.0, 0.5, 1.0, 1.0, 0.0, 0.5, 1.0, 1.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, self.IGNORE_INDEX, 20, 21, 22]])
        with pytest.raises(AssertionError, match="fractional weights don't match thinking_scale"):
            validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN, thinking_scale=0.3)

    def test_no_fractional_weights_passes(self):
        """Only 0.0 and 1.0 weights, no fractional → pass regardless of thinking_scale."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(1, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 10, 11, 12, self.IGNORE_INDEX, 20, 21, 22]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN, thinking_scale=0.7)


class TestPackingStrategyScenarios:
    """Cross-cutting scenarios for different packing strategies."""

    IGNORE_INDEX = -100
    SEQ_LEN = 8

    def test_no_pack_single_doc_with_padding(self):
        """NO_PACK: single doc shorter than seq_len, padded with -1 positions."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.tensor([[10, 11, 12, 13, 14, 0, 0, 0]]),
            "positions": torch.tensor([[0, 1, 2, 3, 4, -1, -1, -1]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 11, 12, 13, 14,
                                self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_full_sample_partial_fill(self):
        """FULL_SAMPLE: 1 complete doc + padding (doc preserved whole)."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.tensor([[10, 11, 12, 13, 14, 15, 0, 0]]),
            "positions": torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]]),
            "loss_weights": torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, self.IGNORE_INDEX, 12, 13, 14, 15,
                                self.IGNORE_INDEX, self.IGNORE_INDEX]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_hard_truncate_no_padding(self):
        """HARD_TRUNCATE: row fully packed (no -1 padding), docs may be split."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]]),
            "positions": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]),
        }
        labels = torch.tensor([[self.IGNORE_INDEX, 11, 12, 13, self.IGNORE_INDEX, 21, 22, 23]])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)

    def test_multi_batch_mixed_fill(self):
        """B=2: row 0 fully packed, row 1 has padding → both valid."""
        from titaninfra.eval.pretrain_checks import validate_sft_batch

        input_dict = {
            "input": torch.zeros(2, self.SEQ_LEN, dtype=torch.long),
            "positions": torch.tensor([
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 4, -1, -1, -1],
            ]),
            "loss_weights": torch.tensor([
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]),
        }
        labels = torch.tensor([
            [self.IGNORE_INDEX, 1, 2, 3, self.IGNORE_INDEX, 5, 6, 7],
            [self.IGNORE_INDEX, 1, 2, 3, 4, self.IGNORE_INDEX, self.IGNORE_INDEX, self.IGNORE_INDEX],
        ])
        validate_sft_batch(1, input_dict, labels, seq_len=self.SEQ_LEN)
