"""Unit tests for token estimator."""

import math

import pytest

# ============================================================================
# estimate_epoch_size
# ============================================================================


class TestEstimateEpochSize:
    """Tests for test estimate epoch size."""

    def _call(self, **overrides):
        from titaninfra.data.token_estimator import estimate_epoch_size

        defaults = dict(
            num_samples=10000,
            avg_tokens_per_sample=500.0,
            pack_strategy="hard_truncate",
            normalized_weight=1.0,
            training_steps=1000,
            seq_len=4096,
            local_batch_size=4,
            gas=1,
            dp_world_size=8,
            mosaicml_world_size=8,
            margin=0.05,
            calibration_steps=0,
        )
        defaults.update(overrides)
        return estimate_epoch_size(**defaults)


    def test_hard_truncate_basic(self):
        result = self._call()
        assert result is not None
        assert result > 0
        # tokens_per_rank = 1000 * 4 * 4096 * 1.0 = 16,384,000
        # samples_per_rank = 16,384,000 / 500 = 32,768
        # epoch_size = ceil(32768 * 8 * 1.05) = ceil(275251.2) = 275252
        assert result == math.ceil(32768 * 8 * 1.05)

    def test_sentence_boundary_basic(self):
        result = self._call(pack_strategy="sentence_boundary")
        assert result is not None
        assert result == self._call(pack_strategy="hard_truncate")

    def test_no_pack_basic(self):
        result = self._call(pack_strategy="no_pack")
        assert result is not None
        # sample-based: samples_per_rank = 1000 * 4 * 1 * 1.0 = 4000
        # epoch_size = ceil(4000 * 8 * 1.05) = ceil(33600) = 33600
        assert result == math.ceil(4000 * 8 * 1.05)

    def test_full_sample_basic(self):
        result = self._call(pack_strategy="full_sample")
        assert result is not None
        # full_sample uses token-based math (same as hard_truncate)
        assert result == self._call(pack_strategy="hard_truncate")

    # --- GAS ---

    def test_gas_1_matches_default(self):
        result_gas1 = self._call(gas=1)
        result_default = self._call()
        assert result_gas1 == result_default

    def test_gas_2_doubles_epoch_size(self):
        result_gas1 = self._call(gas=1, margin=0.0)
        result_gas2 = self._call(gas=2, margin=0.0)
        assert result_gas2 == result_gas1 * 2


    def test_training_steps_proportion(self):
        result = self._call(
            training_steps=500,
            normalized_weight=0.5,
            margin=0.0,
        )
        assert result is not None
        # tokens_per_rank = 500 * 4 * 4096 * 0.5 = 4,096,000
        # samples_per_rank = 4,096,000 / 500 = 8192
        # epoch_size = ceil(8192 * 8) = 65536
        assert result == 65536

    # --- Calibration ---

    def test_calibration_steps_included(self):
        without_cal = self._call(calibration_steps=0, margin=0.0)
        with_cal = self._call(calibration_steps=100, margin=0.0)
        assert with_cal > without_cal


    def test_very_small_dataset(self):
        result = self._call(
            num_samples=10,
            training_steps=10,
        )
        assert result is not None
        assert result >= 10

    def test_very_low_proportion(self):
        result = self._call(normalized_weight=0.001, margin=0.0)
        assert result is not None
        assert result > 0

    def test_margin_zero(self):
        result = self._call(margin=0.0)
        result_with_margin = self._call(margin=0.1)
        assert result_with_margin > result

    def test_single_sample(self):
        result = self._call(
            num_samples=1,
            training_steps=10,
        )
        assert result is not None
        assert result >= 1

    def test_mosaicml_scaling(self):
        result_dp_only = self._call(dp_world_size=4, mosaicml_world_size=4, margin=0.0)
        result_with_tp = self._call(dp_world_size=4, mosaicml_world_size=8, margin=0.0)
        assert result_with_tp == result_dp_only * 2


    # --- Input validation ---

    def test_invalid_num_samples(self):
        with pytest.raises(ValueError, match="num_samples"):
            self._call(num_samples=0)

    def test_invalid_gas(self):
        with pytest.raises(ValueError, match="gas"):
            self._call(gas=0)

    def test_invalid_avg_tokens(self):
        with pytest.raises(ValueError, match="avg_tokens"):
            self._call(avg_tokens_per_sample=0)

    def test_proportion_allows_below_num_samples(self):
        result = self._call(
            training_steps=1, normalized_weight=0.001,
            num_samples=100000, margin=0.0,
        )
        assert result < 100000

    # --- Prefetch floor ---

    def test_prefetch_floor_token_based_large_samples(self):
        """LIMO-like: avg_tokens=13112 >> seq_len=2048 → packing_ratio≈6.4."""
        without = self._call(
            avg_tokens_per_sample=13112.0, seq_len=2048,
            training_steps=20, normalized_weight=0.2,
            local_batch_size=1, gas=1, dp_world_size=8, mosaicml_world_size=8,
            margin=2.0, prefetch_batches=0,
        )
        with_prefetch = self._call(
            avg_tokens_per_sample=13112.0, seq_len=2048,
            training_steps=20, normalized_weight=0.2,
            local_batch_size=1, gas=1, dp_world_size=8, mosaicml_world_size=8,
            margin=2.0, prefetch_batches=16,
        )
        assert with_prefetch > without

    def test_prefetch_floor_token_based_normal_samples(self):
        """AIME-like: avg_tokens≈seq_len → packing_ratio≈1, prefetch adds 16."""
        without = self._call(
            avg_tokens_per_sample=2048.0, seq_len=2048,
            training_steps=20, normalized_weight=1.0,
            local_batch_size=1, gas=1, dp_world_size=8, mosaicml_world_size=8,
            margin=0.0, prefetch_batches=0,
        )
        with_prefetch = self._call(
            avg_tokens_per_sample=2048.0, seq_len=2048,
            training_steps=20, normalized_weight=1.0,
            local_batch_size=1, gas=1, dp_world_size=8, mosaicml_world_size=8,
            margin=0.0, prefetch_batches=16,
        )
        assert with_prefetch > without
        assert with_prefetch - without == (math.ceil(16 * 8))

    def test_prefetch_floor_sample_based(self):
        """no_pack: 1:1 sample→batch, so prefetch adds prefetch_batches directly."""
        without = self._call(
            pack_strategy="no_pack",
            training_steps=20, normalized_weight=1.0,
            local_batch_size=1, gas=1, dp_world_size=8, mosaicml_world_size=8,
            margin=0.0, prefetch_batches=0,
        )
        with_prefetch = self._call(
            pack_strategy="no_pack",
            training_steps=20, normalized_weight=1.0,
            local_batch_size=1, gas=1, dp_world_size=8, mosaicml_world_size=8,
            margin=0.0, prefetch_batches=16,
        )
        assert with_prefetch > without
        assert with_prefetch - without == 16 * 8

    def test_prefetch_zero_is_noop(self):
        """prefetch_batches=0 produces same result as before."""
        result = self._call(prefetch_batches=0)
        baseline = self._call()
        assert result == baseline


# ============================================================================
# estimate_total_steps
# ============================================================================


# ============================================================================
# estimate_total_steps_from_repeat
# ============================================================================


class TestEstimateTotalStepsFromRepeat:
    """Tests for test estimate total steps from repeat."""

    def _make_cfg(self, local, repeat=None, pack_strategy="hard_truncate"):
        from titaninfra.config.data import StreamConfig

        return StreamConfig(tokenizer_name="qwen3", local=local, repeat=repeat, pack_strategy=pack_strategy)

    def _call(self, *, stream_stats, stream_configs, **overrides):
        from titaninfra.data.token_estimator import estimate_total_steps_from_repeat

        defaults = dict(
            local_batch_size=4,
            gas=1,
            seq_len=4096,
            dp_world_size=8,
        )
        defaults.update(overrides)
        return estimate_total_steps_from_repeat(
            stream_stats=stream_stats, stream_configs=stream_configs, **defaults,
        )

    def test_single_stream_token_based(self):
        stats = [{"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", repeat=2.0)]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        # weight = 1.0 (single stream)
        # tokens_per_rank = 10000 * 500 / 8 = 625,000
        # tokens_per_step = 4 * 1 * 4096 * 1.0 = 16,384
        # steps_per_epoch = ceil(625000 / 16384) = 39
        # stream_total = ceil(39 * 2.0) = 78
        assert result.per_stream[0].steps_per_epoch == 39
        assert result.total_steps == 78

    def test_single_stream_sample_based(self):
        stats = [{"num_samples": 1000, "total_tokens": 500_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", repeat=3.0, pack_strategy="no_pack")]
        result = self._call(stream_stats=stats, stream_configs=cfgs, dp_world_size=2)
        assert result is not None
        # weight = 1.0
        # samples_per_rank = 1000 / 2 = 500
        # samples_per_step = 4 * 1 * 1.0 = 4
        # steps_per_epoch = ceil(500 / 4) = 125
        # stream_total = ceil(125 * 3.0) = 375
        assert result.per_stream[0].steps_per_epoch == 125
        assert result.total_steps == 375

    def test_multi_stream_bottleneck_by_repeat(self):
        stats = [
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", repeat=1.0),
            self._make_cfg("/data/b", repeat=5.0),
        ]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        assert len(result.per_stream) == 2
        # Stream with higher repeat × steps_per_epoch is the bottleneck
        total_a = math.ceil(result.per_stream[0].steps_per_epoch * 1.0)
        total_b = math.ceil(result.per_stream[1].steps_per_epoch * 5.0)
        assert result.total_steps == max(total_a, total_b)

    def test_multi_stream_bottleneck_by_dataset_size(self):
        stats = [
            {"num_samples": 20000, "total_tokens": 10_000_000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 5000, "total_tokens": 10_000_000, "avg_tokens_per_sample": 2000.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", repeat=2.0),
            self._make_cfg("/data/b", repeat=2.0),
        ]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        assert len(result.per_stream) == 2
        # weights: raw = [2.0*20000*500, 2.0*5000*2000] = [20M, 20M], w_a=0.5, w_b=0.5
        # Equal total tokens → equal steps_per_epoch → both finish simultaneously
        assert result.per_stream[0].steps_per_epoch == result.per_stream[1].steps_per_epoch
        total_a = math.ceil(result.per_stream[0].steps_per_epoch * 2.0)
        total_b = math.ceil(result.per_stream[1].steps_per_epoch * 2.0)
        assert result.total_steps == max(total_a, total_b)

    def test_gas_halves_steps(self):
        stats = [{"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", repeat=2.0)]
        result_gas1 = self._call(stream_stats=stats, stream_configs=cfgs, gas=1)
        result_gas2 = self._call(stream_stats=stats, stream_configs=cfgs, gas=2)
        assert result_gas1 is not None and result_gas2 is not None
        # gas=2 doubles tokens_per_step → halves steps_per_epoch
        assert result_gas2.total_steps < result_gas1.total_steps
        assert result_gas2.per_stream[0].steps_per_epoch < result_gas1.per_stream[0].steps_per_epoch

    def test_missing_stats_returns_none(self):
        cfgs = [self._make_cfg("/data/a", repeat=2.0)]
        result = self._call(stream_stats=[None], stream_configs=cfgs)
        assert result is None

    def test_not_all_repeat_returns_none(self):
        from titaninfra.config.data import StreamConfig

        stats = [
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 5000, "total_tokens": 2_500_000, "avg_tokens_per_sample": 500.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", repeat=2.0),
            StreamConfig(tokenizer_name="qwen3", local="/data/b", proportion=0.5),
        ]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is None

    def test_empty_configs_returns_none(self):
        result = self._call(stream_stats=[], stream_configs=[])
        assert result is None

    def test_zero_avg_tokens_skipped(self):
        stats = [{"num_samples": 1000, "total_tokens": 0, "avg_tokens_per_sample": 0.0}]
        cfgs = [self._make_cfg("/data/a", repeat=2.0)]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is None

    def test_data_passes_computed(self):
        stats = [{"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", repeat=2.0)]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        assert result.per_stream[0].data_passes == 2.0

    def test_data_passes_exact_at_various_dp(self):
        """data_passes = repeat, independent of dp_world_size."""
        stats = [{"num_samples": 20000, "total_tokens": 10_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", repeat=3.0)]
        for dp in [1, 4, 8, 16]:
            result = self._call(stream_stats=stats, stream_configs=cfgs, dp_world_size=dp)
            assert result is not None
            assert result.per_stream[0].data_passes == 3.0, (
                f"dp_world_size={dp}: expected 3.0, got {result.per_stream[0].data_passes}"
            )

    def test_equal_repeat_different_avg_tokens_simultaneous_finish(self):
        """Token-proportional weights ensure all streams finish simultaneously."""
        stats = [
            {"num_samples": 20000, "avg_tokens_per_sample": 4308.88},
            {"num_samples": 20000, "avg_tokens_per_sample": 11911.68},
            {"num_samples": 20000, "avg_tokens_per_sample": 23323.47},
        ]
        cfgs = [
            self._make_cfg("/data/lt8k", repeat=3.0),
            self._make_cfg("/data/8k16k", repeat=3.0),
            self._make_cfg("/data/16k32k", repeat=3.0),
        ]
        result = self._call(
            stream_stats=stats, stream_configs=cfgs,
            dp_world_size=16, local_batch_size=4, gas=4, seq_len=32768,
        )
        assert result is not None
        steps = [
            math.ceil(s.steps_per_epoch * 3.0) for s in result.per_stream
        ]
        assert max(steps) - min(steps) <= 1, (
            f"Streams should finish simultaneously, got steps={steps}"
        )


# ============================================================================
# estimate_total_steps_from_choose
# ============================================================================


class TestEstimateTotalStepsFromChoose:
    """Tests for test estimate total steps from choose."""

    def _make_cfg(self, local, choose=None, pack_strategy="hard_truncate"):
        from titaninfra.config.data import StreamConfig

        return StreamConfig(tokenizer_name="qwen3", local=local, choose=choose, pack_strategy=pack_strategy)

    def _call(self, *, stream_stats, stream_configs, **overrides):
        from titaninfra.data.token_estimator import estimate_total_steps_from_choose

        defaults = dict(
            local_batch_size=4,
            gas=1,
            seq_len=4096,
            dp_world_size=8,
        )
        defaults.update(overrides)
        return estimate_total_steps_from_choose(
            stream_stats=stream_stats, stream_configs=stream_configs, **defaults,
        )

    def test_single_stream_token_based(self):
        stats = [{"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", choose=5000)]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        # weight = 1.0 (single stream)
        # total_tokens = 5000 * 500 = 2,500,000
        # tokens_per_rank = 2,500,000 / 8 = 312,500
        # tokens_per_step = 4 * 1 * 4096 * 1.0 = 16,384
        # steps = ceil(312500 / 16384) = 20
        assert result.per_stream[0].steps_per_epoch == 20
        assert result.total_steps == 20

    def test_single_stream_sample_based(self):
        stats = [{"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", choose=400, pack_strategy="no_pack")]
        result = self._call(stream_stats=stats, stream_configs=cfgs, dp_world_size=2)
        assert result is not None
        # weight = 1.0
        # samples_per_rank = 400 / 2 = 200
        # samples_per_step = 4 * 1 * 1.0 = 4
        # steps = ceil(200 / 4) = 50
        assert result.per_stream[0].steps_per_epoch == 50
        assert result.total_steps == 50

    def test_multi_stream_bottleneck(self):
        stats = [
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 5000, "total_tokens": 10_000_000, "avg_tokens_per_sample": 2000.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", choose=1000),
            self._make_cfg("/data/b", choose=1000),
        ]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        assert len(result.per_stream) == 2
        # Token-proportional weights: w_a=1000*500/(500K+2M)=0.2, w_b=0.8
        # Both streams should finish at similar steps (token-proportional → simultaneous exhaust)
        assert result.total_steps == max(
            result.per_stream[0].steps_per_epoch,
            result.per_stream[1].steps_per_epoch,
        )

    def test_multi_stream_unequal_choose(self):
        stats = [
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", choose=1000),
            self._make_cfg("/data/b", choose=3000),
        ]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        # Same avg_tokens → weights: 1000*500/(1000*500+3000*500)=0.25, 0.75
        assert result.total_steps >= 1

    def test_token_proportional_weight_same_avg(self):
        """Same avg_tokens → choose ratio = weight ratio (token-proportional is neutral)."""
        from titaninfra.data.token_estimator import _compute_weights

        stats = [
            {"num_samples": 10000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 10000, "avg_tokens_per_sample": 500.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", choose=1000),
            self._make_cfg("/data/b", choose=3000),
        ]
        weights = _compute_weights(cfgs, stats)
        assert abs(weights[0] - 0.25) < 1e-6
        assert abs(weights[1] - 0.75) < 1e-6

    def test_token_proportional_weight_different_avg(self):
        """Different avg_tokens → weight reflects token volume, not sample count."""
        from titaninfra.data.token_estimator import _compute_weights

        stats = [
            {"num_samples": 10000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 5000, "avg_tokens_per_sample": 2000.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", choose=1000),
            self._make_cfg("/data/b", choose=1000),
        ]
        weights = _compute_weights(cfgs, stats)
        # w_a = 1000*500 / (500000+2000000) = 0.2
        # w_b = 1000*2000 / 2500000 = 0.8
        assert abs(weights[0] - 0.2) < 1e-6
        assert abs(weights[1] - 0.8) < 1e-6

    def test_simultaneous_exhaust_different_avg(self):
        """With token-proportional weights, streams with different avg_tokens
        should produce similar steps (simultaneous exhaustion, no distribution drift)."""
        stats = [
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 5000, "total_tokens": 10_000_000, "avg_tokens_per_sample": 2000.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", choose=1000),
            self._make_cfg("/data/b", choose=1000),
        ]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        s0, s1 = result.per_stream[0].steps_per_epoch, result.per_stream[1].steps_per_epoch
        # Both should be equal (same total tokens = 1000*500 vs 1000*2000,
        # but weight compensates: 0.2 vs 0.8 → tokens_per_step scales inversely)
        assert s0 == s1

    def test_gas_halves_steps(self):
        stats = [{"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", choose=5000)]
        result_gas1 = self._call(stream_stats=stats, stream_configs=cfgs, gas=1)
        result_gas2 = self._call(stream_stats=stats, stream_configs=cfgs, gas=2)
        assert result_gas1 is not None and result_gas2 is not None
        assert result_gas2.total_steps < result_gas1.total_steps

    def test_missing_stats_skipped(self):
        stats = [
            None,
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", choose=1000),
            self._make_cfg("/data/b", choose=1000),
        ]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        assert result.per_stream[0].steps_per_epoch is None
        assert result.per_stream[1].steps_per_epoch is not None

    def test_all_stats_none_returns_none(self):
        cfgs = [self._make_cfg("/data/a", choose=1000)]
        result = self._call(stream_stats=[None], stream_configs=cfgs)
        assert result is None

    def test_not_all_choose_returns_none(self):
        from titaninfra.config.data import StreamConfig

        stats = [
            {"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0},
            {"num_samples": 5000, "total_tokens": 2_500_000, "avg_tokens_per_sample": 500.0},
        ]
        cfgs = [
            self._make_cfg("/data/a", choose=1000),
            StreamConfig(tokenizer_name="qwen3", local="/data/b", proportion=0.5),
        ]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is None

    def test_empty_configs_returns_none(self):
        result = self._call(stream_stats=[], stream_configs=[])
        assert result is None

    def test_zero_avg_tokens_skipped(self):
        stats = [{"num_samples": 1000, "total_tokens": 0, "avg_tokens_per_sample": 0.0}]
        cfgs = [self._make_cfg("/data/a", choose=500)]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is None

    def test_data_passes_computed(self):
        stats = [{"num_samples": 10000, "total_tokens": 5_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs = [self._make_cfg("/data/a", choose=5000)]
        result = self._call(stream_stats=stats, stream_configs=cfgs)
        assert result is not None
        assert result.per_stream[0].data_passes == 0.5

    def test_data_passes_exact_at_various_dp(self):
        """data_passes = choose/num_samples, independent of dp_world_size."""
        stats = [{"num_samples": 20000, "total_tokens": 86_000_000, "avg_tokens_per_sample": 4300.0}]
        cfgs = [self._make_cfg("/data/a", choose=60000)]
        for dp in [1, 4, 8, 16]:
            result = self._call(stream_stats=stats, stream_configs=cfgs, dp_world_size=dp)
            assert result is not None
            assert result.per_stream[0].data_passes == 3.0, (
                f"dp_world_size={dp}: expected 3.0, got {result.per_stream[0].data_passes}"
            )

    def test_data_passes_multi_stream_dp16(self):
        """Real-world scenario: 3 length-split streams, dp=16, choose=60000 each."""
        stats = [
            {"num_samples": 20000, "total_tokens": 86_177_562, "avg_tokens_per_sample": 4308.88},
            {"num_samples": 20000, "total_tokens": 238_233_506, "avg_tokens_per_sample": 11911.68},
            {"num_samples": 20000, "total_tokens": 466_469_391, "avg_tokens_per_sample": 23323.47},
        ]
        cfgs = [
            self._make_cfg("/data/lt8k", choose=60000),
            self._make_cfg("/data/8k_16k", choose=60000),
            self._make_cfg("/data/16k_32k", choose=60000),
        ]
        result = self._call(
            stream_stats=stats, stream_configs=cfgs,
            local_batch_size=4, gas=4, seq_len=32768, dp_world_size=16,
        )
        assert result is not None
        for ps in result.per_stream:
            assert ps.data_passes == 3.0, f"{ps.stream_id}: expected 3.0, got {ps.data_passes}"

    def test_choose_less_than_dataset(self):
        """choose < num_samples: should use only choose samples, not full dataset."""
        stats = [{"num_samples": 100000, "total_tokens": 50_000_000, "avg_tokens_per_sample": 500.0}]
        cfgs_full = [self._make_cfg("/data/a", choose=100000)]
        cfgs_partial = [self._make_cfg("/data/a", choose=10000)]
        result_full = self._call(stream_stats=stats, stream_configs=cfgs_full)
        result_partial = self._call(stream_stats=stats, stream_configs=cfgs_partial)
        assert result_full is not None and result_partial is not None
        assert result_partial.total_steps < result_full.total_steps


# ============================================================================
# estimate_packing_efficiency
# ============================================================================


class TestEstimatePackingEfficiency:
    """Tests for test estimate packing efficiency."""

    def _call(self, fine_dist, seq_len=2048, pack_strategy="full_sample"):
        from titaninfra.data.token_estimator import estimate_packing_efficiency
        return estimate_packing_efficiency(fine_dist, seq_len, pack_strategy)

    def test_none_returns_1(self):
        assert self._call(None) == 1.0

    def test_empty_returns_1(self):
        assert self._call({}) == 1.0

    def test_hard_truncate_returns_1(self):
        dist = {"0-64": 100, "64-128": 200}
        assert self._call(dist, pack_strategy="hard_truncate") == 1.0

    def test_no_pack_returns_1(self):
        dist = {"0-64": 100}
        assert self._call(dist, pack_strategy="no_pack") == 1.0

    def test_sentence_boundary_fixed_correction(self):
        dist = {"512-576": 100}
        eff = self._call(dist, seq_len=2048, pack_strategy="sentence_boundary")
        expected = 1.0 - 100 / 2048
        assert abs(eff - expected) < 1e-9

    def test_sentence_boundary_short_seq(self):
        eff = self._call({"0-64": 10}, seq_len=200, pack_strategy="sentence_boundary")
        expected = 1.0 - min(100, 200 * 0.05) / 200
        assert abs(eff - expected) < 1e-9

    def test_uniform_short_samples(self):
        dist = {"64-128": 1000}
        eff = self._call(dist, seq_len=2048)
        assert eff > 0.9

    def test_all_samples_exceed_seq_len(self):
        dist = {"4096-4160": 100}
        eff = self._call(dist, seq_len=2048)
        assert 0.1 <= eff <= 1.0

    def test_single_bin_half_seq_len(self):
        dist = {"992-1056": 100}
        eff = self._call(dist, seq_len=2048)
        assert eff > 0.45

    def test_extreme_long_tail(self):
        dist = {"64-128": 990, "4096-4160": 10}
        eff = self._call(dist, seq_len=2048)
        assert eff > 0.85

    def test_all_zero_counts(self):
        dist = {"0-64": 0, "64-128": 0}
        assert self._call(dist) == 1.0

    def test_efficiency_increases_epoch_size(self):
        from titaninfra.data.token_estimator import estimate_epoch_size

        base = dict(
            num_samples=10000,
            avg_tokens_per_sample=500.0,
            pack_strategy="full_sample",
            normalized_weight=1.0,
            training_steps=1000,
            seq_len=2048,
            local_batch_size=4,
            gas=1,
            dp_world_size=8,
            mosaicml_world_size=8,
            margin=0.0,
        )
        epoch_pe1 = estimate_epoch_size(**base, packing_efficiency=1.0)
        epoch_pe07 = estimate_epoch_size(**base, packing_efficiency=0.7)
        assert epoch_pe07 > epoch_pe1


class TestFineHistBucketFunction:
    """Tests for test fine hist bucket function."""

    def _call(self, tc):
        from titaninfra.data.converter import _tc_to_fine_bucket
        return _tc_to_fine_bucket(tc)

    def test_zero(self):
        assert self._call(0) == "0-64"

    def test_boundary(self):
        assert self._call(64) == "64-128"

    def test_mid(self):
        assert self._call(100) == "64-128"

    def test_large(self):
        assert self._call(2048) == "2048-2112"

    def test_overflow(self):
        assert self._call(70000) == "65536+"
