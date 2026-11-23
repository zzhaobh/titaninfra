"""Unit tests for scheduling invariants."""

import math

import pytest

from titaninfra.config.data import StreamConfig
from titaninfra.data.token_estimator import (
    _compute_weights,
    estimate_epoch_size,
    estimate_total_steps_from_choose,
    estimate_total_steps_from_repeat,
)

# ============================================================================
# Real dataset stats fixtures (from stream_stats.json)
# ============================================================================

AIME_STATS = {
    "num_samples": 96,
    "total_tokens": 206_673,
    "avg_tokens_per_sample": 2152.84375,
}

LIMO_STATS = {
    "num_samples": 640,
    "total_tokens": 8_391_453,
    "avg_tokens_per_sample": 13111.6453125,
}

ALL_STATS = [
    pytest.param(AIME_STATS, id="AIME-96s-2153t"),
    pytest.param(LIMO_STATS, id="LIMO-640s-13112t"),
]

DP_VALUES = [1, 4, 8]
GAS_VALUES = [1, 2, 4]
SEQ_LEN_VALUES = [2048, 4096, 8192]


def _samples_consumed_per_rank(
    total_steps: int,
    local_batch_size: int,
    gas: int,
    seq_len: int,
    weight: float,
    avg_tokens: float,
    sample_based: bool = False,
) -> float:
    if sample_based:
        return total_steps * local_batch_size * gas * weight
    tokens = total_steps * local_batch_size * gas * seq_len * weight
    return tokens / avg_tokens


# ============================================================================
# Class 1: Epoch Size Sufficiency
# ============================================================================


class TestEpochSizeSufficiency:
    """Tests for test epoch size sufficiency."""

    @pytest.mark.parametrize("stats", ALL_STATS)
    @pytest.mark.parametrize("dp", DP_VALUES)
    @pytest.mark.parametrize("gas", GAS_VALUES)
    @pytest.mark.parametrize("seq_len", SEQ_LEN_VALUES)
    def test_proportion_epoch_covers_steps(self, stats, dp, gas, seq_len):
        training_steps = 200
        local_batch_size = 4
        margin = 0.05

        epoch_size = estimate_epoch_size(
            num_samples=stats["num_samples"],
            avg_tokens_per_sample=stats["avg_tokens_per_sample"],
            pack_strategy="hard_truncate",
            normalized_weight=1.0,
            training_steps=training_steps,

            seq_len=seq_len,
            local_batch_size=local_batch_size,
            gas=gas,
            dp_world_size=dp,
            mosaicml_world_size=dp,
            margin=margin,
        )
        assert epoch_size is not None

        consumed = _samples_consumed_per_rank(
            training_steps, local_batch_size, gas, seq_len,
            weight=1.0, avg_tokens=stats["avg_tokens_per_sample"],
        )
        per_rank_available = epoch_size / dp
        assert per_rank_available >= consumed, (
            f"epoch_size/dp={per_rank_available:.1f} < consumed={consumed:.1f}"
        )

    @pytest.mark.parametrize("stats", ALL_STATS)
    @pytest.mark.parametrize("dp", DP_VALUES)
    @pytest.mark.parametrize("repeat_val", [0.5, 1.0, 2.0])
    def test_repeat_epoch_covers_steps(self, stats, dp, repeat_val):
        """Verify repeat epoch covers steps."""
        seq_len = 4096
        local_batch_size = 4
        gas = 1
        margin = 0.8

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/test", repeat=repeat_val)
        weights = _compute_weights([cfg], [stats])
        w = weights[0]

        result = estimate_total_steps_from_repeat(
            stream_stats=[stats],
            stream_configs=[cfg],
            local_batch_size=local_batch_size,
            gas=gas,
            seq_len=seq_len,
            dp_world_size=dp,
        )
        assert result is not None

        epoch_size = math.ceil(
            stats["num_samples"] * repeat_val * (1.0 + margin)
        )
        consumed = _samples_consumed_per_rank(
            result.total_steps, local_batch_size, gas, seq_len,
            weight=w, avg_tokens=stats["avg_tokens_per_sample"],
        )
        per_rank_available = epoch_size / dp
        assert per_rank_available >= consumed, (
            f"repeat={repeat_val}: epoch/dp={per_rank_available:.1f} < consumed={consumed:.1f}"
        )

    @pytest.mark.parametrize("stats", ALL_STATS)
    @pytest.mark.parametrize("dp", DP_VALUES)
    @pytest.mark.parametrize("choose_val", [50, 200, 500])
    def test_choose_epoch_covers_steps(self, stats, dp, choose_val):
        """Verify choose epoch covers steps."""
        seq_len = 4096
        local_batch_size = 4
        gas = 1
        margin = 0.8

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/test", choose=choose_val)
        weights = _compute_weights([cfg], [stats])
        w = weights[0]

        result = estimate_total_steps_from_choose(
            stream_stats=[stats],
            stream_configs=[cfg],
            local_batch_size=local_batch_size,
            gas=gas,
            seq_len=seq_len,
            dp_world_size=dp,
        )
        assert result is not None

        epoch_size = math.ceil(choose_val * (1.0 + margin))
        consumed = _samples_consumed_per_rank(
            result.total_steps, local_batch_size, gas, seq_len,
            weight=w, avg_tokens=stats["avg_tokens_per_sample"],
        )
        per_rank_available = epoch_size / dp
        assert per_rank_available >= consumed, (
            f"choose={choose_val}: epoch/dp={per_rank_available:.1f} < consumed={consumed:.1f}"
        )

    def test_small_dataset_needs_large_margin(self):
        """Verify small dataset needs large margin."""
        dp = 8
        margin_small = 0.05
        margin_recommended = 0.8

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/aime", repeat=0.5)
        result = estimate_total_steps_from_repeat(
            stream_stats=[AIME_STATS], stream_configs=[cfg],
            local_batch_size=4, gas=1, seq_len=4096, dp_world_size=dp,
        )
        assert result is not None

        consumed = _samples_consumed_per_rank(
            result.total_steps, 4, 1, 4096,
            weight=1.0, avg_tokens=AIME_STATS["avg_tokens_per_sample"],
        )

        epoch_small = math.ceil(96 * 0.5 * (1.0 + margin_small)) / dp
        epoch_large = math.ceil(96 * 0.5 * (1.0 + margin_recommended)) / dp

        assert epoch_small < consumed, "margin=0.05 should be insufficient for AIME"
        assert epoch_large >= consumed, "margin=0.8 should be sufficient for AIME"

    @pytest.mark.parametrize("stats", ALL_STATS)
    def test_epoch_size_monotone_with_margin(self, stats):
        margins = [0.0, 0.05, 0.5, 3.0]
        sizes = []
        for m in margins:
            s = estimate_epoch_size(
                num_samples=stats["num_samples"],
                avg_tokens_per_sample=stats["avg_tokens_per_sample"],
                pack_strategy="hard_truncate",
                normalized_weight=1.0,
                training_steps=100,

                seq_len=4096,
                local_batch_size=4,
                gas=1,
                dp_world_size=8,
                mosaicml_world_size=8,
                margin=m,
            )
            sizes.append(s)
        for i in range(len(sizes) - 1):
            assert sizes[i] <= sizes[i + 1], (
                f"margin={margins[i]}→{sizes[i]} > margin={margins[i+1]}→{sizes[i+1]}"
            )


# ============================================================================
# Class 2: Simultaneous Exhaustion
# ============================================================================


class TestSimultaneousExhaustion:
    """Tests for test simultaneous exhaustion."""

    def test_choose_streams_finish_together(self):
        """same choose=100, different avg_tokens → steps should be equal."""
        cfgs = [
            StreamConfig(tokenizer_name="qwen3", local="/data/aime", choose=100),
            StreamConfig(tokenizer_name="qwen3", local="/data/limo", choose=100),
        ]
        result = estimate_total_steps_from_choose(
            stream_stats=[AIME_STATS, LIMO_STATS],
            stream_configs=cfgs,
            local_batch_size=4, gas=1, seq_len=4096, dp_world_size=8,
        )
        assert result is not None
        s0 = result.per_stream[0].steps_per_epoch
        s1 = result.per_stream[1].steps_per_epoch
        assert s0 is not None and s1 is not None
        assert abs(s0 - s1) <= 1, f"steps differ: {s0} vs {s1}"

    def test_choose_unequal_samples_finish_together(self):
        """choose=50 vs choose=200, different avg_tokens → still simultaneous."""
        cfgs = [
            StreamConfig(tokenizer_name="qwen3", local="/data/aime", choose=50),
            StreamConfig(tokenizer_name="qwen3", local="/data/limo", choose=200),
        ]
        result = estimate_total_steps_from_choose(
            stream_stats=[AIME_STATS, LIMO_STATS],
            stream_configs=cfgs,
            local_batch_size=4, gas=1, seq_len=4096, dp_world_size=8,
        )
        assert result is not None
        s0 = result.per_stream[0].steps_per_epoch
        s1 = result.per_stream[1].steps_per_epoch
        assert s0 is not None and s1 is not None
        assert abs(s0 - s1) <= 1, f"steps differ: {s0} vs {s1}"

    @pytest.mark.parametrize("dp", DP_VALUES)
    @pytest.mark.parametrize("gas", GAS_VALUES)
    def test_choose_simultaneous_across_configs(self, dp, gas):
        """Parametrized: choose=100/100 across dp/gas combos."""
        cfgs = [
            StreamConfig(tokenizer_name="qwen3", local="/data/aime", choose=100),
            StreamConfig(tokenizer_name="qwen3", local="/data/limo", choose=100),
        ]
        result = estimate_total_steps_from_choose(
            stream_stats=[AIME_STATS, LIMO_STATS],
            stream_configs=cfgs,
            local_batch_size=4, gas=gas, seq_len=4096, dp_world_size=dp,
        )
        assert result is not None
        s0 = result.per_stream[0].steps_per_epoch
        s1 = result.per_stream[1].steps_per_epoch
        assert s0 is not None and s1 is not None
        assert abs(s0 - s1) <= 1, f"dp={dp},gas={gas}: {s0} vs {s1}"

    def test_repeat_bottleneck_is_max(self):
        """repeat mode: total_steps = max(stream_total_steps)."""
        cfgs = [
            StreamConfig(tokenizer_name="qwen3", local="/data/aime", repeat=1.0),
            StreamConfig(tokenizer_name="qwen3", local="/data/limo", repeat=3.0),
        ]
        result = estimate_total_steps_from_repeat(
            stream_stats=[AIME_STATS, LIMO_STATS],
            stream_configs=cfgs,
            local_batch_size=4, gas=1, seq_len=4096, dp_world_size=8,
        )
        assert result is not None
        s0 = result.per_stream[0].steps_per_epoch
        s1 = result.per_stream[1].steps_per_epoch
        assert s0 is not None and s1 is not None
        total_0 = math.ceil(s0 * 1.0)
        total_1 = math.ceil(s1 * 3.0)
        assert result.total_steps == max(total_0, total_1)


# ============================================================================
# Class 3: Scaling Consistency
# ============================================================================


class TestScalingConsistency:
    """Tests for test scaling consistency."""

    def _steps(self, dp=8, gas=1, local_batch_size=4, stats=None):
        s = stats or LIMO_STATS
        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/test", repeat=2.0)
        result = estimate_total_steps_from_repeat(
            stream_stats=[s], stream_configs=[cfg],
            local_batch_size=local_batch_size, gas=gas,
            seq_len=4096, dp_world_size=dp,
        )
        return result.total_steps if result else None

    def test_dp_doubles_halves_steps(self):
        """dp×2 → roughly steps/2 (same global batch, more ranks)."""
        steps_dp4 = self._steps(dp=4)
        steps_dp8 = self._steps(dp=8)
        assert steps_dp4 is not None and steps_dp8 is not None
        ratio = steps_dp4 / steps_dp8
        assert 1.8 <= ratio <= 2.2, f"dp 4→8 ratio={ratio:.2f}, expected ~2.0"

    def test_gas_doubles_halves_steps(self):
        """GAS×2 → steps/2 (same tokens per step)."""
        steps_gas1 = self._steps(gas=1)
        steps_gas2 = self._steps(gas=2)
        assert steps_gas1 is not None and steps_gas2 is not None
        ratio = steps_gas1 / steps_gas2
        assert 1.8 <= ratio <= 2.2, f"GAS 1→2 ratio={ratio:.2f}, expected ~2.0"

    def test_batch_size_doubles_halves_steps(self):
        """batch_size×2 → steps/2."""
        steps_bs4 = self._steps(local_batch_size=4)
        steps_bs8 = self._steps(local_batch_size=8)
        assert steps_bs4 is not None and steps_bs8 is not None
        ratio = steps_bs4 / steps_bs8
        assert 1.8 <= ratio <= 2.2, f"bs 4→8 ratio={ratio:.2f}, expected ~2.0"

    @pytest.mark.parametrize("stats", ALL_STATS)
    @pytest.mark.parametrize("dp", [1, 4, 8])
    def test_total_tokens_invariant(self, stats, dp):
        """total_steps × batch × GAS × seq_len × weight ≈ expected total tokens."""
        repeat_val = 2.0
        gas = 1
        seq_len = 4096
        local_batch_size = 4

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/test", repeat=repeat_val)
        weights = _compute_weights([cfg], [stats])
        w = weights[0]

        result = estimate_total_steps_from_repeat(
            stream_stats=[stats], stream_configs=[cfg],
            local_batch_size=local_batch_size, gas=gas,
            seq_len=seq_len, dp_world_size=dp,
        )
        assert result is not None

        expected_tokens = stats["num_samples"] * stats["avg_tokens_per_sample"] * repeat_val
        actual_tokens_approx = result.total_steps * local_batch_size * gas * seq_len * w * dp
        ratio = actual_tokens_approx / expected_tokens
        assert 0.9 <= ratio <= 1.5, (
            f"token ratio={ratio:.2f}, expected ~1.0 "
            f"(actual={actual_tokens_approx:.0f}, expected={expected_tokens:.0f})"
        )


# ============================================================================
# Class 4: Cross-Path Consistency
# ============================================================================


class TestCrossPathConsistency:
    """Tests for test cross path consistency."""

    @pytest.mark.parametrize("stats", ALL_STATS)
    @pytest.mark.parametrize("dp", DP_VALUES)
    @pytest.mark.parametrize("repeat_val", [0.5, 1.0, 2.0, 3.0])
    def test_repeat_epoch_vs_estimator(self, stats, dp, repeat_val):
        margin = 0.8
        seq_len = 4096
        local_batch_size = 4
        gas = 1

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/test", repeat=repeat_val)
        weights = _compute_weights([cfg], [stats])
        w = weights[0]

        result = estimate_total_steps_from_repeat(
            stream_stats=[stats], stream_configs=[cfg],
            local_batch_size=local_batch_size, gas=gas,
            seq_len=seq_len, dp_world_size=dp,
        )
        assert result is not None

        epoch_size = math.ceil(
            stats["num_samples"] * repeat_val * (1.0 + margin)
        )

        consumed = _samples_consumed_per_rank(
            result.total_steps, local_batch_size, gas, seq_len,
            weight=w, avg_tokens=stats["avg_tokens_per_sample"],
        )
        per_rank = epoch_size / dp
        assert per_rank >= consumed, (
            f"repeat={repeat_val},dp={dp}: "
            f"epoch/dp={per_rank:.1f} < consumed={consumed:.1f}"
        )

    @pytest.mark.parametrize("stats", ALL_STATS)
    @pytest.mark.parametrize("dp", DP_VALUES)
    @pytest.mark.parametrize("choose_val", [50, 200, 500])
    def test_choose_epoch_vs_estimator(self, stats, dp, choose_val):
        margin = 0.8
        seq_len = 4096
        local_batch_size = 4
        gas = 1

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/test", choose=choose_val)
        weights = _compute_weights([cfg], [stats])
        w = weights[0]

        result = estimate_total_steps_from_choose(
            stream_stats=[stats], stream_configs=[cfg],
            local_batch_size=local_batch_size, gas=gas,
            seq_len=seq_len, dp_world_size=dp,
        )
        assert result is not None

        epoch_size = math.ceil(choose_val * (1.0 + margin))

        consumed = _samples_consumed_per_rank(
            result.total_steps, local_batch_size, gas, seq_len,
            weight=w, avg_tokens=stats["avg_tokens_per_sample"],
        )
        per_rank = epoch_size / dp
        assert per_rank >= consumed, (
            f"choose={choose_val},dp={dp}: "
            f"epoch/dp={per_rank:.1f} < consumed={consumed:.1f}"
        )

    @pytest.mark.parametrize("stats", ALL_STATS)
    @pytest.mark.parametrize("dp", DP_VALUES)
    def test_proportion_epoch_self_consistent(self, stats, dp):
        """Verify proportion epoch self consistent."""
        training_steps = 100
        seq_len = 4096
        local_batch_size = 4
        gas = 1
        margin = 0.05

        epoch_size = estimate_epoch_size(
            num_samples=stats["num_samples"],
            avg_tokens_per_sample=stats["avg_tokens_per_sample"],
            pack_strategy="hard_truncate",
            normalized_weight=1.0,
            training_steps=training_steps,

            seq_len=seq_len,
            local_batch_size=local_batch_size,
            gas=gas,
            dp_world_size=dp,
            mosaicml_world_size=dp,
            margin=margin,
        )
        assert epoch_size is not None

        consumed = _samples_consumed_per_rank(
            training_steps, local_batch_size, gas, seq_len,
            weight=1.0, avg_tokens=stats["avg_tokens_per_sample"],
        )
        per_rank = epoch_size / dp
        assert per_rank >= consumed, (
            f"proportion,dp={dp}: epoch/dp={per_rank:.1f} < consumed={consumed:.1f}"
        )
