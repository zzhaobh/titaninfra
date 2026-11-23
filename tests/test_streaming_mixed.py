"""Unit tests for streaming mixed."""

import pytest
from pydantic import ValidationError

# ===========================================================================
# ===========================================================================


class TestStreamConfig:
    """Tests for test stream config."""

    def test_valid_proportion(self):
        from titaninfra.config.data import PackStrategy, StreamConfig

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/train", proportion=0.3)
        assert cfg.proportion == 0.3
        assert cfg.pack_strategy == PackStrategy.HARD_TRUNCATE

    def test_valid_repeat(self):
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/train", repeat=2.0)
        assert cfg.repeat == 2.0

    def test_valid_choose(self):
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/train", choose=5000)
        assert cfg.choose == 5000

    def test_valid_sentence_boundary(self):
        from titaninfra.config.data import PackStrategy, StreamConfig

        cfg = StreamConfig(
            tokenizer_name="qwen3",
            local="/data/train",
            proportion=0.3,
            pack_strategy="sentence_boundary",
            sentence_end_tokens=["\u3002", "\n", "."],
        )
        assert cfg.pack_strategy == PackStrategy.SENTENCE_BOUNDARY
        assert len(cfg.sentence_end_tokens) == 3

    def test_valid_full_sample(self):
        from titaninfra.config.data import PackStrategy, StreamConfig

        cfg = StreamConfig(
            tokenizer_name="qwen3",
            local="/data/code", proportion=0.3, pack_strategy="full_sample",
        )
        assert cfg.pack_strategy == PackStrategy.FULL_SAMPLE

    def test_error_mixing_modes(self):
        """Verify error mixing modes."""
        from titaninfra.config.data import StreamConfig

        with pytest.raises(ValidationError, match="互斥"):
            StreamConfig(tokenizer_name="qwen3", local="/data/train", proportion=0.3, repeat=2.0)

    def test_error_proportion_and_choose(self):
        from titaninfra.config.data import StreamConfig

        with pytest.raises(ValidationError, match="互斥"):
            StreamConfig(tokenizer_name="qwen3", local="/data/train", proportion=0.3, choose=100)

    def test_error_sentence_tokens_wrong_strategy(self):
        """Verify error sentence tokens wrong strategy."""
        from titaninfra.config.data import StreamConfig

        with pytest.raises(ValidationError, match="sentence_end_tokens"):
            StreamConfig(
                tokenizer_name="qwen3",
                local="/data/train",
                proportion=0.3,
                pack_strategy="hard_truncate",
                sentence_end_tokens=["\u3002"],
            )

    def test_to_mosaicml_kwargs(self):
        """Verify to mosaicml kwargs."""
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(
            tokenizer_name="qwen3",
            local="/data/train",
            remote="s3://bucket/train",
            proportion=0.3,
            split="train",
            pack_strategy="full_sample",
            stream_id="my_stream",
        )
        kwargs = cfg.to_mosaicml_kwargs()
        assert kwargs == {
            "local": "/data/train",
            "remote": "s3://bucket/train",
            "split": "train",
        }
        assert "proportion" not in kwargs
        assert "pack_strategy" not in kwargs
        assert "stream_id" not in kwargs

    def test_default_stream_id_none(self):
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/train", proportion=0.3)
        assert cfg.stream_id is None

    def test_resolve_stream_id_from_local(self):
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/train", proportion=0.3)
        assert cfg.resolve_stream_id() == "train"

    def test_resolve_stream_id_explicit(self):
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/train", proportion=0.3, stream_id="my_id")
        assert cfg.resolve_stream_id() == "my_id"

    def test_no_mixing_mode_allowed(self):
        """Verify no mixing mode allowed."""
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/train")
        assert cfg.proportion is None
        assert cfg.repeat is None
        assert cfg.choose is None

    def test_data_type_text_default(self):
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/train")
        assert cfg.data_type == "text"

    def test_data_type_messages(self):
        from titaninfra.config.data import StreamConfig

        cfg = StreamConfig(
            tokenizer_name="qwen3",
            local="/data/sft/train",
            data_type="messages",
            family="qwen3",
            add_bos_eos=False,
            proportion=0.3,
        )
        assert cfg.data_type == "messages"
        assert cfg.family == "qwen3"


# ===========================================================================
# ===========================================================================


class TestPackStrategyMapping:
    """Tests for test pack strategy mapping."""

    def test_hard_truncate(self):
        from titaninfra.data.packer import PackMode, strategy_to_pack_mode

        assert strategy_to_pack_mode("hard_truncate") == PackMode.SIMPLE

    def test_sentence_boundary(self):
        from titaninfra.data.packer import PackMode, strategy_to_pack_mode

        assert strategy_to_pack_mode("sentence_boundary") == PackMode.PRETRAIN

    def test_full_sample(self):
        from titaninfra.data.packer import PackMode, strategy_to_pack_mode

        assert strategy_to_pack_mode("full_sample") == PackMode.SFT

    def test_invalid_strategy(self):
        from titaninfra.data.packer import strategy_to_pack_mode

        with pytest.raises(ValueError, match="unknown"):
            strategy_to_pack_mode("unknown")


# ===========================================================================
# ===========================================================================

try:
    import torchtitan  # noqa: F401
    _HAS_TORCHTITAN = True
except ImportError:
    _HAS_TORCHTITAN = False


@pytest.mark.skipif(not _HAS_TORCHTITAN, reason="torchtitan 未安装")
class TestPerStreamWeights:
    """Tests for test per stream weights."""

    def test_proportion_mode(self):
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_weights

        streams = [
            StreamConfig(tokenizer_name="qwen3", local="/data/a", proportion=0.4),
            StreamConfig(tokenizer_name="qwen3", local="/data/b", proportion=0.6),
        ]
        weights = _compute_per_stream_weights(streams)
        assert len(weights) == 2
        assert abs(weights[0] - 0.4) < 1e-6
        assert abs(weights[1] - 0.6) < 1e-6

    def test_choose_mode(self):
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_weights

        streams = [
            StreamConfig(tokenizer_name="qwen3", local="/data/a", choose=100),
            StreamConfig(tokenizer_name="qwen3", local="/data/b", choose=300),
        ]
        weights = _compute_per_stream_weights(streams)
        assert abs(weights[0] - 0.25) < 1e-6
        assert abs(weights[1] - 0.75) < 1e-6

    def test_no_weight_equal(self):
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_weights

        streams = [
            StreamConfig(tokenizer_name="qwen3", local="/data/a"),
            StreamConfig(tokenizer_name="qwen3", local="/data/b"),
            StreamConfig(tokenizer_name="qwen3", local="/data/c"),
        ]
        weights = _compute_per_stream_weights(streams)
        assert len(weights) == 3
        for w in weights:
            assert abs(w - 1.0 / 3) < 1e-6


# ===========================================================================
# ===========================================================================


@pytest.mark.skipif(not _HAS_TORCHTITAN, reason="torchtitan 未安装")
class TestPerStreamEpochSize:
    """Tests for test per stream epoch size."""

    def test_repeat_mode(self):
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", repeat=3.0)
        stats = {
            "num_samples": 1000,
            "total_tokens": 500000,
            "avg_tokens_per_sample": 500.0,
        }
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=stats, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.05,
        )
        # ceil(ceil(1000 * 3.0) * 1.05) = ceil(3150) = 3150
        assert epoch_size == 3150

    def test_choose_mode(self):
        import math

        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", choose=2500)
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=None, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.05,
        )
        # ceil(2500 * 1.05) = 2625
        assert epoch_size == math.ceil(2500 * 1.05)

    def test_choose_mode_margin_zero(self):
        """Verify choose mode margin zero."""
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", choose=2500)
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=None, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.0,
        )
        assert epoch_size == 2500

    def test_choose_mode_margin_effect(self):
        """Verify choose mode margin effect."""
        import math

        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", choose=1000)
        kwargs = dict(
            stream_cfg=cfg, stats=None, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8,
        )
        no_margin = _compute_per_stream_epoch_size(**kwargs, margin=0.0)
        with_margin = _compute_per_stream_epoch_size(**kwargs, margin=0.8)
        assert no_margin == 1000
        assert with_margin == math.ceil(1000 * 1.8)
        assert with_margin > no_margin

    def test_choose_mode_large_margin(self):
        """Verify choose mode large margin."""
        import math

        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", choose=500)
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=None, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=3.0,
        )
        # ceil(500 * 4.0) = 2000
        assert epoch_size == math.ceil(500 * 4.0)

    def test_proportion_mode(self):
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", proportion=0.5)
        stats = {
            "num_samples": 10000,
            "total_tokens": 5000000,
            "avg_tokens_per_sample": 500.0,
        }
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=stats, normalized_weight=0.5,
            training_steps=1000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.05,
        )
        # tokens_per_rank = 1000 x 4 x 4096 x 0.5 = 8,192,000
        # samples_per_rank = 8,192,000 / 500 = 16,384
        # epoch_size = ceil(16384 x 8 x 1.05) = ceil(137625.6) = 137626
        # max(137626, 10000) = 137626
        assert epoch_size is not None
        assert epoch_size >= 10000
        expected = 137626
        assert epoch_size == expected

    def test_proportion_no_stats_returns_none(self):
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", proportion=0.5)
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=None, normalized_weight=0.5,
            training_steps=1000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.05,
        )
        assert epoch_size is None

    def test_no_steps_returns_none(self):
        """Verify no steps returns none."""
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", proportion=0.5)
        stats = {
            "num_samples": 1000,
            "total_tokens": 500000,
            "avg_tokens_per_sample": 500.0,
        }
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=stats, normalized_weight=0.5,
            training_steps=None,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.05,
        )
        assert epoch_size is None

    def test_repeat_mode_margin_zero(self):
        """Verify repeat mode margin zero."""
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", repeat=3.0)
        stats = {"num_samples": 1000, "total_tokens": 500000, "avg_tokens_per_sample": 500.0}
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=stats, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.0,
        )
        assert epoch_size == 3000  # ceil(1000 * 3.0) * 1.0

    def test_repeat_mode_margin_effect(self):
        """Verify repeat mode margin effect."""
        import math

        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", repeat=3.0)
        stats = {"num_samples": 1000, "total_tokens": 500000, "avg_tokens_per_sample": 500.0}
        kwargs = dict(
            stream_cfg=cfg, stats=stats, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8,
        )
        no_margin = _compute_per_stream_epoch_size(**kwargs, margin=0.0)
        with_margin = _compute_per_stream_epoch_size(**kwargs, margin=0.1)
        assert with_margin > no_margin
        # ceil(ceil(1000 * 3.0) * 1.1) = ceil(3300) = 3300
        assert with_margin == math.ceil(3000 * 1.1)

    def test_repeat_mode_fractional(self):
        """Verify repeat mode fractional."""
        import math

        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", repeat=1.5)
        stats = {"num_samples": 100, "total_tokens": 50000, "avg_tokens_per_sample": 500.0}
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=stats, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.05,
        )
        # ceil(ceil(100 * 1.5) * 1.05) = ceil(150 * 1.05) = ceil(157.5) = 158
        assert epoch_size == math.ceil(math.ceil(100 * 1.5) * 1.05)

    def test_repeat_mode_no_stats_returns_none(self):
        """Verify repeat mode no stats returns none."""
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        cfg = StreamConfig(tokenizer_name="qwen3", local="/data/a", repeat=2.0)
        epoch_size = _compute_per_stream_epoch_size(
            stream_cfg=cfg, stats=None, normalized_weight=0.5,
            training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8, margin=0.05,
        )
        assert epoch_size is None

    def test_all_modes_margin_strictly_increases_epoch_size(self):
        """Verify all modes margin strictly increases epoch size."""
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import _compute_per_stream_epoch_size

        stats = {"num_samples": 1000, "total_tokens": 500000, "avg_tokens_per_sample": 500.0}
        base_kwargs = dict(
            normalized_weight=0.5, training_steps=10000,
            seq_len=4096, local_batch_size=4, dp_world_size=8,
        )

        configs = [
            StreamConfig(tokenizer_name="qwen3", local="/data/a", repeat=2.0),
            StreamConfig(tokenizer_name="qwen3", local="/data/a", choose=1000),
            StreamConfig(tokenizer_name="qwen3", local="/data/a", proportion=0.5),
        ]
        for cfg in configs:
            needs_stats = cfg.choose is None
            s = stats if needs_stats else None
            zero = _compute_per_stream_epoch_size(stream_cfg=cfg, stats=s, **base_kwargs, margin=0.0)
            pos = _compute_per_stream_epoch_size(stream_cfg=cfg, stats=s, **base_kwargs, margin=0.5)
            assert zero is not None and pos is not None, f"mode={cfg}"
            assert pos > zero, f"margin 未生效: mode={cfg}, zero={zero}, pos={pos}"


requires_torchtitan = pytest.mark.skipif(
    not _HAS_TORCHTITAN, reason="torchtitan 未安装"
)


@requires_torchtitan
class TestConfigStreamFormat:
    """Tests for test config stream format."""

    def test_stream_config_format(self):
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import StreamingTextDataLoader

        config = StreamingTextDataLoader.Config(
            streams=[
                StreamConfig(
                    tokenizer_name="qwen3",
                    local="/data/math", proportion=0.3,
                    pack_strategy="full_sample",
                ),
                StreamConfig(
                    tokenizer_name="qwen3",
                    local="/data/code", proportion=0.7,
                    pack_strategy="hard_truncate",
                ),
            ],
        )
        assert len(config.streams) == 2

    def test_proportion_sum_validation(self):
        """Verify proportion sum validation."""
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import StreamingTextDataLoader

        with pytest.raises(ValueError, match="proportion"):
            StreamingTextDataLoader.Config(
                streams=[
                    StreamConfig(tokenizer_name="qwen3", local="/data/a", proportion=0.6),
                    StreamConfig(tokenizer_name="qwen3", local="/data/b", proportion=0.6),
                ],
            )

    def test_empty_streams_error(self):
        """Verify empty streams error."""
        from titaninfra.data.streaming_dataset import StreamingTextDataLoader

        with pytest.raises(ValueError):
            StreamingTextDataLoader.Config(streams=[])

    def test_single_stream_valid(self):
        """Verify single stream valid."""
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import StreamingTextDataLoader

        config = StreamingTextDataLoader.Config(
            streams=[StreamConfig(tokenizer_name="qwen3", local="/data/train")],
        )
        assert len(config.streams) == 1

    def test_messages_stream_config(self):
        """Verify messages stream config."""
        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import StreamingTextDataLoader

        config = StreamingTextDataLoader.Config(
            streams=[
                StreamConfig(
                    tokenizer_name="qwen3",
                    local="/data/sft/train",
                    data_type="messages",
                    family="qwen3",
                    add_bos_eos=False,
                    proportion=0.3,
                ),
                StreamConfig(tokenizer_name="qwen3", local="/data/pretrain/train", proportion=0.7),
            ],
            hf_tokenizer_path="/path/to/tokenizer",
        )
        assert len(config.streams) == 2
        assert config.streams[0].data_type == "messages"
        assert config.streams[1].data_type == "text"
