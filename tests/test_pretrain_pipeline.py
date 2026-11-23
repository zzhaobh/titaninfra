"""Unit tests for pretrain pipeline."""

import pytest

from tests.conftest import make_test_train_config

try:
    import torchtitan  # noqa: F401

    HAS_TORCHTITAN = True
except ImportError:
    HAS_TORCHTITAN = False

requires_torchtitan = pytest.mark.skipif(
    not HAS_TORCHTITAN, reason="torchtitan 未安装"
)


# =========================================================================
# =========================================================================


class TestTrainConfigValidation:
    """Tests for test train config validation."""

    def test_val_streams_requires_eval_steps(self):
        """Verify val streams requires eval steps."""
        from pydantic import ValidationError


        with pytest.raises(ValidationError, match="eval_steps"):
            make_test_train_config(
                val_streams=[{"local": "/data/eval", "tokenizer_name": "qwen3"}],
                eval_steps=-1,
            )

    def test_val_streams_with_positive_eval_steps(self):
        """Verify val streams with positive eval steps."""

        config = make_test_train_config(
            val_streams=[{"local": "/data/eval", "tokenizer_name": "qwen3"}],
            eval_steps=10,
            eval_freq=5,
        )
        assert config.eval_steps == 10

    def test_epoch_size_margin_on_stream_config(self):
        """Verify epoch size margin on stream config."""
        from pydantic import ValidationError

        from titaninfra.config.data import StreamConfig

        s = StreamConfig(tokenizer_name="qwen3", local="/data/train")
        assert s.epoch_size_margin == 0.05

        s = StreamConfig(tokenizer_name="qwen3", local="/data/train", epoch_size_margin=3.0)
        assert s.epoch_size_margin == 3.0

        s = StreamConfig(tokenizer_name="qwen3", local="/data/train", epoch_size_margin=10.0)
        assert s.epoch_size_margin == 10.0

        with pytest.raises(ValidationError):
            StreamConfig(tokenizer_name="qwen3", local="/data/train", epoch_size_margin=11.0)


# =========================================================================
# =========================================================================


class TestStreamConfigValidation:
    """Tests for test stream config validation."""

    def test_requires_local(self):
        """Verify requires local."""
        from pydantic import ValidationError

        from titaninfra.config.data import StreamConfig

        with pytest.raises(ValidationError):
            StreamConfig(tokenizer_name="qwen3", proportion=0.1)

    def test_mixing_modes_mutually_exclusive(self):
        """Verify mixing modes mutually exclusive."""
        from pydantic import ValidationError

        from titaninfra.config.data import StreamConfig

        with pytest.raises(ValidationError, match="互斥"):
            StreamConfig(
                tokenizer_name="qwen3",
                local="./data/train",
                proportion=0.1,
                repeat=2.0,
            )

    def test_proportion_valid(self):
        """Verify proportion valid."""
        from titaninfra.config.data import StreamConfig

        config = StreamConfig(
            tokenizer_name="qwen3",
            local="./data/train",
            proportion=0.1,
        )
        assert config.proportion == 0.1

    def test_repeat_valid(self):
        """Verify repeat valid."""
        from titaninfra.config.data import StreamConfig

        config = StreamConfig(
            tokenizer_name="qwen3",
            local="./data/train",
            repeat=2.0,
        )
        assert config.repeat == 2.0

    def test_choose_valid(self):
        """Verify choose valid."""
        from titaninfra.config.data import StreamConfig

        config = StreamConfig(
            tokenizer_name="qwen3",
            local="./data/train",
            choose=1000,
        )
        assert config.choose == 1000

    def test_no_mixing_mode(self):
        """Verify no mixing mode."""
        from titaninfra.config.data import StreamConfig

        config = StreamConfig(tokenizer_name="qwen3", local="./data/train")
        assert config.proportion is None
        assert config.repeat is None
        assert config.choose is None


# =========================================================================
# =========================================================================


@requires_torchtitan
class TestPackedStream:
    """Tests for test packed stream."""

    def _make_source(self, n_docs=5, doc_len=100):
        """Make source."""
        docs = []
        for i in range(n_docs):
            doc = [(i * doc_len + j, i * doc_len + j) for j in range(doc_len)]
            docs.append(doc)
        return iter(docs)

    def test_iteration_produces_packed_samples(self):
        """Verify iteration produces packed samples."""
        from titaninfra.data.packer import PackedSample, PackMode, StreamingPacker
        from titaninfra.data.streaming_dataset import PackedStream

        source = self._make_source(n_docs=3, doc_len=50)
        packer = StreamingPacker(seq_len=64, mode=PackMode.SIMPLE)
        stream = PackedStream(source, packer, name="test")

        samples = list(stream)
        assert len(samples) > 0
        for s in samples:
            assert isinstance(s, PackedSample)
            assert len(s.input_ids) <= 65
            assert len(s.labels) == len(s.input_ids)

    def test_state_dict_round_trip(self):
        """Verify state dict round trip."""
        from titaninfra.data.packer import PackMode, StreamingPacker
        from titaninfra.data.streaming_dataset import PackedStream

        source = self._make_source(n_docs=3, doc_len=50)
        packer = StreamingPacker(seq_len=64, mode=PackMode.SIMPLE)
        stream = PackedStream(source, packer, name="test")

        state = stream.state_dict()
        assert "packer" not in state

        stream.load_state_dict(state)

    def test_next_uses_lazy_iterator(self):
        """Verify next uses lazy iterator."""
        from titaninfra.data.packer import PackedSample, PackMode, StreamingPacker
        from titaninfra.data.streaming_dataset import PackedStream

        source = self._make_source(n_docs=3, doc_len=200)
        packer = StreamingPacker(seq_len=64, mode=PackMode.SIMPLE)
        stream = PackedStream(source, packer, name="test")

        sample = next(stream)
        assert isinstance(sample, PackedSample)

    def test_epoch_end_residue_emitted_simple(self):
        """Verify epoch end residue emitted simple."""
        from titaninfra.data.packer import PackMode, StreamingPacker
        from titaninfra.data.streaming_dataset import PackedStream

        source = self._make_source(n_docs=3, doc_len=50)
        packer = StreamingPacker(seq_len=64, mode=PackMode.SIMPLE)
        stream = PackedStream(source, packer, name="test")

        samples = list(stream)
        assert len(samples) == 3, (
            f"期望 3 个 PackedSample（2 full + 1 residual），实际 {len(samples)}。"
            "若为 2，说明 flush_all() 未在 epoch 末调用。"
        )
        assert len(samples[0].input_ids) == 65
        assert len(samples[1].input_ids) == 65
        assert len(samples[2].input_ids) == 20

    def test_epoch_end_residue_emitted_sft(self):
        """Verify epoch end residue emitted sft."""
        from titaninfra.data.packer import PackMode, StreamingPacker
        from titaninfra.data.streaming_dataset import PackedStream

        docs = [[(j, j) for j in range(200)] for _ in range(5)]
        packer = StreamingPacker(seq_len=4096, mode=PackMode.SFT)
        stream = PackedStream(source=iter(docs), packer=packer, name="test-sft")

        samples = list(stream)
        assert len(samples) == 1, (
            f"期望 1 个 PackedSample（5 docs bin-packed），实际 {len(samples)}。"
            "若为 0，说明 flush_all() 未在 epoch 末调用。"
        )
        assert len(samples[0].input_ids) == 1000
        expected_positions = list(range(200)) * 5
        assert samples[0].positions == expected_positions


# =========================================================================
# =========================================================================


@requires_torchtitan
class TestStreamMixer:
    """Tests for test stream mixer."""

    def _make_finite_stream(self, n_samples, stream_id):
        """Make finite stream."""
        from titaninfra.data.packer import PackMode, StreamingPacker
        from titaninfra.data.streaming_dataset import PackedStream

        docs = []
        for i in range(n_samples):
            doc = [(stream_id * 1000 + i * 10 + j, stream_id * 1000 + i * 10 + j) for j in range(10)]
            docs.append(doc)

        packer = StreamingPacker(seq_len=8, mode=PackMode.SIMPLE)
        return PackedStream(iter(docs), packer, name=f"stream-{stream_id}")

    def test_weighted_mixing(self):
        """Verify weighted mixing."""
        from titaninfra.data.streaming_dataset import StreamMixer

        stream_a = self._make_finite_stream(10, stream_id=0)
        stream_b = self._make_finite_stream(10, stream_id=1)

        mixer = StreamMixer(
            streams=[stream_a, stream_b],
            weights=[0.7, 0.3],
            rng_seed=42,
        )

        samples = list(mixer)
        assert len(samples) > 0

    def test_exhaustion_terminates(self):
        """Verify exhaustion terminates."""
        from titaninfra.data.streaming_dataset import StreamMixer

        stream_a = self._make_finite_stream(3, stream_id=0)
        stream_b = self._make_finite_stream(3, stream_id=1)

        mixer = StreamMixer(
            streams=[stream_a, stream_b],
            weights=[0.5, 0.5],
            rng_seed=42,
        )

        samples = list(mixer)
        assert len(samples) > 0

    def test_rng_determinism(self):
        """Verify rng determinism."""
        from titaninfra.data.streaming_dataset import StreamMixer

        def _run(seed):
            stream_a = self._make_finite_stream(5, stream_id=0)
            stream_b = self._make_finite_stream(5, stream_id=1)
            mixer = StreamMixer(
                streams=[stream_a, stream_b],
                weights=[0.5, 0.5],
                rng_seed=seed,
            )
            return [s.input_ids for s in mixer]

        run1 = _run(42)
        run2 = _run(42)
        assert run1 == run2

    def test_mismatched_lengths_raises(self):
        """Verify mismatched lengths raises."""
        from titaninfra.data.streaming_dataset import StreamMixer

        stream_a = self._make_finite_stream(3, stream_id=0)

        with pytest.raises(ValueError, match="长度不匹配"):
            StreamMixer(
                streams=[stream_a],
                weights=[0.5, 0.5],
            )

    def test_state_dict_round_trip(self):
        """Verify state dict round trip."""
        from titaninfra.data.streaming_dataset import StreamMixer

        stream_a = self._make_finite_stream(5, stream_id=0)
        stream_b = self._make_finite_stream(5, stream_id=1)

        mixer = StreamMixer(
            streams=[stream_a, stream_b],
            weights=[0.5, 0.5],
            rng_seed=42,
        )

        state = mixer.state_dict()
        assert "rng_state" in state
        assert "stream_states" in state
        assert len(state["stream_states"]) == 2

        mixer.load_state_dict(state)
