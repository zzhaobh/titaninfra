"""Unit tests for checkpoint state audit."""

import logging
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from tests.conftest import make_test_model_config, make_test_train_config
from titaninfra.data.packer import StreamingPacker
from titaninfra.data.streaming_dataset import PackedStream, StreamMixer

# ============================================================================
# Helpers
# ============================================================================


def _make_doc_source(docs: list[list[tuple[int, int]]]):
    """Make doc source."""

    class DocSource:
        def __init__(self):
            self._idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._idx >= len(docs):
                raise StopIteration
            doc = docs[self._idx]
            self._idx += 1
            return doc

        def state_dict(self):
            return {"idx": self._idx}

        def load_state_dict(self, state):
            self._idx = state["idx"]

    return DocSource


def _make_infinite_doc_source(docs: list[list[tuple[int, int]]]):
    """Make infinite doc source."""

    class InfiniteDocSource:
        def __init__(self):
            self._idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            doc = docs[self._idx % len(docs)]
            self._idx += 1
            return doc

        def state_dict(self):
            return {"idx": self._idx}

        def load_state_dict(self, state):
            self._idx = state["idx"]

    return InfiniteDocSource


# ============================================================================
# 3.1 StreamMixer state_dict roundtrip
# ============================================================================


class TestStreamMixerStateRoundtrip:
    """Tests for test stream mixer state roundtrip."""

    @staticmethod
    def _build_mixer(seq_len: int = 8):
        """Build mixer."""
        docs_a = [[( i,  i) for i in range(seq_len + 1)] for _ in range(50)]
        docs_b = [[(100 + i, 100 + i) for i in range(seq_len + 1)] for _ in range(50)]

        source_a_cls = _make_infinite_doc_source(docs_a)
        source_b_cls = _make_infinite_doc_source(docs_b)

        packer_a = StreamingPacker(seq_len=seq_len)
        packer_b = StreamingPacker(seq_len=seq_len)

        stream_a = PackedStream(source=source_a_cls(), packer=packer_a)
        stream_b = PackedStream(source=source_b_cls(), packer=packer_b)

        mixer = StreamMixer(streams=[stream_a, stream_b], weights=[0.6, 0.4], rng_seed=42)
        return mixer

    def test_rng_state_deterministic_after_restore(self):
        """Verify rng state deterministic after restore."""
        seq_len = 8
        checkpoint_step = 10
        verify_steps = 10

        baseline_mixer = self._build_mixer(seq_len)
        baseline_iter = iter(baseline_mixer)
        for _ in range(checkpoint_step):
            next(baseline_iter)
        baseline_samples = [next(baseline_iter) for _ in range(verify_steps)]

        test_mixer = self._build_mixer(seq_len)
        test_iter = iter(test_mixer)
        for _ in range(checkpoint_step):
            next(test_iter)

        state = test_mixer.state_dict()

        restored_mixer = self._build_mixer(seq_len)
        restored_mixer.load_state_dict(state)
        restored_iter = iter(restored_mixer)
        restored_samples = [next(restored_iter) for _ in range(verify_steps)]

        for i, (baseline, restored) in enumerate(zip(baseline_samples, restored_samples, strict=False)):
            assert baseline.input_ids == restored.input_ids, (
                f"Step {checkpoint_step + i}: input_ids 不一致"
            )
            assert baseline.labels == restored.labels, (
                f"Step {checkpoint_step + i}: labels 不一致"
            )

    def test_rng_state_preserved_in_state_dict(self):
        """Verify rng state preserved in state dict."""
        mixer = self._build_mixer()
        state = mixer.state_dict()

        assert "rng_state" in state
        assert "stream_states" in state
        assert len(state["stream_states"]) == 2

    def test_stream_source_state_preserved(self):
        """Verify stream source state preserved."""
        mixer = self._build_mixer()
        mixer_iter = iter(mixer)
        for _ in range(5):
            next(mixer_iter)

        state = mixer.state_dict()
        for stream_state in state["stream_states"]:
            assert "source" in stream_state
            assert stream_state["source"]["idx"] > 0


# ============================================================================
# 3.2 StreamingTextDataset state_dict roundtrip
# ============================================================================


class TestStreamingTextDatasetState:
    """Tests for test streaming text dataset state."""

    def test_per_stream_counters_saved_and_restored(self):
        """Verify per stream counters saved and restored."""
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        ds = MagicMock(spec=StreamingTextDataset)
        ds._per_stream_tokens = {"zh": 5000, "en": 3000}
        ds._per_stream_samples = {"zh": 10, "en": 6}
        ds._streaming_samples_consumed = 16
        ds._mixer = None
        ds._streaming_ds = None

        state = StreamingTextDataset.state_dict(ds)

        assert state["per_stream_tokens"] == {"zh": 5000, "en": 3000}
        assert state["per_stream_samples"] == {"zh": 10, "en": 6}
        assert state["streaming_samples_consumed"] == 16

    def test_mixer_branch_in_state_dict(self):
        """Verify mixer branch in state dict."""
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        mock_mixer = MagicMock()
        mock_mixer.state_dict.return_value = {"rng_state": (3, (), None)}

        ds = MagicMock(spec=StreamingTextDataset)
        ds._mixer = mock_mixer
        ds._streaming_ds = None
        ds._streaming_samples_consumed = 0
        ds._per_stream_tokens = {}
        ds._per_stream_samples = {}

        state = StreamingTextDataset.state_dict(ds)

        assert "mixer" in state
        assert "streaming" not in state

    def test_streaming_branch_in_state_dict(self):
        """Verify streaming branch in state dict."""
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        mock_streaming_ds = MagicMock()
        mock_streaming_ds.state_dict.return_value = {"epoch": 0, "sample_in_epoch": 42}

        ds = MagicMock(spec=StreamingTextDataset)
        ds._mixer = None
        ds._streaming_ds = mock_streaming_ds
        ds._streaming_samples_consumed = 42
        ds._per_stream_tokens = {}
        ds._per_stream_samples = {}

        state = StreamingTextDataset.state_dict(ds)

        assert "streaming" in state
        assert "mixer" not in state

    def test_load_state_dict_restores_counters(self):
        """Verify load state dict restores counters."""
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        ds = MagicMock(spec=StreamingTextDataset)
        ds._mixer = None
        ds._streaming_ds = None

        saved_state = {
            "streaming_samples_consumed": 100,
            "per_stream_tokens": {"code": 8000},
            "per_stream_samples": {"code": 20},
        }

        StreamingTextDataset.load_state_dict(ds, saved_state)

        assert ds._streaming_samples_consumed == 100
        assert ds._per_stream_tokens == {"code": 8000}
        assert ds._per_stream_samples == {"code": 20}

    def test_load_state_dict_with_mixer(self):
        """Verify load state dict with mixer."""
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        mock_mixer = MagicMock()
        ds = MagicMock(spec=StreamingTextDataset)
        ds._mixer = mock_mixer
        ds._streaming_ds = None

        saved_state = {
            "streaming_samples_consumed": 50,
            "per_stream_tokens": {},
            "per_stream_samples": {},
            "mixer": {"rng_state": (3, (), None)},
        }

        StreamingTextDataset.load_state_dict(ds, saved_state)

        mock_mixer.load_state_dict.assert_called_once_with({"rng_state": (3, (), None)})


# ============================================================================
# ============================================================================


# ============================================================================
# 3.4 patch_dataloader_resilient_resume
# ============================================================================


class TestResilientResume:
    """Tests for test resilient resume."""

    @staticmethod
    def _make_mock_trainer_for_patch():
        """Make mock trainer for patch."""
        trainer = MagicMock()
        trainer.dataloader.dp_world_size = 4

        original_calls = []

        def original_load(state_dict):
            original_calls.append(state_dict)

        trainer.dataloader.load_state_dict = original_load

        return trainer, original_calls

    def test_world_size_match_calls_original(self):
        """Verify world size match calls original."""
        from titaninfra.engine._pretrain_helpers import patch_dataloader_resilient_resume

        trainer, original_calls = self._make_mock_trainer_for_patch()
        patch_dataloader_resilient_resume(trainer)

        state = {"world_size": 4, "data": "some_state"}
        trainer.dataloader.load_state_dict(state)

        assert len(original_calls) == 1
        assert original_calls[0] is state

    def test_world_size_mismatch_skips_with_warning(self, caplog):
        """Verify world size mismatch skips with warning."""
        from titaninfra.engine._pretrain_helpers import patch_dataloader_resilient_resume

        trainer, original_calls = self._make_mock_trainer_for_patch()
        patch_dataloader_resilient_resume(trainer)

        with caplog.at_level(logging.WARNING):
            trainer.dataloader.load_state_dict({"world_size": 8, "data": "state"})

        assert len(original_calls) == 0
        assert "world_size mismatch" in caplog.text.lower()

    def test_empty_state_dict_passes_through(self):
        """Verify empty state dict passes through."""
        from titaninfra.engine._pretrain_helpers import patch_dataloader_resilient_resume

        trainer, original_calls = self._make_mock_trainer_for_patch()
        patch_dataloader_resilient_resume(trainer)

        trainer.dataloader.load_state_dict({})

        assert len(original_calls) == 1
        assert original_calls[0] == {}

    def test_none_world_size_passes_through(self):
        """Verify none world size passes through."""
        from titaninfra.engine._pretrain_helpers import patch_dataloader_resilient_resume

        trainer, original_calls = self._make_mock_trainer_for_patch()
        patch_dataloader_resilient_resume(trainer)

        trainer.dataloader.load_state_dict({"data": "state"})

        assert len(original_calls) == 1


# ============================================================================
# ============================================================================


class TestCheckpointConfigMapping:
    """Tests for test checkpoint config mapping."""

    @staticmethod
    def _build_config(**overrides):
        """Build config."""
        from titaninfra.engine._pretrain_helpers import build_torchtitan_trainer_config
        from titaninfra.utils.resume import CheckpointSetup

        train_config = make_test_train_config(**overrides)
        model_config = make_test_model_config(name="qwen3/0.6B")
        ckpt_setup = CheckpointSetup(
            ckpt_subdir="checkpoints",
            dump_folder="./outputs",
            initial_load_path=None,
            initial_load_in_hf=False,
        )

        model_spec = MagicMock()
        model_spec.cls = MagicMock()
        model_spec.config = {}
        model_spec.display_name = "test"

        try:
            tt_config = build_torchtitan_trainer_config(
                model_spec=model_spec,
                hf_assets_path="/fake/tokenizer",
                dump_folder="./outputs",
                dataloader_cfg=MagicMock(),
                validator_cfg=MagicMock(),
                train_config=train_config,
                model_config=model_config,
                ckpt_setup=ckpt_setup,
                ckpt_dir="checkpoints",
                build_quantization_config_fn=lambda mc: {},
            )
            return tt_config.checkpointer
        except ImportError:
            pytest.skip("torchtitan not available")

    def test_keep_latest_k_mapping(self):
        """Verify keep latest k mapping."""
        ckpt_cfg = self._build_config(keep_latest_k=5)
        assert ckpt_cfg.keep_latest_k == 5

    def test_keep_latest_k_default(self):
        """Verify keep latest k default."""
        ckpt_cfg = self._build_config()
        assert ckpt_cfg.keep_latest_k == 3

    def test_async_checkpoint_mapping(self):
        """Verify async checkpoint mapping."""
        ckpt_cfg = self._build_config(async_checkpoint="async")
        assert ckpt_cfg.async_mode == "async"

    def test_async_checkpoint_disabled_default(self):
        """Verify async checkpoint disabled default."""
        ckpt_cfg = self._build_config()
        assert ckpt_cfg.async_mode == "disabled"

    def test_exclude_from_loading_mapping(self):
        """Verify exclude from loading mapping."""
        ckpt_cfg = self._build_config(exclude_from_loading=["dataloader", "lr_scheduler"])
        assert "dataloader" in ckpt_cfg.exclude_from_loading
        assert "lr_scheduler" in ckpt_cfg.exclude_from_loading

    def test_ckpt_interval_mapping(self):
        """Verify ckpt interval mapping."""
        ckpt_cfg = self._build_config(ckpt_interval=200)
        assert ckpt_cfg.interval == 200

    def test_initial_load_hf_mode(self):
        """Verify initial load hf mode."""
        from titaninfra.engine._pretrain_helpers import build_torchtitan_trainer_config
        from titaninfra.utils.resume import CheckpointSetup

        ckpt_setup = CheckpointSetup(
            ckpt_subdir="checkpoints",
            dump_folder="./outputs",
            initial_load_path="/fake/hf/model",
            initial_load_in_hf=True,
        )

        try:
            tt_config = build_torchtitan_trainer_config(
                model_spec=MagicMock(),
                hf_assets_path="/fake/tokenizer",
                dump_folder="./outputs",
                dataloader_cfg=MagicMock(),
                validator_cfg=MagicMock(),
                train_config=make_test_train_config(),
                model_config=make_test_model_config(name="qwen3/0.6B"),
                ckpt_setup=ckpt_setup,
                ckpt_dir="checkpoints",
                build_quantization_config_fn=lambda mc: {},
            )
        except ImportError:
            pytest.skip("torchtitan not available")

        assert tt_config.checkpointer.initial_load_in_hf is True
        assert tt_config.checkpointer.initial_load_model_only is True
        assert tt_config.checkpointer.initial_load_path == "/fake/hf/model"


# ============================================================================
# ============================================================================


class TestDataMetricsPluginResume:
    """Tests for test data metrics plugin resume."""

    def test_plugin_no_state_dict_method(self):
        """Verify plugin no state dict method."""
        from titaninfra.utils.metrics_plugin import DataMetricsPlugin

        plugin = DataMetricsPlugin(tokens_per_step=4096)
        assert not hasattr(plugin, "state_dict")
        assert not hasattr(plugin, "load_state_dict")

    def test_plugin_works_from_zero(self):
        """Verify plugin works from zero."""
        from titaninfra.utils.metrics_plugin import DataMetricsPlugin

        plugin = DataMetricsPlugin(
            tokens_per_step=8192,
            stream_total_tokens={"zh": 1_000_000, "en": 500_000},
            stream_proportions={"zh": 0.7, "en": 0.3},
        )

        metrics = plugin.collect(step=0, trainer=None)
        assert metrics["data/total_tokens_seen"] == 0.0

        metrics = plugin.collect(step=100, trainer=None)
        assert metrics["data/total_tokens_seen"] == 100 * 8192
        assert metrics["data/zh/tokens"] == 100 * 8192 * 0.7
        assert metrics["data/en/tokens"] == 100 * 8192 * 0.3

    def test_plugin_epoch_progress_from_zero(self):
        """Verify plugin epoch progress from zero."""
        from titaninfra.utils.metrics_plugin import DataMetricsPlugin

        plugin = DataMetricsPlugin(
            tokens_per_step=4096,
            stream_total_tokens={"data": 40960},
            stream_proportions={"data": 1.0},
        )

        # 10 steps × 4096 tokens = 40960 = 1 epoch
        metrics = plugin.collect(step=10, trainer=None)
        assert abs(metrics["data/data/epoch_progress"] - 1.0) < 1e-6

    def test_plugin_protocol_compliance(self):
        """Verify plugin protocol compliance."""
        from titaninfra.utils.metrics_plugin import DataMetricsPlugin, MetricsPlugin

        plugin = DataMetricsPlugin(tokens_per_step=4096)
        assert isinstance(plugin, MetricsPlugin)


# ============================================================================
# 4. _all_streams_have_repeat
# ============================================================================


class TestAllStreamsHaveRepeat:
    """Tests for test all streams have repeat."""

    def _call(self, streams):
        from titaninfra.engine._pretrain_helpers import _all_streams_have_repeat

        return _all_streams_have_repeat(streams)

    def test_all_have_repeat(self):
        a = MagicMock()
        a.repeat = 1.0
        b = MagicMock()
        b.repeat = 2.0
        assert self._call([a, b]) is True

    def test_one_missing_repeat(self):
        a = MagicMock()
        a.repeat = 1.0
        b = MagicMock()
        b.repeat = None
        assert self._call([a, b]) is False

    def test_empty_list(self):
        assert self._call([]) is False

    def test_single_with_repeat(self):
        s = MagicMock()
        s.repeat = 3.0
        assert self._call([s]) is True

    def test_single_without_repeat(self):
        s = MagicMock()
        s.repeat = None
        assert self._call([s]) is False

    def test_no_repeat_attribute(self):
        s = MagicMock(spec=[])
        assert self._call([s]) is False

    def test_mixed_modes_all_false(self):
        a = MagicMock()
        a.repeat = None
        b = MagicMock()
        b.repeat = None
        assert self._call([a, b]) is False


class TestAllStreamsHaveChoose:
    """Tests for test all streams have choose."""

    def _call(self, streams):
        from titaninfra.engine._pretrain_helpers import _all_streams_have_choose

        return _all_streams_have_choose(streams)

    def test_all_have_choose(self):
        a = MagicMock()
        a.choose = 1000
        b = MagicMock()
        b.choose = 2000
        assert self._call([a, b]) is True

    def test_one_missing_choose(self):
        a = MagicMock()
        a.choose = 1000
        b = MagicMock()
        b.choose = None
        assert self._call([a, b]) is False

    def test_empty_list(self):
        assert self._call([]) is False

    def test_single_with_choose(self):
        s = MagicMock()
        s.choose = 500
        assert self._call([s]) is True

    def test_single_without_choose(self):
        s = MagicMock()
        s.choose = None
        assert self._call([s]) is False

    def test_no_choose_attribute(self):
        s = MagicMock(spec=[])
        assert self._call([s]) is False


# ============================================================================
# ============================================================================


class TestRepeatSchedulingStateSync:
    """Tests for test repeat scheduling state sync."""

    @staticmethod
    def _make_mock_trainer(total_tokens: int, num_samples: int, dp_world_size: int = 1):
        trainer = MagicMock()
        # upstream Trainer/TrainingEngine split: built runtime objects
        # (parallelism_context/optim.lr_schedulers/checkpointer/...) now live
        # on trainer.engine, not on trainer directly — see
        # _pretrain_helpers.py's matching trainer.engine.X reads.
        # trainer.config.X (config FIELD access) is unaffected: Trainer.Config
        # inherits TrainingEngine.Config, so those stay on trainer.config.
        # lr_scheduler itself nests one level deeper under optim= (see
        # build_torchtitan_trainer_config's Optim.Config(...) construction):
        # trainer.config.optim.lr_scheduler / trainer.engine.optim.lr_schedulers.
        # parallel_dims → parallelism_context (pure rename, same dp_replicate/
        # dp_shard fields, confirmed via dataclasses.fields() introspection).
        trainer.engine.parallelism_context.dp_replicate = 1
        trainer.engine.parallelism_context.dp_shard = dp_world_size

        @dataclass
        class _Training:
            steps: int = 100

        trainer.config.training = _Training()

        new_schedulers = MagicMock(name="new_lr_schedulers")
        lr_sched_config = MagicMock()
        lr_sched_config.warmup_steps = 200
        lr_sched_config.build.return_value = new_schedulers
        trainer.config.optim.lr_scheduler = lr_sched_config

        trainer.config.checkpointer.exclude_from_loading = []
        trainer.gradient_accumulation_steps = 1
        trainer.engine.checkpointer.states = {"lr_scheduler": MagicMock(name="old_lr_schedulers")}

        return trainer, new_schedulers

    @staticmethod
    def _make_stream_stats_file(tmp_path, total_tokens, num_samples):
        from titaninfra.data.schema import StreamStats, TokenizerStats

        avg_tokens = total_tokens / max(num_samples, 1)
        stats = StreamStats(
            stream_id="test_stream",
            stream_type="pretrain",
            num_samples=num_samples,
            default_tokenizer="qwen3",
            tokenizer_stats={
                "qwen3": TokenizerStats(
                    total_tokens=total_tokens,
                    avg_tokens_per_sample=avg_tokens,
                ),
            },
        )
        stats.save(str(tmp_path))
        return str(tmp_path)

    @staticmethod
    def _make_stream_mock(stream_path: str, **overrides):
        stream = MagicMock()
        stream.local = stream_path
        stream.proportion = None
        stream.repeat = None
        stream.choose = None
        stream.tokenizer_name = "qwen3"
        for key, value in overrides.items():
            setattr(stream, key, value)
        return stream

    def test_steps_overwritten(self, tmp_path):
        from titaninfra.engine._pretrain_helpers import apply_repeat_scheduling

        total_tokens = 100_000
        num_samples = 1000
        trainer, _ = self._make_mock_trainer(total_tokens, num_samples)

        stream_path = self._make_stream_stats_file(tmp_path, total_tokens, num_samples)
        stream = self._make_stream_mock(stream_path, repeat=2.0, tokenizer_name=None)

        original_steps = trainer.config.training.steps
        apply_repeat_scheduling(
            trainer, streams=[stream],
            local_batch_size=1, seq_len=4096,
        )

        assert trainer.config.training.steps != original_steps
        assert trainer.config.training.steps > 0

    def test_lr_scheduler_rebuilt(self, tmp_path):
        from titaninfra.engine._pretrain_helpers import apply_repeat_scheduling

        trainer, new_schedulers = self._make_mock_trainer(100_000, 1000)
        stream_path = self._make_stream_stats_file(tmp_path, 100_000, 1000)
        stream = self._make_stream_mock(stream_path, repeat=2.0, tokenizer_name=None)

        apply_repeat_scheduling(
            trainer, streams=[stream],
            local_batch_size=1, seq_len=4096,
        )

        assert trainer.engine.optim.lr_schedulers is new_schedulers

    def test_lr_scheduler_excluded_from_loading(self, tmp_path):
        from titaninfra.engine._pretrain_helpers import apply_repeat_scheduling

        trainer, _ = self._make_mock_trainer(100_000, 1000)
        stream_path = self._make_stream_stats_file(tmp_path, 100_000, 1000)
        stream = self._make_stream_mock(stream_path, repeat=2.0, tokenizer_name=None)

        apply_repeat_scheduling(
            trainer, streams=[stream],
            local_batch_size=1, seq_len=4096,
        )

        assert "lr_scheduler" in trainer.config.checkpointer.exclude_from_loading

    def test_lr_scheduler_written_to_checkpointer_states(self, tmp_path):
        from titaninfra.engine._pretrain_helpers import apply_repeat_scheduling

        trainer, new_schedulers = self._make_mock_trainer(100_000, 1000)
        stream_path = self._make_stream_stats_file(tmp_path, 100_000, 1000)
        stream = self._make_stream_mock(stream_path, repeat=2.0, tokenizer_name=None)

        apply_repeat_scheduling(
            trainer, streams=[stream],
            local_batch_size=1, seq_len=4096,
        )

        assert trainer.engine.checkpointer.states["lr_scheduler"] is new_schedulers

    def test_no_stats_does_not_crash(self, tmp_path):
        from titaninfra.engine._pretrain_helpers import apply_repeat_scheduling

        trainer, _ = self._make_mock_trainer(100_000, 1000)
        stream = self._make_stream_mock("/nonexistent/path", repeat=2.0, tokenizer_name=None)

        apply_repeat_scheduling(
            trainer, streams=[stream],
            local_batch_size=1, seq_len=4096,
        )

        assert trainer.config.training.steps == 100

    def test_warmup_ratio_applied(self, tmp_path):
        from titaninfra.engine._pretrain_helpers import apply_repeat_scheduling

        trainer, _ = self._make_mock_trainer(100_000, 1000)
        stream_path = self._make_stream_stats_file(tmp_path, 100_000, 1000)
        stream = self._make_stream_mock(stream_path, repeat=2.0, tokenizer_name=None)

        apply_repeat_scheduling(
            trainer, streams=[stream],
            local_batch_size=1, seq_len=4096,
            warmup_ratio=0.1,
        )

        total_steps = trainer.config.training.steps
        expected_warmup = int(0.1 * total_steps)
        assert trainer.config.optim.lr_scheduler.warmup_steps == expected_warmup
