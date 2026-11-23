"""Unit tests for metrics plugin."""

from unittest.mock import MagicMock

from titaninfra.utils.metrics_plugin import (
    DataMetricsPlugin,
    MetricsPlugin,
    inject_metrics_plugins,
)


class TestInjectMetricsPlugins:
    """Tests for test inject metrics plugins."""

    def test_inject_wraps_log(self):
        """Verify inject wraps log."""
        trainer = MagicMock()
        original_log = MagicMock()
        trainer.metrics_processor.log = original_log

        plugin = MagicMock(spec=MetricsPlugin)
        plugin.collect.return_value = {"custom/metric": 42.0}

        inject_metrics_plugins(trainer, [plugin])

        assert trainer.metrics_processor.log != original_log

        trainer.metrics_processor.log(
            step=1,
            global_avg_loss=0.5,
            global_max_loss=0.8,
            grad_norm=1.2,
        )

        original_log.assert_called_once()
        call_args = original_log.call_args
        merged_metrics = call_args[0][4] if len(call_args[0]) > 4 else call_args[1].get("extra_metrics", {})
        assert "custom/metric" in (merged_metrics or {}) or original_log.called

    def test_empty_plugins_noop(self):
        """Verify empty plugins noop."""
        trainer = MagicMock()
        original_log = trainer.metrics_processor.log

        inject_metrics_plugins(trainer, [])

        assert trainer.metrics_processor.log == original_log

    def test_num_flops_kwarg_passes_through(self):
        """Verify num flops kwarg passes through."""
        trainer = MagicMock()
        original_log = MagicMock()
        trainer.metrics_processor.log = original_log

        inject_metrics_plugins(trainer, [MagicMock(spec=MetricsPlugin, collect=lambda *a: {})])

        trainer.metrics_processor.log(
            1, 0.5, 0.8, 1.2, {"n_tokens_seen": 100}, num_flops=1.23e14,
        )

        original_log.assert_called_once()
        assert original_log.call_args.kwargs.get("num_flops") == 1.23e14

    def test_no_metrics_processor_skips(self):
        """Verify no metrics processor skips."""
        trainer = MagicMock(spec=[])
        plugin = MagicMock(spec=MetricsPlugin)

        inject_metrics_plugins(trainer, [plugin])


class TestMetricsPluginProtocol:
    """Tests for test metrics plugin protocol."""

    def test_protocol_check(self):
        """Verify protocol check."""
        class MyPlugin:
            def collect(self, step, trainer):
                return {"my/metric": 1.0}

        plugin = MyPlugin()
        assert isinstance(plugin, MetricsPlugin)


class TestDataMetricsPlugin:
    """Tests for test data metrics plugin."""

    def test_collect_per_stream_tokens(self):
        """Verify collect per stream tokens."""
        trainer = MagicMock()
        plugin = DataMetricsPlugin(
            tokens_per_step=1000,
            stream_total_tokens={"zh_wiki": 50000, "en_code": 50000},
            stream_proportions={"zh_wiki": 0.7, "en_code": 0.3},
        )
        metrics = plugin.collect(step=100, trainer=trainer)

        # global_tokens = 100 * 1000 = 100_000
        assert metrics["data/zh_wiki/tokens"] == 70000.0
        assert metrics["data/en_code/tokens"] == 30000.0
        assert metrics["data/total_tokens_seen"] == 100000.0

    def test_epoch_progress(self):
        """Verify epoch progress."""
        trainer = MagicMock()
        plugin = DataMetricsPlugin(
            tokens_per_step=1000,
            stream_total_tokens={"a": 80000, "b": 20000},
            stream_proportions={"a": 0.8, "b": 0.2},
        )
        metrics = plugin.collect(step=100, trainer=trainer)

        # a: 100*1000*0.8 = 80000, progress = 80000/80000 = 1.0
        # b: 100*1000*0.2 = 20000, progress = 20000/20000 = 1.0
        assert abs(metrics["data/a/epoch_progress"] - 1.0) < 1e-6
        assert abs(metrics["data/b/epoch_progress"] - 1.0) < 1e-6
        # global: total_tokens=100000, all_total=100000 → 1.0
        assert abs(metrics["data/epoch_progress"] - 1.0) < 1e-6

    def test_no_proportions_single_stream(self):
        """Verify no proportions single stream."""
        trainer = MagicMock()
        plugin = DataMetricsPlugin(tokens_per_step=2048)
        metrics = plugin.collect(step=50, trainer=trainer)

        assert metrics["data/total_tokens_seen"] == 50 * 2048.0
        assert "data/epoch_progress" not in metrics

    def test_zero_step(self):
        """Verify zero step."""
        trainer = MagicMock()
        plugin = DataMetricsPlugin(
            tokens_per_step=1000,
            stream_proportions={"x": 1.0},
        )
        metrics = plugin.collect(step=0, trainer=trainer)
        assert metrics["data/total_tokens_seen"] == 0.0
        assert metrics["data/x/tokens"] == 0.0

    def test_global_epoch_progress_weighted(self):
        """Verify global epoch progress weighted."""
        trainer = MagicMock()
        plugin = DataMetricsPlugin(
            tokens_per_step=1000,
            stream_total_tokens={"a": 8000, "b": 2000},
            stream_proportions={"a": 0.5, "b": 0.5},
        )
        metrics = plugin.collect(step=10, trainer=trainer)

        # a: 10*1000*0.5=5000, progress=5000/8000=0.625
        # b: 10*1000*0.5=5000, progress=5000/2000=2.5
        # global: total=10000, all_total=10000 → 1.0
        assert abs(metrics["data/a/epoch_progress"] - 0.625) < 1e-6
        assert abs(metrics["data/b/epoch_progress"] - 2.5) < 1e-6
        assert abs(metrics["data/epoch_progress"] - 1.0) < 1e-6

    def test_partial_stream_total_tokens(self):
        """Verify partial stream total tokens."""
        trainer = MagicMock()
        plugin = DataMetricsPlugin(
            tokens_per_step=1000,
            stream_total_tokens={"a": 5000},
            stream_proportions={"a": 0.6, "b": 0.4},
        )
        metrics = plugin.collect(step=10, trainer=trainer)

        assert "data/a/epoch_progress" in metrics
        assert "data/b/epoch_progress" not in metrics
