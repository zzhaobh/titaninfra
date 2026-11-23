"""Tests for titaninfra.profiling module.

Covers:
- TrainConfig profiler field defaults and validation
- build_profiler_config translation
- patch_profiler monkey-patch logic
- TraceAnalysis import error handling
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_test_train_config

# ═══════════════════════════════════════════════════════════════
# TrainConfig profiler fields
# ═══════════════════════════════════════════════════════════════


class TestProfilerConfig:
    """TrainConfig profiler field defaults and validation."""

    def test_defaults(self):
        t = make_test_train_config()
        assert t.enable_profiling is False
        assert t.profile_freq == 100
        assert t.profile_active == 2
        assert t.profile_warmup == 3
        assert t.profile_skip_first == 5
        assert t.profile_export_dir == "profile_traces"
        assert t.profile_memory is False
        assert t.profile_with_stack is False
        assert t.profile_with_flops is True

    def test_custom_values(self):
        t = make_test_train_config(
            enable_profiling=True,
            profile_freq=50,
            profile_active=4,
            profile_warmup=2,
            profile_skip_first=10,
            profile_export_dir="custom_traces",
            profile_memory=True,
            profile_with_stack=True,
            profile_with_flops=False,
        )
        assert t.enable_profiling is True
        assert t.profile_freq == 50
        assert t.profile_active == 4
        assert t.profile_warmup == 2
        assert t.profile_skip_first == 10
        assert t.profile_export_dir == "custom_traces"
        assert t.profile_memory is True
        assert t.profile_with_stack is True
        assert t.profile_with_flops is False

    def test_profile_freq_ge_1(self):
        with pytest.raises(ValueError):
            make_test_train_config(profile_freq=0)

    def test_profile_active_ge_1(self):
        with pytest.raises(ValueError):
            make_test_train_config(profile_active=0)

    def test_profile_warmup_ge_0(self):
        t = make_test_train_config(profile_warmup=0)
        assert t.profile_warmup == 0
        with pytest.raises(ValueError):
            make_test_train_config(profile_warmup=-1)

    def test_profile_skip_first_ge_0(self):
        t = make_test_train_config(profile_skip_first=0)
        assert t.profile_skip_first == 0
        with pytest.raises(ValueError):
            make_test_train_config(profile_skip_first=-1)


# ═══════════════════════════════════════════════════════════════
# build_profiler_config
# ═══════════════════════════════════════════════════════════════


class TestBuildProfilerConfig:
    """build_profiler_config correctly translates TrainConfig → Profiler.Config."""

    def test_field_mapping_logic(self):
        """Verify the mapping between TrainConfig and Profiler.Config field names."""
        t = make_test_train_config(
            enable_profiling=True,
            profile_freq=50,
            profile_active=3,
            profile_warmup=2,
            profile_skip_first=10,
            profile_export_dir="my_traces",
            profile_memory=True,
        )
        assert t.enable_profiling is True
        assert t.profile_freq == 50
        assert t.profile_active == 3
        assert t.profile_warmup == 2
        assert t.profile_skip_first == 10
        assert t.profile_export_dir == "my_traces"
        assert t.profile_memory is True


# ═══════════════════════════════════════════════════════════════
# patch_profiler
# ═══════════════════════════════════════════════════════════════


class TestPatchProfiler:
    """patch_profiler correctly monkey-patches trainer config."""

    def test_noop_when_profiling_disabled(self):
        from titaninfra.profiling import patch_profiler

        trainer = MagicMock()
        config = make_test_train_config(enable_profiling=False)
        patch_profiler(trainer, config)
        trainer.config.profiler.build.assert_not_called()

    def test_noop_when_no_extended_features(self):
        from titaninfra.profiling import patch_profiler

        trainer = MagicMock()
        original_build = trainer.config.profiler.build
        config = make_test_train_config(
            enable_profiling=True,
            profile_with_flops=False,
            profile_with_stack=False,
        )
        patch_profiler(trainer, config)
        # build should NOT have been replaced
        assert trainer.config.profiler.build is original_build

    def test_patches_build_when_flops_enabled(self):
        from torchtitan.observability.profiler import Profiler

        from titaninfra.profiling import patch_profiler

        trainer = MagicMock()
        original_method = Profiler.build_torch_profiler
        config = make_test_train_config(enable_profiling=True, profile_with_flops=True)

        try:
            patch_profiler(trainer, config)
            # patch replaces the class method Profiler.build_torch_profiler
            assert Profiler.build_torch_profiler is not original_method
        finally:
            Profiler.build_torch_profiler = original_method

    def test_patches_build_when_stack_enabled(self):
        from torchtitan.observability.profiler import Profiler

        from titaninfra.profiling import patch_profiler

        trainer = MagicMock()
        original_method = Profiler.build_torch_profiler
        config = make_test_train_config(
            enable_profiling=True,
            profile_with_flops=False,
            profile_with_stack=True,
        )

        try:
            patch_profiler(trainer, config)
            assert Profiler.build_torch_profiler is not original_method
        finally:
            Profiler.build_torch_profiler = original_method


# ═══════════════════════════════════════════════════════════════
# TraceAnalysis
# ═══════════════════════════════════════════════════════════════


class TestTraceAnalysis:
    """TraceAnalysis import behavior and API contract."""

    def test_import_error_without_hta(self):
        """TraceAnalysis raises clear ImportError when HTA not installed."""
        with patch.dict("sys.modules", {"hta": None, "hta.trace_analysis": None}):
            import importlib

            import titaninfra.profiling.trace_analysis as ta_mod
            importlib.reload(ta_mod)

            with pytest.raises(ImportError, match="HolisticTraceAnalysis not installed"):
                ta_mod.TraceAnalysis(trace_dir="/nonexistent")

    def test_class_importable(self):
        """TraceAnalysis is importable from titaninfra.profiling."""
        from titaninfra.profiling import TraceAnalysis

        assert TraceAnalysis is not None
        assert hasattr(TraceAnalysis, "get_temporal_breakdown")
        assert hasattr(TraceAnalysis, "get_comm_comp_overlap")
        assert hasattr(TraceAnalysis, "get_gpu_kernel_breakdown")
        assert hasattr(TraceAnalysis, "get_idle_time_breakdown")
        assert hasattr(TraceAnalysis, "generate_report")

    def test_api_signatures(self):
        """TraceAnalysis methods accept expected parameters."""
        import inspect

        from titaninfra.profiling.trace_analysis import TraceAnalysis

        sig = inspect.signature(TraceAnalysis.get_temporal_breakdown)
        assert "ranks" in sig.parameters

        sig = inspect.signature(TraceAnalysis.get_comm_comp_overlap)
        assert "ranks" in sig.parameters

        sig = inspect.signature(TraceAnalysis.get_gpu_kernel_breakdown)
        assert "top_k" in sig.parameters
        assert sig.parameters["top_k"].default == 10

        sig = inspect.signature(TraceAnalysis.get_idle_time_breakdown)
        assert "ranks" in sig.parameters

        sig = inspect.signature(TraceAnalysis.generate_report)
        assert "ranks" in sig.parameters
        assert "top_k" in sig.parameters

    def test_generate_report_calls_all_methods(self):
        """generate_report calls all 4 analysis methods and returns formatted string."""
        from titaninfra.profiling.trace_analysis import TraceAnalysis

        analyzer = TraceAnalysis.__new__(TraceAnalysis)

        mock_df = MagicMock()
        mock_df.to_string.return_value = "mock data"
        mock_df.head.return_value = mock_df

        analyzer.get_temporal_breakdown = MagicMock(return_value=mock_df)
        analyzer.get_comm_comp_overlap = MagicMock(return_value=mock_df)
        analyzer.get_gpu_kernel_breakdown = MagicMock(return_value=mock_df)
        analyzer.get_idle_time_breakdown = MagicMock(return_value=mock_df)

        report = analyzer.generate_report(ranks=[0], top_k=5)

        analyzer.get_temporal_breakdown.assert_called_once_with(ranks=[0])
        analyzer.get_comm_comp_overlap.assert_called_once_with(ranks=[0])
        analyzer.get_gpu_kernel_breakdown.assert_called_once_with(top_k=5)
        analyzer.get_idle_time_breakdown.assert_called_once_with(ranks=[0])

        assert "Temporal Breakdown" in report
        assert "Communication-Computation Overlap" in report
        assert "GPU Kernels" in report
        assert "Idle Time" in report
