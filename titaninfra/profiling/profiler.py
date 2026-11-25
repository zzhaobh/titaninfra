"""Extended Kineto profiling built on torchtitan's per-tool profiler configs.

Torchtitan's ``Profiler`` owns independent trace and memory-snapshot tools.
TitanInfra supplies a ``KinetoTraceProfiler`` subclass for the two options the
upstream config does not expose (``with_flops`` / ``with_stack``) and for the
rank-0 text summary. No global Profiler method is monkey-patched.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=1)
def _extended_kineto_profiler_cls():
    """Define the upstream-dependent subclass lazily for Mac/test imports."""
    import torch
    from torchtitan.observability.profiler import KinetoTraceProfiler

    class ExtendedKinetoTraceProfiler(KinetoTraceProfiler):
        """Kineto traces with titaninfra's extra profiler options."""

        @dataclass(kw_only=True, slots=True)
        class Config(KinetoTraceProfiler.Config):
            with_flops: bool = False
            with_stack: bool = False
            write_rank0_summary: bool = True

        def __init__(self, config, **kwargs):
            if not config.with_flops and not config.with_stack:
                super().__init__(config, **kwargs)
                return

            # Upstream TraceProfiler owns schedule/activity/trace-handler setup
            # but does not expose these two torch.profiler kwargs. Scope the
            # injection to this constructor so the rest of the process keeps
            # the real factory and all upstream lifecycle behavior.
            original_profile = torch.profiler.profile

            def extended_profile(*args, **profile_kwargs):
                profile_kwargs["with_flops"] = config.with_flops
                profile_kwargs["with_stack"] = config.with_stack
                return original_profile(*args, **profile_kwargs)

            torch.profiler.profile = extended_profile
            try:
                super().__init__(config, **kwargs)
            finally:
                torch.profiler.profile = original_profile

        def export_trace(self, prof, trace_dir):
            super().export_trace(prof, trace_dir)
            if self._rank != 0 or not self._config.write_rank0_summary:
                return
            summary_path = os.path.join(trace_dir, "summary_rank0.txt")
            with open(summary_path, "w") as f:
                f.write(
                    prof.key_averages().table(
                        sort_by="cuda_time_total", row_limit=30
                    )
                )

    return ExtendedKinetoTraceProfiler


def build_profiler_config(train_config):
    """Translate TrainConfig fields into upstream's per-tool Profiler.Config."""
    from torchtitan.observability.profiler import MemorySnapshotProfiler, Profiler

    trace = None
    if train_config.enable_profiling:
        trace_cls = _extended_kineto_profiler_cls()
        trace = trace_cls.Config(
            save_folder=train_config.profile_export_dir,
            freq=train_config.profile_freq,
            warmup=train_config.profile_warmup,
            active=train_config.profile_active,
            skip_first=train_config.profile_skip_first,
            with_flops=train_config.profile_with_flops,
            with_stack=train_config.profile_with_stack,
        )

    # Memory snapshots remain independently switchable, matching the previous
    # titaninfra contract and upstream's new per-tool ownership model.
    memory_snapshot = (
        MemorySnapshotProfiler.Config(freq=train_config.profile_freq)
        if train_config.profile_memory
        else None
    )
    return Profiler.Config(trace=trace, memory_snapshot=memory_snapshot)
