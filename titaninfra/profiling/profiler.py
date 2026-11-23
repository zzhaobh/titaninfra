"""Extended torch.profiler integration for titaninfra.

Wraps torchtitan's Profiler to add with_flops/with_stack support and
rank-0 text summary export. The profiler lifecycle is managed by
torchtitan's trainer.train() — titaninfra passes the config and patches
the build method to produce our extended profiler.

Integration:
    1. build_profiler_config(train_config) → Profiler.Config
    2. Pass to TorchTitanTrainer.Config(profiler=...)
    3. patch_profiler(trainer, train_config) → monkey-patches config.profiler.build()
"""

import logging
import os
import time

logger = logging.getLogger(__name__)


def build_profiler_config(train_config):
    """Translate titaninfra TrainConfig profiler fields → torchtitan Profiler.Config."""
    from torchtitan.observability.profiler import Profiler

    return Profiler.Config(
        enable_profiling=train_config.enable_profiling,
        save_traces_folder=train_config.profile_export_dir,
        profile_freq=train_config.profile_freq,
        profiler_warmup=train_config.profile_warmup,
        profiler_active=train_config.profile_active,
        profiler_skip_first=train_config.profile_skip_first,
        enable_memory_snapshot=train_config.profile_memory,
    )


def patch_profiler(trainer, train_config) -> None:
    """Patch ``Profiler.build_torch_profiler`` to add with_flops/with_stack
    support and a rank-0 text summary.

    Upstream torchtitan builds the profiler via ``config.profiler.build()``
    (Configurable factory) which returns a ``Profiler`` instance; that
    instance's ``__enter__`` calls ``self.build_torch_profiler(...)`` to create
    the live ``torch.profiler.profile``. We cannot assign to
    ``config.profiler.build`` — ``Profiler.Config`` is a ``slots=True``
    dataclass, so its ``build`` attribute is read-only. Instead we replace the
    ``Profiler.build_torch_profiler`` *method on the class*, which is the real
    hook ``__enter__`` invokes.

    Must be called AFTER TorchTitanTrainer construction, BEFORE trainer.train().
    """
    if not train_config.enable_profiling:
        return

    with_flops = train_config.profile_with_flops
    with_stack = train_config.profile_with_stack

    if not with_flops and not with_stack:
        return

    from torchtitan.observability.profiler import Profiler

    extended = _make_extended_build(with_flops, with_stack)

    def build_torch_profiler(self, *, global_step, base_folder, leaf_folder):
        return extended(
            self, global_step=global_step, base_folder=base_folder,
            leaf_folder=leaf_folder,
        )

    Profiler.build_torch_profiler = build_torch_profiler


def _make_extended_build(with_flops: bool, with_stack: bool):
    """Create a replacement Profiler.build_torch_profiler with extended options.

    Returns a function taking ``(profiler_instance, *, global_step, base_folder,
    leaf_folder)`` — ``profiler_instance`` is the bound ``self`` supplied by the
    class-method wrapper in :func:`patch_profiler`.
    """

    def build_torch_profiler(profiler_instance, *, global_step, base_folder, leaf_folder):
        import torch

        cfg = profiler_instance._config
        if not cfg.enable_profiling:
            return None

        trace_dir = os.path.join(base_folder, cfg.save_traces_folder)
        profile_freq = cfg.profile_freq
        warmup = cfg.profiler_warmup
        active = cfg.profiler_active

        additional_params = {
            key: val
            for key, val in [
                ("repeat", cfg.profiler_repeat),
                ("skip_first", cfg.profiler_skip_first),
                ("skip_first_wait", cfg.profiler_skip_first_wait),
            ]
            if val is not None
        }

        rank = torch.distributed.get_rank()
        post_processor = (
            cfg.trace_post_processor.build() if cfg.trace_post_processor else None
        )

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(
                trace_dir, curr_trace_dir_name, leaf_folder
            )
            os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            output_file = os.path.join(curr_trace_dir, f"rank{rank}_trace.json")
            prof.export_chrome_trace(output_file)

            if rank == 0:
                summary_path = os.path.join(curr_trace_dir, "summary_rank0.txt")
                with open(summary_path, "w") as f:
                    f.write(
                        prof.key_averages().table(
                            sort_by="cuda_time_total", row_limit=30
                        )
                    )

            if post_processor is not None:
                post_processor(output_file)

            logger.info(
                f"Finished dumping profiler traces in "
                f"{time.monotonic() - begin:.2f}s"
            )

        logger.info(
            f"Profiling active (with_flops={with_flops}, "
            f"with_stack={with_stack}). Traces → {trace_dir}"
        )
        os.makedirs(trace_dir, exist_ok=True)

        wait = profile_freq - (active + warmup)
        assert wait >= 0, (
            f"profile_freq ({profile_freq}) must be >= warmup ({warmup}) + "
            f"active ({active})"
        )

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif torch.xpu.is_available():
            activities.append(torch.profiler.ProfilerActivity.XPU)

        torch_profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, **additional_params
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_flops=with_flops,
            with_stack=with_stack,
        )
        torch_profiler.__enter__()
        torch_profiler.step_num = global_step
        return torch_profiler

    return build_torch_profiler
