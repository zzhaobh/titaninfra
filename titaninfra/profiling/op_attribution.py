"""Non-invasive per-operator attribution for the profiler.

Registers forward hooks that wrap selected submodules (MLA attention, MoE
router / experts / shared experts) in ``torch.profiler.record_function`` regions
named ``refmoe/<region>``. Those regions then surface as named rows in the
profiler trace and ``key_averages().table()`` summary, so CUDA time rolls up per
operator without touching ``models/deepseek_v3/model.py`` (which is under a
no-refactor rule).

Why hooks instead of editing the model:
    - models/*/model.py inherits from torchtitan and must not be structurally
      refactored (two production checkpoint-compat incidents).
    - forward hooks are torch-native and survive across families.

Interaction with ``compile=True`` (per-submodule compile, MoE excluded):
    - ``attention`` is compiled → its *inner* children (wq_b, inner_attention…)
      are traced away, but the hook on ``attention`` itself fires at the compiled
      boundary → attention shows up as ONE ``refmoe/attn`` region. Split it by
      kernel name (flex_attention vs GEMM) from the trace.
    - ``moe`` is NOT compiled → hooks on router / experts / shared_experts fire
      normally → fine-grained MoE attribution (where the EP AllToAll lives).

Usage:
    handles = install_op_region_hooks(trainer.engine.model_parts)
    ...                       # run training under torch.profiler
    remove_op_region_hooks(handles)   # optional teardown
"""

import logging

logger = logging.getLogger(__name__)

# Suffix of a module's qualified name (from ``named_modules()``) → region label.
# Order matters: longer/more-specific suffixes are matched first.
_REGION_BY_SUFFIX: tuple[tuple[str, str], ...] = (
    (".moe.shared_experts", "moe.shared"),
    (".moe.experts", "moe.experts"),
    (".moe.router", "moe.router"),
    (".attention", "attn"),
)

_NORM_REGION_BY_SUFFIX: tuple[tuple[str, str], ...] = (
    (".attention_norm", "attn_norm"),
    (".ffn_norm", "ffn_norm"),
)


def _region_for(name: str, suffixes: tuple[tuple[str, str], ...]) -> str | None:
    for suffix, region in suffixes:
        if name.endswith(suffix):
            return region
    return None


def _make_hooks(region: str):
    """Build (pre, post) hooks that bracket a module's forward in a
    record_function region. A per-call stack handles AC recompute re-entry
    (forward runs again in backward as an independent enter/exit pair).
    """
    import torch

    ctx_stack: list = []

    def _pre_hook(_module, _args):
        ctx = torch.profiler.record_function(f"refmoe/{region}")
        ctx.__enter__()
        ctx_stack.append(ctx)

    def _post_hook(_module, _args, _output):
        if ctx_stack:
            ctx_stack.pop().__exit__(None, None, None)

    return _pre_hook, _post_hook


def install_op_region_hooks(model_parts, *, include_norms: bool = False) -> list:
    """Wrap attention / MoE submodules in named profiler regions.

    Args:
        model_parts: list of nn.Module (``trainer.engine.model_parts``).
        include_norms: also tag attention_norm / ffn_norm regions.

    Returns:
        List of hook handles (call ``.remove()`` or pass to
        ``remove_op_region_hooks`` to tear down). Empty if torch is unavailable.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.warning("torch unavailable; op-region hooks not installed.")
        return []

    suffixes = _REGION_BY_SUFFIX + (_NORM_REGION_BY_SUFFIX if include_norms else ())

    handles: list = []
    region_counts: dict[str, int] = {}
    for part in model_parts:
        for name, module in part.named_modules():
            region = _region_for(name, suffixes)
            if region is None:
                continue
            pre_hook, post_hook = _make_hooks(region)
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))
            region_counts[region] = region_counts.get(region, 0) + 1

    if region_counts:
        summary = ", ".join(f"{r}×{c}" for r, c in sorted(region_counts.items()))
        logger.info(f"Installed op-region hooks: {summary}")
    else:
        logger.warning(
            "No op-region hooks matched — model has no attention/moe submodules "
            "with expected names. Attribution will be empty."
        )
    return handles


def remove_op_region_hooks(handles: list) -> None:
    """Remove all hooks returned by ``install_op_region_hooks``."""
    for handle in handles:
        handle.remove()
