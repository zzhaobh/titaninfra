"""Optional Helion dependency guard, configuration buckets, and kernel cache.

This shared utility mirrors TorchTitan's optional-import and bound-kernel
patterns without registering operators or causing import side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static analyzers resolve helion; runtime availability is determined by the
    # guarded import below.
    import helion

    _HELION_IMPORT_ERROR: ImportError | None = None
else:
    try:
        import helion
        import helion.language as hl  # noqa: F401

        _HELION_IMPORT_ERROR = None
    except ImportError as e:
        _HELION_IMPORT_ERROR = e

HELION_AVAILABLE = _HELION_IMPORT_ERROR is None

# Cache key: (kernel id, config repr, shapes, strides, dtypes, device). Training
# uses a small number of layouts, so the cache stays small. Dictionary
# operations are atomic under the GIL for forward and autograd threads.
_BOUND_KERNELS: dict[tuple, Any] = {}


def run_tuned(kernel: Any, config: Any, *args: Any) -> Any:
    """Bind once per shape/config key and reuse the resulting kernel.

    Non-tensor arguments such as scalar eps values are keyed by value but do not
    contribute shape/stride/dtype fields.
    """
    tensor_args = [a for a in args if hasattr(a, "shape")]
    key = (
        id(kernel),
        repr(config),
        tuple(tuple(a.shape) for a in tensor_args),
        tuple(tuple(a.stride()) for a in tensor_args),
        tuple(a.dtype for a in tensor_args),
        tuple(a for a in args if not hasattr(a, "shape")),
        tensor_args[0].device,
    )
    bound = _BOUND_KERNELS.get(key)
    if bound is None:
        bound = kernel.bind(args)
        bound.set_config(config)
        _BOUND_KERNELS[key] = bound
    return bound(*args)


if HELION_AVAILABLE:

    def helion_config_by_tokens(num_tokens: int) -> helion.Config:
        """Return conservative pinned configurations that avoid autotuning."""
        if num_tokens <= 2048:
            return helion.Config(block_sizes=[1, 16, 8, 1], num_warps=4)
        if num_tokens <= 8192:
            return helion.Config(block_sizes=[1, 16, 4, 1], num_warps=2, num_stages=2)
        return helion.Config(block_sizes=[1, 8, 4, 1], num_warps=2, num_stages=2)

else:

    def helion_config_by_tokens(num_tokens: int) -> Any:
        raise ImportError(f"helion 未安装: {_HELION_IMPORT_ERROR}")
