"""Opt-in Triton, Helion, and torch.compile kernels for TitanInfra.

Public helpers resolve implementations by ModelConfig.kernel_impls, register
KernelSpec metadata, validate enabled names, and check parity against PyTorch
references. Importing this package registers the operators under ops/ as a side
effect; optional dependency failures leave the affected operators unavailable
without breaking the base import.
"""

from __future__ import annotations

# Register operators only after registry symbols are available. Operator modules
# tolerate missing optional dependencies and report them through registry state.
from . import ops as _ops  # noqa: F401,E402  (registration side effect)
from .config import validate_enabled
from .parity import assert_parity
from .registry import (
    KernelSpec,
    get_kernel,
    get_spec,
    is_registered,
    list_kernels,
    register_kernel,
)

__all__ = [
    "KernelSpec",
    "get_kernel",
    "get_spec",
    "is_registered",
    "list_kernels",
    "register_kernel",
    "validate_enabled",
    "assert_parity",
]
