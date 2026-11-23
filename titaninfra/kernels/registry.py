"""Registry mapping operator names to reference and custom implementations.

Every KernelSpec contains a PyTorch reference used by default and an optional
Triton, Helion, or torch.compile implementation. get_kernel selects the custom
implementation only when explicitly enabled, available, and supported by the
current CUDA capability. This module has no TorchTitan dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    """Dual implementations and parity metadata for one operator.

    pytorch_ref is the always-available oracle and default path. custom_impl is
    optional; None means only the reference exists. rtol/atol define parity
    tolerances, and min_cuda_capability controls hardware fallback.
    """

    name: str
    pytorch_ref: Callable
    custom_impl: Callable | None = None
    rtol: float = 1e-2
    atol: float = 1e-2
    min_cuda_capability: tuple[int, int] = (9, 0)


_REGISTRY: dict[str, KernelSpec] = {}


def register_kernel(spec: KernelSpec) -> None:
    """Register a kernel once, rejecting duplicate names and silent overrides."""
    if spec.name in _REGISTRY:
        raise ValueError(f"kernel '{spec.name}' 已注册，不允许重复注册（防静默覆盖）")
    _REGISTRY[spec.name] = spec


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def list_kernels() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_spec(name: str) -> KernelSpec:
    if name not in _REGISTRY:
        raise KeyError(f"kernel '{name}' 未注册。已注册：{list_kernels()}")
    return _REGISTRY[name]


def _has_cuda_capability(major: int, minor: int) -> bool:
    """Return whether the current device meets the required CUDA capability."""
    import torch

    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap >= (major, minor)


def get_kernel(name: str, enabled: list[str] | None) -> Callable:
    """Resolve the implementation selected by ModelConfig.kernel_impls.

    Return the reference when the operator is not enabled, no custom
    implementation exists, or the device is below min_cuda_capability.
    Otherwise return custom_impl.
    """
    spec = get_spec(name)
    if not enabled or name not in enabled:
        return spec.pytorch_ref
    if spec.custom_impl is None:
        import logging

        logging.getLogger(__name__).warning(
            "kernel '%s' 请求 custom_impl 但未实现，回退 pytorch_ref", name
        )
        return spec.pytorch_ref
    if not _has_cuda_capability(*spec.min_cuda_capability):
        import logging

        logging.getLogger(__name__).warning(
            "kernel '%s' 请求 custom_impl 但算力 < sm%d%d，回退 pytorch_ref",
            name, spec.min_cuda_capability[0], spec.min_cuda_capability[1],
        )
        return spec.pytorch_ref
    return spec.custom_impl
