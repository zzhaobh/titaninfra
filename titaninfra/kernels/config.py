"""Validate operator names in ModelConfig.kernel_impls against the registry.

An empty list selects PyTorch reference implementations. This module is
stateless; the registry remains the source of truth for custom Triton, Helion,
and torch.compile implementations.
"""

from __future__ import annotations

from .registry import is_registered, list_kernels


def validate_enabled(enabled: list[str]) -> None:
    """Raise ValueError if any enabled operator name is not registered."""
    unknown = [n for n in enabled if not is_registered(n)]
    if unknown:
        raise ValueError(
            f"kernel_impls 含未注册算子 {unknown}。已注册：{list_kernels()}"
        )
