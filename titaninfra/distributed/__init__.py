"""Distributed-training helpers, including MoE-safe per-submodule compilation.

Activation checkpointing uses TorchTitan's native FullAC, SelectiveAC, and
MemoryBudgetAC policies; TitanInfra does not maintain a separate AC policy.
"""

from .compile import apply_compile_moe_safe

__all__ = ["apply_compile_moe_safe"]
