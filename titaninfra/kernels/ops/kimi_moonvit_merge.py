"""Opt-in MoonViT patch merge: eager reference or torch.compile implementation.

This memory-bound reshape is disabled by default; profiling determines whether
the custom implementation provides a useful speedup.
"""
from __future__ import annotations

import logging

import torch

from titaninfra.kernels.registry import KernelSpec, register_kernel

logger = logging.getLogger(__name__)

try:
    from titaninfra.models.kimi_vl.vision_encoder import patch_merger
except Exception as e:
    patch_merger = None
    logger.info("moonvit_patch_merge 不可用(%s: %s),跳过注册", type(e).__name__, e)


def _merge_compiled(x, grid_hws, merge_kernel_size=(2, 2)):
    return patch_merger(x, grid_hws, merge_kernel_size)


if patch_merger is not None:
    _custom = torch.compile(_merge_compiled, dynamic=True) if torch.cuda.is_available() else None
    register_kernel(
        KernelSpec(
            name="moonvit_patch_merge",
            pytorch_ref=patch_merger,
            custom_impl=_custom,
            rtol=0.0, atol=0.0,  # pure reshape → bit-identical
            min_cuda_capability=(7, 0),
        )
    )
