"""Register the existing CSA gather Triton kernel without rewiring attention.

DeepSeekV4Attention dispatches sparse_attn_gather through use_gather_kernel.
Registry metadata supplies parity tolerances, compute-capability gating, and
visibility in list_kernels().
"""

from __future__ import annotations

import logging

from titaninfra.kernels.registry import KernelSpec, register_kernel

logger = logging.getLogger(__name__)

try:
    from titaninfra.models.deepseek_v4.sparse_attn import sparse_attn_gather as _gather
except Exception as e:  # Optional Triton/TorchTitan dependencies are unavailable.
    _gather = None
    logger.info("csa_gather custom_impl 不可用(%s),登记为 ref-only", type(e).__name__)


def _gather_ref(*args, **kwargs):
    raise NotImplementedError(
        "csa_gather 的 ref 是 DeepSeekV4Attention._compressed_attention(dense 分支);"
        "gather 由 use_gather_kernel flag 派发,不经 registry callable 调用。"
    )


register_kernel(
    KernelSpec(
        name="csa_gather",
        pytorch_ref=_gather_ref,
        custom_impl=_gather,
        rtol=2e-3,
        atol=2e-3,
        min_cuda_capability=(9, 0),
    )
)
