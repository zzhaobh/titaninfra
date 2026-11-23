"""Manifold-Constrained Hyper-Connection (mHC) mixing math for DeepSeek V4.

Ported from the DeepSeek-V4 reference inference/model.py (hc_split_sinkhorn +
Block.hc_pre/hc_post), reshaped for batched training (no KV cache). All math
runs in float32; callers cast back to the activation dtype.
"""
from __future__ import annotations

import torch


def HC_MIX_WIDTH(hc_mult: int) -> int:
    return (2 + hc_mult) * hc_mult


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the mix projection into (pre, post, comb).

    mixes: [..., (2 + hc_mult) * hc_mult]  (already RMS-scaled by caller)
    hc_scale: [3]  gates for (pre, post, comb); hc_base: [(2 + hc_mult) * hc_mult]
    Returns pre [..., hc_mult], post [..., hc_mult], comb [..., hc_mult, hc_mult].
    """
    expected_width = HC_MIX_WIDTH(hc_mult)
    if mixes.shape[-1] != expected_width:
        raise ValueError(
            f"mixes last dim must be HC_MIX_WIDTH(hc_mult={hc_mult})={expected_width}, "
            f"got {mixes.shape[-1]}"
        )
    if hc_scale.numel() != 3:
        raise ValueError(
            f"hc_scale must have 3 elements (pre, post, comb gates), got {hc_scale.numel()}"
        )
    m = mixes.float()
    pre_raw, post_raw, comb_raw = torch.split(
        m, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    base_pre, base_post, base_comb = torch.split(
        hc_base.float(), [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.sigmoid(pre_raw * hc_scale[0] + base_pre) + eps
    # 2x sigmoid (NOT +eps) — confirmed against the real DeepSeek-V4-Flash
    # reference kernel.py (hc_split_sinkhorn_kernel_, line 394:
    # `post[i,j] = 2 * sigmoid(...)`) and upstream torchtitan's own
    # HcSplitSinkhorn.forward(), both of which agree and disagree with the
    # `sigmoid(x) + eps` formula this file previously had — an exactly 2x
    # numerical error (verified: ratio == 2.0 elementwise, not rounding
    # noise), never caught because no test here compares against the real
    # reference numerically (see docs/torchtitan_upgrade_plan_2026-11.md §34.2).
    post = 2 * torch.sigmoid(post_raw * hc_scale[1] + base_post)
    comb = (comb_raw * hc_scale[2] + base_comb).view(
        *comb_raw.shape[:-1], hc_mult, hc_mult
    )
    comb = torch.exp(comb - comb.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Fused, torch.compile-friendly HC ops (pure functions — no ``self``).
#
# These are the optimization seam (see titaninfra/kernels/): they carry the full
# hc_pre / hc_post / hc_head math as pure functions so the kernel registry can
# wrap them with ``torch.compile`` (fusing the launch-bound RMS→linear→Sinkhorn
# →weighted-sum chain — ~45 tiny kernels/call collapse toward 1). They are NOT
# ``autograd.Function`` on purpose: that would sever compile fusion (the lesson
# from the abandoned rmsnorm kernel, see titaninfra/kernels/wiring.py). Autograd is
# left to the tracer, so the compiled graph owns fwd+bwd.
#
# The block/model call these directly; the eager path IS the pytorch_ref, the
# compiled wrapper IS the fused_impl — parity is exact by construction (same
# function, compiled vs eager), so assert_parity guards compile correctness.
# ---------------------------------------------------------------------------


def hc_pre_fn(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce ``hc_mult`` copies to one. x: [B,L,hc,d] -> ([B,L,d], post, comb).

    Pure-function form of ``DeepSeekV4TransformerBlock.hc_pre`` (reference
    ``Block.hc_pre``). All math fp32; output ``y`` cast back to ``x.dtype``.
    """
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    # hc_fn is fp32-authored but FSDP2 mixed precision casts params to the bf16
    # compute dtype; the HC math is all-fp32, so cast the weight to match x.
    mixes = torch.nn.functional.linear(x, hc_fn.float()) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb


def hc_post_fn(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Expand one stream back to ``hc_mult`` copies.

    x: [B,L,d], residual: [B,L,hc,d], post: [B,L,hc], comb: [B,L,hc,hc]
    -> [B,L,hc,d]. Pure-function form of ``DeepSeekV4TransformerBlock.hc_post``.
    """
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def hc_head_fn(
    x: torch.Tensor,
    head_fn: torch.Tensor,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Collapse ``hc_mult`` copies to one (no Sinkhorn). x: [B,L,hc,d] -> [B,L,d].

    Pure-function form of ``DeepSeekV4Model._hc_head`` (reference
    ``ParallelHead.hc_head``): RMS-scaled sigmoid-gated weighted sum.
    """
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    # head_fn cast to fp32 to match x under FSDP2 mixed precision (see hc_pre_fn).
    mixes = torch.nn.functional.linear(x, head_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * head_scale + head_base) + eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype)
