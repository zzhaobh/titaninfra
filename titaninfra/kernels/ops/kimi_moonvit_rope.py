"""Opt-in Helion forward/backward kernels for MoonViT 2D RoPE.

The interface matches apply_rope and splits complex values into real and
imaginary components. DTensor inputs fail loudly. If Helion is unavailable,
custom_impl remains None and the registry falls back to the eager reference.
"""
from __future__ import annotations

import logging

import torch

from titaninfra.kernels.ops._helion import HELION_AVAILABLE, run_tuned
from titaninfra.kernels.registry import KernelSpec, register_kernel

logger = logging.getLogger(__name__)

try:
    from titaninfra.models.kimi_vl.rope import apply_rope
except Exception as e:
    apply_rope = None
    logger.info("moonvit_rope 不可用(%s: %s),跳过注册", type(e).__name__, e)


if HELION_AVAILABLE:
    import helion
    import helion.language as hl

    def _rope_config(num_tokens: int):
        """Pin config for the moonvit_rope kernels' 3-tiled-dim grid.

        The grid is ``hl.tile([L, H])`` + inner ``hl.tile(half)`` = 3 tiled
        dims, so block_sizes must have 3 entries. The shared
        ``helion_config_by_tokens`` emits 4 entries and raises
        ``InvalidConfig`` here. Conservative buckets to avoid runtime
        autotune; a GPU node may tune finer.
        """
        if num_tokens <= 2048:
            return helion.Config(block_sizes=[1, 16, 8], num_warps=4)
        if num_tokens <= 8192:
            return helion.Config(block_sizes=[1, 16, 4], num_warps=2, num_stages=2)
        return helion.Config(block_sizes=[1, 8, 4], num_warps=2, num_stages=2)

    @helion.kernel(static_shapes=True)
    def _rope_fwd(x_real, x_imag, cis_real, cis_imag):
        """Compute (a+bi)(c+di)=(ac-bd)+(ad+bc)i over tiled inputs."""
        L, H, half = x_real.size()
        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)
        for tile_l, th in hl.tile([L, H]):
            for tk in hl.tile(half):
                a = x_real[tile_l, th, tk].to(torch.float32)
                b = x_imag[tile_l, th, tk].to(torch.float32)
                c = cis_real[tile_l, None, tk].to(torch.float32)
                dd = cis_imag[tile_l, None, tk].to(torch.float32)
                out_real[tile_l, th, tk] = (a * c - b * dd).to(x_real.dtype)
                out_imag[tile_l, th, tk] = (a * dd + b * c).to(x_imag.dtype)
        return out_real, out_imag

    @helion.kernel(static_shapes=True)
    def _rope_bwd(go_real, go_imag, cis_real, cis_imag):
        """Apply the conjugate rotation during the backward pass."""
        L, H, half = go_real.size()
        gx_real = torch.empty_like(go_real)
        gx_imag = torch.empty_like(go_imag)
        for tile_l, th in hl.tile([L, H]):
            for tk in hl.tile(half):
                gr = go_real[tile_l, th, tk].to(torch.float32)
                gi = go_imag[tile_l, th, tk].to(torch.float32)
                c = cis_real[tile_l, None, tk].to(torch.float32)
                dd = cis_imag[tile_l, None, tk].to(torch.float32)
                gx_real[tile_l, th, tk] = (gr * c + gi * dd).to(go_real.dtype)
                gx_imag[tile_l, th, tk] = (-gr * dd + gi * c).to(go_imag.dtype)
        return gx_real, gx_imag

    def _split_complex(x):
        xr = x.reshape(*x.shape[:-1], -1, 2)
        return xr[..., 0].contiguous(), xr[..., 1].contiguous()

    def _merge_complex(xr, xi):
        return torch.stack([xr, xi], dim=-1).flatten(-2)

    class _RopeFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xq, xk, cis):  # noqa: ANN001
            cr, ci = cis.real.contiguous(), cis.imag.contiguous()
            qr, qi = _split_complex(xq)
            kr, ki = _split_complex(xk)
            cfg = _rope_config(xq.shape[0])
            oqr, oqi = run_tuned(_rope_fwd, cfg, qr, qi, cr, ci)
            okr, oki = run_tuned(_rope_fwd, cfg, kr, ki, cr, ci)
            ctx.save_for_backward(cr, ci)
            ctx.token = xq.shape[0]
            return _merge_complex(oqr, oqi).type_as(xq), _merge_complex(okr, oki).type_as(xk)

        @staticmethod
        def backward(ctx, gq, gk):  # noqa: ANN001
            cr, ci = ctx.saved_tensors
            gqr, gqi = _split_complex(gq)
            gkr, gki = _split_complex(gk)
            cfg = _rope_config(ctx.token)
            dqr, dqi = run_tuned(_rope_bwd, cfg, gqr, gqi, cr, ci)
            dkr, dki = run_tuned(_rope_bwd, cfg, gkr, gki, cr, ci)
            return _merge_complex(dqr, dqi).type_as(gq), _merge_complex(dkr, dki).type_as(gk), None

    def moonvit_rope_helion(xq, xk, freqs_cis):
        """Match apply_rope and reject DTensor inputs outside the DP-only path."""
        from torch.distributed.tensor import DTensor
        if isinstance(xq, DTensor) or isinstance(xk, DTensor):
            raise AssertionError(
                "moonvit_rope Helion kernel is dp-only: got DTensor. Disable 'moonvit_rope' "
                "in kernel_impls under TP/CP so eager apply_rope (DTensor-safe) runs."
            )
        return _RopeFn.apply(xq, xk, freqs_cis)
else:
    moonvit_rope_helion = None


if apply_rope is not None:
    register_kernel(
        KernelSpec(
            name="moonvit_rope",
            pytorch_ref=apply_rope,
            custom_impl=moonvit_rope_helion if HELION_AVAILABLE else None,
            rtol=1e-3, atol=1e-3,
            min_cuda_capability=(9, 0),
        )
    )
