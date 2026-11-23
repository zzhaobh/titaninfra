"""MoonViT 2D RoPE with an eager reference and _rope_impl seam.

Matches HF Rope2DPosEmb and apply_rope: torch.polar builds complex frequencies
for the x/y axes, while view_as_complex/view_as_real applies the rotation.
The embedding dimension must be divisible by four.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.models.base.module import Module


def precompute_freqs_cis(dim: int, max_h: int, max_w: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim % 4 == 0, "dim must be divisible by 4 (2D-RoPE x/y each dim//2, complex halves)"
    n = max_h * max_w
    flat = torch.arange(n)
    x_pos = (flat % max_w).float()
    y_pos = (flat // max_w).float()
    dim_range = torch.arange(0, dim, 4)[: dim // 4].float()
    freqs = 1.0 / (theta ** (dim_range / dim))
    x_cis = torch.polar(torch.ones(n, dim // 4), torch.outer(x_pos, freqs))
    y_cis = torch.polar(torch.ones(n, dim // 4), torch.outer(y_pos, freqs))
    # Interleave x/y per band (x0,y0,x1,y1,...) to match HF Rope2DPosEmb oracle
    # (height axis -> ret[...,2*i], width axis -> ret[...,2*i+1]). NOT block-concat.
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)  # (n, dim//4, 2)
    return freqs_cis.reshape(max_h, max_w, dim // 2)


def get_freqs_cis(freqs_cis: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
    dim_half = freqs_cis.shape[-1]
    out = []
    for hw in grid_hws.tolist():
        h, w = int(hw[0]), int(hw[1])
        out.append(freqs_cis[:h, :w].reshape(-1, dim_half))
    return torch.cat(out, dim=0)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply RoPE to xq/xk shaped (L, H, dim) using complex64 frequencies."""
    cis = freqs_cis[:, None, :]  # (L,1,dim//2)
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    oq = torch.view_as_real(xq_c * cis).flatten(-2)
    ok = torch.view_as_real(xk_c * cis).flatten(-2)
    return oq.type_as(xq), ok.type_as(xk)


class Rope2DPosEmb(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        max_height: int = 512
        max_width: int = 512
        theta: float = 10000.0

    def __init__(self, config: Rope2DPosEmb.Config):
        super().__init__()
        self.dim = config.dim
        self.max_height = config.max_height
        self.max_width = config.max_width
        self.theta = config.theta
        self._freqs_cis: torch.Tensor | None = None

    def get_freqs_cis(self, grid_hws: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self._freqs_cis is None or self._freqs_cis.device != device:
            self._freqs_cis = precompute_freqs_cis(
                self.dim, self.max_height, self.max_width, self.theta
            ).to(device)
        return get_freqs_cis(self._freqs_cis, grid_hws)
