"""MoonViT vision tower aligned with HF modeling_kimi_vl.py.

This family-owned module uses FlexInnerAttention with block masks; temporal
pooling is not part of the current path.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from titaninfra.models.kimi_vl.rope import Rope2DPosEmb, apply_rope


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(self, height: int, width: int, dim: int, mode: str = "bicubic"):
        super().__init__()
        self.height, self.width, self.dim, self.mode = height, width, dim, mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for hw in grid_hws.tolist():
            h, w = int(hw[0]), int(hw[1])
            if (h, w) == (self.height, self.width):
                pos_embs.append(self.weight.reshape(h * w, self.dim))
            else:
                src = self.weight.permute(2, 0, 1).unsqueeze(0)  # (1,dim,H,W)
                interp = F.interpolate(src, size=(h, w), mode=self.mode, align_corners=False)
                pos_embs.append(interp.squeeze(0).permute(1, 2, 0).reshape(h * w, self.dim))
        return x + torch.cat(pos_embs, dim=0)


class PatchEmbed(nn.Module):
    def __init__(self, out_dim: int, in_dim: int = 3, patch_size: int = 14, pos_emb_hw: int = 14):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = Learnable2DInterpPosEmb(pos_emb_hw, pos_emb_hw, out_dim)

    def forward(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixel_values).view(pixel_values.size(0), -1)  # (L,out_dim)
        return self.pos_emb(x, grid_hws)


class VisionMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, bias: bool = True):
        super().__init__()
        self.fc0 = nn.Linear(hidden_dim, mlp_dim, bias=bias)
        self.fc1 = nn.Linear(mlp_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


def build_cu_seqlens(grid_hws: torch.Tensor) -> torch.Tensor:
    lengths = grid_hws[:, 0] * grid_hws[:, 1]
    zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    return torch.cat([zero, lengths]).cumsum(0).to(torch.int32)


def get_vision_block_mask_mod(cu_seqlens: torch.Tensor) -> Callable:
    """Build a non-causal block-diagonal mask for patches within each image."""
    boundaries = cu_seqlens[1:-1]

    def mask_mod(b, h, q_idx, kv_idx):
        q_doc = torch.searchsorted(boundaries, q_idx, right=True)
        kv_doc = torch.searchsorted(boundaries, kv_idx, right=True)
        return q_doc == kv_doc

    return mask_mod


def _sdpa_block_attn(xq, xk, xv, cu_seqlens):
    """Eager SDPA fallback with a non-causal boolean block-diagonal mask."""
    L = xq.shape[0]
    doc = torch.zeros(L, dtype=torch.long)
    for i in range(len(cu_seqlens) - 1):
        doc[cu_seqlens[i] : cu_seqlens[i + 1]] = i
    mask = doc[:, None] == doc[None, :]  # (L,L)
    q, k, v = (t.transpose(0, 1) for t in (xq, xk, xv))  # (H,L,d)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None])
    return out.transpose(0, 1)  # (L,H,d)


class VisionEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, attn_impl="flex", attn_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attn_impl = attn_impl
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
        self.mlp = VisionMLP(hidden_dim, mlp_dim)
        self._rope_impl = apply_rope  # swappable seam

    def _attn(self, x, freqs_cis, cu_seqlens, block_mask):
        L = x.shape[0]
        qkv = self.wqkv(x).view(L, 3, self.num_heads, self.head_dim)
        xq, xk, xv = qkv.unbind(dim=1)  # each (L,H,d)
        xq, xk = self._rope_impl(xq, xk, freqs_cis)
        if self.attn_impl == "flex":
            from torchtitan.models.common.attention import FlexInnerAttention
            # New API takes [L,H,d] directly and reshapes to [1,H,L,d] internally
            # (old API expected the caller to do that reshape); attention_metadata
            # replaces attention_masks as the kwarg name, and must be the raw
            # BlockMask (asserted inside forward()), not wrapped in anything.
            out = FlexInnerAttention(FlexInnerAttention.Config())(
                xq, xk, xv, attention_metadata=block_mask
            )
        else:
            out = _sdpa_block_attn(xq, xk, xv, cu_seqlens)
        return self.wo(out.reshape(L, -1))

    def forward(self, x, *, freqs_cis, cu_seqlens, block_mask=None):
        x = x + self._attn(self.norm0(x), freqs_cis, cu_seqlens, block_mask)
        x = x + self.mlp(self.norm1(x))
        return x



def patch_merger(x: torch.Tensor, grid_hws: torch.Tensor, merge_kernel_size=(2, 2)) -> list[torch.Tensor]:
    kh, kw = merge_kernel_size
    d = x.shape[-1]
    out, pre = [], 0
    for hw in grid_hws.tolist():
        h, w = int(hw[0]), int(hw[1])
        seq = x[pre : pre + h * w]
        pre += h * w
        new_h, new_w = h // kh, w // kw
        seq = seq.view(new_h, kh, new_w, kw, d).permute(0, 2, 1, 3, 4).contiguous()
        out.append(seq.view(new_h * new_w, kh * kw * d))
    return out


class PatchMerger(nn.Module):
    def __init__(self, merge_kernel_size=(2, 2)):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self._merge_impl = patch_merger  # swappable seam

    def forward(self, x, grid_hws):
        return self._merge_impl(x, grid_hws, self.merge_kernel_size)


class MoonViT(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, mlp_dim, patch_size=14,
                 merge_kernel_size=(2, 2), init_pos_emb_hw=14, attn_impl="flex"):
        super().__init__()
        self.patch_embed = PatchEmbed(hidden_dim, 3, patch_size, init_pos_emb_hw)
        self.rope_2d = Rope2DPosEmb(Rope2DPosEmb.Config(dim=hidden_dim // num_heads))
        self.blocks = nn.ModuleList([
            VisionEncoderLayer(hidden_dim, num_heads, mlp_dim, attn_impl=attn_impl)
            for _ in range(num_layers)
        ])
        self.final_layernorm = nn.LayerNorm(hidden_dim)
        self.merger = PatchMerger(merge_kernel_size)

    def forward(self, pixel_values, grid_hws):
        x = self.patch_embed(pixel_values, grid_hws)
        freqs_cis = self.rope_2d.get_freqs_cis(grid_hws, x.device)
        cu_seqlens = build_cu_seqlens(grid_hws)
        block_mask = None
        if self.blocks[0].attn_impl == "flex":
            from torch.nn.attention.flex_attention import create_block_mask
            L = x.shape[0]
            block_mask = torch.compile(create_block_mask)(
                get_vision_block_mask_mod(cu_seqlens), B=1, H=None, Q_LEN=L, KV_LEN=L
            )
        for blk in self.blocks:
            x = blk(x, freqs_cis=freqs_cis, cu_seqlens=cu_seqlens, block_mask=block_mask)
        x = self.final_layernorm(x)
        return self.merger(x, grid_hws)
