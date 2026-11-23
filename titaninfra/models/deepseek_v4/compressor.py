# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DeepSeek-V4 compressors with upstream math and legacy checkpoint names.

The registered parameter tree intentionally remains ``kv_proj``/``gate_proj``/
``kv_norm``/``position_bias``. Read-only properties expose the names expected by
upstream ``Compressor.forward`` without registering duplicate module paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder_sharding import (
    replicated_state_sharding_config,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v4.compressor import Compressor

from titaninfra.models.deepseek_v4.sharding import compressor_sharding_config


def _rope_cos_sin(
    rope_dim: int,
    positions: Tensor,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Half-sized interleaved-RoPE cos/sin used by the V4 HF oracle."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rope_dim, 2, dtype=torch.int64, device=device).float() / rope_dim)
    )
    inv_freq_expanded = inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1)
    positions_expanded = positions[:, None, :].float()
    freqs = (inv_freq_expanded @ positions_expanded).transpose(1, 2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    """V4 interleaved rotate-half: ``stack((-x[1::2], x[0::2])).flatten(-2)``."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_trailing(x: Tensor, cos: Tensor, sin: Tensor, rope_dim: int) -> Tensor:
    """Apply interleaved RoPE to the trailing channels of ``[B,H,T,D]`` input."""
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (_rotate_half_interleaved(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class _PlainCompressRoPE:
    """Stateless upstream-RoPE-shaped adapter; adds no module or buffer FQN."""

    def __init__(self, rope_dim: int, theta: float):
        self.rope_dim = rope_dim
        self.theta = theta

    def __call__(
        self,
        query: Tensor,
        key: Tensor | None = None,
        positions: Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> Tensor:
        if key is not None:
            raise ValueError("Plain compress RoPE only accepts one tensor at this boundary")
        if positions is None:
            raise ValueError("Plain compress RoPE requires positions")
        cos, sin = _rope_cos_sin(
            self.rope_dim,
            positions.unsqueeze(0),
            self.theta,
            query.device,
            query.dtype,
        )
        rotated = _apply_rope_trailing(
            query.transpose(0, 1).unsqueeze(0),
            cos,
            -sin if inverse else sin,
            self.rope_dim,
        )
        return rotated.squeeze(0).transpose(0, 1)


class _LegacyCompressor(Compressor):
    """Upstream compressor runtime over the production legacy parameter tree."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        kv_proj: Linear.Config
        gate_proj: Linear.Config
        kv_norm: RMSNorm.Config
        head_dim: int
        compress_rate: int
        compress_rope_theta: float
        max_seq_len: int
        rope_head_dim: int = 64
        preserve_dtype_params: tuple[str, ...] = ("position_bias",)

    def __init__(self, config: Config):
        nn.Module.__init__(self)
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.compress_rate = config.compress_rate
        self.compress_ratio = config.compress_rate
        self.compress_rope_theta = config.compress_rope_theta
        self.max_seq_len = config.max_seq_len
        self.overlap = config.compress_rate == 4
        self.rope = _PlainCompressRoPE(config.rope_head_dim, config.compress_rope_theta)
        self.kv_proj = config.kv_proj.build()
        self.gate_proj = config.gate_proj.build()
        self.kv_norm = config.kv_norm.build()
        self.position_bias = nn.Parameter(
            torch.empty(config.compress_rate, self.kv_proj.out_features)
        )

    @property
    def wkv(self):
        return self.kv_proj

    @property
    def wgate(self):
        return self.gate_proj

    @property
    def norm(self):
        return self.kv_norm

    @property
    def ape(self):
        return self.position_bias


class HCACompressor(_LegacyCompressor):
    @dataclass(kw_only=True, slots=True)
    class Config(_LegacyCompressor.Config):
        pass


class CSACompressor(_LegacyCompressor):
    @dataclass(kw_only=True, slots=True)
    class Config(_LegacyCompressor.Config):
        pass


def make_compressor_config(
    *,
    dim: int,
    head_dim: int,
    compress_rate: int,
    compress_rope_theta: float,
    max_seq_len: int,
    rope_head_dim: int,
    norm_eps: float,
    overlap: bool,
) -> HCACompressor.Config | CSACompressor.Config:
    """Build declarative sharding while preserving every legacy state name."""
    out_features = head_dim * (2 if overlap else 1)
    linear_init = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
    fields = {
        "kv_proj": Linear.Config(
            in_features=dim,
            out_features=out_features,
            bias=False,
            param_init=linear_init,
            inplace_wgrad_accum=False,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        "gate_proj": Linear.Config(
            in_features=dim,
            out_features=out_features,
            bias=False,
            param_init=linear_init,
            inplace_wgrad_accum=False,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        "kv_norm": RMSNorm.Config(
            normalized_shape=head_dim,
            eps=norm_eps,
            param_init={"weight": nn.init.ones_},
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        "head_dim": head_dim,
        "compress_rate": compress_rate,
        "compress_rope_theta": compress_rope_theta,
        "max_seq_len": max_seq_len,
        "rope_head_dim": rope_head_dim,
        "param_init": {"position_bias": nn.init.zeros_},
        "sharding_config": compressor_sharding_config(),
    }
    config_cls = CSACompressor.Config if overlap else HCACompressor.Config
    return config_cls(**fields)
