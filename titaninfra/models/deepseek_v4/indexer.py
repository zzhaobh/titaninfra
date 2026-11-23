# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Legacy-FQN indexer shell over upstream Indexer and InnerIndexer.

Upstream nests its mini-compressor below ``indexer.compressor``. TitanInfra's
checkpoint contract stores those tensors directly below ``indexer``. A stateless
view calls upstream ``Compressor.forward`` with this module as the state owner,
so no alias or duplicate registration path is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder_sharding import (
    replicated_state_sharding_config,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v4.compressor import (
    Compressor,
    Indexer,
    InnerIndexer,
)
from torchtitan.models.deepseek_v4.sharding import (
    inner_indexer_sharding_config,
)

from titaninfra.models.deepseek_v4.compressor import (
    _PlainCompressRoPE,
    make_compressor_config,
)
from titaninfra.models.deepseek_v4.sharding import indexer_sharding_config


class _IndexerCompressorView:
    """Dispatch upstream compression without registering a nested module."""

    def __init__(self, owner: LightningIndexer):
        self.owner = owner

    def __call__(self, *args, **kwargs):
        return Compressor.forward(self.owner, *args, **kwargs)


class LightningIndexer(Indexer, Compressor):
    """Upstream indexer runtime with the legacy flat mini-compressor states."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        kv_proj: Linear.Config
        gate_proj: Linear.Config
        kv_norm: RMSNorm.Config
        q_b_proj: Linear.Config
        weights_proj: Linear.Config
        inner_indexer: InnerIndexer.Config
        dim: int
        q_lora_rank: int
        compress_rate: int
        index_n_heads: int
        index_head_dim: int
        compress_rope_theta: float
        max_seq_len: int
        rope_head_dim: int
        preserve_dtype_params: tuple[str, ...] = ("position_bias",)

    def __init__(self, config: Config):
        nn.Module.__init__(self)
        self.dim = config.dim
        self.q_lora_rank = config.q_lora_rank
        self.compress_rate = config.compress_rate
        self.compress_ratio = config.compress_rate
        self.num_heads = config.index_n_heads
        self.num_index_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.inner_indexer.topk
        self.compress_rope_theta = config.compress_rope_theta
        self.max_seq_len = config.max_seq_len
        self.rope_head_dim = config.rope_head_dim
        self.softmax_scale = config.index_head_dim**-0.5
        self.weights_scaling = config.index_n_heads**-0.5
        self.overlap = True
        self.rope = _PlainCompressRoPE(
            config.rope_head_dim, config.compress_rope_theta
        )
        self.kv_proj = config.kv_proj.build()
        self.gate_proj = config.gate_proj.build()
        self.kv_norm = config.kv_norm.build()
        self.position_bias = nn.Parameter(
            torch.empty(config.compress_rate, 2 * config.index_head_dim)
        )
        self.q_b_proj = config.q_b_proj.build()
        self.weights_proj = config.weights_proj.build()
        self.inner_indexer = config.inner_indexer.build()
        self._compressor_view = _IndexerCompressorView(self)

    @property
    def wq_b(self):
        return self.q_b_proj

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

    @property
    def compressor(self):
        return self._compressor_view


def make_indexer_config(
    *,
    dim: int,
    q_lora_rank: int,
    compress_rate: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    compress_rope_theta: float,
    max_seq_len: int,
    rope_head_dim: int,
    norm_eps: float,
) -> LightningIndexer.Config:
    """Build upstream selection and legacy flat states with full sharding."""
    compressor = make_compressor_config(
        dim=dim,
        head_dim=index_head_dim,
        compress_rate=compress_rate,
        compress_rope_theta=compress_rope_theta,
        max_seq_len=max_seq_len,
        rope_head_dim=rope_head_dim,
        norm_eps=norm_eps,
        overlap=True,
    )
    linear_init = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
    return LightningIndexer.Config(
        kv_proj=compressor.kv_proj,
        gate_proj=compressor.gate_proj,
        kv_norm=compressor.kv_norm,
        q_b_proj=Linear.Config(
            in_features=q_lora_rank,
            out_features=index_n_heads * index_head_dim,
            bias=False,
            param_init=linear_init,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        weights_proj=Linear.Config(
            in_features=dim,
            out_features=index_n_heads,
            bias=False,
            param_init=linear_init,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        inner_indexer=InnerIndexer.Config(
            ratio=compress_rate,
            topk=index_topk,
            sharding_config=inner_indexer_sharding_config(),
        ),
        dim=dim,
        q_lora_rank=q_lora_rank,
        compress_rate=compress_rate,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        compress_rope_theta=compress_rope_theta,
        max_seq_len=max_seq_len,
        rope_head_dim=rope_head_dim,
        param_init={"position_bias": nn.init.zeros_},
        sharding_config=indexer_sharding_config(),
    )
