# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.
#
# ============================================================================
# 2026-11 torchtitan upstream-tip migration (pilot family, goal items 4/5).
# ============================================================================
# Ported against torchtitan upstream tip (post #4913 "Reorganize training
# recipes" + #5277 repo reorg): ModelSpec/torchtitan.protocols dissolved;
# parallelize_fn/state_dict_adapter are no longer a separately-constructed
# bundle, they are now class-level attributes/methods on the model itself
# (BaseModel.parallelize() is inherited; state_dict_adapter_cls is a
# ClassVar). See docs/torchtitan_upgrade_plan_2026-11.md Sec 12 for the full
# migration write-up and per-family unit-of-work sizing this pilot is based
# on. Model-architecture semantics (attention/FFN/residual math) are
# unchanged from the pre-migration titaninfra llama3 implementation; only
# the upstream base-class plumbing this file participates in has moved.

from dataclasses import dataclass, field

import torch
import torch_remat as remat
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention import (
    AttentionMetadata,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.feed_forward import DenseFeedForward

from .state_dict_adapter import Llama3StateDictAdapter


class Llama3TransformerBlock(TransformerBlock):
    """
    Llama3 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Llama3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert isinstance(config.feed_forward, DenseFeedForward.Config)
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: AttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        # llama3 has no MoE (aux_loss_denominator is MoE-router-only) and no
        # padding-aware dispatch to thread padding_mask into (dense FFN) --
        # both new upstream TransformerBlock.forward kwargs are accept-and-
        # drop here, same pattern already used pre-migration for
        # padding_mask (see the W2/W3 cross-family padding_mask audit in
        # docs/torchtitan_upgrade_plan_2026-11.md Sec 8).
        del aux_loss_denominator, padding_mask
        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # torch_remat: the residual add reads attn_out with bare ops, so it
        # must survive to backward even if attention's internals are
        # selectively dropped by an enclosing AC region.
        remat.recompute_needs_tensor(attn_out)
        h = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(h))
        # Trailing add, always saved: costs nothing extra for backward, so
        # recompute replay can skip it and its inputs don't need pinning --
        # matches upstream's own llama3/qwen3/etc. convention for this exact
        # residual-add site (see torchtitan's TransformerBlock subclasses).
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(h, ffn_out)


class Llama3Model(Decoder):
    """
    Llama3Model Module

    Args:
        config (Llama3Model.Config): Model configuration.
    """

    state_dict_adapter_cls = Llama3StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 128256
        # 2026-11: llama3's Config never declared this field (inherited the
        # empty-list default from Decoder.Config/BaseModel.Config), leaving
        # every @local_compile-registered function this family actually
        # reaches running eager. Matches upstream llama3/model.py:83-86
        # exactly -- "loss" (torchtitan/components/loss.py, every family),
        # "fused_binary_activation" (common/activation.py, DenseFeedForward's
        # gate*up elementwise product), "rope" (common/rope.py's
        # ComplexRoPE.forward -- this family's actual RoPE class, see
        # __init__.py's ComplexRoPE.Config usage).
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=["loss", "fused_binary_activation", "rope"]
            )
        )
