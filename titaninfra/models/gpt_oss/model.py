# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch_remat as remat
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention import (
    Attention as BaseAttention,
)
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    InnerAttention,
    QKVLinear,
    VarlenInnerAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.rope import RoPE

from .state_dict_adapter import GptOssStateDictAdapter


def apply_attention_sink_rescale(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """Rescale attention output by the learned per-head sink term."""
    sinks = sinks.view(*([1] * (lse.ndim - 1)), -1)
    sink_scale = torch.sigmoid(lse - sinks).unsqueeze(-1)
    return out * sink_scale.to(out.dtype)


class Attention(BaseAttention):
    """
    Multi-head attention (MLA) module with sink attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int = 64
        n_kv_heads: int = 8
        head_dim: int = 64
        dim: int
        qkv_linear: QKVLinear.Config
        wo: Linear.Config  # output projection
        inner_attention: InnerAttention.Config = field(default_factory=VarlenInnerAttention.Config)
        sliding_window_size: int | None = None
        """Per-layer causal sliding-window size"""
        rope: RoPE.Config

        def flops_per_token(self, seq_len: int) -> int:
            """Estimate attention FLOPs per token for MFU reporting.

            Match upstream Attention.Config semantics: attended-token counts
            are bounded by sliding_window_size when configured and use global
            dense attention otherwise. Training initializes this estimator
            unconditionally, so it must exist even when MFU is not inspected.
            """
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            return quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                seq_len=seq_len,
                sliding_window_size=self.sliding_window_size,
            )

        def flops(self, positions: torch.Tensor) -> int:
            """Position-aware FLOPs estimator for the Decoder API."""
            from torchtitan.models.common.attention.attention import (
                SlidingWindowFlexInnerAttention,
                SlidingWindowVarlenInnerAttention,
            )
            from torchtitan.models.common.flops import quadratic_attention_flops

            num_keys = positions + 1
            if isinstance(
                self.inner_attention,
                (
                    SlidingWindowFlexInnerAttention.Config,
                    SlidingWindowVarlenInnerAttention.Config,
                ),
            ):
                num_keys = num_keys.clamp(max=self.inner_attention.sliding_window_size)
            elif self.sliding_window_size is not None:
                num_keys = num_keys.clamp(max=self.sliding_window_size)
            return quadratic_attention_flops(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                num_keys=num_keys,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard attention softmax scale (1/sqrt(head_dim))
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        if config.rope.scaling == "yarn" and config.rope.rope_factor > 1.0:
            mscale = 0.1 * math.log(config.rope.rope_factor) + 1.0
            # Merge YaRN attention mscale into softmax_scale, with
            # m**2 being equivalent to scaling q / k each by mscale.
            self.softmax_scale *= mscale * mscale

        self.qkv_linear = config.qkv_linear.build()
        self.wo = config.wo.build()
        self.sinks = nn.Parameter(torch.empty(config.n_heads))
        self.inner_attention = config.inner_attention.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: AttentionMetadataMap,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x: Input tensor with shape [T, D] (flat token stream, no batch dim).
            attention_metadata: metadata selected for this layer's inner attention
                (FlexAttentionMetadata or VarlenAttentionMetadata).
            positions: Optional position indices (unused, for API compatibility).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        q, k, v = self.qkv_linear(x)

        q, k = self.rope(q, k, positions)

        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q,
            k,
            v,
            attention_metadata=attention_metadata,
            scale=self.softmax_scale,
            enable_gqa=self.enable_gqa,
            out_transform=self._apply_sinks,
        )

        # The reshape below copies the inner_attention output with bare ops.
        remat.recompute_needs_tensor(output)
        # Reshape and project output
        output = output.reshape(output.shape[0], -1).contiguous()
        return self.wo(output)

    def _apply_sinks(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        """out_transform hook: rescale attention output by this layer's sinks."""
        sinks = self.sinks
        if isinstance(sinks, DTensor):
            sinks = sinks.to_local(grad_placements=sinks.placements)
        return apply_attention_sink_rescale(out, lse, sinks)


class GptOssTransformerBlock(TransformerBlock):
    """
    GptOss Transformer block with sliding window attention support.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        assert isinstance(config.attention, Attention.Config)
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        assert isinstance(config.feed_forward, MoE.Config)
        self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: AttentionMetadataMap | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (num_tokens, dim).
            attention_metadata: metadata selected for this layer's inner attention.
                Sliding-window layers carry a SlidingWindowFlexInnerAttention/
                SlidingWindowVarlenInnerAttention inner_attention whose own
                build_attention_metadata already bakes in the window — this
                block no longer picks a mask by dict key (the upstream
                get_attention_masks()/attn_mask_key dispatch this file used to
                do is gone; Decoder._get_attention_metadata() now handles that
                per-layer dispatch generically for every family).
            positions: Optional position indices.
            padding_mask: Boolean (T,) padding mask threaded into the
                always-on MoE's token dispatcher (see moe.forward).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        x = x + attn_out
        moe_out = self.feed_forward(
            self.ffn_norm(x),
            padding_mask_T=padding_mask,
            aux_loss_denominator=aux_loss_denominator,
        )
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x, moe_out)
        return x


class GptOssModel(Decoder):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    state_dict_adapter_cls = GptOssStateDictAdapter

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2880
        vocab_size: int = 201088
        # 2026-11: never declared (inherited the empty-list default from
        # Decoder.Config/BaseModel.Config). Copied verbatim from upstream
        # gpt_oss/model.py:231-240: "loss" (every family), "fused_binary_
        # activation" (common/activation.py, this family's MoE gate*up),
        # "rope" (common/rope.py's RoPE.forward -- this family's actual RoPE
        # class, see __init__.py's RoPE.Config usage), "fp32_to_bf16_split"
        # (HiMidLoLinear, used by this family's gate Linear, see
        # __init__.py's HiMidLoLinear.Config usage).
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "fused_binary_activation",
                    "rope",
                    "fp32_to_bf16_split",
                ]
            )
        )

    def __init__(self, config: Config):
        super().__init__(config)
