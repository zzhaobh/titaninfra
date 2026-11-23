# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 hybrid-attention multimodal model.

Shape suffixes: T = packed tokens, D = model dimension, H = attention heads,
K = query/key head dimension, V = value head dimension, C = projection
channels, R = rotary dimension, P = non-rotary dimension."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn
from torchtitan.distributed.local_compile import LocalCompileConfig, local_compile
from torchtitan.models.base.module import Module
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    Attention,
    AttentionMetadata,
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
    local_head_split,
)
from torchtitan.models.common.attention.attention import InnerAttention
from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.multimodal import (
    MultimodalModel,
    get_vision_positions,
    scatter_vision_embeds,
)

from titaninfra.models._tt_compat import get_moe_model_nparams_and_flops

from .rope import MRoPE
from .state_dict_adapter import Qwen35StateDictAdapter
from .vision_encoder import Qwen35VisionEncoder


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 norm using rsqrt(sum(x²) + eps), not x/max(norm, eps) like F.normalize, to match FLA kernel."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_native_gated_delta(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    v_THV: torch.Tensor,
    g_TH: torch.Tensor,
    beta_TH: torch.Tensor,
) -> torch.Tensor:
    """Standalone math reference for the gated delta rule recurrence.

    Sequential O(seqlen) loop over a flat token stream (no batch dim) — use
    FLA/Attention-Gym kernels for GPU efficiency; this is the CPU-safe
    fallback. Not varlen-aware (no document resets): fine for CPU
    construction/forward-pass verification, which never exercises
    multi-document packing with this backend.

    Args:
        q_THK, k_THK: (T, H, key_head_dim), already head-expanded to match v
        v_THV: (T, H, value_head_dim)
        g_TH: (T, H) — log-space decay, always negative
        beta_TH: (T, H) — update gate ∈ (0, 1)
    """
    T, H, key_head_dim = q_THK.shape
    value_head_dim = v_THV.shape[-1]
    dtype = q_THK.dtype
    device = q_THK.device

    # Upcast to float32 — recurrence accumulates over T steps
    q_THK = _l2norm(q_THK.float(), dim=-1) * (key_head_dim**-0.5)
    k_THK = _l2norm(k_THK.float(), dim=-1)
    v_THV, g_TH, beta_TH = v_THV.float(), g_TH.float(), beta_TH.float()

    output_THV = torch.zeros(T, H, value_head_dim, dtype=torch.float32, device=device)
    state_HKV = torch.zeros(H, key_head_dim, value_head_dim, dtype=torch.float32, device=device)

    for t in range(T):
        g_t = g_TH[t].exp().unsqueeze(-1).unsqueeze(-1)
        b_t = beta_TH[t].unsqueeze(-1)
        state_HKV = state_HKV * g_t
        kv_mem_HV = torch.einsum("hkv,hk->hv", state_HKV, k_THK[t])
        delta_HV = (v_THV[t] - kv_mem_HV) * b_t
        state_HKV = state_HKV + torch.einsum("hk,hv->hkv", k_THK[t], delta_HV)
        output_THV[t] = torch.einsum("hkv,hk->hv", state_HKV, q_THK[t])

    return output_THV.to(dtype)


class GatedDeltaKernel(InnerAttention):
    """Dense Gated DeltaNet computation behind the vLLM replacement boundary.

    CPU-safe stand-in for upstream's ``InnerGDN``/``GDNKernel``, which hard-require
    CUDA (Attention Gym's ``l2norm``/``chunk_gdn`` reject CPU tensors — see
    ``docs/torchtitan_upgrade_plan_2026-11.md`` §28.2/§30 for the full investigation).
    Plugs into the real upstream ``GDN``/``GDN.Config`` unmodified as its
    ``inner_attention`` field, so the generic base-class state-dict helpers
    (``_split_gdn_in_proj``/``_merge_gdn_in_proj``, keyed on ``GDN.Config`` via
    ``model_config.traverse``) still find and correctly pack/unpack the fused
    ``in_proj`` — only the kernel dispatch is titaninfra-owned, not the whole
    attention wrapper.

    ``mixed_qkv_TC`` holds ``[q | k | v]`` columns, grouped by key head (see
    ``GDN.Config.in_proj_split_sizes``); the causal conv runs on each block
    separately, same math as ``InnerGDN`` just via plain ``F.conv1d`` instead
    of the CUDA-only ``causal_conv1d`` Attention-Gym kernel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        # "fla_chunked"/"fla_fused_recurrent": GPU, via FLA kernels (not yet wired --
        # the CPU-only verification bar for this migration only requires
        # "torch_native" to work; GPU backends are a follow-up).
        # "torch_native": pure-Python reference, CPU-safe, used for CPU tests.
        backend: Literal["torch_native"] = "torch_native"
        conv_kernel_size: int = 4

    def __init__(self, config: Config):
        super().__init__()
        self.backend = config.backend
        self.conv_kernel_size = config.conv_kernel_size

    def _causal_conv(self, x_TC: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Depthwise causal conv with left-padding (no per-document reset)."""
        x_1CT = F.pad(x_TC.unsqueeze(0).transpose(1, 2), [self.conv_kernel_size - 1, 0])
        out_1CT = F.conv1d(x_1CT, weight, None, groups=weight.shape[0])
        return F.silu(out_1CT).transpose(1, 2).squeeze(0)

    def forward(
        self,
        mixed_qkv_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        key_dim = conv_q_weight_C1W.shape[0]
        value_dim = conv_v_weight_C1W.shape[0]
        q_TC, k_TC, v_TC = mixed_qkv_TC.split([key_dim, key_dim, value_dim], dim=-1)
        q_TC = self._causal_conv(q_TC, conv_q_weight_C1W)
        k_TC = self._causal_conv(k_TC, conv_k_weight_C1W)
        v_TC = self._causal_conv(v_TC, conv_v_weight_C1W)

        num_tokens = mixed_qkv_TC.shape[0]
        n_key_heads = key_dim // key_head_dim
        n_value_heads = value_dim // value_head_dim
        q_THK = q_TC.view(num_tokens, n_key_heads, key_head_dim)
        k_THK = k_TC.view(num_tokens, n_key_heads, key_head_dim)
        v_THV = v_TC.view(num_tokens, n_value_heads, value_head_dim)

        if n_key_heads != n_value_heads:
            # in_proj groups rows by key head (see GDN.Config.in_proj_split_sizes),
            # so after the TC->THK view each key head's block is followed by its
            # `repeat` value-head group in consecutive order: repeat_interleave
            # duplicates each key head's data `repeat` times contiguously,
            # aligning it with its corresponding value-head group.
            repeat = n_value_heads // n_key_heads
            q_THK = q_THK.repeat_interleave(repeat, dim=1)
            k_THK = k_THK.repeat_interleave(repeat, dim=1)

        # Gating signals, shape (T, H):
        #   g:    decay rate per head, always negative
        #   beta: update gate in (0, 1)
        g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
        beta_TH = torch.sigmoid(b_TH)

        if self.backend == "torch_native":
            return _torch_native_gated_delta(q_THK, k_THK, v_THV, g_TH, beta_TH)
        raise ValueError(
            f"Unknown GatedDeltaKernel backend '{self.backend}'. Valid: 'torch_native'."
        )


class OffsetRMSNorm(Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``.

    Weight is zero-initialized so the norm starts as identity-scaled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    @local_compile("offset_rmsnorm", batch_invariant=False, recompile_limit=16)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = input.dtype
        input = input.float()
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * input).to(input_dtype)


class Qwen35Attention(Attention):
    """Full attention with output gating and partial RoPE for Qwen3.5.

    Differences from GQAttention:
    - wq is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses OffsetRMSNorm

    Uses separate ``wq``/``wk``/``wv`` instead of the common fused ``qkv_linear``
    (so this subclasses ``Attention``, not ``GQAttention``): the 2x-wide,
    gated ``wq`` doesn't fit a fused QKV projection that TP-shards by head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        n_kv_heads: int
        head_dim: int
        rotary_dim: int
        rope: MRoPE.Config
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config

        def flops_per_token(self, seq_len: int) -> int:
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            return quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                seq_len=seq_len,
                sliding_window_size=None,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()

        self.rope = config.rope.build()

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.scaling = self.head_dim**-0.5

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_TD = maybe_gather_tp_input(self, x_TD)
        num_tokens = x_TD.shape[0]

        # wq is 2x wider: produces query + gate
        xq_gate_THC = local_head_split(self.wq(x_TD), self.head_dim * 2, cp_shard_dim=0)
        xq_THK, gate_THV = xq_gate_THC.chunk(2, dim=-1)
        xk_THK = local_head_split(self.wk(x_TD), self.head_dim, cp_shard_dim=0)
        xv_THV = local_head_split(self.wv(x_TD), self.head_dim, cp_shard_dim=0)

        # QK norm (before RoPE). The norms read the wq and wk projection
        # outputs with bare ops.
        remat.recompute_needs_tensor(xq_THK, xk_THK)
        xq_THK = self.q_norm(xq_THK)
        xk_THK = self.k_norm(xk_THK)

        xq_THK, xk_THK = self._partial_rope(xq_THK, xk_THK, positions)

        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            xq_THK,
            xk_THK,
            xv_THV,
            attention_metadata=attention_metadata,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        # The output gating reads the inner_attention and wq gate outputs with
        # bare ops.
        remat.recompute_needs_tensor(out_THV, gate_THV)
        out_THV = out_THV.contiguous()

        # Output gating
        out_THV = out_THV * torch.sigmoid(gate_THV)
        out_TD = out_THV.view(num_tokens, -1)
        return self.wo(out_TD)

    @local_compile("partial_rope", batch_invariant=True)
    def _partial_rope(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to the first ``rotary_dim`` channels of each head; keep the rest."""
        assert self.rotary_dim <= self.head_dim
        xq_THR, xq_THP = xq_THK[..., : self.rotary_dim], xq_THK[..., self.rotary_dim :]
        xk_THR, xk_THP = xk_THK[..., : self.rotary_dim], xk_THK[..., self.rotary_dim :]
        xq_THR, xk_THR = self.rope(xq_THR, xk_THR, positions)
        xq_THK = torch.cat([xq_THR, xq_THP], dim=-1)
        xk_THK = torch.cat([xk_THR, xk_THP], dim=-1)
        return xq_THK, xk_THK


class Qwen35TransformerBlock(TransformerBlock):
    """Hybrid transformer block for Qwen3.5.

    Each layer uses either full attention (Qwen35Attention) or linear
    attention (GDN), determined by the attention config type. Both types
    share the same FFN/MoE structure.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.full_attn = isinstance(config.attention, Qwen35Attention.Config)
        self.attention = config.attention.build()

        self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: AttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_TD = self.attention_norm(x_TD)
        if self.full_attn:
            h_TD = self.attention(h_TD, attention_metadata, positions)
        else:
            h_TD = self.attention(h_TD, attention_metadata)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(h_TD)
        x_TD = x_TD + h_TD

        h_TD = self.ffn_norm(x_TD)
        if isinstance(self.feed_forward, MoE):
            h_TD = self.feed_forward(
                h_TD,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            h_TD = self.feed_forward(h_TD)
        # Trailing add, always saved: it saves nothing for backward, so replay
        # skips it and its inputs need no persisting, matching checkpoint
        # early stop.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x_TD, h_TD)


class Qwen35Model(Decoder):
    """Qwen3.5 language model with hybrid attention.

    Key architectural features:
    - Hybrid attention: 75% GatedDeltaNet (linear, upstream's real ``GDN``
      with a titaninfra-owned CPU-safe kernel) + 25% full attention
    - Output gating on full attention: ``attn_out * sigmoid(gate)``
    - Partial RoPE: only first ``rotary_dim`` elements get positional encoding
    - OffsetRMSNorm: ``(1 + weight) * norm(x)`` with zero-init weight
    - MoE variant: routed experts + shared expert with sigmoid gate

    ``Qwen35MultimodalModel`` adds the vision encoder.
    """

    state_dict_adapter_cls = Qwen35StateDictAdapter

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
        # pyrefly: ignore [bad-override]
        layers: list[Qwen35TransformerBlock.Config]
        # Compile only region names registered by actual @local_compile
        # decorators. Include all shared-module boundaries plus partial_rope;
        # an incomplete list silently leaves work in eager mode.
        #
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "fused_binary_activation",
                    "gated_rmsnorm",
                    "gdn_split_in_proj",
                    "offset_rmsnorm",
                    "partial_rope",
                    "shared_expert_gate",
                    "fp32_to_bf16_split",
                ]
            )
        )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attn_cfg = self.first_attention
            return get_moe_model_nparams_and_flops(
                self,
                model,
                attn_cfg.n_heads,
                2 * attn_cfg.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)


class Qwen35MultimodalModel(Qwen35Model, MultimodalModel):
    """Qwen35 multimodal model configuration and behavior."""

    multimodal_encoder_fqns = ("vision_encoder",)

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config, MultimodalModel.Config):
        vision_encoder: Qwen35VisionEncoder.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the vision encoder; return packed embeddings with per-item token counts."""
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds_VD = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit
        return vision_embeds_VD, num_tokens_per_item

    def _prepare_multimodal_embeds(
        self,
        tokens_T: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        """Embed tokens, run the vision encoder, scatter vision into the text stream.

        Args:
            tokens_T: Input token IDs, flat packed token stream ``(T,)``.
        """
        inputs_embeds_TD = (
            self.tok_embeddings(tokens_T) if self.tok_embeddings is not None else tokens_T
        )

        if pixel_values is not None and grid_thw is not None:
            assert special_tokens is not None and "image_id" in special_tokens
            vision_embeds_VD, num_tokens_per_item = self.get_vision_embeds(
                pixel_values, grid_thw=grid_thw
            )
            image_positions = get_vision_positions(
                tokens_T, num_tokens_per_item, special_tokens["image_id"]
            )
            inputs_embeds_TD = scatter_vision_embeds(
                inputs_embeds_TD,
                vision_embeds=vision_embeds_VD,
                vision_positions=image_positions,
            )

        if pixel_values_videos is not None and grid_thw_videos is not None:
            assert special_tokens is not None and "video_id" in special_tokens
            vision_embeds_VD, num_tokens_per_item = self.get_vision_embeds(
                pixel_values_videos, grid_thw=grid_thw_videos
            )
            video_positions = get_vision_positions(
                tokens_T, num_tokens_per_item, special_tokens["video_id"]
            )
            inputs_embeds_TD = scatter_vision_embeds(
                inputs_embeds_TD,
                vision_embeds=vision_embeds_VD,
                vision_positions=video_positions,
            )

        return inputs_embeds_TD

    def preprocess_inputs(
        self,
        input_dict: dict,
        *,
        parallelism_context,
        parallelism,
        max_num_documents: int | None = None,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs,
    ):
        """Preprocess VLM inputs with metadata, positions, and SPMD declarations.

Fold mrope_positions into positions, declare sharding for
pixel_values/grid_thw/mrope_positions, then delegate to the base annotator.
CP+VLM sharding is intentionally rejected until that exact combination has
been validated; CP=1 follows the supported path."""
        from torchtitan.distributed.spmd_types import annotate_input_spmd_types
        from torchtitan.models.common.decoder_sharding import decoder_input_sharding
        from torchtitan.models.common.vision_encoder_sharding import (
            multimodal_input_sharding,
        )

        loss_kwargs = input_dict.pop("loss_kwargs", {})
        padding_mask = input_dict.get("padding_mask", None)

        positions = input_dict.get("positions")
        if positions is not None:
            input_dict["attention_metadata"] = self._get_attention_metadata(
                positions=positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
            )

        if parallelism_context.cp_enabled and self.tok_embeddings is not None:
            if input_dict.get("pixel_values") is not None or input_dict.get(
                "pixel_values_videos"
            ) is not None:
                raise NotImplementedError(
                    "Qwen35MultimodalModel.preprocess_inputs: CP (context "
                    "parallel) + 真实图像输入（pixel_values/pixel_values_videos）"
                    "组合尚未移植/验证（需要 vision_bank_indices 的 CP 版本，"
                    "titaninfra 没有真实 GPU case 覆盖这个组合，拒绝静默跑出"
                    "可能错误的结果）。"
                )

        input_shardings = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(),
        }

        mrope_positions = input_dict.pop("mrope_positions", None)
        if mrope_positions is None:
            rope_positions = positions
        else:
            rope_positions = mrope_positions
            # MRoPE positions fold to (tokens, 3): replicate the three
            # components on the last dimension instead of 1D-position layout.
            import spmd_types as spmd
            from spmd_types import SpmdType
            from torchtitan.distributed.parallelism_context import MeshAxisName

            input_shardings["positions"] = SpmdType(
                {
                    MeshAxisName.DP: spmd.V,
                    MeshAxisName.CP: spmd.V,
                    MeshAxisName.TP: spmd.R,
                },
                partition_spec=spmd.PartitionSpec(
                    (MeshAxisName.DP, MeshAxisName.CP), None
                ),
            )
        assert rope_positions is not None, (
            "Qwen3.5 needs RoPE positions: the batch must provide "
            "'positions' or 'mrope_positions'."
        )
        input_dict["positions"] = rope_positions

        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)
        if parallelism_context.cp_enabled:
            input_dict = self._cp_shard(
                input_dict,
                input_shardings=input_shardings,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
        input_dict = annotate_input_spmd_types(
            parallelism_context, input_dict, input_shardings
        )

        loss_kwargs = {name: input_dict.pop(name) for name in loss_kwargs}
        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        if any(
            isinstance(layer.feed_forward, MoE.Config) for layer in self.config.layers
        ):
            input_dict["aux_loss_denominators"] = global_routing_token_counts
        return inputs, labels, input_dict, loss_kwargs

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_metadata: AttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        mrope_positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        if self.tok_embeddings is not None:
            x_TD = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                grid_thw=grid_thw,
                grid_thw_videos=grid_thw_videos,
                special_tokens=special_tokens,
            )
        else:
            x_TD = tokens

        # 3D MRoPE positions for multimodal batches, else plain 1D text positions.
        rope_positions = mrope_positions if mrope_positions is not None else positions
        assert rope_positions is not None, (
            "Qwen3.5 needs RoPE positions: pass 'positions' or 'mrope_positions'."
        )
        return self.forward_embeddings(
            x_TD,
            rope_positions,
            attention_metadata,
            padding_mask=padding_mask,
            aux_loss_denominators=aux_loss_denominators,
        )
