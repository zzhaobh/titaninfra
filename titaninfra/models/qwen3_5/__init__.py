# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Literal

import spmd_types as spmd
import torch.nn as nn
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import (
    ColumnParallelLinear,
    Conv1d,
    GatedRMSNorm,
    Linear,
    RowParallelLinear,
    SiLU,
    Softmax,
    VocabParallelEmbedding,
)  # noqa: F401
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.decoder_sharding import (
    head_norm_sharding_config,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    replicated_input_colwise_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe.sharding import shared_experts_sharding_config
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.param_init import depth_scaled_std  # noqa: F401
from torchtitan.models.common.rope import CosSinVisionRoPE
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_sharding_config,
    vision_attention_sharding_config,
    vision_colwise_sharding_config,
    vision_inner_attention_sharding_config,
    vision_invariant_linear_sharding_config,
    vision_rope_sharding_config,
    vision_rowwise_sharding_config,
)

from .model import (
    GatedDeltaKernel,
    OffsetRMSNorm,
    Qwen35Attention,
    Qwen35Model,
    Qwen35MultimodalModel,
    Qwen35TransformerBlock,
)
from .moe import SigmoidGatedFeedForward
from .rope import MRoPE
from .sharding import (
    attention_sharding_config,
    conv_weight_sharding_config,
    decoder_norm_sharding_config,
    gated_delta_net_sharding_config,
    gated_rms_norm_sharding_config,
    inner_gated_delta_net_sharding_config,
    sigmoid_gate_sharding_config,
    sigmoid_gated_w13_sharding_config,
    transformer_block_sharding_config,
    vision_encoder_sharding_config,
    vision_merger_sharding_config,
)
from .state_dict_adapter import Qwen35StateDictAdapter
from .vision_encoder import PatchMerger, Qwen35VisionEncoder

__all__ = [
    "Qwen35Model",
    "Qwen35MultimodalModel",
    "Qwen35StateDictAdapter",
    "qwen3_5_configs",
    "QWEN3_5_SPECIAL_TOKENS",
]

QWEN3_5_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_OFFSET_NORM_INIT = {"weight": nn.init.zeros_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}

_EPS = 1e-6


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _a_log_init(param: nn.Parameter) -> None:
    param.data.uniform_(1e-6, 16.0).log_()


def _linear(
    in_features: int, out_features: int, *, sharding_config
) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
        sharding_config=sharding_config,
    )


def _vision_column_parallel_linear(
    in_features: int, out_features: int
) -> ColumnParallelLinear.Config:
    return ColumnParallelLinear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
        sharding_config=vision_colwise_sharding_config(),
    )


def _vision_row_parallel_linear(
    in_features: int, out_features: int
) -> RowParallelLinear.Config:
    return RowParallelLinear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
        sharding_config=vision_rowwise_sharding_config(),
    )


def _offset_norm(dim: int, *, sharding_config) -> OffsetRMSNorm.Config:
    return OffsetRMSNorm.Config(
        dim=dim, eps=_EPS, param_init=_OFFSET_NORM_INIT, sharding_config=sharding_config
    )


def _shared_experts_config(
    *, dim: int, hidden_dim: int, layer_id: int, enable_sp: bool
) -> SigmoidGatedFeedForward.Config:
    """Build Qwen3.5's sigmoid-gated shared-expert config (SwiGLU FFN + gate)."""
    from torchtitan.models.common.config_utils import (
        fused_gate_up_param_init,
        make_shared_expert_down_proj_config,
    )

    depth_init = _depth_init(layer_id)
    return SigmoidGatedFeedForward.Config(
        # The shared expert gathers once because w13 and the sigmoid gate
        # consume the same input.
        w13=Linear.Config(
            in_features=dim,
            out_features=hidden_dim,
            num_linears=2,
            param_init=fused_gate_up_param_init(_LINEAR_INIT, _LINEAR_INIT),
            sharding_config=sigmoid_gated_w13_sharding_config(),
        ),
        w2=make_shared_expert_down_proj_config(
            in_features=hidden_dim,
            out_features=dim,
            param_init=depth_init,
            enable_sp=enable_sp,
        ),
        gate=Linear.Config(
            in_features=dim,
            out_features=1,
            param_init=_LINEAR_INIT,
            sharding_config=sigmoid_gate_sharding_config(),
        ),
        sharding_config=shared_experts_sharding_config(enable_sp=enable_sp),
    )


def _qwen35_vision_encoder_config(
    *,
    dim: int,
    ffn_dim: int,
    num_layers: int,
    num_heads: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    out_hidden_size: int,
    num_position_embeddings: int,
    layer_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    # The default 16M-pixel budget with aspect ratio < 200 gives <= 3,620 patches per side.
    max_rope_positions: int = 4096,
    in_channels: int = 3,
) -> Qwen35VisionEncoder.Config:
    """Build a fully-specified Qwen35VisionEncoder.Config."""
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    merged_hidden_size = dim * (spatial_merge_size**2)
    head_dim = dim // num_heads
    _norm = LayerNorm.Config(
        normalized_shape=dim,
        eps=layer_norm_eps,
        sharding_config=invariant_norm_sharding_config(),
    )
    return Qwen35VisionEncoder.Config(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        spatial_merge_size=spatial_merge_size,
        num_position_embeddings=num_position_embeddings,
        patch_embed_proj=_linear(
            patch_dim, dim, sharding_config=vision_invariant_linear_sharding_config()
        ),
        block=VisionTransformerBlock.Config(
            norm1=_norm,
            norm2=_norm,
            attn=VisionAttention.Config(
                dim=dim,
                num_heads=num_heads,
                rope=CosSinVisionRoPE.Config(
                    dim=head_dim,
                    max_context_length=max_rope_positions,
                    theta=rope_theta,
                    sharding_config=vision_rope_sharding_config(),
                ),
                wq=_linear(
                    dim, dim, sharding_config=vision_colwise_sharding_config(input_tp=spmd.R)
                ),
                wk=_linear(
                    dim, dim, sharding_config=vision_colwise_sharding_config(input_tp=spmd.R)
                ),
                wv=_linear(
                    dim, dim, sharding_config=vision_colwise_sharding_config(input_tp=spmd.R)
                ),
                proj=_vision_row_parallel_linear(dim, dim),
                inner_attention=FlexInnerAttention.Config(
                    sharding_config=vision_inner_attention_sharding_config()
                ),
                sharding_config=vision_attention_sharding_config(positions_dp=spmd.V),
            ),
            mlp=VisionMLP.Config(
                fc1=_vision_column_parallel_linear(dim, ffn_dim),
                fc2=_vision_row_parallel_linear(ffn_dim, dim),
            ),
        ),
        merger=PatchMerger.Config(
            spatial_merge_size=spatial_merge_size,
            merged_hidden_size=merged_hidden_size,
            norm=LayerNorm.Config(
                normalized_shape=dim,
                eps=layer_norm_eps,
                sharding_config=invariant_norm_sharding_config(),
            ),
            fc1=_vision_column_parallel_linear(merged_hidden_size, merged_hidden_size),
            fc2=_vision_row_parallel_linear(merged_hidden_size, out_hidden_size),
            sharding_config=vision_merger_sharding_config(),
        ),
        param_init=_POS_EMBED_INIT,
        sharding_config=vision_encoder_sharding_config(),
    )


def _qwen35_attention_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: MRoPE.Config,
    attn_backend: str,
    layer_id: int,
    enable_sp: bool,
) -> Qwen35Attention.Config:
    """Build a fully-specified, sharded Qwen35Attention.Config."""
    import dataclasses

    inner_attention = dataclasses.replace(
        get_attention_config(attn_backend),
        sharding_config=inner_attention_local_spmd_sharding_config(),
    )
    return Qwen35Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        # The per-layer rope cache buffer is replicated on TP.
        rope=dataclasses.replace(rope, sharding_config=rope_cache_sharding_config()),
        wq=Linear.Config(
            in_features=dim,
            out_features=n_heads * head_dim * 2,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wk=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wv=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        # RowParallelLinear reduce-scatters with SP and all-reduces otherwise.
        wo=RowParallelLinear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        q_norm=_offset_norm(head_dim, sharding_config=head_norm_sharding_config()),
        k_norm=_offset_norm(head_dim, sharding_config=head_norm_sharding_config()),
        inner_attention=inner_attention,
        sharding_config=attention_sharding_config(enable_sp=enable_sp),
    )


def _qwen35_deltanet_config(
    *,
    dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    layer_id: int,
    enable_sp: bool,
    conv_kernel_size: int = 4,
    fla_backend: Literal[
        "fla_chunked", "fla_fused_recurrent", "torch_native"
    ] = "fla_chunked",
) -> GDN.Config:
    """Build a fully-specified, head-sharded GDN.Config (upstream's real class --
    see model.py's GatedDeltaKernel docstring for why: this makes the generic
    base-class state_dict_adapter in_proj packing automatic via
    model_config.traverse(GDN.Config), instead of reimplementing the
    per-key-head-interleaved pack/unpack logic by hand).

    ``fla_backend`` is accepted for call-site compatibility with the old
    signature but only 'torch_native' (CPU-safe) is wired so far --
    upstream's FLA/Attention-Gym GPU kernels are a follow-up, not needed
    for this migration's CPU verification bar.
    """
    key_dim = n_key_heads * key_head_dim
    value_dim = n_value_heads * value_head_dim
    # All six logical projections' rows, grouped by key head (see
    # GDN.Config.in_proj_split_sizes): q|k|v|z|a|b, repeated per key head.
    heads_per_key = n_value_heads // n_key_heads
    in_proj_dim = n_key_heads * (
        2 * key_head_dim + 2 * heads_per_key * value_head_dim + 2 * heads_per_key
    )

    def _proj(in_f: int, out_f: int, init: dict) -> Linear.Config:
        return Linear.Config(
            in_features=in_f,
            out_features=out_f,
            bias=False,
            param_init=init,
            sharding_config=replicated_input_colwise_sharding_config(),
        )

    def _conv(channels: int) -> Conv1d.Config:
        # Depthwise causal conv (groups == channels). Causal left-padding is
        # applied in the forward, so padding=0 here.
        return Conv1d.Config(
            in_channels=channels,
            out_channels=channels,
            kernel_size=conv_kernel_size,
            groups=channels,
            padding=0,
            bias=False,
            sharding_config=conv_weight_sharding_config(),
        )

    return GDN.Config(
        n_heads=n_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        n_key_heads=n_key_heads,
        conv_kernel_size=conv_kernel_size,
        in_proj=_proj(dim, in_proj_dim, _LINEAR_INIT),
        conv_q=_conv(key_dim),
        conv_k=_conv(key_dim),
        conv_v=_conv(value_dim),
        # CPU-safe stand-in for upstream's InnerGDN/GDNKernel (CUDA-only
        # Attention-Gym kernels) -- see GatedDeltaKernel's docstring in
        # model.py. Same forward contract (mixed_qkv_TC/a_TH/b_TH/
        # conv_*_weight_C1W/A_log_H/dt_bias_H), so the upstream sharding
        # declaration applies unchanged.
        inner_attention=GatedDeltaKernel.Config(
            backend="torch_native",
            sharding_config=inner_gated_delta_net_sharding_config(),
        ),
        norm=GatedRMSNorm.Config(
            dim=value_head_dim,
            eps=1e-6,
            activation_fn=SiLU.Config(),
            param_init={"weight": nn.init.ones_},
            sharding_config=gated_rms_norm_sharding_config(),
        ),
        # RowParallelLinear reduce-scatters with SP and all-reduces otherwise.
        out_proj=RowParallelLinear.Config(
            in_features=value_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        param_init={
            "A_log": _a_log_init,
            "dt_bias": nn.init.ones_,
        },
        sharding_config=gated_delta_net_sharding_config(enable_sp=enable_sp),
    )


def _build_qwen35_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: MRoPE.Config,
    hidden_dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    enable_sp: bool,
    full_attention_interval: int = 4,
    attn_backend: str,
    fla_backend: Literal[
        "fla_chunked", "fla_fused_recurrent", "torch_native"
    ] = "fla_chunked",
) -> list[Qwen35TransformerBlock.Config]:
    """Build per-layer configs for dense Qwen3.5 models."""
    layers = []
    for layer_id in range(n_layers):
        is_full = (layer_id + 1) % full_attention_interval == 0

        attention = (
            _qwen35_attention_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rope=rope,
                attn_backend=attn_backend,
                layer_id=layer_id,
                enable_sp=enable_sp,
            )
            if is_full
            else None
        )
        deltanet = (
            _qwen35_deltanet_config(
                dim=dim,
                n_key_heads=n_key_heads,
                n_value_heads=n_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                layer_id=layer_id,
                enable_sp=enable_sp,
                fla_backend=fla_backend,
            )
            if not is_full
            else None
        )

        layers.append(
            Qwen35TransformerBlock.Config(
                attention=attention if is_full else deltanet,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w13_param_init=_LINEAR_INIT,
                    w2_param_init=_depth_init(layer_id),
                    enable_sp=enable_sp,
                ),
                attention_norm=_offset_norm(
                    dim, sharding_config=decoder_norm_sharding_config(enable_sp=enable_sp)
                ),
                ffn_norm=_offset_norm(
                    dim, sharding_config=decoder_norm_sharding_config(enable_sp=enable_sp)
                ),
                sharding_config=transformer_block_sharding_config(enable_sp=enable_sp),
            )
        )
    return layers


def _qwen35_routed_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    param_init: dict,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    enable_ep: bool,
):
    """make_routed_experts_config + optional non-standard token_dispatcher
    swap -- make_routed_experts_config itself no longer takes comm_backend/
    non_blocking_capacity_factor (confirmed via config_utils.py); matches
    deepseek_v3's already-migrated pattern for this exact situation."""
    import dataclasses

    from titaninfra.models._tt_compat import make_token_dispatcher_config

    routed_experts_cfg = make_routed_experts_config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        param_init=param_init,
        enable_ep=enable_ep,
    )
    if moe_comm_backend != "standard":
        routed_experts_cfg = dataclasses.replace(
            routed_experts_cfg,
            token_dispatcher=make_token_dispatcher_config(
                num_experts=num_experts,
                top_k=top_k,
                comm_backend=moe_comm_backend,
                hidden_dim=dim,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
            ),
        )
    return routed_experts_cfg


def _build_qwen35_moe_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: MRoPE.Config,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    shared_expert_hidden_dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    enable_sp: bool,
    enable_ep: bool,
    full_attention_interval: int = 4,
    attn_backend: str,
    fla_backend: Literal[
        "fla_chunked", "fla_fused_recurrent", "torch_native"
    ] = "fla_chunked",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
) -> list[Qwen35TransformerBlock.Config]:
    """Build per-layer configs for MoE Qwen3.5 models with shared expert."""
    layers = []
    for layer_id in range(n_layers):
        is_full = (layer_id + 1) % full_attention_interval == 0

        attention = (
            _qwen35_attention_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rope=rope,
                attn_backend=attn_backend,
                layer_id=layer_id,
                enable_sp=enable_sp,
            )
            if is_full
            else None
        )
        deltanet = (
            _qwen35_deltanet_config(
                dim=dim,
                n_key_heads=n_key_heads,
                n_value_heads=n_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                layer_id=layer_id,
                enable_sp=enable_sp,
                fla_backend=fla_backend,
            )
            if not is_full
            else None
        )

        layers.append(
            Qwen35TransformerBlock.Config(
                attention=attention if is_full else deltanet,
                feed_forward=make_moe_config(
                    num_experts=num_experts,
                    enable_sp=enable_sp,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=_depth_init(layer_id),
                        top_k=top_k,
                        score_func=Softmax.Config(),
                        route_norm=True,
                    ),
                    routed_experts=_qwen35_routed_experts_config(
                        dim=dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        param_init=_depth_experts_init(layer_id),
                        moe_comm_backend=moe_comm_backend,
                        non_blocking_capacity_factor=non_blocking_capacity_factor,
                        enable_ep=enable_ep,
                    ),
                    shared_experts=_shared_experts_config(
                        dim=dim,
                        hidden_dim=shared_expert_hidden_dim,
                        layer_id=layer_id,
                        enable_sp=enable_sp,
                    ),
                ),
                attention_norm=_offset_norm(
                    dim, sharding_config=decoder_norm_sharding_config(enable_sp=enable_sp)
                ),
                ffn_norm=_offset_norm(
                    dim, sharding_config=decoder_norm_sharding_config(enable_sp=enable_sp)
                ),
                sharding_config=transformer_block_sharding_config(enable_sp=enable_sp),
            )
        )
    return layers


def _debugmodel(
    attn_backend: str, parallelism: ParallelismConfig | None = None
) -> Qwen35MultimodalModel.Config:
    """Debug config for Qwen3.5 with vision encoder."""
    dim = 256
    head_dim = 64
    rotary_dim = 16
    n_layers = 8
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    # mrope_section sum must equal rotary_dim / 2 (8 for rotary_dim=16).
    # Real models use [11, 11, 10] with rotary_dim=64.
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=4096,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=4096,
                theta=10_000_000.0,
                mrope_section=[3, 3, 2],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=512,
            n_key_heads=2,
            n_value_heads=4,
            key_head_dim=64,
            value_head_dim=64,
            fla_backend="fla_chunked",
            enable_sp=enable_sp,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            num_layers=4,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
        ),
    )


def _debugmodel_moe(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    parallelism: ParallelismConfig | None = None,
) -> Qwen35MultimodalModel.Config:
    """Debug MoE config for Qwen3.5 with shared expert."""
    dim = 256
    head_dim = 64
    rotary_dim = 16
    n_layers = 4
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=4096,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=4096,
                theta=10_000_000.0,
                mrope_section=[3, 3, 2],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=256,
            num_experts=8,
            top_k=2,
            shared_expert_hidden_dim=256,
            n_key_heads=2,
            n_value_heads=4,
            key_head_dim=64,
            value_head_dim=64,
            moe_comm_backend=moe_comm_backend,
            fla_backend="fla_chunked",
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=256,
            ffn_dim=512,
            num_layers=2,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
            num_position_embeddings=1024,
        ),
    )


def _0_8b(
    attn_backend: str, parallelism: ParallelismConfig | None = None
) -> Qwen35MultimodalModel.Config:
    """Qwen3.5-0.8B dense config with vision encoder.

    NOTE: HF config has tie_word_embeddings=true. Torchtitan doesn't support
    tied embeddings yet, so we use a separate lm_head. Checkpoint conversion
    must handle this.
    """
    dim = 1024
    head_dim = 256
    rotary_dim = 64  # partial_rotary_factor=0.25 → head_dim * 0.25
    n_layers = 24
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=262144,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=8,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=3584,
            n_key_heads=16,
            n_value_heads=16,
            key_head_dim=128,
            value_head_dim=128,
            enable_sp=enable_sp,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=768,
            ffn_dim=3072,
            num_layers=12,
            num_heads=12,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=1024,
            num_position_embeddings=2304,
        ),
    )


def _2b(
    attn_backend: str, parallelism: ParallelismConfig | None = None
) -> Qwen35MultimodalModel.Config:
    """Qwen3.5-2B dense config with vision encoder.

    NOTE: HF config has tie_word_embeddings=true. Torchtitan doesn't support
    tied embeddings yet, so we use a separate lm_head. Checkpoint conversion
    must handle this.
    """
    dim = 2048
    head_dim = 256
    rotary_dim = 64  # partial_rotary_factor=0.25
    n_layers = 24
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=262144,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=8,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=6144,
            n_key_heads=16,
            n_value_heads=16,
            key_head_dim=128,
            value_head_dim=128,
            enable_sp=enable_sp,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1024,
            ffn_dim=4096,
            num_layers=24,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
        ),
    )


def _4b(
    attn_backend: str, parallelism: ParallelismConfig | None = None
) -> Qwen35MultimodalModel.Config:
    """Qwen3.5-4B dense config with vision encoder.

    NOTE: HF config has tie_word_embeddings=true. Torchtitan doesn't support
    tied embeddings yet, so we use a separate lm_head.
    """
    dim = 2560
    head_dim = 256
    rotary_dim = 64
    n_layers = 32
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=262144,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=4,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=9216,
            n_key_heads=16,
            n_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
            enable_sp=enable_sp,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1024,
            ffn_dim=4096,
            num_layers=24,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2560,
            num_position_embeddings=2304,
        ),
    )


def _9b(
    attn_backend: str, parallelism: ParallelismConfig | None = None
) -> Qwen35MultimodalModel.Config:
    """Qwen3.5-9B dense config with vision encoder."""
    dim = 4096
    head_dim = 256
    rotary_dim = 64
    n_layers = 32
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=262144,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=4,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=12288,
            n_key_heads=16,
            n_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
            enable_sp=enable_sp,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
        ),
    )


def _27b(
    attn_backend: str, parallelism: ParallelismConfig | None = None
) -> Qwen35MultimodalModel.Config:
    """Qwen3.5-27B dense config with vision encoder."""
    dim = 5120
    head_dim = 256
    rotary_dim = 64
    n_layers = 64
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=262144,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=24,
            n_kv_heads=4,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=17408,
            n_key_heads=16,
            n_value_heads=48,
            key_head_dim=128,
            value_head_dim=128,
            enable_sp=enable_sp,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=5120,
            num_position_embeddings=2304,
        ),
    )


def _35b_a3b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    parallelism: ParallelismConfig | None = None,
) -> Qwen35MultimodalModel.Config:
    """Qwen3.5-35B-A3B MoE config with vision encoder."""
    dim = 2048
    head_dim = 256
    rotary_dim = 64
    n_layers = 40
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=262144,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=512,
            num_experts=256,
            top_k=8,
            shared_expert_hidden_dim=512,
            n_key_heads=16,
            n_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
            moe_comm_backend=moe_comm_backend,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
        ),
    )


def _122b_a10b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    parallelism: ParallelismConfig | None = None,
) -> Qwen35MultimodalModel.Config:
    """Qwen3.5-122B-A10B MoE config with vision encoder."""
    dim = 3072
    head_dim = 256
    rotary_dim = 64
    n_layers = 48
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=262144,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=1024,
            num_experts=256,
            top_k=8,
            shared_expert_hidden_dim=1024,
            n_key_heads=16,
            n_value_heads=64,
            key_head_dim=128,
            value_head_dim=128,
            moe_comm_backend=moe_comm_backend,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=3072,
            num_position_embeddings=2304,
        ),
    )


def _397b_a17b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    parallelism: ParallelismConfig | None = None,
) -> Qwen35MultimodalModel.Config:
    """Qwen3.5-397B-A17B MoE config with vision encoder."""
    dim = 4096
    head_dim = 256
    rotary_dim = 64
    n_layers = 60
    vocab_size = 248320
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    return Qwen35MultimodalModel.Config(
        vocab_size=vocab_size,
        max_context_length=262144,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=1024,
            num_experts=512,
            top_k=10,
            shared_expert_hidden_dim=1024,
            n_key_heads=16,
            n_value_heads=64,
            key_head_dim=128,
            value_head_dim=128,
            moe_comm_backend=moe_comm_backend,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            num_position_embeddings=2304,
        ),
    )


qwen3_5_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_moe": _debugmodel_moe,
    "0.8B": _0_8b,
    "2B": _2b,
    "4B": _4b,
    "9B": _9b,
    "27B": _27b,
    "35B-A3B": _35b_a3b,
    "122B-A10B": _122b_a10b,
    "397B-A17B": _397b_a17b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    parallelism: ParallelismConfig | None = None,
):
    """Build Qwen35Model.Config or Qwen35MultimodalModel.Config.

    Sharding declarations are constructed from
    parallelism.enable_sequence_parallel and fall back to enable_sp=False when
    parallelism is None. Parallelize and checkpoint adapters are inherited from
    BaseModel/model classes rather than wrapped in a separate ModelSpec.
    """
    kwargs = dict(attn_backend=attn_backend, parallelism=parallelism)
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    config = qwen3_5_configs[flavor](**kwargs)
    return config
