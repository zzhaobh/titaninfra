# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Literal

import spmd_types as spmd
import torch.nn as nn
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import (
    ComplexRoPE,
    HiMidLoLinear,
    Linear,
    RMSNorm,
    RoPE,
    RowParallelLinear,
    Sigmoid,
    Softmax,
    VocabParallelEmbedding,
)
from torchtitan.models.common.attention import MLAAttention
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    mla_attention_local_spmd_sharding_config,
    mla_kv_projection_sharding_config,
    norm_sharding_config,
    replicated_input_colwise_sharding_config,
    replicated_state_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe.sharding import router_gate_sharding_config, router_sharding_config
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.rope import YaRNScalingConfig

from titaninfra.models._tt_compat import make_token_dispatcher_config

from .attention import DeepSeekMLAAttention
from .model import DeepSeekV3Model, DeepSeekV3TransformerBlock
from .moe import DeepSeekV3Router
from .sharding import mla_attention_sharding_config
from .state_dict_adapter import (
    DeepSeekV3StateDictAdapter,
    KimiVLLanguageStateDictAdapter,
)

__all__ = [
    "DeepSeekV3Model",
    "DeepSeekV3StateDictAdapter",
    "KimiVLLanguageStateDictAdapter",
    "deepseekv3_configs",
    "model_registry",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


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
        "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
    }


def _make_dsv3_router_config(
    *,
    dim: int,
    num_experts: int,
    layer_id: int,
    top_k: int,
    score_func: Literal["sigmoid", "softmax"],
    num_expert_groups: int | None = None,
    num_limited_groups: int | None = None,
    route_scale: float = 1.0,
    route_norm: bool = False,
) -> DeepSeekV3Router.Config:
    """DeepSeekV3Router (not the generic make_router_config/TokenChoiceTopKRouter):
    upstream moved group-limited routing (num_expert_groups/num_limited_groups)
    out of the common router into this DeepSeek-specific subclass."""
    return DeepSeekV3Router.Config(
        num_experts=num_experts,
        gate=HiMidLoLinear.Config(
            in_features=dim,
            out_features=num_experts,
            backward_mode="hi_mid_lo",
            bias=False,
            param_init=_depth_init(layer_id),
            sharding_config=router_gate_sharding_config(),
        ),
        top_k=top_k,
        score_func=Sigmoid.Config() if score_func == "sigmoid" else Softmax.Config(),
        num_expert_groups=num_expert_groups,
        num_limited_groups=num_limited_groups,
        route_scale=route_scale,
        route_norm=route_norm,
        sharding_config=router_sharding_config(),
    )


def _make_mla_attention_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float = 1.0,
    attn_backend: str,
    rope: RoPE.Config,
    enable_sp: bool = False,
) -> DeepSeekMLAAttention.Config:
    """Build a fully-specified, sharded DeepSeekMLAAttention.Config.

    All Linear and RMSNorm sub-configs have their dimensional fields set.
    When q_lora_rank == 0, sets wq (not wq_a/wq_b); when q_lora_rank > 0, sets
    wq_a/wq_b (not wq). wkv_b (the K/V up-projection) lives inside the new
    MLAAttention inner-attention wrapper, not directly on this Config.
    """
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    x_layout = residual_activation_placement(enable_sp=enable_sp)

    wkv_b = Linear.Config(
        in_features=kv_lora_rank,
        out_features=n_heads * (qk_nope_head_dim + v_head_dim),
        param_init=_LINEAR_INIT,
        sharding_config=mla_kv_projection_sharding_config(cp=spmd.S(0)),
    )
    inner_attention = MLAAttention.Config(
        wkv_b=wkv_b,
        packed_kv_head_dim=qk_nope_head_dim + v_head_dim,
        inner_attention=get_attention_config(attn_backend),
        sharding_config=mla_attention_local_spmd_sharding_config(),
    )

    if q_lora_rank == 0:
        wq = Linear.Config(
            in_features=dim,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        )
        wq_a = None
        wq_b = None
        # q_norm is unused when q_lora_rank == 0 (never built), but the field is
        # required on DeepSeekMLAAttention.Config so we supply a placeholder.
        q_norm = RMSNorm.Config(normalized_shape=1, param_init=_NORM_INIT)
    else:
        wq = None
        wq_a = Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        )
        wq_b = Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        )
        q_norm = RMSNorm.Config(
            normalized_shape=q_lora_rank,
            param_init=_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        )

    return DeepSeekMLAAttention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        wq=wq,
        wq_a=wq_a,
        wq_b=wq_b,
        q_norm=q_norm,
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=kv_lora_rank,
            param_init=_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        wo=RowParallelLinear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(output_layout=x_layout),
        ),
        inner_attention=inner_attention,
        rope=dataclasses.replace(rope, sharding_config=rope_cache_sharding_config()),
        sharding_config=mla_attention_sharding_config(enable_sp=enable_sp),
    )


def _build_dsv3_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: Literal["sigmoid", "softmax"],
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    rope: RoPE.Config,
    enable_sp: bool = False,
    enable_ep: bool = False,
) -> list[DeepSeekV3TransformerBlock.Config]:
    """Build the list of per-layer TransformerBlock configs.

    Layers with layer_id < n_dense_layers get a dense FeedForward and no MoE.
    Layers with layer_id >= n_dense_layers get a MoE and no FeedForward.

    Router and expert inits are constructed per-layer so depth-scaled
    initializers are correct for each layer's position.
    """
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_mla_attention_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            attn_backend=attn_backend,
            rope=rope,
            enable_sp=enable_sp,
        )

        if layer_id < n_dense_layers:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_depth_init(layer_id),
                enable_sp=enable_sp,
            )
        else:
            routed_experts_cfg = make_routed_experts_config(
                dim=dim,
                hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=router_top_k,
                param_init=_depth_experts_init(layer_id),
                enable_ep=enable_ep,
            )
            if moe_comm_backend != "standard":
                routed_experts_cfg = dataclasses.replace(
                    routed_experts_cfg,
                    token_dispatcher=make_token_dispatcher_config(
                        num_experts=num_experts,
                        top_k=router_top_k,
                        comm_backend=moe_comm_backend,
                        hidden_dim=dim,
                        non_blocking_capacity_factor=non_blocking_capacity_factor,
                    ),
                )
            ffn_cfg = make_moe_config(
                num_experts=num_experts,
                router=_make_dsv3_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    layer_id=layer_id,
                    top_k=router_top_k,
                    score_func=router_score_func,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                routed_experts=routed_experts_cfg,
                shared_experts=make_shared_expert_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w13_param_init=_LINEAR_INIT,
                    w2_param_init=_depth_init(layer_id),
                    enable_sp=enable_sp,
                ),
                load_balance_coeff=1e-3,
                enable_sp=enable_sp,
            )

        layers.append(
            DeepSeekV3TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                    sharding_config=norm_sharding,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                    sharding_config=norm_sharding,
                ),
                feed_forward=ffn_cfg,
            )
        )
    return layers


def _debugmodel(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV3Model.Config:
    dim = 256
    n_layers = 6
    vocab_size = 2048
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    num_experts = 8
    n_dense_layers = 1
    max_context_length = 4096 * 4
    # parallelism may be None for fallback or legacy callers; disable SP in
    # that case to preserve previous behavior.
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    # Derive enable_ep from the real expert_parallel_degree. Sparse layouts are
    # invalid for EP=1 because dense-axis validation requires dp_shard.
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=max_context_length,
            theta=10000.0,
            scaling=YaRNScalingConfig(
                factor=40.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_seq_len=4096,
            ),
        ),
    )
    return DeepSeekV3Model.Config(
        max_context_length=max_context_length,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
    )


def _16b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV3Model.Config:
    dim = 2048
    n_layers = 27
    vocab_size = 102400
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 64
    n_dense_layers = 1
    max_context_length = 4096 * 4
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    # Derive enable_ep from the real expert_parallel_degree. Sparse layouts are
    # invalid for EP=1 because dense-axis validation requires dp_shard.
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=max_context_length,
            theta=10000.0,
            scaling=YaRNScalingConfig(
                factor=40.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_seq_len=4096,
            ),
        ),
    )
    return DeepSeekV3Model.Config(
        max_context_length=max_context_length,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
    )


def _236b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV3Model.Config:
    dim = 5120
    n_layers = 60
    vocab_size = 102400
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 1536
    num_shared_experts = 2
    dense_hidden_dim = 12288
    rope_dim = 64
    num_experts = 160
    n_dense_layers = 1
    max_context_length = 4096 * 4
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    # Derive enable_ep from the real expert_parallel_degree. Sparse layouts are
    # invalid for EP=1 because dense-axis validation requires dp_shard.
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        router_num_expert_groups=8,
        router_num_limited_groups=3,
        router_route_scale=16.0,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=max_context_length,
            theta=10000.0,
            scaling=YaRNScalingConfig(
                factor=40.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_seq_len=4096,
            ),
        ),
    )
    return DeepSeekV3Model.Config(
        max_context_length=max_context_length,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
    )


def _671b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV3Model.Config:
    dim = 7168
    n_layers = 61
    vocab_size = 129280
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 2048
    num_shared_experts = 1
    dense_hidden_dim = 18432
    rope_dim = 64
    num_experts = 256
    n_dense_layers = 3
    max_context_length = 4096 * 4
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    # Derive enable_ep from the real expert_parallel_degree. Sparse layouts are
    # invalid for EP=1 because dense-axis validation requires dp_shard.
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=8,
        router_score_func="sigmoid",
        router_num_expert_groups=8,
        router_num_limited_groups=4,
        router_route_scale=2.5,
        router_route_norm=True,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=max_context_length,
            theta=10000.0,
            scaling=YaRNScalingConfig(
                factor=40.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_seq_len=4096,
            ),
        ),
    )
    return DeepSeekV3Model.Config(
        max_context_length=max_context_length,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
    )


def _dsv3_lite_16b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    *,
    theta: float,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV3Model.Config:
    """Shared builder for the 16B-A3B-shaped DeepSeek-V3 lite flavors.

    Kimi-VL-A3B's language tower and Moonshot's standalone Moonlight-16B-A3B
    pretrain checkpoint are bit-for-bit the same architecture (dims, MoE
    shape, MLA config) — confirmed against both HF ``config.json`` files —
    but were trained as separate runs with a different RoPE ``theta``
    (Kimi-VL-A3B 800000.0, Moonlight 50000.0). Parametrize on ``theta``
    rather than duplicating the whole builder.
    """
    dim = 2048
    n_layers = 27
    vocab_size = 163840
    n_heads = 16
    q_lora_rank = 0  # HF null → non-LoRA wq path
    kv_lora_rank = 512
    moe_hidden_dim = 1408  # moe_intermediate_size
    num_shared_experts = 2
    dense_hidden_dim = 11264  # intermediate_size
    rope_dim = 64  # qk_rope_head_dim
    num_experts = 64
    n_dense_layers = 1  # first_k_dense_replace
    max_context_length = 4096 * 4
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    # Derive enable_ep from the real expert_parallel_degree. Sparse layouts are
    # invalid for EP=1 because dense-axis validation requires dp_shard.
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="sigmoid",
        router_num_expert_groups=1,
        router_num_limited_groups=1,
        router_route_scale=2.446,
        router_route_norm=True,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=max_context_length,
            theta=theta,
            # NO YaRN: scaling left at its "none" default (do not set "yarn").
        ),
    )
    return DeepSeekV3Model.Config(
        max_context_length=max_context_length,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
    )


def _kimi_vl_a3b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV3Model.Config:
    """Kimi-VL-A3B language tower (Kimi-VL-A3B ``text_config``).

    The Kimi-VL-A3B language tower IS DeepSeek-V3 architecture (MLA + MoE), so
    it is a flavor here rather than a new model family. See
    ``_dsv3_lite_16b`` for the shared 16B-A3B builder and its one delta vs
    ``_moonlight_16b_a3b`` (``theta``).
    """
    return _dsv3_lite_16b(
        attn_backend,
        moe_comm_backend,
        non_blocking_capacity_factor,
        theta=800000.0,
        parallelism=parallelism,
    )


def _moonlight_16b_a3b(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV3Model.Config:
    """moonshotai/Moonlight-16B-A3B — standalone pretrain checkpoint, same
    16B-A3B DeepSeek-V3 architecture as the Kimi-VL-A3B language tower (see
    ``_dsv3_lite_16b``) but trained separately with ``rope_theta=50000.0``
    (vs Kimi-VL-A3B's ``800000.0``) and no vision tower / no
    ``language_model.`` HF key prefix — plain ``DeepseekV3ForCausalLM``.
    Global MoE case model (replaces qwen3 as the flagship MoE pretrain case).
    """
    return _dsv3_lite_16b(
        attn_backend,
        moe_comm_backend,
        non_blocking_capacity_factor,
        theta=50000.0,
        parallelism=parallelism,
    )


deepseekv3_configs = {
    "debugmodel": _debugmodel,
    "16B": _16b,
    "236B": _236b,
    "671B": _671b,
    "kimi_vl_a3b": _kimi_vl_a3b,
    "moonlight_16b_a3b": _moonlight_16b_a3b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV3Model.Config:
    """Build DeepSeekV3Model.Config with topology-derived SP and EP layouts.

enable_sp comes from parallelism.enable_sequence_parallel so shared-expert,
attention, and norm sharding follows the actual TP topology. enable_ep comes
from expert_parallel_degree so EP=1 uses dense layouts. parallelism=None
preserves the previous non-SP/non-EP behavior."""
    config = deepseekv3_configs[flavor](
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        parallelism=parallelism,
    )
    return config
