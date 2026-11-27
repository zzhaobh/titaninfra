# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial

import torch.nn as nn
from torchtitan.models.common import (
    ColumnParallelLinear,
    CosSinRoPE,
    HiMidLoLinear,
    Linear,
    RMSNorm,
    RoPE,
    RowParallelLinear,
    Softmax,
    TransformerBlock,
    VocabParallelEmbedding,
)
from torchtitan.models.common.activation import ClampedSwiGLU
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    QKVLinear,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.decoder_sharding import (
    colwise_sharding_config,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.moe.sharding import (
    moe_sharding_config,
    routed_experts_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
)
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.rope import YaRNScalingConfig

from titaninfra.models._tt_compat import make_token_dispatcher_config

from .model import Attention, GptOssModel, GptOssTransformerBlock
from .moe import GptOssGroupedLinear
from .sharding import attention_sharding_config, routed_expert_linear_sharding_config

__all__ = [
    "GptOssModel",
    "gptoss_configs",
    "model_registry",
]

_NORM_INIT = {"weight": nn.init.ones_}
# GPT-OSS uses std=0.02 for embeddings (model-specific)
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


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


def _make_gptoss_attn_config(
    *,
    dim: int,
    layer_id: int,
    attn_backend: str = "flex",
    n_heads: int = 64,
    n_kv_heads: int = 8,
    head_dim: int = 64,
    sliding_window_size: int | None = None,
    rope: RoPE.Config,
    enable_sp: bool = False,
) -> Attention.Config:
    """Build a fully-specified, sharded GPT-OSS Attention.Config for a layer.

    All linear sub-configs have their in_features/out_features set.
    All linear params use depth-scaled init (including wq/wkv/wo).
    Sinks also use depth-scaled init.
    """
    inner_attention = get_attention_config(attn_backend)

    # Sliding-window layers now carry a dedicated InnerAttention subclass
    # (upstream dropped the old "uniform FlexInnerAttention + runtime
    # mask-dict lookup by attn_mask_key" pattern this file used before this
    # migration -- see model.py's forward(), which no longer picks a mask by
    # key). Build the windowed variant by copying the base config's fields
    # and adding sliding_window_size, mirroring upstream's own flavors.py.
    if sliding_window_size is not None and isinstance(
        inner_attention, VarlenInnerAttention.Config
    ):
        inner_attention = SlidingWindowVarlenInnerAttention.Config(
            **{
                f.name: getattr(inner_attention, f.name)
                for f in dataclasses.fields(inner_attention)
            },
            sliding_window_size=sliding_window_size,
        )
    elif sliding_window_size is not None and isinstance(
        inner_attention, FlexInnerAttention.Config
    ):
        inner_attention = SlidingWindowFlexInnerAttention.Config(
            block_size=inner_attention.block_size,
            kernel_options=inner_attention.kernel_options,
            sliding_window_size=sliding_window_size,
        )
    inner_attention = dataclasses.replace(
        inner_attention, sharding_config=inner_attention_local_spmd_sharding_config()
    )
    x_layout = residual_activation_placement(enable_sp=enable_sp)

    sinks_init = {
        "sinks": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))
    }

    qkv = QKVLinear.Config(
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        wqkv=ColumnParallelLinear.Config(
            in_features=dim,
            out_features=(n_heads + 2 * n_kv_heads) * head_dim,
            bias=True,
            param_init=_depth_init(layer_id),
            sharding_config=colwise_sharding_config(input_layout=x_layout),
        ),
    )

    return Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=qkv,
        wo=RowParallelLinear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            bias=True,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(output_layout=x_layout),
        ),
        sliding_window_size=sliding_window_size,
        inner_attention=inner_attention,
        param_init=sinks_init,
        rope=dataclasses.replace(rope, sharding_config=rope_cache_sharding_config()),
        sharding_config=attention_sharding_config(enable_sp=enable_sp),
    )


def _make_gptoss_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    layer_id: int,
    top_k: int,
    enable_ep: bool,
) -> RoutedExperts.Config:
    """Build a fully-specified, sharded RoutedExperts.Config for one GPT-OSS layer.

    2026-11-22: ``enable_ep`` no longer defaults to a hardcoded ``True`` --
    its sole real caller (``_build_gptoss_layers``) now always threads it
    from ``parallelism.expert_parallel_degree``, matching the fix already
    applied to deepseek_v3/qwen3/qwen3_5. The old docstring here cited
    ``ShardingConfig.resolve_placements()``'s runtime Shard/Partial-on-
    size-1-axis normalization (docs/torchtitan_upgrade_plan_2026-11.md
    §17.4) as justification for ``True`` being safe unconditionally -- that
    reasoning covered the pre-#5995 world. Upstream #5995 (merged into this
    fork 2026-11-22, see §64) added
    ``_validate_declared_parallel_axes()``, which checks the *declared*
    sharding axes at config-construction time rather than the runtime-
    resolved placement; at ``expert_parallel_degree == 1`` it requires
    these weights to declare ``dp_shard``, which the sparse EP layout
    (``{dp_replicate, edp_shard, ep}``) does not have. moe_comm_backend
    (standard/deepep/hybridep) is applied post-hoc via dataclasses.replace
    below, same pattern as qwen3 -- AllToAllTokenDispatcher is upstream's
    hardcoded default with no comm_backend choice.
    """
    std = depth_scaled_std(0.02, layer_id)
    experts_init = {
        "weight": partial(nn.init.trunc_normal_, std=std),
        "bias": partial(nn.init.trunc_normal_, std=std),
    }
    return RoutedExperts.Config(
        w13=GptOssGroupedLinear.Config(
            group_size=num_experts,
            in_features=dim,
            out_features=hidden_dim,
            num_linears=2,
            param_init=experts_init,
            sharding_config=routed_expert_linear_sharding_config(enable_ep=enable_ep),
        ),
        w2=GptOssGroupedLinear.Config(
            group_size=num_experts,
            in_features=hidden_dim,
            out_features=dim,
            param_init=experts_init,
            sharding_config=routed_expert_linear_sharding_config(enable_ep=enable_ep),
        ),
        activation_fn=ClampedSwiGLU.Config(),
        token_dispatcher=AllToAllTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
        ),
        sharding_config=routed_experts_sharding_config(),
    )


def _build_gptoss_layers(
    *,
    dim: int,
    n_layers: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    load_balance_coeff: float | None,
    attn_backend: str = "flex",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    rope: RoPE.Config,
    enable_sp: bool = False,
    enable_ep: bool = False,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for GPT-OSS.

    Even-indexed layers (0, 2, 4, ...) use sliding window attention.
    All dimensional fields are set directly.
    """
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_gptoss_attn_config(
            dim=dim,
            layer_id=layer_id,
            attn_backend=attn_backend,
            sliding_window_size=128 if layer_id % 2 == 0 else None,
            rope=rope,
            enable_sp=enable_sp,
        )
        routed_experts_cfg = _make_gptoss_experts_config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            layer_id=layer_id,
            top_k=top_k,
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
        moe_cfg = MoE.Config(
            num_experts=num_experts,
            load_balance_coeff=load_balance_coeff,
            routed_experts=routed_experts_cfg,
            router=TokenChoiceTopKRouter.Config(
                num_experts=num_experts,
                score_func=Softmax.Config(),
                route_norm=True,
                gate=HiMidLoLinear.Config(
                    in_features=dim,
                    out_features=num_experts,
                    backward_mode="hi_mid_lo",
                    bias=True,
                    param_init=_depth_init(layer_id),
                    sharding_config=router_gate_sharding_config(),
                ),
                top_k=top_k,
                sharding_config=router_sharding_config(),
            ),
            sharding_config=moe_sharding_config(enable_sp=enable_sp),
        )
        layer_cfg = GptOssTransformerBlock.Config(
            attention=attn_cfg,
            attention_norm=RMSNorm.Config(
                normalized_shape=dim, param_init=_NORM_INIT, sharding_config=norm_sharding
            ),
            ffn_norm=RMSNorm.Config(
                normalized_shape=dim, param_init=_NORM_INIT, sharding_config=norm_sharding
            ),
            feed_forward=moe_cfg,
        )
        layers.append(layer_cfg)
    return layers


def _debugmodel(
    moe_comm_backend: str = "standard",
    attn_backend: str = "flex",
    parallelism=None,
) -> GptOssModel.Config:
    dim = 256
    hidden_dim = 2880
    n_layers = 4
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    # Derive enable_sp from the actual parallelism config rather than fixing it
    # to False. Construction-time checks distinguish replicated and ignored TP
    # placements even before a dedicated TP>1 GPU case is added.
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return GptOssModel.Config(
        max_context_length=131072,
        vocab_size=2048,
        dim=dim,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=2048,
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
            out_features=2048,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=8,
            top_k=4,
            load_balance_coeff=1e-3,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            rope=CosSinRoPE.Config(
                dim=64,
                max_context_length=131072,
                theta=150000.0,
                scaling=YaRNScalingConfig(
                    factor=32,
                    beta_fast=32.0,
                    beta_slow=1.0,
                    truncate=False,
                    original_seq_len=4096,
                ),
            ),
        ),
    )


def _20b(
    moe_comm_backend: str = "standard",
    attn_backend: str = "flex",
    parallelism=None,
) -> GptOssModel.Config:
    dim = 2880
    hidden_dim = 2880
    n_layers = 24
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return GptOssModel.Config(
        max_context_length=131072,
        dim=dim,
        vocab_size=201088,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=201088,
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
            out_features=201088,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=32,
            top_k=4,
            load_balance_coeff=1e-3,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            rope=CosSinRoPE.Config(
                dim=64,
                max_context_length=131072,
                theta=150000.0,
                scaling=YaRNScalingConfig(
                    factor=32,
                    beta_fast=32.0,
                    beta_slow=1.0,
                    truncate=False,
                    original_seq_len=4096,
                ),
            ),
        ),
    )


def _120b(
    moe_comm_backend: str = "standard",
    attn_backend: str = "flex",
    parallelism=None,
) -> GptOssModel.Config:
    dim = 2880
    hidden_dim = 2880
    n_layers = 36
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return GptOssModel.Config(
        max_context_length=131072,
        dim=dim,
        vocab_size=201088,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=201088,
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
            out_features=201088,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=128,
            top_k=4,
            load_balance_coeff=1e-3,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            rope=CosSinRoPE.Config(
                dim=64,
                max_context_length=131072,
                theta=150000.0,
                scaling=YaRNScalingConfig(
                    factor=32,
                    beta_fast=32.0,
                    beta_slow=1.0,
                    truncate=False,
                    original_seq_len=4096,
                ),
            ),
        ),
    )


gptoss_configs = {
    "debugmodel": _debugmodel,
    "20b": _20b,
    "120b": _120b,
}


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    attn_backend: str = "flex",
    parallelism=None,
) -> GptOssModel.Config:
    """Build GptOssModel.Config with topology-derived EP and SP declarations.

    enable_ep comes from parallelism.expert_parallel_degree and enable_sp from
    parallelism.enable_sequence_parallel. These declarations affect root
    embeddings/norms and every layer's attention/FFN sharding. GPT-OSS has no
    shared experts; TP>1 GPU validation is tracked separately.
    """
    config = gptoss_configs[flavor](
        moe_comm_backend=moe_comm_backend,
        attn_backend=attn_backend,
        parallelism=parallelism,
    )
    return config
