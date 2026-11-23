# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from functools import partial

import torch.nn as nn
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common import (
    CosSinRoPE,
    Linear,
    RoPE,
    Softmax,
    TransformerBlock,
    VocabParallelEmbedding,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init

from titaninfra.models._tt_compat import make_token_dispatcher_config

from .model import Qwen3Model, Qwen3TransformerBlock

__all__ = [
    "Qwen3Model",
    "qwen3_configs",
    "model_registry",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}

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


def _qwen3_norm(
    dim: int, *, sharding_config: ShardingConfig | None = None
) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT,
        sharding_config=sharding_config,
    )


def _build_qwen3_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    attn_backend: str,
    rope: RoPE.Config,
    enable_sp: bool = False,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for dense Qwen3 models with depth-scaled inits."""
    inner_attention = get_attention_config(attn_backend)
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen3_norm(dim, sharding_config=norm_sharding),
                ffn_norm=_qwen3_norm(dim, sharding_config=norm_sharding),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    rope=rope,
                    qk_norm=_qwen3_norm(head_dim),
                    enable_sp=enable_sp,
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w13_param_init=_LINEAR_INIT,
                    w2_param_init=_depth_init(layer_id),
                    enable_sp=enable_sp,
                ),
            )
        )
    return layers


def _build_qwen3_moe_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    rope: RoPE.Config,
    enable_sp: bool = False,
    enable_ep: bool = False,
) -> list[TransformerBlock.Config]:
    """Build per-layer configs for MoE Qwen3 models with depth-scaled inits."""
    inner_attention = get_attention_config(attn_backend)
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        routed_experts = make_routed_experts_config(
            dim=dim,
            hidden_dim=moe_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init=_depth_experts_init(layer_id),
            enable_ep=enable_ep,
        )
        # make_routed_experts_config() hardcodes AllToAllTokenDispatcher
        # (upstream dropped the comm_backend concept -- it's a
        # titaninfra-only feature). Post-hoc swap the dispatcher config
        # when a non-default backend is requested, exactly mirroring
        # how _apply_moe_comm_backend() overrides it at the engine layer
        # for the ep_backend>1 production path -- this is the
        # construction-time equivalent for direct model_registry() callers
        # (e.g. config_registry.py recipes) that pass moe_comm_backend
        # explicitly instead of relying on the engine override.
        if moe_comm_backend != "standard":
            import dataclasses

            new_dispatcher = make_token_dispatcher_config(
                num_experts=num_experts,
                top_k=top_k,
                comm_backend=moe_comm_backend,
                hidden_dim=moe_hidden_dim,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
            )
            routed_experts = dataclasses.replace(
                routed_experts, token_dispatcher=new_dispatcher
            )
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen3_norm(dim, sharding_config=norm_sharding),
                ffn_norm=_qwen3_norm(dim, sharding_config=norm_sharding),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    rope=rope,
                    qk_norm=_qwen3_norm(head_dim),
                    enable_sp=enable_sp,
                ),
                feed_forward=make_moe_config(
                    num_experts=num_experts,
                    router=make_router_config(
                        dim=dim,
                        num_experts=num_experts,
                        gate_param_init=_depth_init(layer_id),
                        top_k=top_k,
                        score_func=Softmax.Config(),
                        route_norm=True,
                    ),
                    routed_experts=routed_experts,
                    enable_sp=enable_sp,
                ),
            )
        )
    return layers


def _debugmodel(attn_backend: str, parallelism=None) -> Qwen3Model.Config:
    dim = 256
    head_dim = 128
    n_layers = 8
    vocab_size = 2048
    # Derive enable_sp from the actual topology so root norms and per-layer
    # attention/FFN declarations do not remain hardcoded to False.
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=4096,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        enable_weight_tying=True,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=3072,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=4096,
                theta=1000000.0,
            ),
            enable_sp=enable_sp,
        ),
    )


def _0_6b(attn_backend: str, parallelism=None) -> Qwen3Model.Config:
    dim = 1024
    head_dim = 128
    n_layers = 28
    vocab_size = 151936
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=40960,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        enable_weight_tying=True,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=3072,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=40960,
                theta=1000000.0,
            ),
            enable_sp=enable_sp,
        ),
    )


def _1_7b(attn_backend: str, parallelism=None) -> Qwen3Model.Config:
    dim = 2048
    head_dim = 128
    n_layers = 28
    vocab_size = 151936
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=40960,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        enable_weight_tying=True,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=6144,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=40960,
                theta=1000000.0,
            ),
            enable_sp=enable_sp,
        ),
    )


def _4b(attn_backend: str, parallelism=None) -> Qwen3Model.Config:
    dim = 2560
    head_dim = 128
    n_layers = 36
    vocab_size = 151936
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=40960,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        enable_weight_tying=True,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=9728,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=40960,
                theta=1000000.0,
            ),
            enable_sp=enable_sp,
        ),
    )


def _8b(attn_backend: str, parallelism=None) -> Qwen3Model.Config:
    dim = 4096
    head_dim = 128
    n_layers = 36
    vocab_size = 151936
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=40960,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=12288,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=40960,
                theta=1000000.0,
            ),
            enable_sp=enable_sp,
        ),
    )


def _14b(attn_backend: str, parallelism=None) -> Qwen3Model.Config:
    dim = 5120
    head_dim = 128
    n_layers = 40
    vocab_size = 151936
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=40960,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=40,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=17408,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=40960,
                theta=1000000.0,
            ),
            enable_sp=enable_sp,
        ),
    )


def _32b(attn_backend: str, parallelism=None) -> Qwen3Model.Config:
    dim = 5120
    head_dim = 128
    n_layers = 64
    vocab_size = 151936
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=40960,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=25600,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=40960,
                theta=1000000.0,
            ),
            enable_sp=enable_sp,
        ),
    )


# Qwen3-MoE models


def _debugmodel_moe(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    parallelism=None,
) -> Qwen3Model.Config:
    dim = 256
    head_dim = 128
    n_layers = 8
    vocab_size = 2048
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=4096,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=64,
            top_k=8,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=4096,
                theta=1000000.0,
            ),
            moe_comm_backend=moe_comm_backend,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
        ),
    )


def _30b_a3b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    parallelism=None,
) -> Qwen3Model.Config:
    dim = 2048
    head_dim = 128
    n_layers = 48
    vocab_size = 151936
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=40960,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=128,
            top_k=8,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=40960,
                theta=1000000.0,
            ),
            moe_comm_backend=moe_comm_backend,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
        ),
    )


def _235b_a22b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    parallelism=None,
) -> Qwen3Model.Config:
    dim = 4096
    head_dim = 128
    n_layers = 94
    vocab_size = 151936
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    return Qwen3Model.Config(
        max_context_length=40960,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=4,
            head_dim=head_dim,
            moe_hidden_dim=1536,
            num_experts=128,
            top_k=8,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=40960,
                theta=5000000.0,
            ),
            moe_comm_backend=moe_comm_backend,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
        ),
    )


qwen3_configs = {
    "debugmodel": _debugmodel,
    "0.6B": _0_6b,
    "1.7B": _1_7b,
    "4B": _4b,
    "8B": _8b,
    "14B": _14b,
    "32B": _32b,
    "debugmodel_moe": _debugmodel_moe,
    "30B-A3B": _30b_a3b,
    "235B-A22B": _235b_a22b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    parallelism=None,
) -> Qwen3Model.Config:
    """Build Qwen3Model.Config with topology-derived EP and SP declarations.

enable_ep comes from parallelism.expert_parallel_degree; enable_sp comes
from parallelism.enable_sequence_parallel and reaches root embeddings,
norms, and per-layer sharding. Qwen3 MoE flavors have no shared experts.
TP>1 GPU validation is tracked separately from construction-time checks."""
    kwargs = dict(attn_backend=attn_backend, parallelism=parallelism)
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    config = qwen3_configs[flavor](**kwargs)
    return config
