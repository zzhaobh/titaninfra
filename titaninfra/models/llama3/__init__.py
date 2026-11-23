# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# ============================================================================
# 2026-11 torchtitan upstream-tip migration (pilot family, goal items 4/5).
# ============================================================================
# `ModelSpec`/`torchtitan.protocols` are gone upstream (commit b09a88b9b,
# "Reorganize training recipes"). The old model_registry(flavor) -> ModelSpec
# pattern (parallelize_fn/pipelining_fn/state_dict_adapter bundled as a
# separate object) is replaced by model_registry(flavor) -> Model.Config:
# parallelize/pipeline are now inherited BaseModel methods, and
# state_dict_adapter_cls is a ClassVar set directly on the model class (see
# model.py). titaninfra keeps the function name `model_registry` (the
# titaninfra-side consumer in titaninfra/models/__init__.py calls it by
# this name) but its return type changes from ModelSpec to Llama3Model.Config.
#
# Sharding: upstream's make_gqa_config()/make_ffn_config() now build their
# ShardingConfig fields internally from an `enable_sp: bool` kwarg (returned,
# not mutated-in-place) -- the separate titaninfra sharding.py mutation-pass
# file (set_llama3_sharding_config) is gone; enable_sp is threaded straight
# from parallelism.enable_sequence_parallel down into _build_llama3_layers.
# parallelize.py is also gone: model.parallelize() is an inherited BaseModel
# method (see model.py's Decoder._apply_fsdp), no per-family wrapper needed.
#
# The CP+Varlen restriction from the pre-migration version
# ("Context Parallel only supports SDPA and FlexInnerAttention") is dropped
# here, not ported forward: upstream's models/common/attention/cp_attention.py
# now has dedicated support for Varlen-attention CP (see e.g.
# "K/V all-gather varlen CP" in that file) -- carrying the old restriction
# forward would be reintroducing a now-stale limitation, not preserving a
# real one.
#
# See docs/torchtitan_upgrade_plan_2026-11.md Sec 12 for the full write-up
# this pilot is based on, including what is explicitly NOT yet done (the
# titaninfra engine-layer adaptation that would let this file's output
# actually flow through Trainer/_load_model_spec/convert.py -- those still
# expect the old ModelSpec shape and are a separate, one-time task that
# benefits every family, not just this one).

from collections.abc import Callable
from functools import partial

import torch.nn as nn
from torchtitan.models.common import (
    ComplexRoPE,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
    VocabParallelEmbedding,
    compute_ffn_hidden_dim,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rope import Llama3ScalingConfig

from .model import Llama3Model, Llama3TransformerBlock

__all__ = [
    "Llama3Model",
    "llama3_configs",
    "model_registry",
]


# titaninfra deliberately keeps std=0.02 (industry-standard) rather than
# upstream's own default of std=1.0 for tok_embeddings -- see CLAUDE.md
# "Embedding init std" (0.02 is used across every titaninfra family;
# upstream's flavors.py for this family still uses 1.0 at the time of this
# migration, so this divergence must be preserved explicitly, not dropped
# as if it were migration noise).
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


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


def _build_llama3_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    rope: RoPE.Config,
    n_kv_heads: int | None = None,
    attn_backend: str,
    enable_sp: bool,
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer TransformerBlock configs with depth-scaled inits.

    ``enable_sp`` threads straight into make_gqa_config/make_ffn_config,
    which build each sub-config's ShardingConfig internally -- there is no
    separate sharding-mutation pass to run after this (see module docstring).
    ``attention_norm``/``ffn_norm`` are not covered by those two builders
    (they build the attention/FFN sub-trees, not the surrounding block's own
    norm layers), so this function must declare their sharding_config itself
    -- same per-layer-norm gap class found and fixed on qwen2_5/qwen3/gemma4
    this round (see docs/torchtitan_upgrade_plan_2026-11.md Sec 52/58),
    following qwen3/__init__.py's identical pattern of computing
    norm_sharding_config(enable_sp=enable_sp) once and reusing it per layer.
    """
    inner_attention = get_attention_config(attn_backend)
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT,
                    sharding_config=norm_sharding,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT,
                    sharding_config=norm_sharding,
                ),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    rope=rope,
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


def _debugmodel(attn_backend: str, *, enable_sp: bool) -> Llama3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    vocab_size = 2048
    return Llama3Model.Config(
        max_context_length=2048,
        dim=dim,
        vocab_size=vocab_size,
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
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=256, ffn_dim_multiplier=1.0
            ),
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_context_length=2048,
                theta=500000,
                scaling=Llama3ScalingConfig(),
            ),
            attn_backend=attn_backend,
            enable_sp=enable_sp,
        ),
    )


def _8b(attn_backend: str, *, enable_sp: bool) -> Llama3Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    vocab_size = 128256
    return Llama3Model.Config(
        max_context_length=131072,
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=False,
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
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=1024, ffn_dim_multiplier=1.3
            ),
            rope=ComplexRoPE.Config(
                dim=dim // n_heads,
                max_context_length=131072,
                theta=500000,
                scaling=Llama3ScalingConfig(),
            ),
            attn_backend=attn_backend,
            enable_sp=enable_sp,
        ),
    )


# PILOT SCOPE NOTE: only debugmodel and 8B are ported in this pilot (enough
# to prove the migration pattern on both the tiny CPU-testable flavor and one
# real production-sized flavor). 1B/3B/70B/405B carry the exact same pattern
# (see the pre-migration titaninfra __init__.py for their dims) and are
# deliberately deferred to the follow-up round that migrates this family for
# real, to keep this pilot's diff reviewable.
llama3_configs = {
    "debugmodel": _debugmodel,
    "8B": _8b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    *,
    enable_sp: bool | None = None,
    parallelism=None,
) -> Llama3Model.Config:
    """Build Llama3 model configuration with topology-aware SP declarations.

    enable_sp defaults to parallelism.enable_sequence_parallel when supplied.
    An explicit enable_sp argument overrides that derived value for backward
    compatibility. Llama3 is dense and has no expert-parallel layout.
    """
    if enable_sp is None:
        enable_sp = (
            parallelism.enable_sequence_parallel if parallelism is not None else False
        )
    config = llama3_configs[flavor](attn_backend=attn_backend, enable_sp=enable_sp)
    return config
