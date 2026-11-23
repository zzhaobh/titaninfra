"""Qwen2.5 model configuration and registration.

model_registry returns Qwen2_5Model.Config for debugmodel and 7B-Instruct."""

import dataclasses
from collections.abc import Callable
from functools import partial

import torch.nn as nn
from torchtitan.models.common import (
    CosSinRoPE,
    Linear,
    RoPE,
    TransformerBlock,
    VocabParallelEmbedding,
)
from torchtitan.models.common.attention import GQAttention
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
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std

from .model import Qwen2_5Model, Qwen2_5TransformerBlock

__all__ = [
    "Qwen2_5Model",
    "Qwen2_5TransformerBlock",
    "qwen2_5_configs",
    "model_registry",
]


# ========================================
# Initialization policies.
# ========================================

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
# Embedding init std=0.02 matches the Qwen HF initializer and linear layers.
# Larger legacy values can slow embedding learning after RMSNorm.
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}

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


def _qwen2_5_norm(dim: int, *, sharding_config) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT,
        sharding_config=sharding_config,
    )


def _make_gqa_config_with_qkv_bias(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    wqkv_param_init: dict[str, Callable],
    wo_param_init: dict[str, Callable],
    inner_attention,
    rope: RoPE.Config,
    enable_sp: bool,
) -> GQAttention.Config:
    """Build Qwen2.5's GQA config from the upstream declaration, adding only QKV bias.

    Qwen2.5 has Q/K/V bias (o_proj does not), which upstream's
    ``make_gqa_config`` does not expose as a keyword. Reuse the complete
    upstream declaration -- including its SP-aware TP collectives -- then
    replace only ``wqkv.bias`` with the real architectural delta instead of
    maintaining a second copy of the GQA sharding logic.
    """
    config = make_gqa_config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        wqkv_param_init=wqkv_param_init,
        wo_param_init=wo_param_init,
        inner_attention=inner_attention,
        rope=rope,
        qk_norm=None,
        enable_sp=enable_sp,
    )
    return dataclasses.replace(
        config,
        qkv_linear=dataclasses.replace(
            config.qkv_linear,
            wqkv=dataclasses.replace(config.qkv_linear.wqkv, bias=True),
        ),
    )


def _build_qwen2_5_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    rope: RoPE.Config,
    attn_backend: str = "flex",
    enable_sp: bool = False,
) -> list[TransformerBlock.Config]:
    inner_attention = get_attention_config(attn_backend)
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        attention = _make_gqa_config_with_qkv_bias(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            wqkv_param_init=_LINEAR_INIT,
            wo_param_init=_depth_init(layer_id),
            inner_attention=inner_attention,
            rope=rope,
            enable_sp=enable_sp,
        )
        feed_forward = make_ffn_config(
            dim=dim,
            hidden_dim=hidden_dim,
            w13_param_init=_LINEAR_INIT,
            w2_param_init=_depth_init(layer_id),
            enable_sp=enable_sp,
        )

        layers.append(
            Qwen2_5TransformerBlock.Config(
                attention_norm=_qwen2_5_norm(dim, sharding_config=norm_sharding),
                ffn_norm=_qwen2_5_norm(dim, sharding_config=norm_sharding),
                attention=attention,
                feed_forward=feed_forward,
            )
        )
    return layers


# ========================================
# Model factory functions.
# ========================================

def _make_dense_model(
    *,
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    max_seq_len: int = 32768,
    theta: float = 1000000.0,
    attn_backend: str = "flex",
    parallelism=None,
) -> Qwen2_5Model.Config:
    # 2026-11-23: derive SP from the real topology instead of hardcoding False.
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    rope = CosSinRoPE.Config(
        dim=head_dim,
        max_context_length=max_seq_len,
        theta=theta,
    )
    return Qwen2_5Model.Config(
        max_context_length=max_seq_len,
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen2_5_norm(dim, sharding_config=norm_sharding_config(enable_sp=enable_sp)),
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
        layers=_build_qwen2_5_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            rope=rope,
            attn_backend=attn_backend,
            enable_sp=enable_sp,
        ),
    )


# ========================================
# Qwen2.5 configurations maintained by TitanInfra.
# ========================================

qwen2_5_configs = {
    "debugmodel": lambda parallelism=None: _make_dense_model(
        vocab_size=151936, dim=256, n_layers=8, n_heads=16, n_kv_heads=8,
        head_dim=128, hidden_dim=3072, max_seq_len=4096,
        parallelism=parallelism,
    ),

    "7B-Instruct": lambda parallelism=None: _make_dense_model(
        vocab_size=152064, dim=3584, n_layers=28, n_heads=28, n_kv_heads=4,
        head_dim=128, hidden_dim=18944, max_seq_len=32768, theta=1000000.0,
        parallelism=parallelism,
    ),
}


def model_registry(flavor: str, parallelism=None) -> Qwen2_5Model.Config:
    """
    Return a Qwen2_5Model.Config for the requested flavor.

    parallelism is forwarded to every flavor constructor and derives enable_sp
    from enable_sequence_parallel. Root and per-layer declarations use that
    value rather than a hardcoded False.
    """
    return qwen2_5_configs[flavor](parallelism=parallelism)
