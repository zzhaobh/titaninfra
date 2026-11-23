# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Declarative sharding configs for titaninfra's DeepSeek V4 components.

The helpers mirror upstream DeepSeek V4 while retaining titaninfra's fused
Hyper-Connection blocks and direct ``attn_sink`` parameter.
"""

import spmd_types as spmd
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    residual_activation_placement,
)
from torchtitan.models.common.moe.sharding import router_sharding_config


def attention_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Attention boundary plus the direct per-head sink parameter."""
    return ShardingConfig(
        state_shardings={"attn_sink": dense_param_placement(tp=spmd.S(0))},
        input_shardings={
            "x": residual_activation_placement(enable_sp=enable_sp),
        },
    )


def wo_a_sharding_config() -> ShardingConfig:
    """Shard the grouped LoRA-A weight along its output dimension."""
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.S(0))})


def compressor_sharding_config() -> ShardingConfig:
    """Replicate the legacy direct position-bias parameter across TP."""
    return ShardingConfig(
        state_shardings={"position_bias": dense_param_placement(tp=spmd.R)}
    )


def indexer_sharding_config() -> ShardingConfig:
    """Upstream indexer boundary plus the legacy direct position-bias state."""
    from torchtitan.models.deepseek_v4.sharding import (
        indexer_sharding_config as upstream_indexer_sharding_config,
    )

    config = upstream_indexer_sharding_config()
    config.state_shardings["position_bias"] = dense_param_placement(tp=spmd.R)
    return config


def deepseek_v4_router_sharding_config(*, layer_id: int, n_hash_layers: int) -> ShardingConfig:
    """Router sharding; hash-routed layers also replicate ``tid2eid``."""
    config = router_sharding_config()
    if layer_id < n_hash_layers:
        config.state_shardings["tid2eid"] = dense_param_placement(tp=spmd.R)
    return config


def hc_head_sharding_config() -> ShardingConfig:
    """Replicate the root Hyper-Connection mixing coefficients."""
    return ShardingConfig(
        state_shardings={
            name: dense_param_placement(tp=spmd.R) for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")
        }
    )


def transformer_block_sharding_config() -> ShardingConfig:
    """Replicate both sets of block-level Hyper-Connection coefficients."""
    return ShardingConfig(
        state_shardings={
            name: dense_param_placement(tp=spmd.R)
            for name in (
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
            )
        }
    )
