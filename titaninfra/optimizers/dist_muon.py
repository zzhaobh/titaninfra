# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Upstream DistMuon layout algorithm bound to TitanInfra's K3 config types."""

from typing import cast

import torch
from torch.distributed.tensor import Shard
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.feed_forward import DenseFeedForward

from titaninfra.models.kimi_k3.attention import KimiMLAAttention
from titaninfra.models.kimi_k3.model import KimiK3Model


def _on_dp_axes(placement) -> ComputeLayout:
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: placement,
            f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": placement,
        },
        compute_dtype=torch.bfloat16,
    )


def _per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    ep_size = parallelism.expert_parallel_degree
    if ep_size <= 0:
        raise ValueError("expert_parallel_degree must be positive")
    if ep_size == 1:
        return _on_dp_axes(Shard(0))
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EDP_SHARD.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
        },
        compute_dtype=torch.bfloat16,
    )


def build_kimi_k3_dist_muon_layout(
    model_config: KimiK3Model.Config,
    parallelism: ParallelismConfig,
) -> tuple[str, dict[str, ComputeLayout], tuple[BucketConfig, ...]]:
    attention = cast(KimiMLAAttention.Config, model_config.first_full_attention)
    delta_attention = next(
        layer.attention
        for layer in model_config.layers
        if isinstance(layer.attention, KDA.Config)
    )
    owned = _on_dp_axes(Owned())

    def blocks_of(*num_rows: int) -> ComputeLayout:
        return _on_dp_axes(BlockShard(dim=0, block_sizes=num_rows))

    attention_shardings = {
        "wq_a": owned,
        "wq_b": blocks_of(attention.qk_nope_head_dim, attention.qk_rope_head_dim),
        "wkv_a": blocks_of(attention.kv_lora_rank, attention.qk_rope_head_dim),
        "inner_attention.wkv_b": blocks_of(
            attention.qk_nope_head_dim, attention.v_head_dim
        ),
        "gate": blocks_of(attention.v_head_dim),
        "wo": owned,
    }
    per_kda_head = blocks_of(delta_attention.head_dim)
    delta_attention_shardings = {
        "q_proj": per_kda_head,
        "k_proj": per_kda_head,
        "v_proj": per_kda_head,
        "forget_a": owned,
        "forget_b": per_kda_head,
        "output_gate": per_kda_head,
        "output_proj": owned,
    }
    feed_forward_shardings = {"w13": _on_dp_axes(Shard(0)), "w2": owned}
    per_expert = _per_expert_compute_layout(parallelism)
    moe_shardings = {
        "routed_experts.w13": per_expert,
        "routed_experts.w2": per_expert,
        "router.gate": owned,
        **{
            f"shared_experts.{name}": layout
            for name, layout in feed_forward_shardings.items()
        },
        "routed_down": owned,
        "routed_up": owned,
    }

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    routed_fqns_per_layer, non_routed_fqns_per_layer = [], []
    for layer_id, layer in enumerate(model_config.layers):
        attention_table = (
            attention_shardings
            if isinstance(layer.attention, KimiMLAAttention.Config)
            else delta_attention_shardings
        )
        feed_forward_table = (
            feed_forward_shardings
            if isinstance(layer.feed_forward, DenseFeedForward.Config)
            else moe_shardings
        )
        layer_shardings = {
            f"layers.{layer_id}.{module}.{name}.weight": layout
            for module, table in (
                ("attention", attention_table),
                ("feed_forward", feed_forward_table),
            )
            for name, layout in table.items()
        }
        compute_sharding_by_fqn |= layer_shardings
        routed_fqns_per_layer.append(
            [fqn for fqn, layout in layer_shardings.items() if layout is per_expert]
        )
        non_routed_fqns_per_layer.append(
            [fqn for fqn, layout in layer_shardings.items() if layout is not per_expert]
        )

    num_layers = len(model_config.layers)
    bucket_layer_ids = [[0]] + [
        list(range(first, min(first + 2, num_layers)))
        for first in range(1, num_layers, 2)
    ]
    bucket_configs = []
    for layer_ids in bucket_layer_ids:
        name = "layers." + "-".join(map(str, layer_ids))
        non_routed = tuple(f for i in layer_ids for f in non_routed_fqns_per_layer[i])
        routed = tuple(f for i in layer_ids for f in routed_fqns_per_layer[i])
        bucket_configs.append(BucketConfig(name=name, patterns=non_routed))
        if routed:
            bucket_configs.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=routed)
            )

    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_shardings)}|"
        rf"{'|'.join(delta_attention_shardings)})\.weight|"
        r"routed_experts\.w(?:13|2)\.weight|"
        r"feed_forward\.w(?:13|2)\.weight|"
        r"feed_forward\.router\.gate\.weight|"
        r"feed_forward\.shared_experts\.w(?:13|2)\.weight|"
        r"feed_forward\.routed_(?:down|up)\.weight"
        r")$"
    )
    return muon_pattern, compute_sharding_by_fqn, tuple(bucket_configs)
