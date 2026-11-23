# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Qwen3.5 hybrid attention model.

Declarations only: ``__init__.py`` sets these on each sub-config as it builds
it (construction-time, inline ``sharding_config=``), and ``model.parallelize()``
applies them through the Module protocol. Same pattern as
``deepseek_v3/sharding.py`` and upstream's own
``torchtitan/models/qwen3_5/sharding.py`` (this file is a near-verbatim port
of that upstream module).

2026-11-19: rewritten from scratch. The previous version of this file used a
"construct bare Config, then mutate ``cfg.sharding_config = ...`` after the
fact via ``set_qwen35_sharding_config()``" pattern, imported from a module
path (``torchtitan.protocols.sharding``) that no longer exists, and had zero
callers anywhere in the repo (confirmed via repo-wide grep before deletion).
``__init__.py`` never called any of it — the entire qwen3.5 family (text and
vision) had zero ``sharding_config`` wiring, which only surfaces as a crash
the first time a real ``Trainer()`` actually tries to construct+FSDP-wrap a
qwen3_5 model (``ValueError: When dp_mesh_dims is provided, all parameters
must be DTensors...``) — real GPU VLM pretrain cases (case_37/38) never got
this far before (blocked by earlier, since-fixed bugs: missing lance fixture,
missing ``build_flops_estimator``, wrong ``local_compile`` region name).

Full-attention layers: TP on wq/wk/wv/wo with local SPMD inner attention;
each layer's MRoPE ``cache`` buffer is sharded Replicate.
GatedDeltaNet layers: head-sharded TP on projections (ColwiseParallel) and
out_proj (RowwiseParallel); the GDN kernel and depthwise Conv1d run on local
tensors in a local SPMD region.
"""

import spmd_types as spmd
from spmd_types import SpmdType
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_activation_placement,
    dense_param_placement,
    replicated_activation_placement,
    residual_activation_placement,
    stacked_colwise_sharding_config,
)

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def transformer_block_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Transformer block boundary in the residual layout."""
    layer_layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        input_shardings={"x_TD": layer_layout},
        output_shardings=layer_layout,
    )


def decoder_norm_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Pre-attention norm: Replicate weight, residual-layout activations."""
    activation_layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        input_shardings={"input": activation_layout},
        output_shardings=activation_layout,
    )


def attention_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Full-attention boundary: ``x_TD`` in and out in the residual layout."""
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        input_shardings={"x_TD": layout},
        output_shardings=layout,
    )


def conv_weight_sharding_config() -> ShardingConfig:
    """Depthwise Conv1d weight sharded Shard(0) on out-channels (head-sharded)."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )


def gated_rms_norm_sharding_config() -> ShardingConfig:
    """GatedDeltaNet output norm over head-sharded ``[T, H, V]`` activations."""
    head_placement = attention_activation_placement()
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        input_shardings={
            "input": head_placement,
            "gate": head_placement,
        },
        output_shardings=head_placement,
    )


def inner_gated_delta_net_sharding_config() -> ShardingConfig:
    """Local SPMD boundary for the head-parallel convolution and recurrence.

    The projections are 2D [T, C], while the recurrence output is 3D
    [T, H, V]. Both shard the feature/head axis on TP.
    """
    projected_placement = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    head_placement = attention_activation_placement()
    parameter_placement = dense_param_placement(tp=spmd.S(0))
    input_shardings = {
        # Nominal tp S(1): each rank holds its own [q | k | v]; never redistribute or gather it.
        "mixed_qkv_TC": projected_placement,
        "a_TH": projected_placement,
        "b_TH": projected_placement,
        "conv_q_weight_C1W": parameter_placement,
        "conv_k_weight_C1W": parameter_placement,
        "conv_v_weight_C1W": parameter_placement,
        "A_log_H": parameter_placement,
        "dt_bias_H": parameter_placement,
    }
    return ShardingConfig(
        input_shardings=input_shardings,
        output_shardings=head_placement,
        local_spmd=True,
    )


def gated_delta_net_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """GatedDeltaNet boundary with head-sharded ``A_log``/``dt_bias``."""
    layout = residual_activation_placement(enable_sp=enable_sp)
    parameter_placement = dense_param_placement(tp=spmd.S(0))
    return ShardingConfig(
        state_shardings={
            "A_log": parameter_placement,
            "dt_bias": parameter_placement,
        },
        input_shardings={"x_TD": layout},
        output_shardings=layout,
    )


def sigmoid_gated_w13_sharding_config() -> ShardingConfig:
    """Shared-expert ``w13`` over the input the shared FFN gathers once."""
    return stacked_colwise_sharding_config(
        input_layout=replicated_activation_placement()
    )


def sigmoid_gate_sharding_config() -> ShardingConfig:
    """Shared-expert sigmoid gate: Replicate weights and activations."""
    replicated = replicated_activation_placement()
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
        input_shardings={"input": replicated},
        output_shardings=replicated,
    )


def vision_encoder_sharding_config() -> ShardingConfig:
    """Vision encoder root: invariant ``pos_embed`` and output.

    All activations flow without SP in the vision encoder. The model converts
    the output to TP- and CP-``Replicate`` where the decoder consumes it.
    """
    return ShardingConfig(
        state_shardings={"pos_embed": SpmdType({DP: spmd.R, CP: spmd.I, TP: spmd.I})},
        output_shardings=SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.I}),
    )


def vision_merger_sharding_config() -> ShardingConfig:
    """Merger output: TP- and CP-invariant."""
    return ShardingConfig(
        output_shardings=SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.I}),
    )
