# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DeepSeek V4 parallelization (FSDP2 / EP / CP / compile).

Ported from ``titaninfra/models/deepseek_v3/parallelize.py`` — same FSDP2/EP/CP/
compile flow; only the model import and function name differ. Do NOT modify
torchtitan; all customization stays here.
"""

from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)

from titaninfra.distributed._tt_compat import validate_config
from titaninfra.models.deepseek_v4.model import DeepSeekV4Model

# DeepSeek-V4 blocks vary along three axes that Dynamo specializes on:
# router kind (hash vs learned), attention type (sliding / CSA / HCA), and the
# transient DTensor state (AsyncCollectiveTensor vs resolved Tensor under
# TP/EP). Across 27 layers this produces far more than the default 8 recompiles
# per compiled block, and fullgraph=True turns the cap into a HARD failure
# (FailOnRecompileLimitHit) instead of an eager fallback. Recompiling is only
# a one-time compile-latency cost — correctness/perf of the compiled path is
# unaffected — so lift both caps effectively out of the way.
# TODO: revisit once https://github.com/pytorch/pytorch/issues/187073 is fixed.
_DSV4_RECOMPILE_LIMIT = 256


def _raise_dynamo_recompile_limit() -> None:
    import torch

    # PyTorch types these as Literal[8]/int but runtime accepts larger ints.
    # pyrefly: ignore [bad-assignment]
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, _DSV4_RECOMPILE_LIMIT
    )
    # accumulated_recompile_limit is the global ceiling across all functions;
    # keep it at least as high as the per-function cap so it never bites first.
    # pyrefly: ignore [bad-assignment]
    torch._dynamo.config.accumulated_recompile_limit = max(
        torch._dynamo.config.accumulated_recompile_limit, _DSV4_RECOMPILE_LIMIT
    )


def parallelize_deepseekv4(
    model: DeepSeekV4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    validate_config(parallel_dims, model)
    model.parallelize(parallel_dims)

    model_compile_enabled = (
        compile_config is not None and "model" in compile_config.components
    )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if model_compile_enabled:
        _raise_dynamo_recompile_limit()
        apply_compile(model, compile_config=compile_config, parallel_dims=parallel_dims)

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
    )

    return model
