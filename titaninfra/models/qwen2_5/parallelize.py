"""Legacy Qwen2.5 parallelization helpers based on stable FSDP2 APIs.

The active runtime entry point is BaseModel.parallelize() rather than this
legacy free-standing wrapper.
"""

import logging

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)

from titaninfra.distributed._converter_compat import find_float8_linear_config
from titaninfra.distributed._tt_compat import NoParallel, apply_cp_to_forward
from titaninfra.distributed.compile import apply_compile_moe_safe

from .model import Qwen2_5Model

logger = logging.getLogger(__name__)


def parallelize_qwen2_5(
    model: Qwen2_5Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters=None,  # list[ModelConfigConverter.Config] | None (converters)
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config,  # ActivationCheckpointingConfig | None (torchtitan policy union)
    dump_folder: str,
):
    """
    Apply parallelization to Qwen2.5 (Dense only, no MoE).

    Processing order:
    1. Tensor Parallelism (TP)
    2. Context Parallelism (CP)
    3. Activation Checkpointing (AC)
    4. torch.compile
    5. FSDP/HSDP
    """
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    model_compile_enabled = (
        compile_config is not None and "model" in compile_config.components
    )

    # 1. Tensor Parallelism (TP)
    if parallel_dims.tp_enabled:
        if (
            compile_config is not None
            and compile_config.enable_async_tensor_parallel
            and not model_compile_enabled
        ):
            raise RuntimeError("Async TP requires torch.compile")

        float8_config = find_float8_linear_config(model_converters or [])
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=compile_config is not None and compile_config.enable_async_tensor_parallel,
            enable_cp=parallel_dims.cp_enabled,
            enable_sp=parallelism.enable_sequence_parallel,
        )

    # 2. Context Parallelism (CP)
    if parallel_dims.cp_enabled:
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    # 3. Activation Checkpointing (AC)
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    # 4. torch.compile
    if model_compile_enabled:
        apply_compile_moe_safe(
            model,
            compile_config,
            tp_mesh=parallel_dims.get_dense_tp_mesh() if parallel_dims.tp_enabled else None,
        )

    # 5. FSDP/HSDP
    if parallel_dims.fsdp_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to Qwen2.5 model")
        else:
            logger.info("Applied FSDP to Qwen2.5 model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to Qwen2.5 model")

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
    enable_cp: bool,
    enable_sp: bool = True,
):
    """Apply Tensor Parallelism to Qwen2.5 (no qk_norm, no MoE)."""
    sp_layout = Shard(1) if enable_sp else Replicate()
    embed_plan = RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=sp_layout,
        use_local_output=False,
    )

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": embed_plan,
            "norm": SequenceParallel(use_local_output=False) if enable_sp else NoParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    positions_sharding = Replicate() if enable_cp else None
    norm_plan = SequenceParallel(use_local_output=False) if enable_sp else NoParallel()
    rowwise_output_plan = rowwise_parallel(
        output_layouts=sp_layout, use_local_output=False
    )

    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": prepare_module_input(
                input_layouts=(sp_layout, Replicate(), None, positions_sharding),
                desired_input_layouts=(
                    Replicate(),
                    Replicate(),
                    None,
                    positions_sharding,
                ),
            ),
            "attention.qkv_linear.wq": colwise_parallel(use_local_output=False),
            "attention.qkv_linear.wk": colwise_parallel(use_local_output=False),
            "attention.qkv_linear.wv": colwise_parallel(use_local_output=False),
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
            "feed_forward": prepare_module_input(
                input_layouts=(sp_layout,),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(use_local_output=False),
            "feed_forward.w2": rowwise_output_plan,
            "feed_forward.w3": colwise_parallel(use_local_output=False),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        f"{'Async ' if enable_async_tp else ''}"
        f"Tensor Parallelism to Qwen2.5 model"
    )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """Apply standard FSDP2 to Qwen2.5 (Dense only, no MoE EP)."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False,
    )
    fsdp_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled,
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings, **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.norm is not None and model.lm_head is not None:
        fully_shard(
            [model.norm, model.lm_head], **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    for transformer_block in model.layers.values():
        fully_shard(
            transformer_block, **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
