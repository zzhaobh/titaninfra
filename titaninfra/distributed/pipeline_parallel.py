"""
TitanInfra pipeline parallelism — wraps torchtitan's PP with backward-compat fixes.

Fixes vs upstream torchtitan:
  - PipelineStage `get_mesh` parameter: added in newer PyTorch for DTensor PP,
    not available on older versions. This module gracefully skips it when absent.
  - `_MeshLayout` import: may not exist in older torch.distributed versions.

All other PP functions (build_pipeline_schedule, generate_llm_fqn_per_model_part)
are re-exported from torchtitan unchanged.
"""

import copy
import inspect
import math
from collections.abc import Callable

import torch
import torch.nn as nn

try:
    from torch.distributed._mesh_layout import _MeshLayout  # noqa: F401
except ImportError:
    pass
import logging

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleSingle,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    _PipelineSchedule,
    get_schedule_class,
)
from torchtitan.components.loss import LossFunction
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline_parallel import (
    _build_pipeline_schedule as build_pipeline_schedule,
)
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part as generate_llm_fqn_per_model_part,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.protocols.module import ModuleDict, ModuleList

logger = logging.getLogger(__name__)

__all__ = [
    "pipeline_llm",
    "build_pipeline_schedule",
    "generate_llm_fqn_per_model_part",
    "pipeline_module_split",
]


def _build_get_mesh_callback(
    parallel_dims: ParallelDims,
) -> Callable:
    """Build a callback that resolves a DeviceMesh from dimension names.

    Backward-compatible: gracefully handles older PyTorch where _MeshLayout
    doesn't exist or mesh._layout is unavailable.
    """

    def _get_mesh(
        mesh_dim_names: tuple[str, ...], mesh_layout=None
    ) -> DeviceMesh | None:
        mesh = parallel_dims.get_mesh(list(mesh_dim_names))
        if mesh_layout is not None and hasattr(mesh, "_layout") and mesh._layout != mesh_layout:
            return None
        return mesh

    return _get_mesh


def pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters=None,  # list[ModelConfigConverter.Config] | None
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config,  # ActivationCheckpointingConfig | None (torchtitan policy union)
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.get_mesh("pp")

    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    if hasattr(model_config, "layers"):
        num_layers = len(model_config.layers)
    else:
        raise ValueError("Model does not have n_layers attribute.")

    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers

    if layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )

        if num_virtual_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
                f"pipeline parallel size ({parallel_dims.pp})."
            )

        stages_per_rank = num_virtual_stages // parallel_dims.pp

        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single stage schedule requires exactly 1 stage per rank, "
                f"but got {stages_per_rank} stages per rank."
            )

        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank, "
                f"but got {stages_per_rank} stages per rank."
            )
    else:
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
        get_mesh=get_mesh_cb,
    )

    for i, m in enumerate(model_parts):
        m = parallelize_fn(
            m,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        model_parts[i] = m
        stages[i].submod = m

    pp_schedule = build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=stages,
        loss_fn=loss_fn,
    )

    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
    get_mesh: Callable | None = None,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """Split model into PP stages. Backward-compatible with older PyTorch
    that lacks PipelineStage's get_mesh parameter."""
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()

    def _build_stage_from_modules(
        stage_idx: int, module_names: list[str], num_stages: int
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)

        modules_to_keep = set(module_names)
        for module_name, module_value in model.named_children():
            if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }
                if layers_to_keep:
                    if isinstance(module_value, nn.ModuleDict):
                        for layer_name in list(module_value.keys()):
                            if layer_name not in layers_to_keep:
                                del module_value[layer_name]
                    elif isinstance(module_value, nn.ModuleList):
                        indices_to_keep = {
                            int(idx) for idx in layers_to_keep if idx.isdigit()
                        }
                        new_layers = ModuleList(
                            [
                                layer
                                for i, layer in enumerate(module_value)
                                if i in indices_to_keep
                            ]
                        )
                        setattr(model, module_name, new_layers)
                else:
                    if isinstance(module_value, nn.ModuleDict):
                        setattr(model, module_name, ModuleDict())
                    elif isinstance(module_value, nn.ModuleList):
                        setattr(model, module_name, ModuleList())
            elif module_name not in modules_to_keep:
                setattr(model, module_name, None)

        kwargs = dict(group=pp_mesh.get_group("pp"))
        if (
            get_mesh is not None
            and "get_mesh" in inspect.signature(PipelineStage.__init__).parameters
        ):
            kwargs["get_mesh"] = get_mesh
        stage = PipelineStage(model, stage_idx, num_stages, device, **kwargs)
        return stage, model

    num_stages = len(module_names_per_stage)
    stages = []
    models = []

    schedule_class = get_schedule_class(pp_schedule)
    style = (
        "v" if schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV) else "loop"
    )

    def _get_stage_indices() -> tuple[int, ...]:
        assert (
            num_stages % pp_degree == 0
        ), f"num_stages {num_stages} must be evenly divisible by pp_degree {pp_degree}"
        stages_per_rank = num_stages // pp_degree
        if style == "loop":
            return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
        elif style == "v":
            assert (
                stages_per_rank == 2
            ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
            stage_v_pairs = list(
                zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1), strict=False)
            )
            return stage_v_pairs[pp_rank]
        else:
            raise ValueError(f"Unknown style {style}")

    for stage_idx in _get_stage_indices():
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            stage_idx,
            module_names,
            num_stages,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models
