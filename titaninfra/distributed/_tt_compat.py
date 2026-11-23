"""Vendored compatibility helpers for removed TorchTitan APIs.

NoParallel replicates computation on a TP mesh without sharding.
validate_config rejects unsupported CP/SDPA/Varlen combinations.
apply_cp_to_forward preserves the older CP wrapper for legacy call sites.
Prefer upstream equivalents when they become available again.
"""

from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


class NoParallel(ParallelStyle):
    """Replicate computation on the TP mesh without sharding.

    This style does nothing other than:
    (1) setting the module parameters as DTensors on the given mesh, and
    (2) inserting hooks at module boundary to convert torch.Tensor to DTensor and back.

    The reason we need this wrapping is to ensure all parameters are on the same 1D/2D mesh,
    which is assumed by (1) gradient norm clipping, and (2) optimizer fused implementation.

    Used for modules like the MoE router gate that need replicated computation on TP mesh.
    """

    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = False,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layout: Placement | None,
        desired_input_layout: Placement | None,
        mod: nn.Module,
        inputs: Any,
        device_mesh: DeviceMesh,
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            assert input_layout is not None
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            assert input_layout is not None
            assert desired_input_layout is not None
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(
        output_layout: Placement,
        use_local_output: bool,
        mod: nn.Module,
        outputs: DTensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor | DTensor:
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn,
                self.input_layout,
                self.desired_input_layout,
            ),
            partial(
                self._prepare_output_fn,
                self.output_layout,
                self.use_local_output,
            ),
        )


def get_spmd_backend() -> str:
    """Return the active SPMD backend.

    Upstream removed the selectable SPMD backend in favor of spmd_types.
    Return that backend name for compatibility with older model code.
    """
    return "spmd_types"


def validate_config(parallel_dims, model: nn.Module) -> None:
    """Validate that the current configuration is compatible with multi-axis backends.

    Walks ``model`` to discover the actual attention modules in use and
    raises ``NotImplementedError`` with a clear message if incompatible.
    """
    from torchtitan.models.common.attention import (
        ScaledDotProductInnerAttention,
        VarlenInnerAttention,
    )

    if parallel_dims.cp_enabled:
        if any(
            isinstance(m, (ScaledDotProductInnerAttention, VarlenInnerAttention))
            for m in model.modules()
        ):
            raise NotImplementedError(
                f"{parallel_dims.spmd_backend} + CP is not supported with "
                "ScaledDotProductInnerAttention or VarlenInnerAttention. "
                "Use FlexInnerAttention + CP or disable CP."
            )


def apply_cp_to_forward(attention_modules, cp_mesh) -> None:
    """Wrap inner attention ``forward`` with CP logic.

    Wrap inner-attention forward with the legacy CP implementation.

    Call before Module.parallelize(). New code should prefer upstream's
    declarative CP sharding instead of this compatibility workaround.
    """
    import logging

    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Shard
    from torchtitan.models.common.attention import (
        FlexInnerAttention,
        ScaledDotProductInnerAttention,
        VarlenInnerAttention,
    )

    log = logging.getLogger(__name__)
    first = attention_modules[0]
    if isinstance(first, FlexInnerAttention):
        from torch.distributed.tensor.experimental._context_parallel._attention import (
            flex_cp_allgather,
        )

        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                pg_name = dist._get_process_group_name(mesh.get_group())

                def cp_forward(q, k, v, **kwargs):
                    if kwargs.get("score_mod") is not None:
                        raise NotImplementedError(
                            "FlexAttention score_mod is not supported with "
                            "Context Parallel yet. It must be sharded before "
                            "use with CP."
                        )
                    k2 = k.contiguous()
                    v2 = v.contiguous()
                    global_k, global_v = flex_cp_allgather(k2, v2, 1, pg_name)
                    return orig_fn(q, global_k, global_v, **kwargs)

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, ScaledDotProductInnerAttention):
        from torch.distributed.tensor.experimental._attention import (
            _enable_context_parallel_dispatcher,
        )

        _enable_context_parallel_dispatcher()

        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                placement = [Shard(1)]

                def cp_forward(q, k, v, **kwargs):
                    if not isinstance(q, DTensor):
                        q = DTensor.from_local(q, mesh, placement, run_check=False)
                    if not isinstance(k, DTensor):
                        k = DTensor.from_local(k, mesh, placement, run_check=False)
                    if not isinstance(v, DTensor):
                        v = DTensor.from_local(v, mesh, placement, run_check=False)
                    output = orig_fn(q, k, v, **kwargs)
                    return output.to_local() if isinstance(output, DTensor) else output

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, VarlenInnerAttention):
        raise NotImplementedError("Variable-length attention CP is not yet supported")
    else:
        raise NotImplementedError(
            f"Context Parallel forward wrapping is not supported for "
            f"{type(first).__name__}"
        )

    log.info("Applied Context Parallel (forward wrapping) to the model")
