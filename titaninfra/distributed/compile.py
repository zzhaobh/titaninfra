"""
MoE-safe per-submodule torch.compile for titaninfra.

Compiles each TransformerBlock's submodules individually, automatically
skipping MoE modules that use DeepEP AllToAll (incompatible with Dynamo
fullgraph). When expert parameters have been converted to FP8 via torchao's
Float8GroupedMMConverter, the MoE grouped GEMM ops ARE Dynamo-traceable and
REQUIRE compile for kernel fusion — in that case, compile is applied.

Compatibility note: upstream now uses LocalCompileConfig for operator-level
compile regions, which is a different concept from this module's per-submodule
backend selection. TitanInfra therefore owns a minimal CompileConfig with only
backend and enable_async_tensor_parallel; it is not a renamed upstream type.
The GPU-only call sites that construct this config require separate validation.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torchtitan.models.common.moe import MoE

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CompileConfig:
    """titaninfra-owned minimal compile-config shape (NOT torchtitan's --
    see module docstring). Carries exactly the 2 fields apply_compile_
    moe_safe/maybe_enable_async_tp read; callers (parallelize.py across
    the migrated families) are responsible for constructing one of these
    -- a GPU-only wiring concern flagged for dedicated follow-up, not
    solved here.
    """

    backend: str = "inductor"
    enable_async_tensor_parallel: bool = False

# Workaround for PyTorch issue #178887: FakeTensorMode.__init__ triggers
# recompilation when tracing. Same patch as torchtitan/distributed/compile.py,
# replicated here so titaninfra doesn't depend on that module's import side-effect.
FakeTensorMode.__init__ = torch.compiler.disable(  # type: ignore[method-assign]
    FakeTensorMode.__init__, recursive=True
)


def _is_fp8_quantized(module: nn.Module) -> bool:
    """Check if module parameters have been converted to ScaledGroupedMMTensor by torchao."""
    try:
        from torchao.prototype.moe_training.scaled_grouped_mm import (
            ScaledGroupedMMTensor,
        )
    except ImportError:
        return False
    return any(isinstance(p, ScaledGroupedMMTensor) for p in module.parameters())


def _unwrap_checkpoint(block: nn.Module) -> nn.Module:
    """Unwrap CheckpointWrapper to get the inner TransformerBlock.

    After AC, model.layers contains CheckpointWrapper instances whose sole
    child is `_checkpoint_wrapped_module` (the original TransformerBlock).
    We need the inner block so we can compile its children individually.
    """
    inner = getattr(block, "_checkpoint_wrapped_module", None)
    return inner if inner is not None else block


def maybe_enable_async_tp(compile_config: CompileConfig, tp_mesh) -> None:
    """Enable Inductor async tensor parallelism when configured.

    This mirrors upstream's opt-in symmetric-memory setup. It is a no-op when
    async TP is disabled or tp_mesh is None.
    """
    if not getattr(compile_config, "enable_async_tensor_parallel", False) or tp_mesh is None:
        return

    import warnings

    group_name = tp_mesh.get_group().group_name
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        enable_symm_mem_for_group(group_name)

    torch._inductor.config._micro_pipeline_tp = True
    logger.info("Async TP is enabled (titaninfra moe-safe path)")


def apply_compile_moe_safe(
    model: nn.Module, compile_config: CompileConfig, tp_mesh=None
) -> None:
    """Per-submodule compile that skips MoE unless FP8-quantized.

    For each TransformerBlock, compiles attention, norms, and dense FFN
    submodules individually. MoE submodules (DeepEP AllToAll) are skipped
    by default. When torchao Float8GroupedMMConverter has already converted
    expert parameters to ScaledGroupedMMTensor, the MoE grouped GEMM ops
    are Dynamo-traceable and require compile for FP8 fusion performance.

    Handles the AC + compile ordering: after AC wraps blocks in
    CheckpointWrapper, we unwrap to reach the actual TransformerBlock
    and compile its direct children (not the whole block).

    Pass the dense TP mesh to enable async TP when applicable.
    """
    maybe_enable_async_tp(compile_config, tp_mesh)

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    compiled_count = 0
    skipped_count = 0

    for _, block in model.layers.named_children():
        inner_block = _unwrap_checkpoint(block)
        for name, submodule in inner_block.named_children():
            # Most families route MoE through feed_forward, while gemma4 also
            # has a separate moe module. Check both forms before skipping.
            is_moe = isinstance(submodule, MoE) or name == "moe"
            if is_moe and not _is_fp8_quantized(submodule):
                skipped_count += 1
                continue
            submodule.compile(backend=compile_config.backend, fullgraph=False)
            compiled_count += 1

    logger.info(
        f"MoE-safe compile: {compiled_count} submodules compiled, "
        f"{skipped_count} MoE modules skipped (backend={compile_config.backend})"
    )
