"""EP-aware FSDP for the Gemma4 MoE routed-expert weights.

``Gemma4MoE`` (gemma4/moe.py) is a self-owned MoE implementation (GELU
activation, dual router/expert input, HF-aligned custom router math). It
does NOT subclass ``torchtitan.models.common.moe.moe.MoE`` -- see that
module's docstring for why reusing the common MoE/RoutedExperts classes is a
poor fit (``Gemma4Experts.w1/w2/w3`` are raw ``nn.Parameter``, not a fused
``GroupedLinear`` ``w13``; changing that would break
``tests/test_gemma4_fqn_regression.py``'s DCP parameter FQNs).

Because of this, ``torchtitan.distributed.fsdp.apply_fsdp_to_decoder``'s own
``isinstance(m, MoE)`` per-block MoE detection never recognizes a gemma4
block's ``.moe`` submodule, so it always takes its dense/non-EP branch --
which shards ``Gemma4Experts.w1/w2/w3`` with a plain ``Shard(0)`` on
``dp_mesh``. That is wrong whenever EP > 1: those parameters are already
constructed (``sharding_config.state_shardings``,
``routed_expert_weight_placement(enable_ep=True)``) as DTensors replicated on
the dp_replicate/edp_shard axes but ``Shard(0)`` on the EP axis, and FSDP2
rejects the conflicting pre-existing placement with "Expected spmd.R on FSDP
DP axis ... but got S(0)".

``apply_fsdp_to_gemma4_experts`` gives each block's ``moe.experts`` its own
``fully_shard()`` call on the sparse (edp) mesh BEFORE ``Gemma4Model``'s
inherited ``Decoder._apply_fsdp`` processes the surrounding block. FSDP2's
own managed-module discovery
(``torch.distributed.fsdp._fully_shard._fsdp_init._get_managed_modules``)
stops recursing into any submodule that already carries FSDP state from a
prior ``fully_shard`` call (confirmed directly against the installed
PyTorch's ``_fsdp_init.py`` source -- the same nested-FSDP-unit composition
``apply_fsdp_to_decoder`` itself already relies on for ``tok_embeddings`` /
``[norm, lm_head]`` / each transformer block / the whole model, and the same
pattern ``apply_fsdp_to_multimodal_encoder`` documents for a vision encoder
wrapped ahead of the decoder, see ``kimi_vl/parallelize.py`` for the real,
GPU-verified precedent of this exact override-``_apply_fsdp`` structure).

Called from ``Gemma4Model._apply_fsdp`` (model.py), mirroring
``MultimodalModel._apply_fsdp`` / ``MTPDecoder._apply_fsdp`` / kimi_vl's
``KimiVLModel._apply_fsdp``: call the family-specific helper first, then
``super()._apply_fsdp(...)`` for the decoder's own dense/EP-oblivious FSDP
pass. Only invoked when ``parallelism_context.ep_enabled``; at EP degree 1
``resolve_sparse_fsdp_mesh`` returns ``(None, None)`` and the inherited
``Decoder._apply_fsdp`` dense branch is already correct, so this module adds
no behavior there.
"""

from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import Shard
from torchtitan.config import TORCH_DTYPE_MAP, PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
    linear_param_shard_placements,
    mixed_precision_policy,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.distributed.parallelism_context import ParallelismContext

from .model import Gemma4Model


def apply_fsdp_to_gemma4_experts(
    model: Gemma4Model,
    *,
    parallelism_context: ParallelismContext,
    precision: PrecisionConfig,
    parallelism: ParallelismConfig,
) -> None:
    """``Gemma4Model._apply_fsdp`` body: EP-aware experts FSDP before decoder FSDP."""
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
    if edp_mesh is None:
        return  # EP disabled: the inherited dense Decoder._apply_fsdp is already correct.

    mp_policy = mixed_precision_policy(
        model,
        param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
        reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallelism_context.pp_enabled
    )
    expert_fsdp_config: dict = {"mesh": edp_mesh, "mp_policy": mp_policy}
    if edp_mesh_dims is not None:
        expert_fsdp_config["dp_mesh_dims"] = edp_mesh_dims
    if parallelism.fsdp_enable_cpu_offloading:
        expert_fsdp_config["offload_policy"] = CPUOffloadPolicy()

    ep_degree = parallelism_context.ep
    expert_sharding_size = edp_mesh["edp_shard"].size() * ep_degree

    for transformer_block in model.layers.values():
        experts = transformer_block.moe.experts
        num_experts = experts.num_experts
        expert_param_placements = {p: Shard(0) for p in experts.parameters()}
        if expert_sharding_size > num_experts:
            # Mirrors apply_fsdp_to_decoder's own overflow fallback, but
            # Gemma4Experts' w1/w2/w3 are raw nn.Parameters (not
            # GroupedLinear-wrapped), so linear_param_shard_placements finds
            # nothing to add here -- a latent gap at topologies where
            # edp_shard_size * ep_degree exceeds num_experts. Not reached by
            # this family's debugmodel (num_experts=16) or 26B-A4B
            # (num_experts=128) flavors at any topology on an 8-GPU node;
            # flagged for follow-up, not fixed here.
            expert_param_placements.update(
                linear_param_shard_placements(experts, include_unstacked_grouped=True)
            )
        fully_shard(
            experts,
            **expert_fsdp_config,
            reshard_after_forward=reshard_after_forward,
            shard_placement_fn=expert_param_placements.get,
        )
