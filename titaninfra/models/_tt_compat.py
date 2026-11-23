"""Compatibility helpers for TorchTitan model construction.

get_moe_model_nparams_and_flops preserves the older parameter/FLOP estimator.
make_token_dispatcher_config is a thin construct-time factory for upstream
AllToAll, DeepEP, and HybridEP dispatcher configs, needed before a complete
Trainer.Config exists for config-tree transforms. It contains no custom
communication or scheduling logic.
"""

import logging

import torch.nn as nn
from torchtitan.models.common.decoder import Decoder

logger = logging.getLogger(__name__)


def make_token_dispatcher_config(
    *,
    num_experts: int,
    top_k: int,
    comm_backend: str,
    hidden_dim: int,
    num_max_tokens_per_rank: int | None = None,
    non_blocking_capacity_factor: float | None = None,
):
    """Build a token-dispatcher ``Config`` from titaninfra's ``ep_backend`` string.

    ``comm_backend`` maps 1:1 onto ``ModelConfig.ep_backend``'s three
    ``Literal`` values; every branch constructs an upstream dispatcher class
    verbatim (no titaninfra-owned dispatch logic):

    - ``"standard"``: ``AllToAllTokenDispatcher`` — plain NCCL all-to-all,
      no RDMA/GIN hardware requirement. Default for EP>1 without DeepEP.
    - ``"deepep"``: ``DeepEPTokenDispatcher`` — DeepEP v2's unified
      ``ElasticBuffer``. Requires NCCL GIN hardware and
      ``num_max_tokens_per_rank`` (hard per-rank buffer-size bound; the
      caller must compute it from the actual training topology — model
      construction time does not know batch size/seq_len, so callers at
      that layer must pass a value computed elsewhere, or this raises).
    - ``"hybridep"``: ``HybridEPTokenDispatcher`` — same buffer-sizing
      requirement as DeepEP, plus the optional
      ``non_blocking_capacity_factor`` knob.

    Raises ``ValueError`` for an unrecognized ``comm_backend`` string (no
    silent fallback to a different backend — a typo must not quietly
    degrade a training run's communication pattern).
    """
    from torchtitan.models.common.moe.token_dispatcher import (
        AllToAllTokenDispatcher,
        DeepEPTokenDispatcher,
        HybridEPTokenDispatcher,
    )

    if comm_backend == "standard":
        return AllToAllTokenDispatcher.Config(num_experts=num_experts, top_k=top_k)
    if comm_backend == "deepep":
        if num_max_tokens_per_rank is None:
            raise ValueError(
                "ep_backend='deepep' requires num_max_tokens_per_rank (DeepEP "
                "preallocates its communication buffer at construction time). "
                "This must be computed from the actual training topology "
                "(local_batch_size * seq_len, or tighter with CP/TP division) "
                "and threaded through by the caller — model-definition time "
                "alone cannot know it."
            )
        return DeepEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
        )
    if comm_backend == "hybridep":
        if num_max_tokens_per_rank is None:
            raise ValueError(
                "ep_backend='hybridep' requires num_max_tokens_per_rank for "
                "the same reason as 'deepep' (see that branch's error)."
            )
        return HybridEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        )
    raise ValueError(
        f"Unknown comm_backend={comm_backend!r}; expected one of "
        "'standard', 'deepep', 'hybridep' (ModelConfig.ep_backend's Literal "
        "values)."
    )


def get_moe_model_nparams_and_flops(
    model_config: Decoder.Config,
    model: nn.Module,
    n_heads: int,
    head_dims: int,
    seq_len: int,
) -> tuple[int, int]:
    """
    Calculate nparams and nflops for MoE models.

    Args:
        model_config: Decoder.Config object containing model configuration parameters including MoE settings.
        model: nn.Module representing the MoE model.
        n_heads: The number of attention heads.
        head_dims: The sum of qk and v head dimensions.
        seq_len: The sequence length in training configs.

    Returns:
        Tuple of (nparams, num_flops_per_token):
            nparams: Total number of model parameters including all experts.
            num_flops_per_token: Estimated number of floating point operations per token
                                based on active parameters only.
    """
    nparams_embedding = 0
    nparams_moe_router = 0
    nparams_shared_experts = 0
    nparams_experts = 0
    nparams_dense = 0
    nparams_vision = 0

    for name, p in model.named_parameters():
        if "vision_encoder" in name:
            nparams_vision += p.numel()
        elif "embedding" in name:
            nparams_embedding += p.numel()
            nparams_dense += p.numel()
        elif "moe.shared_experts" in name:
            nparams_shared_experts += p.numel()
        elif "moe.router" in name:
            nparams_moe_router += p.numel()
        elif "moe.routed_experts" in name:
            nparams_experts += p.numel()
        else:
            nparams_dense += p.numel()

    nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
    nparams = nparams_dense + nparams_sparse + nparams_vision

    moe_config = next((layer.moe for layer in model_config.layers if layer.moe is not None), None)
    if moe_config is not None:
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_experts
            + nparams_experts * moe_config.router.top_k // moe_config.num_experts
        )
    else:
        nparams_sparse_active = 0

    logger.info(
        f"Total parameter count: dense {nparams_dense:,}, "
        f"sparse {nparams_sparse:,}, vision {nparams_vision:,}, "
        f"active {nparams_dense + nparams_sparse_active:,}"
    )

    # With tied embeddings, PyTorch's parameter iterator already counts the
    # shared input/output parameter once. That parameter still participates in
    # the lm_head matmul, so do not subtract it for either size or FLOPs.
    if getattr(model_config, "enable_weight_tying", False):
        nparams_for_matmul = nparams_dense + nparams_sparse_active
    else:
        nparams_for_matmul = nparams_dense - nparams_embedding + nparams_sparse_active
    # Only full attention layers contribute the quadratic O(L²) FLOPs
    # term. Hybrid models mix full attention with linear attention
    # layers whose block leaves ``attention=None``; standard decoders carry
    # full attention on every layer, so this counts all of them.
    num_full_attn = sum(
        1 for layer in model_config.layers if getattr(layer, "attention", None) is not None
    )
    num_flops_per_token = (
        6 * nparams_for_matmul + 6 * num_full_attn * n_heads * head_dims * seq_len
    )

    return nparams, num_flops_per_token


def resolve_max_context_length(config) -> int:
    """Resolve the value ``ModelSpec.max_context_length`` needs from a built
    ``Decoder.Config`` (or family subclass).

    upstream #5277 added ``max_context_length`` as a required ``ModelSpec``
    field (previously absent); titaninfra's 7 family registries construct
    ``ModelSpec`` without it. The RoPE cache length lives on
    ``first_attention.rope`` for 6 of 7 families (gemma4 keeps its own
    top-level ``max_seq_len`` since it carries two RoPE caches — sliding +
    full — with no single ``Decoder.Config.rope`` field to delegate to).
    """
    top_level = getattr(config, "max_seq_len", None)
    if top_level is not None:
        return top_level
    attn = config.first_attention
    if attn is not None and getattr(attn, "rope", None) is not None:
        return attn.rope.max_context_length
    raise ValueError(
        f"Cannot resolve max_context_length for {type(config).__name__}: "
        "no top-level max_seq_len and no layer carries attention.rope."
    )
