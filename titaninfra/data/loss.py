"""Resolve block loss weights and normalize cross-entropy losses.

Weight precedence is block.weight, then type_overrides[block.type], then the
role default (assistant=1, all others=0), multiplied by message weight.
History policies are configured on StreamConfig rather than this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from titaninfra.data.schema import Block, LossConfig

# Hard-coded role defaults; LossConfig no longer owns these values.
_ROLE_DEFAULTS: dict[str, float] = {
    "system": 0, "user": 0, "assistant": 1, "tool": 0,
}


def resolve_weight(
    block: Block,
    role: str,
    loss_config: LossConfig,
    msg_weight: float | None = None,
) -> float:
    """Resolve a block's final loss weight.

    Precedence is explicit block.weight, then loss_config.type_overrides,
    then the role default. The result is base_weight * msg_weight.
    """
    # Resolve the base weight by precedence.
    if block.weight is not None:
        base = block.weight
    elif _block_type(block) in loss_config.type_overrides:
        base = loss_config.type_overrides[_block_type(block)]
    else:
        base = _ROLE_DEFAULTS.get(role, 0.0)

    # Apply the message-level multiplier.
    if msg_weight is None:
        msg_weight = _ROLE_DEFAULTS.get(role, 0.0)

    return base * msg_weight


class _AuxLossMixin:
    """Adds an optional MoE auxiliary-loss term to a loss_fn's returned sum.

    The aux source (MoEAuxLossPlugin) returns an already-scaled, grad-carrying
    scalar from compute_microbatch_aux_loss(); we just add it. No-op when unset.
    """

    _aux_source = None

    def set_aux_loss_source(self, aux_source) -> None:
        self._aux_source = aux_source

    def _add_aux(self, loss_sum):
        if self._aux_source is None:
            return loss_sum
        return loss_sum + self._aux_source.compute_microbatch_aux_loss()


def _normalize_by_gvt(loss_sum, global_loss_token_counts):
    """Normalize by global valid token count when it is supplied.

    Without a count, return the original sum for local/non-distributed use.
    Auxiliary loss is already scaled by the same count before this function.
    """
    if global_loss_token_counts is not None:
        return loss_sum / global_loss_token_counts
    return loss_sum


class WeightedCELoss(_AuxLossMixin):
    """Standard CE with optional per-token loss_weights support.

    Wraps torchtitan's default CE loss to apply thinking scale weights.
    When no weights are passed, behaves identically to the default CE.

    Contract: return (scaled_loss, {}). Compute an unnormalized sum, add any
    auxiliary term, and divide once by global valid token count.

    loss_weights is passed per microbatch through loss_kwargs rather than held
    as mutable external state. This prevents prefetched gradient-accumulation
    microbatches from overwriting one another's weights.
    """

    def __init__(self, inner_loss_fn):
        self._inner = inner_loss_fn

    def __call__(self, pred, labels, global_loss_token_counts=None, loss_weights=None, **loss_inputs):
        if loss_weights is None:
            # Request the inner unnormalized sum so this wrapper can add aux
            # and perform normalization exactly once.
            inner_out = self._inner(pred, labels, None, **loss_inputs)
            raw_sum = inner_out[0] if isinstance(inner_out, tuple) else inner_out
            loss_sum = self._add_aux(raw_sum)
            return _normalize_by_gvt(loss_sum, global_loss_token_counts), {}

        import torch.nn.functional as F

        per_token = F.cross_entropy(
            pred.float(),
            labels,
            reduction="none",
            ignore_index=-100,
        )
        valid_mask = (labels != -100).to(per_token.dtype)
        weighted = per_token * valid_mask * loss_weights
        loss_sum = self._add_aux(weighted.sum())
        return _normalize_by_gvt(loss_sum, global_loss_token_counts), {}


def _block_type(block: Block) -> str:
    """Map block classes to their string type tags."""
    from titaninfra.data.schema import (
        ImageBlock,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    _TYPE_MAP = {
        TextBlock: "text",
        ThinkingBlock: "thinking",
        ToolUseBlock: "tool_use",
        ToolResultBlock: "tool_result",
        ImageBlock: "image",
    }
    return _TYPE_MAP[type(block)]
