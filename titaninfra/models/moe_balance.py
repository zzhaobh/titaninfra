"""Family-level declarations for MoE training-time load balancing.

Balancing is an architectural property, not an arbitrary ModelConfig switch.
Families with both bias_gamma and aux_coeff set to None declare no balancing
and register no expert_bias buffer.

Two orthogonal mechanisms are supported: aux-loss-free expert-bias feedback
and auxiliary-loss gradients, with sequence or global scope. DeepSeek V3/V4
can enable bias feedback and sequence-scoped auxiliary loss together.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoEBalanceSpec:
    """Immutable load-balancing declaration for one MoE family."""

    bias_gamma: float | None = None
    aux_coeff: float | None = None
    aux_scope: Literal["sequence", "global"] = "sequence"


_REGISTRY: dict[str, MoEBalanceSpec] = {}


def register_moe_balance(family: str, spec: MoEBalanceSpec) -> None:
    """Register a family specification and reject duplicate registration."""
    if family in _REGISTRY:
        raise ValueError(f"MoE balance spec for family {family!r} already registered")
    _REGISTRY[family] = spec


def get_moe_balance(family: str) -> MoEBalanceSpec:
    """Return the family spec, or an empty no-balancing spec if unregistered."""
    return _REGISTRY.get(family, MoEBalanceSpec())


def resolve_balance(family: str, model_config) -> MoEBalanceSpec:
    """Merge ModelConfig coefficients over the family baseline.

    Overrides can change declared coefficients but cannot activate a mechanism
    the family does not support; unsupported overrides emit a warning.
    """
    base = get_moe_balance(family)
    overrides: dict[str, float] = {}

    bias_override = getattr(model_config, "moe_load_balance_coeff", None)
    if bias_override is not None:
        if base.bias_gamma is None:
            logger.warning(
                f"⚠️  {family} 在 titaninfra 不支持 expert_bias 均衡，"
                f"moe_load_balance_coeff={bias_override} 被忽略。"
            )
        else:
            overrides["bias_gamma"] = bias_override

    aux_override = getattr(model_config, "moe_aux_loss_coeff", None)
    if aux_override is not None:
        if base.aux_coeff is None:
            logger.warning(
                f"⚠️  {family} 未声明 aux-loss 均衡，"
                f"moe_aux_loss_coeff={aux_override} 被忽略。"
            )
        else:
            overrides["aux_coeff"] = aux_override

    return dataclasses.replace(base, **overrides) if overrides else base


def resolve_gamma_at(
    step: int,
    base_gamma: float | None,
    *,
    calibration_steps: int,
    calibration_multiplier: float,
    freeze_after: int | None,
) -> float | None:
    """Resolve effective bias gamma for a 1-based training step.

    None disables bias updates. Otherwise late freezing takes precedence,
    followed by the cold-start calibration multiplier, then the base value.
    """
    if base_gamma is None:
        return None
    if freeze_after is not None and step > freeze_after:
        return 0.0
    if calibration_steps and step <= calibration_steps:
        return base_gamma * calibration_multiplier
    return base_gamma
