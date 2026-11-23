"""Layer-1 boundary builder: collapses training policy into per-message weight.

This module is the single place where ``sft_mask_mode`` and
``train_on_inputs`` get resolved. Encoders downstream read only
``NormalizedMessage.weight`` and never re-import this policy -- which
keeps protocol code (Layer 2 encoders) free of training-side concerns.

The mask policy has three layers:

    Layer 1 (highest): train_on_inputs=True -> every weight becomes 1.0
    Layer 2:           sft_mask_mode
                         "auto"   -> assistant=1.0, others=0.0,
                                     BUT weight==0 is ALWAYS respected
                                     (allows explicit silencing).
                         "manual" -> pass per-message weight verbatim.
    Layer 3:           msg.weight (role default already applied by
                                   normalize.py upstream)

``type_overrides`` is *not* applied here -- it's a per-block dial
("thinking" scale, etc.) that the per-family encoder consumes when
emitting THINKING / TOOL_CALL segments. The policy carries it so the
encoder has one place to look.

M-alpha only: no callsite imports this yet. Wiring happens in M-beta when
``encoders/<family>.py`` migrates to ``encode_segments``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from titaninfra.data.sft.types import NormalizedSample

SftMaskMode = Literal["auto", "manual"]

_VALID_MODES: frozenset[str] = frozenset({"auto", "manual"})


@dataclass(frozen=True)
class LossPolicy:
    """Training-side policy applied at Layer 1.

    Fields
    ------
    sft_mask_mode:
        ``"auto"`` (default) forces binary assistant-only loss, with the
        single carve-out that ``weight==0`` is always honoured.
        ``"manual"`` passes the per-message weight through unchanged.
    train_on_inputs:
        When ``True``, overrides everything and gives every message a
        weight of ``1.0``. Matches ``_apply_mask_strategy`` Level 1.
    type_overrides:
        Per-block multiplier dict (e.g. ``{"thinking": 0.3}``). Carried
        here so the encoder receives the full policy in one object, but
        this function does NOT apply it to message-level weights.
    """

    sft_mask_mode: SftMaskMode = "auto"
    train_on_inputs: bool = False
    type_overrides: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.sft_mask_mode not in _VALID_MODES:
            raise ValueError(
                f"LossPolicy.sft_mask_mode must be one of {sorted(_VALID_MODES)}, "
                f"got {self.sft_mask_mode!r}"
            )


def resolve_weights(
    sample: NormalizedSample,
    policy: LossPolicy,
) -> NormalizedSample:
    """Return a new NormalizedSample with per-message weight finalised.

    The input ``sample`` is not mutated; ``tools`` / ``system`` and every
    NormalizedMessage field other than ``weight`` is carried through
    unchanged via ``dataclasses.replace``.
    """
    new_messages = tuple(
        replace(m, weight=_resolve_one(m.role, m.weight, policy))
        for m in sample.messages
    )
    return replace(sample, messages=new_messages)


def _resolve_one(role: str, weight: float, policy: LossPolicy) -> float:
    # Layer 1 -- train_on_inputs wins outright.
    if policy.train_on_inputs:
        return 1.0

    # Layer 2 -- sft_mask_mode.
    if policy.sft_mask_mode == "manual":
        return weight

    # "auto" mode below.
    # Always-respected silencing carve-out: explicit zero stays zero.
    if weight == 0.0:
        return 0.0
    # Assistant turns get binary 1.0; everything else is masked.
    return 1.0 if role == "assistant" else 0.0


__all__ = ["LossPolicy", "resolve_weights"]
