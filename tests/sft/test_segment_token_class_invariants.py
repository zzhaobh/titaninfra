"""Phase 1.4 — invariants for ``Segment.token_class`` (required field).

Locks down the A/B/C-class rules defined in
``docs/superpowers/specs/2026-06-02-encoder-layer-refactor.md`` §3.1:

* ``loss_scale > 0`` ⇒ ``token_class == DECODED``
* ``token_class == METADATA`` ⇒ ``loss_scale == 0.0``
* ``token_class == PROTOCOL`` ⇒ ``loss_scale == 0.0``
  EXCEPT for the assistant-turn closing ``MARKER`` segment (the class-B
  ``<|im_end|>`` token the model autonomously emits).

Phase 1.4 (post Phase 3): ``token_class`` is required — no default.
Every encoder callsite must declare intent.
"""

from __future__ import annotations

import pytest

from titaninfra.data.sft.segment import SegKind, Segment, TokenClass


def _make(**overrides):
    # Translate a desired final loss_scale into the scale_factor the
    # constructor now takes (base_mask handles the structural half).
    desired = overrides.pop("loss_scale", None)
    base_kw = dict(
        text="x",
        token_ids=None,
        kind=SegKind.CONTENT,
        role="assistant",
        is_generation=True,
        token_class=TokenClass.DECODED,
    )
    base_kw.update(overrides)
    if desired is not None and desired > 0:
        from titaninfra.data.sft.segment import _base_mask
        if _base_mask(base_kw["kind"], base_kw["role"]) > 0:
            base_kw["scale_factor"] = desired
        # base==0 → derived forced 0; a desired>0 there is now unconstructible
        # (those tests are deleted below).
    return Segment(**base_kw)


def test_token_class_is_required() -> None:
    """Phase 1.4: omitting token_class raises TypeError at construction."""
    with pytest.raises(TypeError, match="token_class"):
        Segment(
            text="hello",
            token_ids=None,
            kind=SegKind.CONTENT,
            role="assistant",
            is_generation=True,
        )


def test_loss_gt_zero_requires_decoded() -> None:
    # METADATA + loss>0 must raise — the model never emits METADATA at
    # inference, so training loss on it would teach a phantom prefix.
    # (The METADATA-specific check fires first and gives a clearer
    # message than the general DECODED-fallback.)
    with pytest.raises(ValueError, match="METADATA must have loss_scale"):
        _make(loss_scale=1.0, token_class=TokenClass.METADATA)


def test_metadata_requires_zero_loss() -> None:
    with pytest.raises(ValueError, match="METADATA must have loss_scale"):
        _make(loss_scale=0.5, token_class=TokenClass.METADATA)


def test_metadata_zero_loss_ok() -> None:
    seg = _make(kind=SegKind.MARKER, token_class=TokenClass.METADATA)
    assert seg.token_class is TokenClass.METADATA
    assert seg.loss_scale == 0.0


def test_protocol_default_requires_zero_loss() -> None:
    # PROTOCOL on a CONTENT segment with loss>0 must raise.
    with pytest.raises(ValueError, match="PROTOCOL must have loss_scale"):
        _make(loss_scale=1.0, token_class=TokenClass.PROTOCOL, kind=SegKind.CONTENT)


def test_protocol_assistant_stop_marker_allowed() -> None:
    # Exception: the assistant's own autonomous turn terminator
    # (<|im_end|>/<|return|>/<|call|>/<|eot_id|>) is class-B in disguise —
    # kind=STOP_MARKER + role=assistant. The model decides to emit it.
    # is_generation is NOT required (a tool-call <|call|> closer is a
    # STOP_MARKER even when the turn is not the RL generation boundary).
    seg = _make(
        loss_scale=1.0,
        token_class=TokenClass.PROTOCOL,
        kind=SegKind.STOP_MARKER,
        role="assistant",
        is_generation=False,
    )
    assert seg.token_class is TokenClass.PROTOCOL
    assert seg.loss_scale == 1.0


# (removed Task 9: framework-kind-with-loss is now unconstructible — base_mask forces derived 0, no explicit loss input exists.)


def test_decoded_with_zero_loss_allowed() -> None:
    seg = _make(scale_factor=0.0, token_class=TokenClass.DECODED)
    assert seg.loss_scale == 0.0


def test_negative_scale_factor_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _make(scale_factor=-0.1)
