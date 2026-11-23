"""Exhaustive base_mask(kind, role) truth table — spec §3.2 / §7.2.

base_mask is the pure structural half of loss_scale:

    loss_scale = base_mask(kind, role) × scale_factor

It returns 1.0 only for tokens the MODEL autonomously emits (assistant
turns whose kind is STOP_MARKER / CONTENT / THINKING / TOOL_CALL) and
0.0 for everything the inference framework injects. This table is the
single source of truth for "which token may carry loss"; pinning every
cell prevents the audit #2/#3 bug class from regressing.
"""

from __future__ import annotations

import itertools

import pytest

from titaninfra.data.sft.segment import Role, SegKind, Segment, TokenClass, _base_mask

# Every Role literal value (segment.py Role).
_ROLES: tuple[Role, ...] = (
    "system",
    "user",
    "assistant",
    "tool",
    "developer",
    "latest_reminder",
)

# The four assistant kinds that the model autonomously decodes → mask 1.
_ASSISTANT_LOSS_KINDS = frozenset({
    SegKind.STOP_MARKER,
    SegKind.CONTENT,
    SegKind.THINKING,
    SegKind.TOOL_CALL,
})


def test_all_segkinds_covered() -> None:
    """Guard: if a new SegKind is added, this test forces a decision."""
    assert set(SegKind) == {
        SegKind.MARKER,
        SegKind.STOP_MARKER,
        SegKind.CONTENT,
        SegKind.THINKING,
        SegKind.TOOL_CALL,
        SegKind.TOOL_RESULT,
        SegKind.SYSTEM,
        SegKind.TOOLS_HEADER,
        SegKind.IMAGE,
    }


@pytest.mark.parametrize("kind,role", list(itertools.product(SegKind, _ROLES)))
def test_base_mask_truth_table(kind: SegKind, role: Role) -> None:
    expected = 1.0 if (role == "assistant" and kind in _ASSISTANT_LOSS_KINDS) else 0.0
    assert _base_mask(kind, role) == expected


def test_non_assistant_always_zero() -> None:
    for role in _ROLES:
        if role == "assistant":
            continue
        for kind in SegKind:
            assert _base_mask(kind, role) == 0.0


def test_assistant_framework_kinds_zero() -> None:
    # MARKER / TOOL_RESULT / SYSTEM / TOOLS_HEADER on assistant → framework, mask 0.
    for kind in (SegKind.MARKER, SegKind.TOOL_RESULT, SegKind.SYSTEM, SegKind.TOOLS_HEADER):
        assert _base_mask(kind, "assistant") == 0.0


def test_assistant_decoded_kinds_one() -> None:
    for kind in _ASSISTANT_LOSS_KINDS:
        assert _base_mask(kind, "assistant") == 1.0


# --- Task 2: derived loss_scale + deprecation window --------------------


def _seg(**overrides):
    base = dict(
        text="x",
        token_ids=None,
        kind=SegKind.CONTENT,
        role="assistant",
        is_generation=False,
        token_class=TokenClass.DECODED,
    )
    base.update(overrides)
    return Segment(**base)


def test_derive_loss_scale_when_omitted_assistant_content() -> None:
    # No loss_scale, no scale_factor → derived = base(1) × 1.0 = 1.0
    seg = _seg()
    assert seg.loss_scale == 1.0


def test_derive_loss_scale_scale_factor_applied() -> None:
    seg = _seg(scale_factor=0.3)
    assert seg.loss_scale == pytest.approx(0.3)


def test_derive_framework_kind_is_zero_even_with_scale_factor() -> None:
    # base(MARKER, assistant) == 0 → derived 0 regardless of scale_factor.
    seg = _seg(kind=SegKind.MARKER, scale_factor=1.0, token_class=TokenClass.PROTOCOL)
    assert seg.loss_scale == 0.0


def test_derive_non_assistant_is_zero() -> None:
    seg = _seg(kind=SegKind.CONTENT, role="user", token_class=TokenClass.PROTOCOL)
    assert seg.loss_scale == 0.0


def test_derive_stop_marker_assistant_trained() -> None:
    seg = _seg(
        kind=SegKind.STOP_MARKER, scale_factor=0.5,
        token_class=TokenClass.PROTOCOL,
    )
    assert seg.loss_scale == pytest.approx(0.5)


def test_loss_scale_is_not_a_constructor_arg() -> None:
    with pytest.raises(TypeError):
        Segment(
            text="x", token_ids=None, kind=SegKind.CONTENT, role="assistant",
            is_generation=False, token_class=TokenClass.DECODED,
            loss_scale=1.0,  # no longer an input
        )
