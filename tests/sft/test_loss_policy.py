"""LossPolicy preset matrix + 3-layer cascade (M-α RED).

The new boundary-builder lives in ``titaninfra/data/sft/loss_policy.py`` and
collapses three independent dials into a single per-message weight on a
fresh ``NormalizedSample`` copy:

    Layer 1  train_on_inputs=True  → force every message weight to 1.0
                                     (CLAUDE.md §SFT Mask: "overrides everything")
    Layer 2  sft_mask_mode
                "auto"   → assistant=1.0, others=0.0, BUT weight==0 is
                           ALWAYS respected (allows explicit silencing).
                "manual" → pass per-message weight through verbatim.
    Layer 3  msg.weight (already role-default-resolved by normalize.py)

Encoder code never re-reads policy — it only sees ``nm.weight``.

These tests pin every cell of the 2x2 preset matrix plus the silencing
exception, and assert that ``resolve_weights`` is immutable (returns a
new NormalizedSample rather than mutating the input).
"""

from __future__ import annotations

import pytest

from titaninfra.data.schema import TextBlock, ThinkingBlock, ToolUseBlock
from titaninfra.data.sft.loss_policy import LossPolicy, resolve_weights
from titaninfra.data.sft.types import NormalizedMessage, NormalizedSample


def _mk(messages: list[tuple[str, float]]) -> NormalizedSample:
    """Build a NormalizedSample from (role, weight) pairs. Text is irrelevant here."""
    return NormalizedSample(
        messages=tuple(
            NormalizedMessage(
                role=role,
                weight=w,
                blocks=(TextBlock(text="x"),),
            ) for role, w in messages
        )
    )


def _weights(sample: NormalizedSample) -> list[float]:
    return [m.weight for m in sample.messages]


# ----------------------------------------------------------------------
# Preset matrix: 2 (sft_mask_mode) x 2 (train_on_inputs) = 4 cells
# ----------------------------------------------------------------------
class TestPresetMatrix:
    def test_auto_no_train_on_inputs(self):
        sample = _mk([("user", 0.0), ("assistant", 1.0), ("user", 0.0), ("assistant", 1.0)])
        out = resolve_weights(sample, LossPolicy(sft_mask_mode="auto", train_on_inputs=False))
        assert _weights(out) == [0.0, 1.0, 0.0, 1.0]

    def test_auto_with_train_on_inputs(self):
        sample = _mk([("user", 0.0), ("assistant", 1.0), ("tool", 0.0)])
        out = resolve_weights(sample, LossPolicy(sft_mask_mode="auto", train_on_inputs=True))
        assert _weights(out) == [1.0, 1.0, 1.0]

    def test_manual_no_train_on_inputs_passthrough(self):
        sample = _mk([("user", 0.5), ("assistant", 0.3), ("tool", 0.7)])
        out = resolve_weights(sample, LossPolicy(sft_mask_mode="manual", train_on_inputs=False))
        assert _weights(out) == [0.5, 0.3, 0.7]

    def test_manual_with_train_on_inputs_overrides(self):
        sample = _mk([("user", 0.5), ("assistant", 0.3), ("tool", 0.7)])
        out = resolve_weights(sample, LossPolicy(sft_mask_mode="manual", train_on_inputs=True))
        assert _weights(out) == [1.0, 1.0, 1.0]


# ----------------------------------------------------------------------
# 3-layer cascade resolution order
# ----------------------------------------------------------------------
class TestCascadeOrder:
    def test_layer1_train_on_inputs_beats_explicit_zero(self):
        # CLAUDE.md: "train_on_inputs=True -> all tokens get loss (overrides everything)"
        sample = _mk([("assistant", 0.0)])
        out = resolve_weights(sample, LossPolicy(train_on_inputs=True))
        assert _weights(out) == [1.0]

    def test_layer2_auto_respects_weight_zero_on_assistant(self):
        # "weight=0 on any message is ALWAYS respected" under auto.
        sample = _mk([("assistant", 0.0), ("assistant", 1.0)])
        out = resolve_weights(sample, LossPolicy(sft_mask_mode="auto"))
        assert _weights(out) == [0.0, 1.0]

    def test_layer2_auto_masks_nonassistant_regardless_of_input(self):
        sample = _mk([("user", 0.8), ("system", 0.9), ("tool", 0.4)])
        out = resolve_weights(sample, LossPolicy(sft_mask_mode="auto"))
        assert _weights(out) == [0.0, 0.0, 0.0]

    def test_layer2_auto_normalises_nonzero_assistant_to_one(self):
        # Auto mode is binary on assistant; non-zero values get clamped to 1.0
        # to match _resolve_weights() in segment encoders (qwen3, gemma4, etc.).
        sample = _mk([("assistant", 0.4)])
        out = resolve_weights(sample, LossPolicy(sft_mask_mode="auto"))
        assert _weights(out) == [1.0]

    def test_layer3_manual_passthrough_preserves_continuous_weights(self):
        sample = _mk([("assistant", 0.42), ("user", 0.13)])
        out = resolve_weights(sample, LossPolicy(sft_mask_mode="manual"))
        assert _weights(out) == [0.42, 0.13]


# ----------------------------------------------------------------------
# Immutability + structural invariants
# ----------------------------------------------------------------------
class TestImmutability:
    def test_input_sample_is_not_mutated(self):
        sample = _mk([("user", 0.0), ("assistant", 1.0)])
        original = list(_weights(sample))
        _ = resolve_weights(sample, LossPolicy(train_on_inputs=True))
        assert _weights(sample) == original

    def test_returns_new_object(self):
        sample = _mk([("assistant", 1.0)])
        out = resolve_weights(sample, LossPolicy())
        assert out is not sample

    def test_preserves_top_level_fields(self):
        sample = NormalizedSample(
            messages=(NormalizedMessage(
                role="assistant",
                blocks=(TextBlock(text="x"),),
                weight=1.0,
            ),),
            tools=({"name": "search"},),
            system="you are helpful",
        )
        out = resolve_weights(sample, LossPolicy(train_on_inputs=True))
        assert out.system == "you are helpful"
        assert out.tools == ({"name": "search"},)

    def test_preserves_per_message_thinking_and_tool_calls(self):
        sample = NormalizedSample(
            messages=(
                NormalizedMessage(
                    role="assistant",
                    weight=1.0,
                    blocks=(
                        ThinkingBlock(text="step 1"),
                        TextBlock(text="answer"),
                        ToolUseBlock(id="c1", name="f", input={"x": 1}),
                    ),
                ),
            )
        )
        out = resolve_weights(sample, LossPolicy())
        m = out.messages[0]
        # Phase 5: legacy text/thinking/tool_calls fields removed; verify
        # blocks-only IR carries the same content through resolve_weights.
        thinking_blocks = [b for b in m.blocks if isinstance(b, ThinkingBlock)]
        tool_blocks = [b for b in m.blocks if isinstance(b, ToolUseBlock)]
        assert len(thinking_blocks) == 1 and thinking_blocks[0].text == "step 1"
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "f"
        assert tool_blocks[0].input == {"x": 1}
        assert tool_blocks[0].id == "c1"


# ----------------------------------------------------------------------
# Policy defaults + type_overrides passthrough
# ----------------------------------------------------------------------
class TestPolicyDefaults:
    def test_default_policy_is_auto_no_train_on_inputs(self):
        p = LossPolicy()
        assert p.sft_mask_mode == "auto"
        assert p.train_on_inputs is False
        assert p.type_overrides is None

    def test_type_overrides_is_passthrough_data_not_consumed_here(self):
        # type_overrides is a per-block dial (e.g. {"thinking": 0.3}).
        # LossPolicy carries it for the encoder; resolve_weights does not
        # apply it to message-level weights.
        sample = _mk([("assistant", 1.0)])
        out = resolve_weights(
            sample, LossPolicy(sft_mask_mode="auto", type_overrides={"thinking": 0.3})
        )
        assert _weights(out) == [1.0]

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            LossPolicy(sft_mask_mode="off")  # type: ignore[arg-type]
