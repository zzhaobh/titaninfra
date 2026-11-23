"""Qwen3 renderers-oracle parity — 2026-11 feasibility prototype.

Parallel to ``test_qwen3_parity.py`` (the Jinja oracle), this file validates
``tokenize_sample`` against the ``renderers`` PyPI package as a second,
independent oracle — same fixtures, same scenarios, one test per scenario in
the original file. **This file does not replace, modify, or skip anything in
``test_qwen3_parity.py``**; both run side by side.

``renderers`` is hand-coded Python (no Jinja), so agreement with it is a
meaningfully different signal than agreement with the Jinja oracle: two
independently-implemented renderers landing on the same ``input_ids`` is
much stronger evidence of encoder correctness than one encoder matching one
template engine.

Result summary for this round (see
``docs/renderers_oracle_feasibility_report.md`` for the full writeup and the
"should we extend to other families" recommendation):

* **11 scenarios fully reproduced**, modulo two small, well-understood,
  documented ``renderers``-side label-mask deviations that are NOT bugs in
  either system (Pattern A: the empty ``<think>`` envelope; Pattern B: the
  ``\\n`` after an assistant's closing ``<|im_end|>`` — see
  ``_renderers_oracle/_renderers_baseline.py`` module docstring). Every
  divergent label position across all 11 scenarios is accounted for by one
  of these two patterns — zero unexplained positions.
* **1 scenario (fixture 04, historical thinking) is structurally
  unsupported**: ``renderers`` has no config path to reproduce titaninfra's
  deliberate "preserve historical thinking by default" policy; it always
  matches the stock-Jinja behavior of stripping it, which fixture 04 forbids.
* **1 scenario (inline mode-off consecutive assistants) exposes a genuine
  ``renderers`` bug**: unlike stock HF ``apply_chat_template``, the library
  injects the empty-think envelope on every post-query assistant turn, not
  only the last — confirmed by comparing directly against the real Jinja
  oracle, not just against native.
* **1 scenario (fractional ``loss_weights`` via ``type_overrides``) is
  structurally unsupported**: ``renderers.RenderedTrainingSample.loss_mask``
  is strictly boolean; there is no per-token weighting concept in its
  public API.
* 3 tests from the original file are intentionally not mirrored here because
  they are not oracle-comparison tests at all (``test_dispatch_smoke``,
  ``test_identity_encoder_rejects_unsupported_features`` — IdentityEncoder
  has nothing to do with any oracle; and the two tool-role-boundary
  structural sanity checks that only inspect native's own output).
"""

from __future__ import annotations

import pytest

renderers = pytest.importorskip("renderers", reason="renderers package not installed")

from tests.oracle._jinja_oracle._jinja_baseline import render_with_jinja  # noqa: E402
from tests.oracle._renderers_oracle._renderers_baseline import render_with_renderers  # noqa: E402
from tests.oracle._renderers_oracle.diff import classify_label_divergence  # noqa: E402
from titaninfra.data.sft.dispatch import tokenize_sample  # noqa: E402


def _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, tokenizer):
    """Shared assertion for every "renderers fully reproduces this" test.

    L1 (``input_ids``) must be byte-identical. Any L2 (``labels``)
    divergence must be entirely explained by Pattern A or Pattern B — a
    single unexplained position fails the test, since that would signal a
    real encoder regression or a new, uncatalogued ``renderers`` deviation.
    """
    assert native_out.input_ids == renderers_out.input_ids, (
        "L1 (input_ids) mismatch between native and renderers oracle"
    )
    result = classify_label_divergence(
        ids=native_out.input_ids,
        native_labels=native_out.labels,
        renderers_labels=renderers_out.labels,
        tokenizer=tokenizer,
    )
    assert result["unexplained_positions"] == [], (
        f"unexplained label divergence(s) vs renderers oracle: "
        f"{result['unexplained_positions']} (counts={result['counts']})"
    )
    return result


# ----------------------------------------------------------------------
# Plain-text / system / thinking (fixtures 01-03)
# ----------------------------------------------------------------------


def test_qwen3_single_turn_plain_renderers_ok(qwen3_tokenizer, load_oracle_fixture):
    """Fixture 01. Pattern A (envelope) + Pattern B (trailing \\n) only."""
    fx = load_oracle_fixture("qwen3/01_single_turn_plain.json")
    native_out = tokenize_sample(fx.sample, qwen3_tokenizer, family="qwen3", max_length=fx.max_length)
    renderers_out = render_with_renderers(fx.sample, qwen3_tokenizer, max_length=fx.max_length)
    _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)


def test_qwen3_with_system_renderers_ok(qwen3_tokenizer, load_oracle_fixture):
    """Fixture 02. Pattern A + Pattern B only."""
    fx = load_oracle_fixture("qwen3/02_with_system.json")
    native_out = tokenize_sample(fx.sample, qwen3_tokenizer, family="qwen3", max_length=fx.max_length)
    renderers_out = render_with_renderers(fx.sample, qwen3_tokenizer, max_length=fx.max_length)
    _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)


def test_qwen3_last_asst_thinking_renderers_ok(qwen3_tokenizer, load_oracle_fixture):
    """Fixture 03. Real reasoning_content present, so Pattern A never fires
    (the envelope is only empty when there is no real thinking); only
    Pattern B (trailing \\n) remains.
    """
    fx = load_oracle_fixture("qwen3/03_thinking_last_assistant.json")
    native_out = tokenize_sample(fx.sample, qwen3_tokenizer, family="qwen3", max_length=fx.max_length)
    renderers_out = render_with_renderers(fx.sample, qwen3_tokenizer, max_length=fx.max_length)
    result = _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)
    assert result["counts"] == {"B": 1}


@pytest.mark.skip(
    reason=(
        "renderers.RenderedTrainingSample.loss_mask is strictly boolean "
        "(list[bool]) — there is no per-token fractional weighting concept "
        "anywhere in the public API (build_training_sample, RenderedTokens, "
        "RenderedTrainingSample all only expose a bool mask). titaninfra's "
        "loss_config.type_overrides (e.g. thinking=0.5) has no renderers "
        "equivalent to compare against; structurally unsupported, not a "
        "renderers bug."
    )
)
def test_qwen3_type_overrides_thinking_loss_weights_renderers_unsupported():
    pass


# ----------------------------------------------------------------------
# History thinking-drop (fixture 04) — structurally unsupported
# ----------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "renderers.Qwen3Renderer has no config path to reproduce "
        "titaninfra's deliberate 'preserve historical thinking by default' "
        "policy (encoders/qwen3.py). Confirmed via source read: "
        "BaseRendererConfig.thinking_retention explicitly documents 'This "
        "does not change full render() output; full renders stay faithful "
        "to the Python chat-template implementation' — i.e. it only "
        "affects the incremental bridge/rollout-extension path, never "
        "build_training_sample(). A full render of this fixture always "
        "strips the historical <think> block (matching stock-Jinja Quirk B, "
        "the exact behavior titaninfra intentionally overrides), producing "
        "a genuine L1 (input_ids) length mismatch (35 renderers tokens vs "
        "42 native tokens) — not just a label-mask difference. Structurally "
        "unsupported scenario, not a bug in either system."
    )
)
def test_qwen3_history_thinking_drop_renderers_unsupported():
    pass


# ----------------------------------------------------------------------
# Step B tool-use parity (fixtures 05/06/07/08/09)
# ----------------------------------------------------------------------


def test_qwen3_tool_call_renderers_ok(qwen3_tokenizer, load_oracle_fixture):
    """Fixture 05. Pattern A + Pattern B only."""
    fx = load_oracle_fixture("qwen3/05_tool_call.json")
    native_out = tokenize_sample(fx.sample, qwen3_tokenizer, family="qwen3", max_length=fx.max_length)
    renderers_out = render_with_renderers(fx.sample, qwen3_tokenizer, max_length=fx.max_length)
    _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)


def test_qwen3_tool_response_renderers_ok(qwen3_tokenizer, load_oracle_fixture):
    """Fixture 06. Assistant turn is NOT last (a tool message follows), so
    Quirk A never fires; only Pattern B remains.
    """
    fx = load_oracle_fixture("qwen3/06_tool_response.json")
    native_out = tokenize_sample(fx.sample, qwen3_tokenizer, family="qwen3", max_length=fx.max_length)
    renderers_out = render_with_renderers(fx.sample, qwen3_tokenizer, max_length=fx.max_length)
    result = _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)
    assert result["counts"] == {"B": 1}


def test_qwen3_tools_top_level_renderers_ok(qwen3_tokenizer, load_oracle_fixture):
    """Fixture 07. Pattern A + Pattern B only (tools-header coalescing is
    pure L1 and already covered by the L1-equality assertion).
    """
    fx = load_oracle_fixture("qwen3/07_tools_top_level.json")
    native_out = tokenize_sample(fx.sample, qwen3_tokenizer, family="qwen3", max_length=fx.max_length)
    renderers_out = render_with_renderers(fx.sample, qwen3_tokenizer, max_length=fx.max_length)
    _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)


def test_qwen3_full_tool_loop_renderers_ok(qwen3_tokenizer, load_oracle_fixture):
    """Fixture 08. Two assistant turns → Pattern B fires twice (once per
    turn's closing <|im_end|>\\n); Pattern A fires once (only the final,
    last, assistant turn triggers the empty-think envelope).
    """
    fx = load_oracle_fixture("qwen3/08_full_tool_loop.json")
    native_out = tokenize_sample(fx.sample, qwen3_tokenizer, family="qwen3", max_length=fx.max_length)
    renderers_out = render_with_renderers(fx.sample, qwen3_tokenizer, max_length=fx.max_length)
    result = _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)
    assert result["counts"] == {"A": 3, "B": 2}


def test_qwen3_math_tools_real_world_renderers_ok(qwen3_tokenizer, load_oracle_fixture):
    """Fixture 09. Real-world sample, both assistant turns carry real
    reasoning_content, so Pattern A never fires; Pattern B fires twice.
    """
    fx = load_oracle_fixture("qwen3/09_math_tools_real_world.json")
    native_out = tokenize_sample(fx.sample, qwen3_tokenizer, family="qwen3", max_length=fx.max_length)
    renderers_out = render_with_renderers(fx.sample, qwen3_tokenizer, max_length=fx.max_length)
    result = _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)
    assert result["counts"] == {"B": 2}


# ----------------------------------------------------------------------
# Tool-role boundary regression — inline samples
# ----------------------------------------------------------------------


def test_qwen3_user_after_tool_chain_renderers_ok(qwen3_tokenizer):
    """Mirrors ``test_qwen3_user_after_tool_chain_keeps_its_marker``:
    user -> asst(tool_call) -> tool -> tool -> user(real). Pattern B only
    (assistant has no text content triggering Quirk A, and is not last).
    """
    sample = {
        "messages": [
            {"role": "user", "content": "trigger the tool"},
            {
                "role": "assistant",
                "blocks": [{"type": "tool_use", "name": "f", "input": {"x": 1}, "id": "c0"}],
            },
            {"role": "tool", "content": "r0", "tool_call_id": "c0"},
            {"role": "tool", "content": "r1", "tool_call_id": "c0"},
            {"role": "user", "content": "now answer in words"},
        ]
    }
    native_out = tokenize_sample(sample, qwen3_tokenizer, family="qwen3", max_length=512)
    renderers_out = render_with_renderers(sample, qwen3_tokenizer, max_length=512)
    result = _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)
    assert result["counts"] == {"B": 1}


def test_qwen3_plain_system_path_renderers_ok(qwen3_tokenizer):
    """Mirrors ``test_qwen3_plain_system_path_unaffected_by_wrapped_fallback``.
    Pattern A + Pattern B only.
    """
    sample = {
        "system": "You are concise.",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
        ],
    }
    native_out = tokenize_sample(sample, qwen3_tokenizer, family="qwen3", max_length=128)
    renderers_out = render_with_renderers(sample, qwen3_tokenizer, max_length=128)
    result = _assert_l1_matches_and_l2_fully_explained(native_out, renderers_out, qwen3_tokenizer)
    assert result["counts"] == {"A": 3, "B": 1}


# ----------------------------------------------------------------------
# mode="off" consecutive assistants — genuine renderers bug (not native's)
# ----------------------------------------------------------------------


def test_qwen3_mode_off_consecutive_assistants_renderers_diverges_from_both(
    qwen3_tokenizer,
):
    """Mirrors ``test_qwen3_mode_off_consecutive_assistants_only_last_injects_envelope``.

    This is NOT a "renderers fully reproduces" test — it is a positive
    assertion that ``renderers`` has a genuine divergence from BOTH native
    AND the real Jinja oracle here, confirmed by checking all three
    independently. Stock HF ``apply_chat_template(enable_thinking=False)``
    injects the empty-think envelope ONLY on the last of two consecutive
    post-query assistant turns (this is exactly what titaninfra's own
    "Audit #5" fix made native match). ``renderers.Qwen3Renderer`` injects
    it on EVERY such turn instead — its ``emit_thinking = ... or (not
    enable_thinking and not reasoning_content)`` condition in
    ``renderers/qwen3.py`` has no ``is_last`` gate for this branch, so it
    fires unconditionally whenever thinking is disabled and no real
    reasoning_content is present, regardless of turn position. This is a
    real bug/limitation in the library (verified against the actual
    rendered HF token stream, not just against native's policy), tracked
    here as a documented, reproducible divergence rather than filed
    upstream as part of this prototype round's scope.
    """
    sample = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "q1"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "a1"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "a2"}]},
        ],
        "thinking_intent": {"mode": "off"},
    }
    native_out = tokenize_sample(sample, qwen3_tokenizer, family="qwen3")
    jinja_out = render_with_jinja(sample, qwen3_tokenizer, max_length=None, template_name="qwen3")
    renderers_out = render_with_renderers(sample, qwen3_tokenizer)

    # Sanity: native and the real Jinja oracle agree (this is exactly what
    # test_qwen3_parity.py's sibling test already asserts — re-confirmed
    # here as the baseline this renderers comparison is measured against).
    assert native_out.input_ids == jinja_out.input_ids

    # renderers must diverge at L1 from both — if a future renderers
    # release fixes this, this assertion starts failing loudly rather than
    # silently losing coverage of the bug.
    assert renderers_out.input_ids != native_out.input_ids, (
        "renderers no longer diverges here — the documented bug in "
        "renderers/qwen3.py appears fixed upstream; update this test and "
        "the feasibility report."
    )
    assert len(renderers_out.input_ids) == len(native_out.input_ids) + 4, (
        "expected exactly one extra empty-think envelope (4 tokens: "
        "<think>, \\n\\n, </think>, \\n\\n) injected on the non-last "
        "assistant turn; got a different-sized divergence — investigate "
        "before assuming this is still the same known bug."
    )
