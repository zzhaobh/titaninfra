"""DeepSeek-V3 native encoder oracle parity.

Mirrors qwen2_5/llama3 patterns. Native encoder (segment path) must
match the Jinja oracle baseline (HF ``apply_chat_template`` +
``DeepSeekV3Adapter`` boundary detection) bit-for-bit on input_ids and
labels for every fixture.

Quirks captured:

* BOS + raw-system prologue (no role markers around system)
* User marker without suffix (``<｜User｜>``)
* Post-tool assistant uses ``<｜tool▁outputs▁end｜>`` instead of
  ``<｜Assistant｜>`` (is_tool state)
* Per-iteration closing-tag emission on tool_calls (jinja bug mirrored)
"""

from __future__ import annotations

import pytest

from tests.oracle._jinja_oracle._jinja_baseline import render_with_jinja
from tests.oracle._jinja_oracle.diff import compare_encoded
from titaninfra.data.sft.dispatch import tokenize_sample


@pytest.mark.parametrize(
    "fixture_path",
    [
        "deepseek_v3/01_single_turn_plain.json",
        "deepseek_v3/02_with_system.json",
        "deepseek_v3/03_multi_turn.json",
        "deepseek_v3/05_two_tool_calls.json",
    ],
)
def test_deepseek_v3_native_matches_jinja_oracle(
    fixture_path, deepseek_v3_tokenizer, load_oracle_fixture
):
    """Native DSv3 encoder output matches Jinja oracle baseline."""
    fx = load_oracle_fixture(fixture_path)
    jinja_out = render_with_jinja(
        fx.sample, deepseek_v3_tokenizer,
        max_length=fx.max_length, template_name="deepseek_v3",
    )
    native_out = tokenize_sample(
        fx.sample, deepseek_v3_tokenizer, family="deepseek_v3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="deepseek_v3", fixture=str(fx.path),
    )
    assert report["ok"] is True, (
        f"DSv3 parity broke at fixture={fixture_path}: {report}"
    )
    assert report["first_divergence"] is None
    assert report["levels"]["L1"]["ok"] is True
    assert report["levels"]["L2"]["ok"] is True
    assert fx.expected["oracle_ok"] is True


def test_deepseek_v3_tool_loop_deliberate_loss_divergence(
    deepseek_v3_tokenizer, load_oracle_fixture
):
    """04_tool_loop: native MASKS <｜tool▁outputs▁begin｜> (audit #3) while
    jinja's crude span-mask trains it — a DELIBERATE, correct divergence.

    Contract: L1 (input_ids / wire structure) stays byte-identical to jinja;
    L2 (labels) diverges at exactly the documented index where the
    framework-injected outputs-begin delimiter sits. The delimiter is
    class-A PROTOCOL (the model never autonomously emits it), so training
    loss on it would teach a phantom token — titaninfra is stricter and more
    correct than the jinja oracle here (same category as the class-C strips).
    """
    fx = load_oracle_fixture("deepseek_v3/04_tool_loop.json")
    assert fx.expected["oracle_ok"] is False
    jinja_out = render_with_jinja(
        fx.sample, deepseek_v3_tokenizer,
        max_length=fx.max_length, template_name="deepseek_v3",
    )
    native_out = tokenize_sample(
        fx.sample, deepseek_v3_tokenizer, family="deepseek_v3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="deepseek_v3", fixture=str(fx.path),
    )
    # Wire structure identical; only the loss mask diverges.
    assert report["levels"]["L1"]["ok"] is True, (
        f"L1 (input_ids) must stay identical to jinja: {report}"
    )
    assert report["levels"]["L2"]["ok"] is False, (
        "L2 must diverge — native deliberately masks the outputs-begin delimiter"
    )
    assert report["first_divergence"] == fx.expected["first_divergence"]
    # The divergent token must be the masked delimiter: native IGNORE, jinja trained.
    from titaninfra.data.sft.types import IGNORE_INDEX
    idx = fx.expected["first_divergence"]
    assert native_out.labels[idx] == IGNORE_INDEX
    assert jinja_out.labels[idx] != IGNORE_INDEX
    tok_str = deepseek_v3_tokenizer.convert_ids_to_tokens([native_out.input_ids[idx]])[0]
    assert "tool▁outputs▁begin" in tok_str, (
        f"expected outputs-begin delimiter at idx {idx}, got {tok_str!r}"
    )


def test_deepseek_v3_warns_and_skips_thinking(deepseek_v3_tokenizer, caplog):
    """Capability mismatch contract: ``thinking`` blocks warn-and-skip.

    DSv3 base template has no native ``<think>`` protocol; the R1
    reasoning variant uses a different template. Per the warn-loud
    contract (see ``feedback_capability_mismatch_must_fail_loud``),
    dispatch logs a structured warning and drops the ThinkingBlock,
    then the encoder runs on the remaining text-only blocks. The
    sample still produces a valid EncodedSample because at least one
    supported assistant block (text) survives pruning.
    """
    import logging

    sample = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "blocks": [
                    {"type": "thinking", "text": "let me think"},
                    {"type": "text", "text": "ok"},
                ],
            },
        ]
    }
    with caplog.at_level(logging.WARNING, logger="titaninfra.sft.capabilities"):
        encoded = tokenize_sample(
            sample, deepseek_v3_tokenizer, family="deepseek_v3",
            max_length=256,
        )
    assert any(
        "family=deepseek_v3" in r.message and "block=ThinkingBlock" in r.message
        for r in caplog.records
    ), f"expected ThinkingBlock skip warning, got: {[r.message for r in caplog.records]}"
    assert len(encoded.input_ids) > 0
