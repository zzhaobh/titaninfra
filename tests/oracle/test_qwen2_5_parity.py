"""Qwen2.5 native encoder oracle parity.

Post jinja-purge: ``tokenize_sample`` is the single runtime path; the
oracle baseline (``render_with_jinja``) is the independent Jinja
reference used for regression detection.

Plain-text parity is structurally tautological (encoder still delegates
to the legacy renderer in M1.x); these tests exist to lock the wiring.
"""

from __future__ import annotations

import pytest

from tests.oracle._jinja_oracle._jinja_baseline import render_with_jinja
from tests.oracle._jinja_oracle.diff import compare_encoded
from titaninfra.data.sft.dispatch import tokenize_sample


def test_qwen2_5_native_matches_jinja_oracle_ok(qwen2_5_tokenizer, load_oracle_fixture):
    """Native encoder output matches Jinja oracle baseline for plain text."""
    fx = load_oracle_fixture("qwen2_5/01_single_turn_plain.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen2_5_tokenizer, max_length=fx.max_length,
    )
    native_out = tokenize_sample(
        fx.sample, qwen2_5_tokenizer, family="qwen2_5",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen2_5", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert report["levels"]["L1"]["ok"] is True
    assert report["levels"]["L2"]["ok"] is True
    assert fx.expected["oracle_ok"] is True


def test_qwen2_5_native_rejects_tool_calls(qwen2_5_tokenizer, caplog):
    """Capability mismatch contract: ``tool_calls`` warn-and-skip.

    Qwen2.5 declares no FEATURE_TOOL_CALL support. Per warn-loud
    contract (see ``feedback_capability_mismatch_must_fail_loud``)
    dispatch surfaces the unsupported feature; the encoder produces
    the assistant turn with whatever structured blocks survive
    (here: empty content, since ``content=""`` and no blocks).
    The encoder must NOT raise — assemble() raises only if the
    resulting segment list contains a 0-token text segment.
    """
    import logging

    sample = {
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                    }
                ],
            },
        ]
    }
    with caplog.at_level(logging.WARNING, logger="titaninfra.sft.capabilities"):
        with pytest.raises(ValueError, match="no learnable text content"):
            tokenize_sample(
                sample, qwen2_5_tokenizer, family="qwen2_5",
                max_length=256,
            )
    assert any(
        "family=qwen2_5" in r.message and "feature=tool_call" in r.message
        for r in caplog.records
    ), f"expected tool_call warn-loud, got: {[r.message for r in caplog.records]}"


def test_qwen2_5_warns_and_skips_thinking(qwen2_5_tokenizer, caplog):
    """Capability mismatch contract: ``thinking`` blocks warn-and-skip.

    Qwen2.5 has no native thinking protocol; per warn-loud contract
    (see ``feedback_capability_mismatch_must_fail_loud``) dispatch
    logs and structurally drops the ThinkingBlock, leaving the
    remaining text-only blocks to encode normally.
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
            sample, qwen2_5_tokenizer, family="qwen2_5",
            max_length=256,
        )
    assert any(
        "family=qwen2_5" in r.message and "block=ThinkingBlock" in r.message
        for r in caplog.records
    ), f"expected ThinkingBlock skip warning, got: {[r.message for r in caplog.records]}"
    assert len(encoded.input_ids) > 0


# ----------------------------------------------------------------------
# Expanded fixture coverage (system prompt)
# ----------------------------------------------------------------------
def test_qwen2_5_with_system_oracle_ok(qwen2_5_tokenizer, load_oracle_fixture):
    """System prompt flows through dispatch -> encoder -> renderer."""
    fx = load_oracle_fixture("qwen2_5/02_with_system.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen2_5_tokenizer, max_length=fx.max_length,
    )
    native_out = tokenize_sample(
        fx.sample, qwen2_5_tokenizer, family="qwen2_5",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen2_5", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert fx.expected["oracle_ok"] is True


# ----------------------------------------------------------------------
# Multi-turn coverage (no system, no thinking)
# ----------------------------------------------------------------------
def test_qwen2_5_multi_turn_oracle_ok(qwen2_5_tokenizer, load_oracle_fixture):
    """Two-exchange user/assistant alternation matches Jinja oracle.

    Exercises the inter-turn delimiter path (``<|im_end|>\\n<|im_start|>``
    boundary on Qwen2.5) that single-turn fixtures cannot reach.
    """
    fx = load_oracle_fixture("qwen2_5/03_multi_turn.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen2_5_tokenizer, max_length=fx.max_length,
    )
    native_out = tokenize_sample(
        fx.sample, qwen2_5_tokenizer, family="qwen2_5",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen2_5", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert fx.expected["oracle_ok"] is True


# ----------------------------------------------------------------------
# Regression: assistant text starting with '\n' must match jinja
# ----------------------------------------------------------------------
def test_qwen2_5_assistant_text_leading_newline_oracle_ok(
    qwen2_5_tokenizer, load_oracle_fixture,
):
    """Regression: ``\\n``-leading assistant body — Option C BPE merge fix.

    Companion to qwen3 fixture 10. Real-world data with assistant text
    that literally begins with ``\\n`` exposed a 1-token BPE drift: when
    MARKER (``...assistant\\n``) and CONTENT (``\\nReply...``) tokenize as
    two separate pieces, the joint render's ``\\n\\n`` BPE merge (token id
    271 in qwen2.5) is lost. Qwen2.5 jinja does NOT lstrip (unlike qwen3),
    so the encoder cannot drop the ``\\n``; instead it relocates the
    trailing ``\\n`` from MARKER to CONTENT head — wire string identical,
    BPE merges across the seam, **L1 drift goes to 0**.

    L2 (loss-mask) divergence is DELIBERATE and documented:
    fixture ``oracle_ok=false`` (Quirk-B-style annotation) because the
    merged ``\\n\\n`` token (id 271) now lives inside CONTENT and earns
    ``loss_scale=msg.weight``, while the jinja-then-mask baseline assigns
    it ``IGNORE_INDEX``. This is consistent with HF Trainer's behaviour
    when ``apply_chat_template`` output is tokenized once and masked by
    response-prefix offset. See fixture 04 _description for the full
    rationale.
    """
    fx = load_oracle_fixture("qwen2_5/04_assistant_text_leading_newline.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen2_5_tokenizer, max_length=fx.max_length,
    )
    native_out = tokenize_sample(
        fx.sample, qwen2_5_tokenizer, family="qwen2_5",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen2_5", fixture=str(fx.path),
    )
    # L1 (input_ids) MUST be byte-exact — the whole point of the BPE fix.
    assert report["levels"]["L1"]["ok"] is True, report
    assert len(native_out.input_ids) == len(jinja_out.input_ids)
    # L2 (labels) is the deliberate divergence at the merged \n\n token.
    assert report["levels"]["L2"]["ok"] is False, report
    expected_idx = fx.expected["first_divergence"]
    assert report["levels"]["L2"]["first_idx"] == expected_idx
    # Pin the exact token id at the divergence point so future BPE-vocab
    # changes or refactors that shift the index can't sneak past silently.
    assert native_out.input_ids[expected_idx] == 271  # qwen2.5 BPE id for '\n\n'
    assert native_out.labels[expected_idx] == 271     # CONTENT side — loss carried
    assert jinja_out.labels[expected_idx] == -100      # PROTOCOL side — masked
    assert fx.expected["oracle_ok"] is False
