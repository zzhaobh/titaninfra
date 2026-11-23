"""Llama3 native encoder oracle parity.

Post jinja-purge: ``tokenize_sample`` is the single runtime path; the
oracle baseline (``render_with_jinja``) is the independent Jinja
reference used for regression detection.

Known intentional divergence (task #14 fix):
  The native encoder correctly emits ``loss_scale=0.0`` for the trailing
  ``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` segment (class-C
  token: vLLM/SGLang inject it as a prompt prefix).  The Jinja
  ``_compute_message_boundaries`` path incorrectly bled the last
  assistant message's loss scale onto this trailer.

  Consequence: L1 (input_ids) is always identical; L2 (labels) diverges
  at the trailer tokens.  Tests assert L1=ok and accept the L2 delta at
  the known trailer position.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tests.oracle._jinja_oracle._jinja_baseline import render_with_jinja
from tests.oracle._jinja_oracle.diff import compare_encoded
from titaninfra.data.sft.dispatch import tokenize_sample
from titaninfra.data.sft.types import IGNORE_INDEX

_LLAMA3_FIXTURE_DIR = pathlib.Path("tests/fixtures/sft/llama3")

_TRAILER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _assert_llama3_l1_ok_l2_trailer_only(report, native_out, tokenizer):
    """L1 must match; L2 divergence must be confined to the trailer."""
    assert report["levels"]["L1"]["ok"] is True, "input_ids mismatch — encoder drifted"

    if report["levels"]["L2"]["ok"]:
        return  # perfect match — fine too

    # L2 diverges: verify it's only the trailing header tokens.
    trailer_ids = tokenizer.encode(_TRAILER, add_special_tokens=False)
    n = len(native_out.input_ids)
    trailer_start = n - len(trailer_ids)

    first_div = report.get("first_divergence")
    assert first_div is not None and first_div >= trailer_start, (
        f"L2 divergence at pos={first_div} is before the trailer "
        f"(trailer starts at pos={trailer_start}). "
        "The encoder has introduced an unexpected label difference."
    )
    # All trailer labels in native must be IGNORE_INDEX.
    for i, lbl in enumerate(list(native_out.labels)[trailer_start:]):
        assert lbl == IGNORE_INDEX, (
            f"trailer token at pos={trailer_start + i} has label={lbl}, "
            "expected IGNORE_INDEX (class-C fix)"
        )


def test_llama3_native_matches_jinja_oracle_ok(llama3_tokenizer, load_oracle_fixture):
    """Native encoder L1 matches Jinja; L2 may diverge only at trailer."""
    fx = load_oracle_fixture("llama3/01_single_turn_plain.json")
    jinja_out = render_with_jinja(fx.sample, llama3_tokenizer, max_length=fx.max_length)
    native_out = tokenize_sample(
        fx.sample, llama3_tokenizer, family="llama3", max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="llama3", fixture=str(fx.path),
    )
    _assert_llama3_l1_ok_l2_trailer_only(report, native_out, llama3_tokenizer)


def test_llama3_native_rejects_tool_calls(llama3_tokenizer, caplog):
    """Capability mismatch contract: ``tool_calls`` warn-and-skip.

    Llama3 declares no FEATURE_TOOL_CALL support. Per warn-loud
    contract dispatch surfaces the unsupported feature; encoder then
    raises on the empty assistant body since silently training on a
    bare ``<|eot_id|>`` would mask the data error.
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
                sample, llama3_tokenizer, family="llama3",
                max_length=256,
            )
    assert any(
        "family=llama3" in r.message and "feature=tool_call" in r.message
        for r in caplog.records
    ), f"expected tool_call warn-loud, got: {[r.message for r in caplog.records]}"


def test_llama3_warns_and_skips_thinking(llama3_tokenizer, caplog):
    """Capability mismatch contract: ``thinking`` blocks warn-and-skip.

    Llama3 has no native thinking protocol; per warn-loud contract
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
            sample, llama3_tokenizer, family="llama3",
            max_length=256,
        )
    assert any(
        "family=llama3" in r.message and "block=ThinkingBlock" in r.message
        for r in caplog.records
    ), f"expected ThinkingBlock skip warning, got: {[r.message for r in caplog.records]}"
    assert len(encoded.input_ids) > 0


# ----------------------------------------------------------------------
# Expanded fixture coverage (system prompt)
# ----------------------------------------------------------------------
def test_llama3_with_system_oracle_ok(llama3_tokenizer, load_oracle_fixture):
    """System prompt: L1 matches Jinja; L2 may diverge only at trailer."""
    fx = load_oracle_fixture("llama3/02_with_system.json")
    jinja_out = render_with_jinja(fx.sample, llama3_tokenizer, max_length=fx.max_length)
    native_out = tokenize_sample(
        fx.sample, llama3_tokenizer, family="llama3", max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="llama3", fixture=str(fx.path),
    )
    _assert_llama3_l1_ok_l2_trailer_only(report, native_out, llama3_tokenizer)


# ----------------------------------------------------------------------
# Multi-turn coverage (no system, no thinking)
# ----------------------------------------------------------------------
def test_llama3_multi_turn_oracle_ok(llama3_tokenizer, load_oracle_fixture):
    """Two-exchange turn: L1 matches Jinja; L2 may diverge only at trailer."""
    fx = load_oracle_fixture("llama3/03_multi_turn.json")
    jinja_out = render_with_jinja(fx.sample, llama3_tokenizer, max_length=fx.max_length)
    native_out = tokenize_sample(
        fx.sample, llama3_tokenizer, family="llama3", max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="llama3", fixture=str(fx.path),
    )
    _assert_llama3_l1_ok_l2_trailer_only(report, native_out, llama3_tokenizer)


# ----------------------------------------------------------------------
# Regression: trailing assistant header must be class-C (loss_scale=0)
# task #14 / commit 551c8f1 fix
# ----------------------------------------------------------------------
def test_llama3_trailing_header_is_masked(llama3_tokenizer):
    """Trailing <|start_header_id|>assistant<|end_header_id|>\\n\\n is class-C.

    vLLM/SGLang inject this token as a prompt prefix at inference time.
    The model should never be trained to predict it.  Asserts that every
    token in that trailer segment carries labels == IGNORE_INDEX.
    """
    from titaninfra.data.sft.dispatch import tokenize_sample
    from titaninfra.data.sft.types import IGNORE_INDEX

    tok = llama3_tokenizer
    trailer = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    trailer_ids = tok.encode(trailer, add_special_tokens=False)
    assert trailer_ids, "tokenizer must produce tokens for the trailer"

    sample = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    enc = tokenize_sample(sample, tok, family="llama3", max_length=256)
    ids = list(enc.input_ids)
    labels = list(enc.labels)

    # The trailer must appear at the end of input_ids.
    assert ids[-len(trailer_ids):] == trailer_ids, (
        "trailer tokens not found at end of input_ids"
    )
    # Every trailer token must be masked.
    for pos, (tid, lbl) in enumerate(
        zip(ids[-len(trailer_ids):], labels[-len(trailer_ids):], strict=True)
    ):
        assert lbl == IGNORE_INDEX, (
            f"trailer token at offset -{len(trailer_ids) - pos}: "
            f"id={tid} ({tok.decode([tid])!r}) has label={lbl}, expected IGNORE_INDEX"
        )


# ----------------------------------------------------------------------
# F11 expanded fixture coverage (golden snapshots)
# multi-turn / weight=0 silencing / long system — text-only, llama3 range.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "fixture_name",
    ["04_multi_turn", "05_weight_zero_silencing", "06_long_system"],
)
def test_llama3_fixture_golden(llama3_tokenizer, fixture_name):
    """Encoder output matches the frozen golden snapshot in the F11 fixture."""
    fx = json.loads(
        (_LLAMA3_FIXTURE_DIR / f"{fixture_name}.json").read_text(encoding="utf-8")
    )
    enc = tokenize_sample(
        fx["sample"], llama3_tokenizer, family="llama3", max_length=fx["max_length"],
    )
    expected = fx["expected"]
    assert len(enc.input_ids) == expected["n_tokens"], (
        f"{fixture_name}: encoder emitted {len(enc.input_ids)} tokens, "
        f"fixture expects {expected['n_tokens']}"
    )
    n_loss = sum(1 for label in enc.labels if label != -100)
    assert n_loss == expected["n_loss_tokens"], (
        f"{fixture_name}: encoder marked {n_loss} loss tokens, "
        f"fixture expects {expected['n_loss_tokens']}"
    )
    decoded = llama3_tokenizer.decode(enc.input_ids, skip_special_tokens=False)
    assert decoded == expected["decoded"], (
        f"{fixture_name}: decoded wire string drifted from golden snapshot"
    )
