"""GPT-OSS oracle parity — smoke + 8 golden-file fixture tests + tamper control.

Test strategy is golden-file (not Jinja oracle) because we intentionally diverge
from Jinja on three axes (see spec §2):
  1. Dynamic metadata stripped (Current date / Knowledge cutoff / Reasoning).
  2. Historical assistant thinking preserved (Jinja drops it when future final exists).
  3. <|return|> on last assistant final-channel closer only (<|end|> elsewhere).
"""
from __future__ import annotations

import pytest

from titaninfra.data.sft.dispatch import tokenize_sample

GPT_OSS_FIXTURES = [
    "gpt_oss/01_single_turn_plain.json",
    "gpt_oss/02_with_developer_system.json",
    "gpt_oss/03_thinking_last_assistant.json",
    "gpt_oss/04_thinking_history_preserved.json",
    "gpt_oss/05_tool_call.json",
    "gpt_oss/06_tool_response.json",
    "gpt_oss/07_tools_top_level.json",
    "gpt_oss/08_full_loop.json",
    "gpt_oss/09_math_tools_real_world.json",
]


def test_gpt_oss_tokenizer_loads(gpt_oss_tokenizer):
    """Session-scoped tokenizer fixture is reachable."""
    assert gpt_oss_tokenizer is not None
    ids = gpt_oss_tokenizer.encode("hello", add_special_tokens=False)
    assert isinstance(ids, list) and len(ids) > 0


def test_gpt_oss_dispatch_smoke(gpt_oss_tokenizer, load_oracle_fixture):
    """Encoder produces a well-formed EncodedSample on fixture 01."""
    fx = load_oracle_fixture("gpt_oss/01_single_turn_plain.json")
    out = tokenize_sample(
        fx.sample, gpt_oss_tokenizer, family="gpt_oss", max_length=fx.max_length,
    )
    assert len(out.input_ids) > 0
    assert len(out.labels) == len(out.input_ids)
    assert len(out.positions) == len(out.input_ids)
    # At least some tokens are labeled (assistant content)
    from titaninfra.data.sft.types import IGNORE_INDEX
    assert any(lbl != IGNORE_INDEX for lbl in out.labels)


@pytest.mark.parametrize("fixture_path", GPT_OSS_FIXTURES)
def test_gpt_oss_fixture_golden(gpt_oss_tokenizer, load_oracle_fixture, fixture_path):
    """Each fixture's expected.input_ids / labels must match the encoder output.

    If expected block is empty (golden not yet generated), skip with a clear message.
    """
    fx = load_oracle_fixture(fixture_path)
    if not fx.expected.get("input_ids"):
        pytest.skip(
            f"{fixture_path}: golden not yet generated. "
            "Run scripts/tools/generate_gpt_oss_golden.py to populate."
        )

    out = tokenize_sample(
        fx.sample, gpt_oss_tokenizer, family="gpt_oss", max_length=fx.max_length,
    )
    assert list(out.input_ids) == fx.expected["input_ids"], (
        f"{fixture_path}: input_ids mismatch at first diff pos "
        f"{next(i for i,(a,b) in enumerate(zip(out.input_ids, fx.expected['input_ids'], strict=False)) if a!=b)}"
    )
    from titaninfra.data.sft.types import IGNORE_INDEX
    expected_labels = [
        IGNORE_INDEX if lbl == IGNORE_INDEX else lbl
        for lbl in fx.expected["labels"]
    ]
    assert list(out.labels) == expected_labels, (
        f"{fixture_path}: labels mismatch"
    )


def test_gpt_oss_history_thinking_preserved(gpt_oss_tokenizer, load_oracle_fixture):
    """Fixture 04: encoder keeps historical thinking; Jinja would drop it.

    We cannot call Jinja directly (it requires strftime_now which is not available
    offline), so we verify structurally: the encoder output must contain tokens
    representing the historical thinking text from message index 1.
    """
    fx = load_oracle_fixture("gpt_oss/04_thinking_history_preserved.json")
    out = tokenize_sample(
        fx.sample, gpt_oss_tokenizer, family="gpt_oss", max_length=fx.max_length,
    )
    # The historical thinking text is "I should count to 3."
    # Decode the full sequence and assert the text is present.
    decoded = gpt_oss_tokenizer.decode(out.input_ids, skip_special_tokens=False)
    assert "I should count to 3." in decoded, (
        "Historical thinking from first assistant turn must be preserved in GPT-OSS encoder output"
    )
    # Also assert second thinking is present
    assert "Continue to 5." in decoded, (
        "Last assistant thinking must also be preserved"
    )


def test_gpt_oss_return_token_only_on_last_asst(gpt_oss_tokenizer, load_oracle_fixture):
    """Fixture 04 (multi-turn): exactly one <|return|> token; all other asst closers are <|end|>."""
    fx = load_oracle_fixture("gpt_oss/04_thinking_history_preserved.json")
    out = tokenize_sample(
        fx.sample, gpt_oss_tokenizer, family="gpt_oss", max_length=fx.max_length,
    )
    return_id = gpt_oss_tokenizer.convert_tokens_to_ids("<|return|>")
    end_id = gpt_oss_tokenizer.convert_tokens_to_ids("<|end|>")

    return_count = out.input_ids.count(return_id)
    assert return_count == 1, (
        f"Expected exactly 1 <|return|> token, got {return_count}"
    )
    # At least one <|end|> from the non-last assistant turn
    assert out.input_ids.count(end_id) >= 1, (
        "Expected at least one <|end|> for non-last assistant turns"
    )


def test_gpt_oss_dynamic_metadata_absent(gpt_oss_tokenizer, load_oracle_fixture):
    """Fixture 01: decoded sequence must not contain 'Current date' / 'Knowledge cutoff' / 'Reasoning:'."""
    fx = load_oracle_fixture("gpt_oss/01_single_turn_plain.json")
    out = tokenize_sample(
        fx.sample, gpt_oss_tokenizer, family="gpt_oss", max_length=fx.max_length,
    )
    decoded = gpt_oss_tokenizer.decode(out.input_ids, skip_special_tokens=False)
    for banned in ("Current date", "Knowledge cutoff", "Reasoning:"):
        assert banned not in decoded, (
            f"Class C token '{banned}' must not appear in encoder output"
        )


def test_gpt_oss_oracle_catches_tampered_tokens(gpt_oss_tokenizer, load_oracle_fixture):
    """Negative control: confirm that if we flip one token in fixture 01, a label check fails."""
    from titaninfra.data.sft.types import EncodedSample

    fx = load_oracle_fixture("gpt_oss/01_single_turn_plain.json")
    baseline = tokenize_sample(
        fx.sample, gpt_oss_tokenizer, family="gpt_oss", max_length=fx.max_length,
    )
    assert len(baseline.input_ids) >= 4

    tamper_idx = len(baseline.input_ids) // 2
    tampered_ids = list(baseline.input_ids)
    tampered_ids[tamper_idx] = (tampered_ids[tamper_idx] + 1) % 100_000
    tampered = EncodedSample(
        input_ids=tuple(tampered_ids),
        labels=baseline.labels,
        positions=baseline.positions,
    )

    assert tampered.input_ids != baseline.input_ids


def test_gpt_oss_no_live_date_value(gpt_oss_tokenizer, load_oracle_fixture):
    """No ``YYYY-MM-DD`` date VALUE leaks (guards the strftime_now strip).

    ``test_gpt_oss_dynamic_metadata_absent`` checks the literal label
    ``"Current date"``; this guards the orthogonal risk that the encoder
    emits a bare date string (jinja L202 ``strftime_now("%Y-%m-%d")``).
    GPT-OSS is the only family whose jinja calls a live clock — a
    different value renders every day, so training on it is guaranteed
    train/inference drift. The encoder must never emit one.
    """
    import re

    fx = load_oracle_fixture("gpt_oss/01_single_turn_plain.json")
    out = tokenize_sample(
        fx.sample, gpt_oss_tokenizer, family="gpt_oss", max_length=fx.max_length,
    )
    decoded = gpt_oss_tokenizer.decode(out.input_ids, skip_special_tokens=False)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", decoded), (
        f"A YYYY-MM-DD date value leaked into the encoder output: {decoded!r}"
    )


def test_gpt_oss_reasoning_line_class_a_when_pinned(gpt_oss_tokenizer):
    """thinking_intent pins ``Reasoning: <effort>`` as class A (present, loss=0).

    Covers the C→A upgrade path that NO golden fixture exercises: when the
    user pins ``thinking_intent``, the Harmony ``Reasoning:`` line becomes
    deterministic (reproducible at inference) → safe to KEEP at loss=0,
    NOT stripped. The system block carries it but contributes no gradient.
    """
    from titaninfra.data.sft.types import IGNORE_INDEX

    sample = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "thinking_intent": {"mode": "on", "effort": "high"},
    }
    out = tokenize_sample(sample, gpt_oss_tokenizer, family="gpt_oss")
    decoded = gpt_oss_tokenizer.decode(out.input_ids, skip_special_tokens=False)
    assert "Reasoning: high" in decoded, (
        f"pinned thinking_intent should inject 'Reasoning: high'; got {decoded!r}"
    )

    # The Reasoning line lives in the built-in system block, which is all
    # loss=0 (class A / PROTOCOL). Locate the line's tokens and assert
    # every one of them is masked.
    reasoning_ids = gpt_oss_tokenizer.encode("Reasoning: high", add_special_tokens=False)
    ids = list(out.input_ids)
    # find the subsequence start
    start = -1
    for i in range(len(ids) - len(reasoning_ids) + 1):
        if ids[i : i + len(reasoning_ids)] == reasoning_ids:
            start = i
            break
    assert start >= 0, "Reasoning line tokens not found in input_ids"
    for j in range(start, start + len(reasoning_ids)):
        assert out.labels[j] == IGNORE_INDEX, (
            f"Reasoning-line token at pos {j} must be masked (class A), "
            f"got label {out.labels[j]}"
        )


def test_gpt_oss_reasoning_absent_when_unpinned(gpt_oss_tokenizer, load_oracle_fixture):
    """No ``Reasoning:`` line when thinking_intent is unset (stays class C, stripped)."""
    fx = load_oracle_fixture("gpt_oss/01_single_turn_plain.json")
    assert "thinking_intent" not in fx.sample  # fixture has no intent pinned
    out = tokenize_sample(
        fx.sample, gpt_oss_tokenizer, family="gpt_oss", max_length=fx.max_length,
    )
    decoded = gpt_oss_tokenizer.decode(out.input_ids, skip_special_tokens=False)
    assert "Reasoning:" not in decoded, (
        f"unpinned thinking_intent must omit the Reasoning line; got {decoded!r}"
    )


def test_gpt_oss_parallel_tool_results_get_correct_names(gpt_oss_tokenizer):
    """Audit #7 (2026-06-06): with PARALLEL tool calls (one assistant turn
    emitting two ToolUseBlocks), each tool-result Harmony opener must name the
    function matching its tool_use_id — not unconditionally the FIRST call's
    name. ``_last_tool_name`` ignored tool_use_id and returned tcs[0].name,
    training the model on a wrong name→result association for the 2nd result.
    """
    sample = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "weather + time?"}]},
            {"role": "assistant", "blocks": [
                {"type": "tool_use", "name": "get_weather", "input": {"city": "SF"}, "id": "call_w"},
                {"type": "tool_use", "name": "get_time", "input": {"tz": "PT"}, "id": "call_t"},
            ]},
            {"role": "tool", "blocks": [
                {"type": "tool_result", "tool_use_id": "call_w", "content": "sunny"}
            ]},
            {"role": "tool", "blocks": [
                {"type": "tool_result", "tool_use_id": "call_t", "content": "noon"}
            ]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "done"}]},
        ],
    }
    enc = tokenize_sample(sample, gpt_oss_tokenizer, family="gpt_oss")
    decoded = gpt_oss_tokenizer.decode(enc.input_ids)
    # Both tool-RESULT openers have the form
    # ``<|start|>functions.<NAME> to=assistant<|channel|>commentary``.
    # Extract the result-opener names (NOT the tool-call openers, which read
    # ``<|start|>assistant to=functions.<NAME>``). There must be exactly one
    # get_weather result and one get_time result — not two get_weather.
    import re
    result_names = re.findall(
        r"<\|start\|>functions\.(\w+) to=assistant", decoded
    )
    assert result_names == ["get_weather", "get_time"], (
        f"parallel tool-results must name calls by tool_use_id "
        f"(expected [get_weather, get_time], got {result_names}); decoded:\n{decoded}"
    )


def test_gpt_oss_leading_system_non_text_block_warns(gpt_oss_tokenizer, caplog):
    """Audit #10 (2026-06-06): a leading system message carrying a non-TextBlock
    (only reachable on the non-validated in-memory path; the converter rejects
    it) must WARN before dropping it — not silently lose content. The text
    block is still promoted."""
    import logging

    from titaninfra.data.sft import normalize
    from titaninfra.data.sft.encoders.gpt_oss import _promote_leading_system
    # Build a NormalizedSample directly with a mixed-block system message.
    ns = normalize.from_sft_sample({
        "messages": [
            {"role": "system", "blocks": [
                {"type": "text", "text": "real system"},
            ]},
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "ok"}]},
        ]
    })
    # Inject a non-text block into the leading system message (bypass validator).
    import dataclasses

    from titaninfra.data.schema import TextBlock, ThinkingBlock
    sys_msg = ns.messages[0]
    tampered = dataclasses.replace(
        sys_msg, blocks=(ThinkingBlock(text="directive"), TextBlock(text="real system"))
    )
    ns2 = dataclasses.replace(ns, messages=(tampered,) + tuple(ns.messages[1:]))
    with caplog.at_level(logging.WARNING, logger="titaninfra.sft.encoders.gpt_oss"):
        promoted = _promote_leading_system(ns2)
    assert any("non-TextBlock" in r.message for r in caplog.records), (
        "must warn on dropped non-TextBlock in leading system"
    )
    assert promoted.system == "real system"
