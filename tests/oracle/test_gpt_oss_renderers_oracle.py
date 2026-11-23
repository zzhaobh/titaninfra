"""GPT-OSS native encoder parity against the official-Harmony renderers oracle.

Unlike qwen3/llama3, ``test_gpt_oss_parity.py`` contains no native-vs-Jinja
comparison: its nine fixture cases are golden snapshots and its other tests are
native-only structural/capability checks. Those golden arrays and native-only
assertions are deliberately NOT copied here. Instead, the same nine fixture
*inputs* are submitted to ``renderers.gpt_oss`` as a genuinely independent
oracle. This is especially strong evidence because that renderer directly wraps
``openai_harmony``, the package used by vLLM's serving path, rather than
reimplementing Harmony itself.

Raw L1 is not identical. The helpers below first prove that the complete raw
difference consists only of explicitly asserted spans, then compare the shared
core token-for-token and scan every core label position:

* ``renderers_dynamic_system_metadata`` — official Harmony's knowledge-cutoff,
  current-date, and (when unpinned) default Reasoning lines. The date is pinned
  through ``conversation_start_date`` so this suite is deterministic.
* ``native_tool_call_json_literal`` — native and direct official Harmony with
  ``with_content_type("json")`` emit ``commentary json``; renderers 0.2.0 omits
  the content type. This is a renderers bug, recorded positively below.
* ``native_developer_trailing_blank_line`` — with instructions but no tools,
  native/HF emits ``\n\n`` before the developer closer; official Harmony does
  not. The whole, exactly asserted developer envelope is removed from both
  cores (tools-bearing developer envelopes remain in the core).
* ``tool_result_name_fallback_divergence`` — when the input lacks an explicit
  tool ``name`` (only ``tool_call_id``), renderers 0.2.0 cannot resolve it and
  emits ``functions.unknown`` where native resolves the real name via the
  assistant's matching tool_use block. This is a renderers bug, not native's;
  the whole native/renderers tool-result envelope pair is asserted exactly
  and masked before being excluded from the shared core ONLY in this case.
  2026-11-16 update: native previously also JSON-quoted string tool-result
  content (``message.content|tojson``, matching the stock HF jinja template)
  while official Harmony renders the raw string -- that divergence has been
  fixed in native (see docs/renderers_oracle_gptoss_report.md Sec 9 and
  encoders/gpt_oss.py::_tool_result_segment); when an explicit tool ``name``
  IS present (fixture 09), native and renderers now emit byte-identical
  tool-result envelopes and the span is left in the shared core like any
  other token run, not excluded.

The only accepted core-L2 pattern is ``renderers_trains_assistant_end``:
renderers trains every assistant analysis/final ``<|end|>``, while native masks
those protocol markers (the terminal ``<|return|>`` and tool ``<|call|>`` remain
trainable on both sides). The detector scans complete Harmony messages in the
forward direction. Any other label difference is UNEXPLAINED and fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

renderers = pytest.importorskip("renderers", reason="renderers package not installed")
openai_harmony = pytest.importorskip(
    "openai_harmony", reason="openai-harmony package not installed"
)

from openai_harmony import (  # noqa: E402
    HarmonyEncoding,
    HarmonyEncodingName,
    Role,
    load_harmony_encoding,
)
from openai_harmony import Message as HarmonyMessage  # noqa: E402
from renderers import GptOssRendererConfig, build_training_sample, create_renderer  # noqa: E402

from tests.oracle._renderers_oracle.diff import classify_label_divergence  # noqa: E402
from titaninfra.data.sft.dispatch import tokenize_sample  # noqa: E402
from titaninfra.data.sft.message_converter import normalize_to_hf_messages  # noqa: E402
from titaninfra.data.sft.types import IGNORE_INDEX, EncodedSample  # noqa: E402

_FIXED_DATE = "2026-11-16"
_METADATA_WITHOUT_REASONING = (
    "Knowledge cutoff: 2024-06\n"
    f"Current date: {_FIXED_DATE}\n\n"
)
_DEFAULT_REASONING = "Reasoning: medium\n\n"


@dataclass(frozen=True)
class _FixtureCase:
    path: str
    native_tokens: int
    renderers_tokens: int
    core_l2: int
    native_json_literals: int = 0
    tool_result_envelopes_dropped: int = 0
    has_developer_trailing_blank_line: bool = False


_FIXTURES = (
    _FixtureCase("01_single_turn_plain.json", 60, 85, 0),
    _FixtureCase(
        "02_with_developer_system.json",
        73,
        98,
        0,
        has_developer_trailing_blank_line=True,
    ),
    _FixtureCase("03_thinking_last_assistant.json", 89, 114, 1),
    _FixtureCase("04_thinking_history_preserved.json", 113, 138, 3),
    _FixtureCase("05_tool_call.json", 79, 103, 1, native_json_literals=1),
    _FixtureCase(
        "06_tool_response.json",
        91,
        116,
        0,
        native_json_literals=1,
        tool_result_envelopes_dropped=1,
    ),
    _FixtureCase("07_tools_top_level.json", 115, 140, 0),
    _FixtureCase(
        "08_full_loop.json",
        193,
        218,
        2,
        native_json_literals=1,
        tool_result_envelopes_dropped=1,
    ),
    _FixtureCase(
        "09_math_tools_real_world.json",
        1196,
        1220,
        2,
        native_json_literals=1,
        tool_result_envelopes_dropped=0,
    ),
)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    text: str


def _encoded(ids: list[int], labels: list[int]) -> EncodedSample:
    return EncodedSample(
        input_ids=tuple(ids),
        labels=tuple(labels),
        positions=tuple(range(len(ids))),
    )


def _render_with_gpt_oss_renderers(
    sample_json: dict,
    tokenizer,
    *,
    max_length: int | None = None,
) -> EncodedSample:
    """Map titaninfra's rich messages to the renderer's Harmony input contract."""
    messages, weights = normalize_to_hf_messages(
        sample_json.get("messages", []),
        system=sample_json.get("system"),
    )

    # normalize_to_hf_messages intentionally targets generic HF/Jinja templates:
    # an assistant with text+thinking but no tool call carries a literal
    # <think>...</think> wrapper. GptOssRenderer instead has a first-class
    # reasoning_content field. Convert that envelope without re-rendering any
    # Harmony bytes; tool-call assistants already use reasoning_content.
    for msg in messages:
        if msg.get("role") != "assistant" or "reasoning_content" in msg:
            continue
        content = msg.get("content") or ""
        if content.startswith("<think>\n") and "\n</think>\n\n" in content:
            reasoning, text = content[len("<think>\n"):].split("\n</think>\n\n", 1)
            msg["reasoning_content"] = reasoning
            msg["content"] = text

    intent = sample_json.get("thinking_intent") or {}
    if intent.get("mode") == "off":
        raise ValueError("GptOssRendererConfig cannot represent Reasoning: none")
    effort = intent.get("effort") or "medium"
    renderer = create_renderer(
        tokenizer,
        GptOssRendererConfig(
            reasoning_effort=effort,
            conversation_start_date=_FIXED_DATE,
            # Native deliberately preserves all current and historical thinking.
            auto_drop_analysis=False,
        ),
    )
    # The installed renderers (0.1.11) takes a role_to_mask(message) -> bool
    # callable, not the old per-message bool list; this dict + lambda preserves the
    # exact prior per-message granularity by keying on object identity (see
    # the llama3 oracle round for the full rationale).
    loss_mask_by_id = {id(m): w > 0 for m, w in zip(messages, weights, strict=True)}
    rendered = build_training_sample(
        renderer,
        messages,
        tools=sample_json.get("tools"),
        role_to_mask=lambda m: loss_mask_by_id[id(m)],
    )
    ids = rendered.token_ids
    mask = rendered.loss_mask
    if max_length is not None:
        ids = ids[:max_length]
        mask = mask[:max_length]
    return _encoded(
        ids,
        [tid if keep else IGNORE_INDEX for tid, keep in zip(ids, mask, strict=True)],
    )


def _message_spans(encoded: EncodedSample, tokenizer) -> list[_Span]:
    """Scan every token and return exact, non-overlapping Harmony envelopes."""
    ids = list(encoded.input_ids)
    start_id = tokenizer.convert_tokens_to_ids("<|start|>")
    stop_ids = {
        tokenizer.convert_tokens_to_ids(token)
        for token in ("<|end|>", "<|return|>", "<|call|>")
    }
    spans: list[_Span] = []
    cursor = 0
    while cursor < len(ids):
        assert ids[cursor] == start_id, (
            f"token {cursor} is outside a Harmony message envelope"
        )
        end = next(
            (i for i in range(cursor + 1, len(ids)) if ids[i] in stop_ids),
            None,
        )
        assert end is not None, f"unterminated Harmony message at token {cursor}"
        spans.append(
            _Span(
                start=cursor,
                end=end + 1,
                text=tokenizer.decode(
                    ids[cursor:end + 1], skip_special_tokens=False
                ),
            )
        )
        cursor = end + 1
    return spans


def _remove_exact_span(
    encoded: EncodedSample,
    tokenizer,
    text: str,
    *,
    expected_count: int = 1,
) -> tuple[EncodedSample, int]:
    """Remove only exact, masked token spans; no difflib or fuzzy alignment."""
    ids = list(encoded.input_ids)
    labels = list(encoded.labels)
    needle = tokenizer.encode(text, add_special_tokens=False)
    positions = [
        i
        for i in range(len(ids) - len(needle) + 1)
        if ids[i:i + len(needle)] == needle
    ]
    assert len(positions) == expected_count, (
        f"expected {expected_count} exact spans for {text!r}, got {positions}"
    )
    removed: set[int] = set()
    for start in positions:
        assert labels[start:start + len(needle)] == [IGNORE_INDEX] * len(needle)
        removed.update(range(start, start + len(needle)))
    return _encoded(
        [tid for i, tid in enumerate(ids) if i not in removed],
        [label for i, label in enumerate(labels) if i not in removed],
    ), len(needle) * expected_count


def _remove_native_tool_call_json_literals(
    native: EncodedSample,
    tokenizer,
    *,
    expected_count: int,
) -> EncodedSample:
    """Remove the exact native `` json`` header token(s), asserting context."""
    ids = list(native.input_ids)
    labels = list(native.labels)
    json_ids = tokenizer.encode(" json", add_special_tokens=False)
    assert len(json_ids) == 1
    json_id = json_ids[0]
    start_id = tokenizer.convert_tokens_to_ids("<|start|>")
    message_id = tokenizer.convert_tokens_to_ids("<|message|>")
    positions: list[int] = []
    for i in range(1, len(ids) - 1):
        if ids[i] != json_id or ids[i + 1] != message_id:
            continue
        opener = ids[:i]
        if start_id not in opener:
            continue
        message_start = len(opener) - 1 - opener[::-1].index(start_id)
        header = tokenizer.decode(ids[message_start:i + 2], skip_special_tokens=False)
        if header.startswith("<|start|>assistant to=functions.") and header.endswith(
            "<|channel|>commentary json<|message|>"
        ):
            positions.append(i)
    assert len(positions) == expected_count, (
        f"expected {expected_count} native tool-call json literals, got {positions}"
    )
    for i in positions:
        assert labels[i] == IGNORE_INDEX
    removed = set(positions)
    return _encoded(
        [tid for i, tid in enumerate(ids) if i not in removed],
        [label for i, label in enumerate(labels) if i not in removed],
    )


def _assert_span_masked(encoded: EncodedSample, span: _Span) -> None:
    assert encoded.labels[span.start:span.end] == (IGNORE_INDEX,) * (
        span.end - span.start
    )


def _without_spans(encoded: EncodedSample, spans: list[_Span]) -> EncodedSample:
    removed = {
        i
        for span in spans
        for i in range(span.start, span.end)
    }
    return _encoded(
        [tid for i, tid in enumerate(encoded.input_ids) if i not in removed],
        [label for i, label in enumerate(encoded.labels) if i not in removed],
    )


def _tool_call_names(messages: list[dict]) -> dict[str, str]:
    names: dict[str, str] = {}
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id")
            function = call.get("function") or {}
            if call_id and function.get("name"):
                names[call_id] = function["name"]
    return names


def _drop_exactly_asserted_divergent_envelopes(
    native: EncodedSample,
    renderers_out: EncodedSample,
    tokenizer,
    *,
    sample: dict,
) -> tuple[EncodedSample, EncodedSample, dict[str, int | bool]]:
    """Assert whole divergent envelopes exactly, then remove them from BOTH cores."""
    native_spans = _message_spans(native, tokenizer)
    renderers_spans = _message_spans(renderers_out, tokenizer)
    native_drop: list[_Span] = []
    renderers_drop: list[_Span] = []
    details: dict[str, int | bool] = {
        "developer_envelopes": 0,
        "tool_result_envelopes": 0,
        "renderers_unknown_tool_names": False,
    }

    # The blank-line difference exists only when instructions are the whole
    # developer message. With tools, both implementations retain the separator
    # before '# Tools', so that entire developer envelope stays in the core.
    if sample.get("system") and not sample.get("tools"):
        native_dev = [s for s in native_spans if s.text.startswith("<|start|>developer")]
        renderers_dev = [s for s in renderers_spans if s.text.startswith("<|start|>developer")]
        assert len(native_dev) == len(renderers_dev) == 1
        system = sample["system"]
        assert native_dev[0].text == (
            "<|start|>developer<|message|># Instructions\n\n"
            f"{system}\n\n<|end|>"
        )
        assert renderers_dev[0].text == (
            "<|start|>developer<|message|># Instructions\n\n"
            f"{system}<|end|>"
        )
        _assert_span_masked(native, native_dev[0])
        _assert_span_masked(renderers_out, renderers_dev[0])
        native_drop.extend(native_dev)
        renderers_drop.extend(renderers_dev)
        details["developer_envelopes"] = 1

    original_tool_messages = [
        msg for msg in sample.get("messages", []) if msg.get("role") == "tool"
    ]
    if original_tool_messages:
        normalized, _weights = normalize_to_hf_messages(
            sample.get("messages", []), system=sample.get("system")
        )
        normalized_tools = [msg for msg in normalized if msg.get("role") == "tool"]
        call_names = _tool_call_names(normalized)
        native_results = [
            s for s in native_spans
            if s.text.startswith("<|start|>functions.") and " to=assistant" in s.text
        ]
        renderers_results = [
            s for s in renderers_spans
            if s.text.startswith("<|start|>functions.") and " to=assistant" in s.text
        ]
        assert len(native_results) == len(renderers_results) == len(original_tool_messages)
        dropped_native: list[_Span] = []
        dropped_renderers: list[_Span] = []
        for original, normalized_tool, native_span, renderers_span in zip(
            original_tool_messages,
            normalized_tools,
            native_results,
            renderers_results,
            strict=True,
        ):
            content = normalized_tool["content"]
            native_name = original.get("name") or call_names[
                normalized_tool["tool_call_id"]
            ]
            # renderers 0.2.0 reads only msg['name']; it does not resolve the
            # HF-standard tool_call_id despite mentioning a client-side helper
            # that does not exist in the installed package.
            renderers_name = original.get("name") or "unknown"
            # 2026-11-16: native no longer json.dumps-quotes string content
            # (fixed to match official Harmony's raw-string rendering; see
            # encoders/gpt_oss.py::_tool_result_segment and
            # docs/renderers_oracle_gptoss_report.md Sec 9). Both sides emit
            # identical raw-content envelope text now; only the resolved
            # function NAME can still differ (the separate, still-open
            # renderers unknown-name bug below).
            assert native_span.text == (
                f"<|start|>functions.{native_name} to=assistant"
                "<|channel|>commentary<|message|>"
                f"{content}<|end|>"
            )
            assert renderers_span.text == (
                f"<|start|>functions.{renderers_name} to=assistant"
                "<|channel|>commentary<|message|>"
                f"{content}<|end|>"
            )
            _assert_span_masked(native, native_span)
            _assert_span_masked(renderers_out, renderers_span)
            if native_name == renderers_name:
                # Explicit tool `name` was given (e.g. fixture 09): content
                # AND name now match exactly, so the envelope is byte-
                # identical on both sides. Leave it in the shared core like
                # any other matching span instead of excluding it -- that
                # keeps the L1/L2 scan honest about what is actually equal,
                # rather than hiding a now-trivial pair behind the "dropped"
                # machinery meant for genuine divergences.
                continue
            if renderers_name == "unknown":
                details["renderers_unknown_tool_names"] = True
            dropped_native.append(native_span)
            dropped_renderers.append(renderers_span)
        native_drop.extend(dropped_native)
        renderers_drop.extend(dropped_renderers)
        details["tool_result_envelopes"] = len(dropped_native)

    return (
        _without_spans(native, native_drop),
        _without_spans(renderers_out, renderers_drop),
        details,
    )


def _assert_raw_l1_patterns_and_core_l2(
    native: EncodedSample,
    renderers_out: EncodedSample,
    tokenizer,
    *,
    sample: dict,
    expected_native_json_literals: int,
    expected_core_l2: int,
) -> dict[str, int | bool]:
    """Prove all raw L1 deltas, require core L1 equality, then scan all core L2."""
    assert native.input_ids != renderers_out.input_ids
    intent = sample.get("thinking_intent") or {}
    metadata = _METADATA_WITHOUT_REASONING
    if not intent:
        metadata += _DEFAULT_REASONING
    renderers_core, metadata_tokens = _remove_exact_span(
        renderers_out, tokenizer, metadata
    )
    native_core = _remove_native_tool_call_json_literals(
        native,
        tokenizer,
        expected_count=expected_native_json_literals,
    )
    native_core, renderers_core, details = _drop_exactly_asserted_divergent_envelopes(
        native_core,
        renderers_core,
        tokenizer,
        sample=sample,
    )
    assert native_core.input_ids == renderers_core.input_ids, (
        "canonical L1 mismatch after removing only exactly asserted spans; "
        f"native={native_core.input_ids}, renderers={renderers_core.input_ids}"
    )

    result = classify_label_divergence(
        ids=native_core.input_ids,
        native_labels=native_core.labels,
        renderers_labels=renderers_core.labels,
        tokenizer=tokenizer,
        family="gpt_oss",
    )
    assert result["unexplained_positions"] == [], f"unexplained core L2: {result}"
    assert result["n_divergent"] == expected_core_l2, result
    if expected_core_l2:
        assert result["counts"] == {"renderers_trains_assistant_end": expected_core_l2}
        for i, (native_label, renderers_label) in enumerate(
            zip(native_core.labels, renderers_core.labels, strict=True)
        ):
            if native_label != renderers_label:
                assert native_label == IGNORE_INDEX
                assert renderers_label == native_core.input_ids[i]
    return {
        "metadata_tokens": metadata_tokens,
        "native_json_literals": expected_native_json_literals,
        "core_tokens": len(native_core.input_ids),
        "core_l2": result["n_divergent"],
        **details,
    }


@pytest.mark.parametrize("case", _FIXTURES, ids=lambda case: case.path.removesuffix(".json"))
def test_gpt_oss_fixture_renderers_raw_spans_and_core_ok(
    case: _FixtureCase,
    gpt_oss_tokenizer,
    load_oracle_fixture,
):
    """Each fixture fails on wrong Harmony bytes, masks, or a new uncatalogued delta."""
    fx = load_oracle_fixture(f"gpt_oss/{case.path}")
    native = tokenize_sample(
        fx.sample,
        gpt_oss_tokenizer,
        family="gpt_oss",
        max_length=fx.max_length,
    )
    renderers_out = _render_with_gpt_oss_renderers(
        fx.sample, gpt_oss_tokenizer, max_length=fx.max_length
    )
    assert len(native.input_ids) == case.native_tokens
    assert len(renderers_out.input_ids) == case.renderers_tokens
    result = _assert_raw_l1_patterns_and_core_l2(
        native,
        renderers_out,
        gpt_oss_tokenizer,
        sample=fx.sample,
        expected_native_json_literals=case.native_json_literals,
        expected_core_l2=case.core_l2,
    )
    assert result["metadata_tokens"] == 25
    assert bool(result["developer_envelopes"]) == case.has_developer_trailing_blank_line
    assert result["tool_result_envelopes"] == case.tool_result_envelopes_dropped


@pytest.mark.parametrize("effort", [None, "low", "medium", "high"])
def test_gpt_oss_reasoning_effort_renderers_core_ok(effort, gpt_oss_tokenizer):
    """Every pinned effort maps one-to-one; only cutoff/date metadata is stripped."""
    intent = {"mode": "on"}
    if effort is not None:
        intent["effort"] = effort
    sample = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "thinking_intent": intent,
    }
    native = tokenize_sample(sample, gpt_oss_tokenizer, family="gpt_oss")
    renderers_out = _render_with_gpt_oss_renderers(sample, gpt_oss_tokenizer)
    expected_effort = effort or "medium"
    reasoning_text = f"Reasoning: {expected_effort}"
    assert reasoning_text in gpt_oss_tokenizer.decode(native.input_ids)
    assert reasoning_text in gpt_oss_tokenizer.decode(renderers_out.input_ids)
    reasoning_ids = gpt_oss_tokenizer.encode(
        reasoning_text, add_special_tokens=False
    )
    for output in (native, renderers_out):
        ids = list(output.input_ids)
        starts = [
            i
            for i in range(len(ids) - len(reasoning_ids) + 1)
            if ids[i:i + len(reasoning_ids)] == reasoning_ids
        ]
        assert len(starts) == 1
        start = starts[0]
        assert output.labels[start:start + len(reasoning_ids)] == (
            IGNORE_INDEX,
        ) * len(reasoning_ids)
    result = _assert_raw_l1_patterns_and_core_l2(
        native,
        renderers_out,
        gpt_oss_tokenizer,
        sample=sample,
        expected_native_json_literals=0,
        expected_core_l2=0,
    )
    assert result["metadata_tokens"] == len(
        gpt_oss_tokenizer.encode(
            _METADATA_WITHOUT_REASONING, add_special_tokens=False
        )
    )


@pytest.mark.skip(
    reason=(
        "GptOssRendererConfig.reasoning_effort is Literal['low','medium','high']; "
        "openai_harmony.ReasoningEffort has no 'none'. Native mode=off emits the "
        "deterministic 'Reasoning: none' class-A line, which renderers 0.2.0 "
        "cannot represent. Structurally unsupported, not a mask discrepancy."
    )
)
def test_gpt_oss_reasoning_mode_off_renderers_unsupported():
    pass


def test_gpt_oss_renderer_really_uses_official_harmony(
    gpt_oss_tokenizer, load_oracle_fixture
):
    """The oracle is the production Harmony package, not another hand-written renderer."""
    renderer = create_renderer(gpt_oss_tokenizer, GptOssRendererConfig())
    assert isinstance(renderer._enc, HarmonyEncoding)
    fx = load_oracle_fixture("gpt_oss/01_single_turn_plain.json")
    rendered = _render_with_gpt_oss_renderers(fx.sample, gpt_oss_tokenizer)
    official_ids = renderer._enc.render(
        HarmonyMessage.from_role_and_content(
            Role.USER, "What is the capital of France?"
        )
    )
    ids = list(rendered.input_ids)
    assert any(
        ids[i:i + len(official_ids)] == official_ids
        for i in range(len(ids) - len(official_ids) + 1)
    )


def test_gpt_oss_renderers_missing_tool_call_content_type_is_renderers_bug(
    gpt_oss_tokenizer, load_oracle_fixture
):
    """Verify gpt oss renderers missing tool call content type is renderers bug."""
    fx = load_oracle_fixture("gpt_oss/05_tool_call.json")
    native = tokenize_sample(fx.sample, gpt_oss_tokenizer, family="gpt_oss")
    rendered = _render_with_gpt_oss_renderers(fx.sample, gpt_oss_tokenizer)
    native_text = gpt_oss_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = gpt_oss_tokenizer.decode(rendered.input_ids, skip_special_tokens=False)
    assert "commentary json<|message|>" in native_text
    assert "commentary json<|message|>" not in renderers_text
    assert "commentary<|message|>" in renderers_text

    # Direct official Harmony proves the native bytes are exactly reproducible
    # when the caller sets content_type; renderers._assistant_to_harmony omits it.
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    official = (
        HarmonyMessage.from_role_and_content(Role.ASSISTANT, '{"expr": "2+3"}')
        .with_channel("commentary")
        .with_recipient("functions.calc")
        .with_content_type("json")
    )
    official_text = gpt_oss_tokenizer.decode(enc.render(official), skip_special_tokens=False)
    assert official_text in native_text


def test_gpt_oss_renderers_unknown_tool_result_name_is_renderers_bug(
    gpt_oss_tokenizer, load_oracle_fixture
):
    """Verify gpt oss renderers unknown tool result name is renderers bug."""
    fx = load_oracle_fixture("gpt_oss/06_tool_response.json")
    native = tokenize_sample(fx.sample, gpt_oss_tokenizer, family="gpt_oss")
    rendered = _render_with_gpt_oss_renderers(fx.sample, gpt_oss_tokenizer)
    native_text = gpt_oss_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = gpt_oss_tokenizer.decode(rendered.input_ids, skip_special_tokens=False)
    assert "<|start|>functions.calc to=assistant" in native_text
    assert "<|start|>functions.unknown to=assistant" in renderers_text


def test_gpt_oss_native_tool_result_matches_official_harmony_raw_string(
    gpt_oss_tokenizer, load_oracle_fixture
):
    """Regression guard: native no longer JSON-quotes tool-result content.

    Until 2026-11-16 this test documented the OPPOSITE: native matched the
    stock HF jinja template's ``tojson`` filter (quoting string content, e.g.
    ``"5"`` instead of ``5``) while official Harmony -- the renderer vLLM's
    serving path actually uses -- renders the raw string. That was a real,
    decided train/inference wire divergence (see
    docs/renderers_oracle_gptoss_report.md Sec 5 L1-D and Sec 9), so
    ``encoders/gpt_oss.py::_tool_result_segment`` was changed to drop the
    ``json.dumps`` call and emit the raw string, matching official Harmony
    byte-for-byte. This test now locks in the corrected behavior so a future
    regression back to JSON-quoting fails loudly instead of silently.
    """
    fx = load_oracle_fixture("gpt_oss/09_math_tools_real_world.json")
    native = tokenize_sample(fx.sample, gpt_oss_tokenizer, family="gpt_oss")
    native_text = gpt_oss_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    tool_message = fx.sample["messages"][2]
    tool_name = tool_message["name"]
    content = tool_message["blocks"][0]["text"]
    native_result = (
        f"<|start|>functions.{tool_name} to=assistant"
        "<|channel|>commentary<|message|>"
        f"{content}<|end|>"
    )
    previously_quoted_result = (
        f"<|start|>functions.{tool_name} to=assistant"
        "<|channel|>commentary<|message|>"
        f"{json.dumps(content, ensure_ascii=False)}<|end|>"
    )
    assert native_result in native_text
    assert previously_quoted_result not in native_text

    # Direct official Harmony confirms native is now byte-identical to it.
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    official = (
        HarmonyMessage.from_author_and_content(
            {"role": "tool", "name": f"functions.{tool_name}"}, content
        )
        .with_recipient("assistant")
        .with_channel("commentary")
    )
    official_text = gpt_oss_tokenizer.decode(enc.render(official), skip_special_tokens=False)
    assert official_text == native_result
    assert official_text in native_text
