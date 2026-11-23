"""Gemma4 native encoder parity against the independent ``renderers`` oracle.

This covers all nine native-vs-Jinja scenarios in ``test_gemma4_parity.py``:
fixtures 01--09. The tokenizer/fixture smoke tests and tamper negative control
are not oracle comparisons and are deliberately not copied. Fixtures 06/08/09
are intentional native-vs-old-Jinja *L2* divergences (the legacy offset mask
lost tool calls and continuation reasoning); their input token streams are
still native-vs-Jinja identical, so all three belong in this second-oracle
comparison rather than being skipped.

Raw L1 differs on every fixture, but the complete difference is decomposed
into exact, named, token-for-token asserted spans before the shared core is
compared:

* ``renderers_empty_thought_prefill`` — renderers deliberately re-emits the
  disabled-thinking ``<|channel>thought\n<channel|>`` generation prefill on
  non-continuation assistant turns for prefix stability. Its own module
  documents this as a deliberate deviation from ``apply_chat_template``;
  native and the official fixture Jinja do not emit it in training history.
* ``tool_call_argument_serialization`` — native preserves the official Jinja
  string-arguments branch (including JSON braces/escapes), while renderers
  parses the JSON string and selects the official mapping-arguments branch.
  The complete native/renderers tool-call envelope pair is asserted exactly
  and trained on both sides before excluding that pair from the shared core.
  This is an input-normalization/wire choice, not counted as a confirmed bug.
* ``native_post_tool_content_turn_close`` — in fixture 08, official Jinja and
  native close an assistant turn when tool responses are followed by assistant
  content. Renderers 0.2.0 wrongly suppresses that ``<turn|>\n`` whenever the
  next non-tool message is also assistant, concatenating ``both.Tokyo``. This
  is a renderers bug, recorded positively below.

After removing only those spans, shared-core L1 is identical on all nine
fixtures. The exhaustive forward L2 scan accepts exactly three patterns:
renderers masks the newline after an assistant ``<turn|>``; renderers trains
the initial ``<|tool_response>`` opener even when an actual tool result follows;
and renderers trains the non-empty thought-channel scaffold which native masks
as protocol. Every other position is UNEXPLAINED and fails. No lookback-window
classifier, sampled positions, fuzzy alignment, or unexplained residue is used.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

renderers = pytest.importorskip("renderers", reason="renderers package not installed")

from renderers import Gemma4RendererConfig, build_training_sample, create_renderer  # noqa: E402

from tests.oracle._renderers_oracle.diff import classify_label_divergence  # noqa: E402
from titaninfra.data.sft import normalize  # noqa: E402
from titaninfra.data.sft.dispatch import tokenize_sample  # noqa: E402
from titaninfra.data.sft.encoders import get_encoder  # noqa: E402
from titaninfra.data.sft.message_converter import normalize_to_hf_messages  # noqa: E402
from titaninfra.data.sft.types import IGNORE_INDEX, EncodedSample  # noqa: E402

_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"
_L2_TURN_NEWLINE = "renderers_masks_assistant_turn_close_newline"
_L2_TOOL_RESPONSE_OPEN = "renderers_trains_initial_tool_response_open"
_L2_THOUGHT_SCAFFOLD = "renderers_trains_thought_channel_scaffold"


@dataclass(frozen=True)
class _FixtureCase:
    path: str
    native_tokens: int
    renderers_tokens: int
    core_tokens: int
    empty_thoughts: int
    tool_calls: int = 0
    native_close_context: str | None = None
    core_l2: dict[str, int] = field(default_factory=dict)


_FIXTURES = (
    _FixtureCase(
        "01_single_turn_plain.json", 25, 29, 25, 1,
        core_l2={_L2_TURN_NEWLINE: 1},
    ),
    _FixtureCase(
        "02_with_system.json", 45, 49, 45, 1,
        core_l2={_L2_TURN_NEWLINE: 1},
    ),
    _FixtureCase(
        "03_thinking_last_assistant.json", 91, 95, 91, 1,
        core_l2={_L2_TURN_NEWLINE: 1},
    ),
    _FixtureCase(
        "04_thinking_history_drop.json", 61, 69, 61, 2,
        core_l2={_L2_TURN_NEWLINE: 2},
    ),
    _FixtureCase("05_tool_call.json", 36, 40, 22, 1, tool_calls=1),
    _FixtureCase(
        "06_tool_response.json", 70, 74, 56, 1, tool_calls=1,
        core_l2={_L2_TOOL_RESPONSE_OPEN: 1, _L2_TURN_NEWLINE: 1},
    ),
    _FixtureCase(
        "07_tools_top_level.json", 86, 90, 86, 1,
        core_l2={_L2_TURN_NEWLINE: 1},
    ),
    _FixtureCase(
        "08_full_tool_loop.json", 170, 172, 140, 1, tool_calls=2,
        native_close_context="Let me check both.",
        core_l2={_L2_TOOL_RESPONSE_OPEN: 1, _L2_TURN_NEWLINE: 1},
    ),
    _FixtureCase(
        "09_math_tools_real_world.json", 1149, 1146, 1076, 0, tool_calls=1,
        core_l2={
            _L2_THOUGHT_SCAFFOLD: 5,
            _L2_TOOL_RESPONSE_OPEN: 1,
            _L2_TURN_NEWLINE: 1,
        },
    ),
)


def _encoded(ids, labels) -> EncodedSample:
    ids = list(ids)
    return EncodedSample(
        input_ids=tuple(ids),
        labels=tuple(labels),
        positions=tuple(range(len(ids))),
    )


def _render_with_gemma4_renderers(
    sample_json: dict,
    tokenizer,
    *,
    max_length: int | None = None,
) -> EncodedSample:
    """Use Gemma4Renderer's own wire and sampled mask, without re-rendering it."""
    messages, weights = normalize_to_hf_messages(
        sample_json.get("messages", []),
        system=sample_json.get("system"),
    )
    intent = sample_json.get("thinking_intent") or {}
    renderer = create_renderer(
        tokenizer,
        Gemma4RendererConfig(
            enable_thinking=intent.get("mode") == "on",
            # Native's explicit channel-thought gate is after the last user,
            # matching the renderer default. Inline <think> history is content
            # on both sides and is unaffected by this tool-cycle setting.
            preserve_thinking=False,
        ),
    )
    # role_to_mask(message) -> bool replaces the old per-message bool list;
    # keyed on object identity to keep the exact per-message granularity
    # (see the llama3 oracle for the full rationale).
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


def _without_positions(encoded: EncodedSample, positions: set[int]) -> EncodedSample:
    return _encoded(
        [tid for i, tid in enumerate(encoded.input_ids) if i not in positions],
        [label for i, label in enumerate(encoded.labels) if i not in positions],
    )


def _matching_starts(ids: list[int], needle: list[int]) -> list[int]:
    return [
        i
        for i in range(len(ids) - len(needle) + 1)
        if ids[i:i + len(needle)] == needle
    ]


def _remove_exact_span(
    encoded: EncodedSample,
    tokenizer,
    text: str,
    *,
    expected_count: int,
    expected_label: str,
) -> tuple[EncodedSample, int]:
    """Remove only exact spans whose complete label direction is also asserted."""
    ids = list(encoded.input_ids)
    needle = tokenizer.encode(text, add_special_tokens=False)
    starts = _matching_starts(ids, needle)
    assert len(starts) == expected_count, (
        f"expected {expected_count} exact spans for {text!r}, got {starts}"
    )
    expected_labels = (
        needle if expected_label == "trained" else [IGNORE_INDEX] * len(needle)
    )
    removed: set[int] = set()
    for start in starts:
        assert encoded.labels[start:start + len(needle)] == tuple(expected_labels)
        removed.update(range(start, start + len(needle)))
    return _without_positions(encoded, removed), len(needle) * expected_count


def _expected_tool_envelopes(sample: dict) -> list[tuple[str, str]]:
    """Hand-derive native string-branch and renderer mapping-branch envelopes."""
    pairs: list[tuple[str, str]] = []
    for message in sample.get("messages", []):
        for block in message.get("blocks", []):
            if block.get("type") != "tool_use":
                continue
            arguments = block["input"]
            # All nine upstream fixtures use flat string-valued arguments. Do
            # not silently generalize this independent expectation to untested
            # nested schemas.
            assert isinstance(arguments, dict)
            assert all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in arguments.items()
            )
            json_arguments = json.dumps(arguments, ensure_ascii=False)
            native = (
                f"<|tool_call>call:{block['name']}{{{json_arguments}}}<tool_call|>"
            )
            mapping_body = "{" + ",".join(
                f"{key}:<|\"|>{arguments[key]}<|\"|>"
                for key in sorted(arguments)
            ) + "}"
            renderer = (
                f"<|tool_call>call:{block['name']}{mapping_body}<tool_call|>"
            )
            pairs.append((native, renderer))
    return pairs


def _drop_tool_argument_envelopes(
    native: EncodedSample,
    renderers_out: EncodedSample,
    tokenizer,
    *,
    sample: dict,
    expected_count: int,
) -> tuple[EncodedSample, EncodedSample]:
    pairs = _expected_tool_envelopes(sample)
    assert len(pairs) == expected_count
    for native_text, renderers_text in pairs:
        native, _ = _remove_exact_span(
            native, tokenizer, native_text,
            expected_count=1, expected_label="trained",
        )
        renderers_out, _ = _remove_exact_span(
            renderers_out, tokenizer, renderers_text,
            expected_count=1, expected_label="trained",
        )
    return native, renderers_out


def _remove_native_post_tool_content_close(
    native: EncodedSample,
    tokenizer,
    *,
    content: str,
) -> EncodedSample:
    """Remove only the official/native close after the exact assistant content."""
    context_ids = tokenizer.encode(content + "<turn|>\n", add_special_tokens=False)
    close_ids = tokenizer.encode("<turn|>\n", add_special_tokens=False)
    assert context_ids[-len(close_ids):] == close_ids
    starts = _matching_starts(list(native.input_ids), context_ids)
    assert len(starts) == 1, f"expected one post-tool content close, got {starts}"
    close_start = starts[0] + len(context_ids) - len(close_ids)
    assert native.labels[close_start:close_start + len(close_ids)] == tuple(close_ids)
    return _without_positions(
        native, set(range(close_start, close_start + len(close_ids)))
    )


def _assert_exactly_one_masked_bos(encoded: EncodedSample, tokenizer) -> None:
    bos_id = tokenizer.convert_tokens_to_ids("<bos>")
    assert encoded.input_ids[0] == bos_id
    assert encoded.input_ids.count(bos_id) == 1
    assert encoded.labels[0] == IGNORE_INDEX


def _assert_core_l2(
    native: EncodedSample,
    renderers_out: EncodedSample,
    tokenizer,
    *,
    expected_counts: dict[str, int],
) -> None:
    result = classify_label_divergence(
        ids=native.input_ids,
        native_labels=native.labels,
        renderers_labels=renderers_out.labels,
        tokenizer=tokenizer,
        family="gemma4",
    )
    assert result["unexplained_positions"] == [], f"unexplained gemma4 core L2: {result}"
    assert result["counts"] == expected_counts, result
    assert result["n_divergent"] == sum(expected_counts.values()), result
    for i, (native_label, renderers_label) in enumerate(
        zip(native.labels, renderers_out.labels, strict=True)
    ):
        if native_label == renderers_label:
            continue
        if native_label == IGNORE_INDEX:
            assert renderers_label == native.input_ids[i]
        else:
            assert native_label == native.input_ids[i]
            assert renderers_label == IGNORE_INDEX


def _assert_raw_l1_patterns_and_core_l2(
    native: EncodedSample,
    renderers_out: EncodedSample,
    tokenizer,
    *,
    sample: dict,
    case: _FixtureCase,
) -> None:
    """Account for every raw-L1 delta, then scan every shared-core label."""
    _assert_exactly_one_masked_bos(native, tokenizer)
    _assert_exactly_one_masked_bos(renderers_out, tokenizer)
    assert native.input_ids != renderers_out.input_ids

    renderers_core, thought_tokens = _remove_exact_span(
        renderers_out, tokenizer, _EMPTY_THOUGHT,
        expected_count=case.empty_thoughts, expected_label="trained",
    )
    assert thought_tokens == 4 * case.empty_thoughts
    native_core, renderers_core = _drop_tool_argument_envelopes(
        native, renderers_core, tokenizer,
        sample=sample, expected_count=case.tool_calls,
    )
    if case.native_close_context is not None:
        native_core = _remove_native_post_tool_content_close(
            native_core, tokenizer, content=case.native_close_context
        )

    assert len(native_core.input_ids) == len(renderers_core.input_ids) == case.core_tokens
    assert native_core.input_ids == renderers_core.input_ids, (
        "gemma4 core L1 mismatch after removing only exactly asserted spans"
    )
    _assert_core_l2(
        native_core, renderers_core, tokenizer, expected_counts=case.core_l2
    )


@pytest.mark.parametrize("case", _FIXTURES, ids=lambda case: case.path.removesuffix(".json"))
def test_gemma4_fixture_renderers_raw_spans_and_core_ok(
    case: _FixtureCase,
    gemma4_tokenizer,
    load_oracle_fixture,
):
    """Any new byte, mask, or uncatalogued span in a mirrored scenario fails here."""
    fx = load_oracle_fixture(f"gemma4/{case.path}")
    native = tokenize_sample(
        fx.sample, gemma4_tokenizer,
        family="gemma4", max_length=fx.max_length,
    )
    renderers_out = _render_with_gemma4_renderers(
        fx.sample, gemma4_tokenizer, max_length=fx.max_length
    )
    assert len(native.input_ids) == case.native_tokens
    assert len(renderers_out.input_ids) == case.renderers_tokens
    _assert_raw_l1_patterns_and_core_l2(
        native, renderers_out, gemma4_tokenizer,
        sample=fx.sample, case=case,
    )


def test_gemma4_thinking_intent_on_has_one_initial_system_think_block(
    gemma4_tokenizer,
):
    """The ephemeral system block is initial, masked, and shared; no thought prefill fires."""
    sample = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
        "thinking_intent": {"mode": "on"},
    }
    native = tokenize_sample(sample, gemma4_tokenizer, family="gemma4")
    renderers_out = _render_with_gemma4_renderers(sample, gemma4_tokenizer)
    prefix = "<bos><|turn>system\n<|think|>\n<turn|>\n"
    prefix_ids = gemma4_tokenizer.encode(prefix, add_special_tokens=False)
    for output in (native, renderers_out):
        _assert_exactly_one_masked_bos(output, gemma4_tokenizer)
        text = gemma4_tokenizer.decode(output.input_ids, skip_special_tokens=False)
        assert text.count("<|turn>system\n") == 1
        assert output.input_ids[:len(prefix_ids)] == tuple(prefix_ids)
        assert output.labels[:len(prefix_ids)] == (IGNORE_INDEX,) * len(prefix_ids)
        assert _EMPTY_THOUGHT not in text
    assert native.input_ids == renderers_out.input_ids
    _assert_core_l2(
        native, renderers_out, gemma4_tokenizer,
        expected_counts={_L2_TURN_NEWLINE: 1},
    )


def test_gemma4_explicit_system_block_is_not_repeated_in_history(
    gemma4_tokenizer, load_oracle_fixture,
):
    """Explicit system is one initial protocol block, not per-turn dynamic metadata."""
    fx = load_oracle_fixture("gemma4/04_thinking_history_drop.json")
    sample = copy.deepcopy(fx.sample)
    sample["system"] = "Answer tersely."
    native = tokenize_sample(sample, gemma4_tokenizer, family="gemma4")
    renderers_out = _render_with_gemma4_renderers(sample, gemma4_tokenizer)
    for output in (native, renderers_out):
        text = gemma4_tokenizer.decode(output.input_ids, skip_special_tokens=False)
        assert text.startswith("<bos><|turn>system\nAnswer tersely.<turn|>\n")
        assert text.count("<|turn>system\n") == 1
        header = text.split("<|turn>user\n", 1)[0]
        header_ids = gemma4_tokenizer.encode(header, add_special_tokens=False)
        assert output.input_ids[:len(header_ids)] == tuple(header_ids)
        assert output.labels[:len(header_ids)] == (IGNORE_INDEX,) * len(header_ids)


def test_gemma4_standalone_tool_declaration_block_is_shared_and_masked(
    gemma4_tokenizer, load_oracle_fixture,
):
    """Top-level tools stay in the core as standalone <|tool>...<tool|> blocks."""
    fx = load_oracle_fixture("gemma4/07_tools_top_level.json")
    native = tokenize_sample(fx.sample, gemma4_tokenizer, family="gemma4")
    renderers_out = _render_with_gemma4_renderers(fx.sample, gemma4_tokenizer)
    native_text = gemma4_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = gemma4_tokenizer.decode(renderers_out.input_ids, skip_special_tokens=False)
    start = native_text.index("<|tool>")
    end = native_text.index("<tool|>", start) + len("<tool|>")
    declaration = native_text[start:end]
    assert declaration.startswith("<|tool>declaration:get_weather{")
    assert declaration in renderers_text
    for output in (native, renderers_out):
        declaration_ids = gemma4_tokenizer.encode(declaration, add_special_tokens=False)
        starts = _matching_starts(list(output.input_ids), declaration_ids)
        assert len(starts) == 1
        span_start = starts[0]
        assert output.labels[span_start:span_start + len(declaration_ids)] == (
            IGNORE_INDEX,
        ) * len(declaration_ids)


def test_gemma4_bos_is_single_despite_tokenizer_auto_bos(
    gemma4_tokenizer, load_oracle_fixture,
):
    """Both implementations explicitly emit BOS and tokenize text with special tokens off."""
    bos_id = gemma4_tokenizer.convert_tokens_to_ids("<bos>")
    plain = gemma4_tokenizer.encode("hello", add_special_tokens=False)
    assert gemma4_tokenizer.encode("hello", add_special_tokens=True) == [bos_id, *plain]
    fx = load_oracle_fixture("gemma4/01_single_turn_plain.json")
    native = tokenize_sample(fx.sample, gemma4_tokenizer, family="gemma4")
    renderers_out = _render_with_gemma4_renderers(fx.sample, gemma4_tokenizer)
    for output in (native, renderers_out):
        _assert_exactly_one_masked_bos(output, gemma4_tokenizer)
    messages, _weights = normalize_to_hf_messages(fx.sample["messages"])
    official = gemma4_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        enable_thinking=False,
    )
    official_ids = official["input_ids"] if isinstance(official, Mapping) else official
    assert official_ids[0] == bos_id
    assert official_ids.count(bos_id) == 1


def test_gemma4_generation_thought_preface_native_strips_renderers_preserves(
    gemma4_tokenizer,
):
    """Native strips only the generation preface; official Jinja/renderers retain it."""
    sample = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }
    messages, _weights = normalize_to_hf_messages(sample["messages"])
    normalized = normalize.from_sft_sample(sample)
    native = get_encoder("gemma4", gemma4_tokenizer).encode(
        normalized, add_generation_prompt=True
    )
    renderer = create_renderer(gemma4_tokenizer, Gemma4RendererConfig())
    rendered = renderer.render(messages, add_generation_prompt=True)
    official_text = gemma4_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    native_text = gemma4_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = gemma4_tokenizer.decode(rendered.token_ids, skip_special_tokens=False)
    trailer = "<|turn>model\n"
    assert native_text.endswith(trailer)
    assert not native_text.endswith(trailer + _EMPTY_THOUGHT)
    assert official_text == native_text + _EMPTY_THOUGHT
    assert renderers_text.endswith(trailer + _EMPTY_THOUGHT)
    # The renderer also re-emits the prefill on the completed assistant turn;
    # that separate training-history behavior is the named raw-L1 span above.
    assert renderers_text.count(_EMPTY_THOUGHT) == 2
    trailer_ids = gemma4_tokenizer.encode(trailer, add_special_tokens=False)
    assert native.input_ids[-len(trailer_ids):] == tuple(trailer_ids)
    assert native.labels[-len(trailer_ids):] == (IGNORE_INDEX,) * len(trailer_ids)


def test_gemma4_tool_arguments_renderers_selects_official_mapping_branch(
    gemma4_tokenizer, load_oracle_fixture,
):
    """String and mapping are distinct official wires; renderers normalizes to mapping."""
    fx = load_oracle_fixture("gemma4/05_tool_call.json")
    messages, _weights = normalize_to_hf_messages(fx.sample["messages"])
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Tokyo"}'
    mapping_messages = copy.deepcopy(messages)
    mapping_messages[1]["tool_calls"][0]["function"]["arguments"] = {"city": "Tokyo"}
    official_string = gemma4_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    official_mapping = gemma4_tokenizer.apply_chat_template(
        mapping_messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    native = tokenize_sample(fx.sample, gemma4_tokenizer, family="gemma4")
    renderers_out = _render_with_gemma4_renderers(fx.sample, gemma4_tokenizer)
    native_text = gemma4_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = gemma4_tokenizer.decode(renderers_out.input_ids, skip_special_tokens=False)
    native_arguments = 'call:get_weather{{"city": "Tokyo"}}'
    mapping_arguments = "call:get_weather{city:<|\"|>Tokyo<|\"|>}"
    assert official_string == native_text
    assert native_arguments in official_string
    assert mapping_arguments in official_mapping
    assert mapping_arguments in renderers_text
    assert native_arguments not in renderers_text


def test_gemma4_renderers_missing_post_tool_content_turn_close_is_renderers_bug(
    gemma4_tokenizer, load_oracle_fixture,
):
    """Verify gemma4 renderers missing post tool content turn close is renderers bug."""
    fx = load_oracle_fixture("gemma4/08_full_tool_loop.json")
    messages, _weights = normalize_to_hf_messages(
        fx.sample["messages"], system=fx.sample.get("system")
    )
    official_text = gemma4_tokenizer.apply_chat_template(
        messages, tools=fx.sample.get("tools"), tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    native = tokenize_sample(fx.sample, gemma4_tokenizer, family="gemma4")
    renderers_out = _render_with_gemma4_renderers(fx.sample, gemma4_tokenizer)
    native_text = gemma4_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = gemma4_tokenizer.decode(renderers_out.input_ids, skip_special_tokens=False)
    correct = "<tool_response|>Let me check both.<turn|>\nTokyo is 18°C, SF is 14°C."
    broken = "<tool_response|>Let me check both.Tokyo is 18°C, SF is 14°C."
    assert official_text == native_text
    assert correct in official_text
    assert correct in native_text
    assert correct not in renderers_text
    assert broken in renderers_text


def test_gemma4_renderer_is_handwritten_and_official_jinja_anchors_plain_wire(
    gemma4_tokenizer, load_oracle_fixture,
):
    """Actual type + independent Jinja distinguish a handwritten mirror from an official wrapper."""
    renderer = create_renderer(gemma4_tokenizer, Gemma4RendererConfig())
    assert type(renderer).__module__ == "renderers.gemma4"
    assert type(renderer).__name__ == "Gemma4Renderer"
    fx = load_oracle_fixture("gemma4/01_single_turn_plain.json")
    messages, _weights = normalize_to_hf_messages(fx.sample["messages"])
    official_text = gemma4_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    native = tokenize_sample(fx.sample, gemma4_tokenizer, family="gemma4")
    native_text = gemma4_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_out = _render_with_gemma4_renderers(fx.sample, gemma4_tokenizer)
    renderers_core, removed = _remove_exact_span(
        renderers_out, gemma4_tokenizer, _EMPTY_THOUGHT,
        expected_count=1, expected_label="trained",
    )
    official_ids = gemma4_tokenizer.encode(official_text, add_special_tokens=False)
    assert official_text == native_text
    assert removed == 4
    assert renderers_core.input_ids == tuple(official_ids)
