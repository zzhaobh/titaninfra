"""DeepSeek-V3 native encoder parity against the independent ``renderers`` oracle.

This mirrors exactly the five scenarios in ``test_deepseek_v3_parity.py`` that
compare native output with the Jinja oracle: fixtures 01/02/03/05 plus the
fixture-04 deliberate-Jinja-loss-divergence case. The thinking warn-and-skip
test is a native capability check, not an oracle comparison, and is not copied.

Fixtures 01/02/03 are raw-L1 and raw-L2 identical. The two tool scenarios have
raw-L1 differences which are fully decomposed into exact, named token spans:

* ``renderers_single_tool_call_close`` — fixture 04's single tool call is left
  without ``<｜tool▁calls▁end｜><｜end▁of▁sentence｜>`` by both native and the
  official fixture Jinja (the known shipped-template bug). Renderers emits the
  well-formed two-token close. Native deliberately matches the shipped
  inference recipe; this is not a native regression and is not automatically a
  renderers bug.
* ``native_inter_tool_call_newline`` — fixture 05's official Jinja/native wire
  has ``\n`` between two calls in one assistant turn. Renderers 0.2.0 omits it,
  despite describing itself as a Jinja mirror. This *is* a renderers
  template-fidelity bug, recorded positively below.

Each span is found by a complete forward scan with exact special-token context
and its label direction is asserted. Only after removing those proven spans do
we require shared-core L1 equality. ``deepseek_v3`` registers no accepted L2
patterns, so the exhaustive core-label scan requires zero divergent positions;
any disagreement is UNEXPLAINED and fails. The excluded L1 spans are trained
wire differences, not masked metadata and not silently ignored noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

renderers = pytest.importorskip("renderers", reason="renderers package not installed")

from renderers import DeepSeekV3RendererConfig, build_training_sample, create_renderer  # noqa: E402

from tests.oracle._renderers_oracle.diff import classify_label_divergence  # noqa: E402
from titaninfra.data.sft.dispatch import tokenize_sample  # noqa: E402
from titaninfra.data.sft.message_converter import normalize_to_hf_messages  # noqa: E402
from titaninfra.data.sft.types import IGNORE_INDEX, EncodedSample  # noqa: E402

_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
_TOOL_CALL_END = "<｜tool▁call▁end｜>"
_TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_TOOL_OUTPUTS_BEGIN = "<｜tool▁outputs▁begin｜>"

_L1_SINGLE_CLOSE = "renderers_single_tool_call_close"
_L1_INTER_CALL_NEWLINE = "native_inter_tool_call_newline"
_RAW_PATTERN_SPAN_TOKENS = {
    _L1_SINGLE_CLOSE: 2,
    _L1_INTER_CALL_NEWLINE: 1,
}


@dataclass(frozen=True)
class _FixtureCase:
    path: str
    native_tokens: int
    renderers_tokens: int
    core_tokens: int
    raw_pattern: str | None = None


_FIXTURES = (
    _FixtureCase("01_single_turn_plain.json", 18, 18, 18),
    _FixtureCase("02_with_system.json", 20, 20, 20),
    _FixtureCase("03_multi_turn.json", 24, 24, 24),
    _FixtureCase("04_tool_loop.json", 35, 37, 35, _L1_SINGLE_CLOSE),
    _FixtureCase("05_two_tool_calls.json", 40, 39, 39, _L1_INTER_CALL_NEWLINE),
)


def _encoded(ids: list[int], labels: list[int]) -> EncodedSample:
    return EncodedSample(
        input_ids=tuple(ids),
        labels=tuple(labels),
        positions=tuple(range(len(ids))),
    )


def _render_with_deepseek_v3_renderers(
    sample_json: dict,
    tokenizer,
    *,
    max_length: int | None = None,
) -> EncodedSample:
    """Use the renderer's own sampled mask, without reimplementing its wire format."""
    messages, weights = normalize_to_hf_messages(
        sample_json.get("messages", []),
        system=sample_json.get("system"),
    )
    renderer = create_renderer(tokenizer, DeepSeekV3RendererConfig())
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


def _assert_exactly_one_masked_bos(encoded: EncodedSample, tokenizer) -> None:
    bos_id = tokenizer.convert_tokens_to_ids(_BOS)
    assert encoded.input_ids[0] == bos_id
    assert encoded.input_ids.count(bos_id) == 1
    assert encoded.labels[0] == IGNORE_INDEX


def _remove_renderers_single_tool_call_close(
    encoded: EncodedSample,
    tokenizer,
) -> EncodedSample:
    """Remove only ``call_end + [calls_end, EOS] + outputs_begin``'s middle pair."""
    ids = list(encoded.input_ids)
    labels = list(encoded.labels)
    call_end = tokenizer.convert_tokens_to_ids(_TOOL_CALL_END)
    calls_end = tokenizer.convert_tokens_to_ids(_TOOL_CALLS_END)
    eos = tokenizer.convert_tokens_to_ids(_EOS)
    outputs_begin = tokenizer.convert_tokens_to_ids(_TOOL_OUTPUTS_BEGIN)
    positions = [
        i
        for i in range(1, len(ids) - 2)
        if ids[i - 1] == call_end
        and ids[i:i + 2] == [calls_end, eos]
        and ids[i + 2] == outputs_begin
    ]
    assert len(positions) == 1, f"expected one renderers single-call close, got {positions}"
    start = positions[0]
    # These are renderer-trained tokens. They are removed solely to compare the
    # remaining wire, not because the two sides agree about their loss semantics.
    assert labels[start:start + 2] == [calls_end, eos]
    return _without_positions(encoded, {start, start + 1})


def _remove_native_inter_tool_call_newline(
    encoded: EncodedSample,
    tokenizer,
) -> EncodedSample:
    """Remove only the native/Jinja newline between adjacent complete tool calls."""
    ids = list(encoded.input_ids)
    labels = list(encoded.labels)
    call_end = tokenizer.convert_tokens_to_ids(_TOOL_CALL_END)
    call_begin = tokenizer.convert_tokens_to_ids(_TOOL_CALL_BEGIN)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline_ids) == 1
    newline = newline_ids[0]
    positions = [
        i
        for i in range(1, len(ids) - 1)
        if ids[i] == newline and ids[i - 1] == call_end and ids[i + 1] == call_begin
    ]
    assert len(positions) == 1, f"expected one native inter-call newline, got {positions}"
    position = positions[0]
    assert labels[position] == newline
    return _without_positions(encoded, {position})


def _assert_raw_l1_patterns_and_core_l2(
    native: EncodedSample,
    renderers_out: EncodedSample,
    tokenizer,
    *,
    case: _FixtureCase,
) -> dict[str, int]:
    """Account for every raw-L1 delta, then exhaustively scan every core label."""
    _assert_exactly_one_masked_bos(native, tokenizer)
    _assert_exactly_one_masked_bos(renderers_out, tokenizer)
    l1_counts: dict[str, int] = {}

    if case.raw_pattern is None:
        assert native.input_ids == renderers_out.input_ids
        native_core = native
        renderers_core = renderers_out
    else:
        assert native.input_ids != renderers_out.input_ids
        if case.raw_pattern == _L1_SINGLE_CLOSE:
            native_core = native
            renderers_core = _remove_renderers_single_tool_call_close(renderers_out, tokenizer)
            l1_counts[_L1_SINGLE_CLOSE] = 2
        elif case.raw_pattern == _L1_INTER_CALL_NEWLINE:
            native_core = _remove_native_inter_tool_call_newline(native, tokenizer)
            renderers_core = renderers_out
            l1_counts[_L1_INTER_CALL_NEWLINE] = 1
        else:  # pragma: no cover - guards the local table, not production behavior
            raise AssertionError(f"unknown fixture raw pattern: {case.raw_pattern}")

    assert len(native_core.input_ids) == len(renderers_core.input_ids) == case.core_tokens
    assert native_core.input_ids == renderers_core.input_ids, (
        "deepseek_v3 core L1 mismatch after removing only exactly asserted spans; "
        f"native={native_core.input_ids}, renderers={renderers_core.input_ids}"
    )
    result = classify_label_divergence(
        ids=native_core.input_ids,
        native_labels=native_core.labels,
        renderers_labels=renderers_core.labels,
        tokenizer=tokenizer,
        family="deepseek_v3",
    )
    assert result["unexplained_positions"] == [], f"unexplained deepseek_v3 core L2: {result}"
    assert result["n_divergent"] == 0, f"unexpected deepseek_v3 core L2: {result}"
    return l1_counts


@pytest.mark.parametrize("case", _FIXTURES, ids=lambda case: case.path.removesuffix(".json"))
def test_deepseek_v3_fixture_renderers_raw_spans_and_core_ok(
    case: _FixtureCase,
    deepseek_v3_tokenizer,
    load_oracle_fixture,
):
    """A new byte, mask, or uncatalogued span in any mirrored oracle scenario fails here."""
    fx = load_oracle_fixture(f"deepseek_v3/{case.path}")
    native = tokenize_sample(
        fx.sample,
        deepseek_v3_tokenizer,
        family="deepseek_v3",
        max_length=fx.max_length,
    )
    renderers_out = _render_with_deepseek_v3_renderers(
        fx.sample,
        deepseek_v3_tokenizer,
        max_length=fx.max_length,
    )
    assert len(native.input_ids) == case.native_tokens
    assert len(renderers_out.input_ids) == case.renderers_tokens
    counts = _assert_raw_l1_patterns_and_core_l2(
        native,
        renderers_out,
        deepseek_v3_tokenizer,
        case=case,
    )
    expected_counts = (
        {}
        if case.raw_pattern is None
        else {case.raw_pattern: _RAW_PATTERN_SPAN_TOKENS[case.raw_pattern]}
    )
    assert counts == expected_counts


def test_deepseek_v3_bos_is_single_and_not_auto_duplicated(deepseek_v3_tokenizer, load_oracle_fixture):
    """The fixture tokenizer adds no implicit BOS; both implementations add exactly one."""
    fx = load_oracle_fixture("deepseek_v3/01_single_turn_plain.json")
    native = tokenize_sample(fx.sample, deepseek_v3_tokenizer, family="deepseek_v3")
    renderers_out = _render_with_deepseek_v3_renderers(fx.sample, deepseek_v3_tokenizer)
    bos_id = deepseek_v3_tokenizer.convert_tokens_to_ids(_BOS)
    assert deepseek_v3_tokenizer.bos_token_id == bos_id
    assert deepseek_v3_tokenizer.encode("hello", add_special_tokens=True) == (
        deepseek_v3_tokenizer.encode("hello", add_special_tokens=False)
    )
    assert bos_id not in deepseek_v3_tokenizer.encode("hello", add_special_tokens=True)
    for output in (native, renderers_out):
        _assert_exactly_one_masked_bos(output, deepseek_v3_tokenizer)

    messages, _weights = normalize_to_hf_messages(fx.sample["messages"])
    official = deepseek_v3_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    official_ids = official["input_ids"] if isinstance(official, Mapping) else official
    assert official_ids[0] == bos_id
    assert official_ids.count(bos_id) == 1


def test_deepseek_v3_tool_arguments_serialize_exactly_once(deepseek_v3_tokenizer, load_oracle_fixture):
    """Native dumps the dict once; renderers accepts either JSON string or dict without double dumps."""
    fx = load_oracle_fixture("deepseek_v3/04_tool_loop.json")
    messages, _weights = normalize_to_hf_messages(fx.sample["messages"])
    arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert arguments == '{"city": "SF"}'
    assert isinstance(arguments, str)

    renderer = create_renderer(deepseek_v3_tokenizer, DeepSeekV3RendererConfig())
    from_string = renderer.render(messages).token_ids
    dict_messages = [dict(message) for message in messages]
    dict_messages[1] = dict(dict_messages[1])
    dict_messages[1]["tool_calls"] = [dict(dict_messages[1]["tool_calls"][0])]
    dict_messages[1]["tool_calls"][0]["function"] = {
        "name": "get_weather",
        "arguments": {"city": "SF"},
    }
    from_dict = renderer.render(dict_messages).token_ids
    assert from_string == from_dict

    native = tokenize_sample(fx.sample, deepseek_v3_tokenizer, family="deepseek_v3")
    native_text = deepseek_v3_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = deepseek_v3_tokenizer.decode(from_string, skip_special_tokens=False)
    for text in (native_text, renderers_text):
        assert text.count('{"city": "SF"}') == 1
        assert '\\"city\\"' not in text


def test_deepseek_v3_renderer_is_handwritten_and_matches_official_jinja_on_plain_wire(
    deepseek_v3_tokenizer,
):
    """Construction plus independent Jinja rendering show a handwritten mirror, not an official-package wrapper."""
    renderer = create_renderer(deepseek_v3_tokenizer, DeepSeekV3RendererConfig())
    assert type(renderer).__module__ == "renderers.deepseek_v3"
    assert type(renderer).__name__ == "DeepSeekV3Renderer"
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    official_text = deepseek_v3_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    official_ids = deepseek_v3_tokenizer.encode(official_text, add_special_tokens=False)
    assert renderer.render(messages).token_ids == official_ids


def test_deepseek_v3_single_tool_close_is_native_official_bug_parity(
    deepseek_v3_tokenizer,
    load_oracle_fixture,
):
    """Native matches shipped Jinja; renderers' well-formed close is a deliberate-design/wire difference."""
    fx = load_oracle_fixture("deepseek_v3/04_tool_loop.json")
    official_messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            }],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "call_0"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    official_text = deepseek_v3_tokenizer.apply_chat_template(
        official_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    native = tokenize_sample(fx.sample, deepseek_v3_tokenizer, family="deepseek_v3")
    renderers_out = _render_with_deepseek_v3_renderers(fx.sample, deepseek_v3_tokenizer)
    native_text = deepseek_v3_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = deepseek_v3_tokenizer.decode(renderers_out.input_ids, skip_special_tokens=False)
    official_without_close = _TOOL_CALL_END + _TOOL_OUTPUTS_BEGIN
    renderers_with_close = _TOOL_CALL_END + _TOOL_CALLS_END + _EOS + _TOOL_OUTPUTS_BEGIN
    assert official_text == native_text
    assert official_without_close in official_text
    assert renderers_with_close in renderers_text
    assert official_text != renderers_text


def test_deepseek_v3_renderers_missing_inter_call_newline_is_renderers_bug(
    deepseek_v3_tokenizer,
    load_oracle_fixture,
):
    """Verify deepseek v3 renderers missing inter call newline is renderers bug."""
    fx = load_oracle_fixture("deepseek_v3/05_two_tool_calls.json")
    official_messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "a", "arguments": '{"x": 1}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "b", "arguments": '{"y": 2}'},
                },
            ],
        },
    ]
    official_text = deepseek_v3_tokenizer.apply_chat_template(
        official_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    native = tokenize_sample(fx.sample, deepseek_v3_tokenizer, family="deepseek_v3")
    renderers_out = _render_with_deepseek_v3_renderers(fx.sample, deepseek_v3_tokenizer)
    native_text = deepseek_v3_tokenizer.decode(native.input_ids, skip_special_tokens=False)
    renderers_text = deepseek_v3_tokenizer.decode(renderers_out.input_ids, skip_special_tokens=False)
    with_newline = _TOOL_CALL_END + "\n" + _TOOL_CALL_BEGIN
    without_newline = _TOOL_CALL_END + _TOOL_CALL_BEGIN
    assert official_text == native_text
    assert with_newline in official_text
    assert with_newline not in renderers_text
    assert without_newline in renderers_text
