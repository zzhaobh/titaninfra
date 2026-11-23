"""Llama3 native encoder parity against the independent ``renderers`` oracle.

This mirrors exactly the three tests in ``test_llama3_parity.py`` that call
``render_with_jinja`` (fixtures 01/02/03). It deliberately does not mirror the
capability negative control, thinking warn-and-skip check, native-only trailer
structure check, or the three golden-snapshot tests: none of those performs a
native-vs-Jinja-oracle comparison.

Raw L1 is NOT identical for the three mirrored scenarios. The complete raw
difference is decomposed here into three exact, named metadata spans:

* ``renderers_system_date_preamble`` — ``Llama3Renderer`` always injects
  ``Cutting Knowledge Date`` + ``Today Date`` into its system block.
* ``renderers_empty_system_envelope`` — when no system message exists, that
  renderer still emits an otherwise-empty system header/EOT envelope.
* ``native_trailing_assistant_header`` — native retains the stock fixture
  template's final assistant generation header, masked as class-C METADATA;
  ``build_training_sample`` calls the renderer without a generation prompt.

The assertions below verify those spans token-for-token and verify every one
is masked wherever it exists. Only then do they compare the remaining shared
core: canonical L1 must be identical and a full L2 scan must have zero
unexplained positions (observed: zero divergent positions at all). BOS is also
checked explicitly: both sides emit exactly one ``<|begin_of_text|>`` at index
0 and both mask it. No raw difference is silently discarded.
"""

from __future__ import annotations

import logging

import pytest

renderers = pytest.importorskip("renderers", reason="renderers package not installed")

from renderers import Llama3RendererConfig, build_training_sample, create_renderer  # noqa: E402

from tests.oracle._renderers_oracle.diff import classify_label_divergence  # noqa: E402
from titaninfra.data.sft.dispatch import tokenize_sample  # noqa: E402
from titaninfra.data.sft.message_converter import normalize_to_hf_messages  # noqa: E402
from titaninfra.data.sft.types import IGNORE_INDEX, EncodedSample  # noqa: E402

_BOS = "<|begin_of_text|>"
_EOT = "<|eot_id|>"
_SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
_DATE_PREAMBLE = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
_TRAILER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

_L1_DATE = "renderers_system_date_preamble"
_L1_EMPTY_SYSTEM = "renderers_empty_system_envelope"
_L1_TRAILER = "native_trailing_assistant_header"


def _render_with_llama3_renderers(
    sample_json: dict,
    tokenizer,
    *,
    renderer=None,
    max_length: int | None = None,
) -> EncodedSample:
    """Adapt a titaninfra sample to ``renderers`` without reimplementing blocks."""
    messages = sample_json.get("messages", [])
    hf_messages, weights = normalize_to_hf_messages(
        messages,
        system=sample_json.get("system"),
    )
    if renderer is None:
        renderer = create_renderer(tokenizer, Llama3RendererConfig())
    # The installed renderers (0.1.11) takes a role_to_mask(message) -> bool
    # callable, not the old per-message bool list. This dict + lambda preserves the
    # exact prior per-message granularity (titaninfra's weight>0 may differ
    # message-to-message within the same role, so a role-level predicate like
    # `m["role"] == "assistant"` is not equivalent) by keying on object
    # identity: build_training_sample invokes the callable once per element
    # of this same `hf_messages` list, so `id(m)` round-trips exactly.
    #
    # The public boolean per-message selector exactly represents llama3's
    # binary label supervision, including weight=0. Fractional loss weights
    # are outside this oracle's list[bool] contract, as in the qwen3 round.
    loss_mask_by_id = {id(m): w > 0 for m, w in zip(hf_messages, weights, strict=True)}
    rendered = build_training_sample(
        renderer,
        hf_messages,
        tools=sample_json.get("tools"),
        role_to_mask=lambda m: loss_mask_by_id[id(m)],
    )
    ids = rendered.token_ids
    loss_mask = rendered.loss_mask
    if max_length is not None:
        ids = ids[:max_length]
        loss_mask = loss_mask[:max_length]
    return EncodedSample(
        input_ids=tuple(ids),
        labels=tuple(
            tid if keep else IGNORE_INDEX
            for tid, keep in zip(ids, loss_mask, strict=True)
        ),
        positions=tuple(range(len(ids))),
    )


def _encoded(ids: list[int], labels: list[int]) -> EncodedSample:
    return EncodedSample(
        input_ids=tuple(ids),
        labels=tuple(labels),
        positions=tuple(range(len(ids))),
    )


def _assert_llama3_raw_l1_patterns_and_core_l2(
    native_out: EncodedSample,
    renderers_out: EncodedSample,
    tokenizer,
    *,
    has_system: bool,
) -> dict[str, int]:
    """Prove raw L1 deltas are exactly named metadata, then exhaustively diff L2."""
    native_ids = list(native_out.input_ids)
    renderers_ids = list(renderers_out.input_ids)
    native_labels = list(native_out.labels)
    renderers_labels = list(renderers_out.labels)

    # L1 is compared first. A future renderer release removing these metadata
    # differences should fail loudly so this test and the report get updated.
    assert native_out.input_ids != renderers_out.input_ids

    bos_id = tokenizer.convert_tokens_to_ids(_BOS)
    assert native_ids[0] == renderers_ids[0] == bos_id
    assert native_labels[0] == renderers_labels[0] == IGNORE_INDEX

    trailer_ids = tokenizer.encode(_TRAILER, add_special_tokens=False)
    trailer_start = len(native_ids) - len(trailer_ids)
    assert native_ids[trailer_start:] == trailer_ids
    assert native_labels[trailer_start:] == [IGNORE_INDEX] * len(trailer_ids)
    native_core = _encoded(native_ids[:trailer_start], native_labels[:trailer_start])

    system_header_ids = tokenizer.encode(_SYSTEM_HEADER, add_special_tokens=False)
    assert renderers_ids[1:1 + len(system_header_ids)] == system_header_ids
    date_start = 1 + len(system_header_ids)
    date_ids = tokenizer.encode(_DATE_PREAMBLE, add_special_tokens=False)
    date_end = date_start + len(date_ids)
    assert renderers_ids[date_start:date_end] == date_ids
    assert renderers_labels[date_start:date_end] == [IGNORE_INDEX] * len(date_ids)

    l1_counts = {_L1_DATE: len(date_ids), _L1_TRAILER: len(trailer_ids)}
    if has_system:
        # Keep the caller's system header/content/EOT; remove only the fixed
        # date preamble, which the native fixture template never contains.
        core_ids = renderers_ids[:date_start] + renderers_ids[date_end:]
        core_labels = renderers_labels[:date_start] + renderers_labels[date_end:]
    else:
        # With no caller system text, remove the entire synthetic system turn.
        # Its non-date residue is header + header-newline + EOT (5 tokens with
        # this tokenizer); date tokens were already accounted for above.
        eot_id = tokenizer.convert_tokens_to_ids(_EOT)
        assert renderers_ids[date_end] == eot_id
        assert renderers_labels[date_end] == IGNORE_INDEX
        core_ids = renderers_ids[:1] + renderers_ids[date_end + 1:]
        core_labels = renderers_labels[:1] + renderers_labels[date_end + 1:]
        empty_envelope_len = len(system_header_ids) + 1
        assert renderers_labels[1:date_start] == [IGNORE_INDEX] * len(system_header_ids)
        l1_counts[_L1_EMPTY_SYSTEM] = empty_envelope_len

    renderers_core = _encoded(core_ids, core_labels)
    assert native_core.input_ids == renderers_core.input_ids, (
        "canonical L1 mismatch after removing only the three asserted metadata "
        f"patterns; native={native_core.input_ids}, renderers={renderers_core.input_ids}"
    )

    # Full forward per-position L2 scan over every shared-core token. llama3
    # registers no accepted L2 patterns, so even one divergence is UNEXPLAINED.
    result = classify_label_divergence(
        ids=native_core.input_ids,
        native_labels=native_core.labels,
        renderers_labels=renderers_core.labels,
        tokenizer=tokenizer,
        family="llama3",
    )
    assert result["unexplained_positions"] == []
    assert result["n_divergent"] == 0, (
        f"unexpected llama3 core label divergence: {result}"
    )
    return l1_counts


def test_llama3_single_turn_plain_renderers_core_ok(llama3_tokenizer, load_oracle_fixture):
    """Fixture 01: 20 date + 5 empty-system + 4 native-trailer L1 tokens."""
    fx = load_oracle_fixture("llama3/01_single_turn_plain.json")
    native_out = tokenize_sample(
        fx.sample, llama3_tokenizer, family="llama3", max_length=fx.max_length,
    )
    renderers_out = _render_with_llama3_renderers(
        fx.sample, llama3_tokenizer, max_length=fx.max_length,
    )
    counts = _assert_llama3_raw_l1_patterns_and_core_l2(
        native_out, renderers_out, llama3_tokenizer, has_system=False,
    )
    assert counts == {_L1_DATE: 20, _L1_EMPTY_SYSTEM: 5, _L1_TRAILER: 4}


def test_llama3_with_system_renderers_core_ok(llama3_tokenizer, load_oracle_fixture):
    """Fixture 02: only the 20 date + 4 native-trailer L1 tokens differ."""
    fx = load_oracle_fixture("llama3/02_with_system.json")
    native_out = tokenize_sample(
        fx.sample, llama3_tokenizer, family="llama3", max_length=fx.max_length,
    )
    renderers_out = _render_with_llama3_renderers(
        fx.sample, llama3_tokenizer, max_length=fx.max_length,
    )
    counts = _assert_llama3_raw_l1_patterns_and_core_l2(
        native_out, renderers_out, llama3_tokenizer, has_system=True,
    )
    assert counts == {_L1_DATE: 20, _L1_TRAILER: 4}


def test_llama3_multi_turn_renderers_core_ok(llama3_tokenizer, load_oracle_fixture):
    """Fixture 03: both assistant bodies/EOTs align after metadata canonicalization."""
    fx = load_oracle_fixture("llama3/03_multi_turn.json")
    native_out = tokenize_sample(
        fx.sample, llama3_tokenizer, family="llama3", max_length=fx.max_length,
    )
    renderers_out = _render_with_llama3_renderers(
        fx.sample, llama3_tokenizer, max_length=fx.max_length,
    )
    counts = _assert_llama3_raw_l1_patterns_and_core_l2(
        native_out, renderers_out, llama3_tokenizer, has_system=False,
    )
    assert counts == {_L1_DATE: 20, _L1_EMPTY_SYSTEM: 5, _L1_TRAILER: 4}


def test_llama3_tools_default_is_first_user_but_native_skips_tools(
    llama3_tokenizer,
    caplog,
):
    """Risk-point probe: this is a capability gap, not a mask-only deviation.

    ``Llama3RendererConfig.tools_in_user_message`` defaults to True and the
    rendered token stream really carries the tool JSON in the first user turn.
    Native llama3 supports text-only samples and dispatch warn-and-skips the
    top-level declaration entirely. Consequently no honest L1/L2 oracle parity
    assertion is possible for a tools sample; this positive test records that
    structural limitation instead of hiding it behind a skip.
    """
    sample = {
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "messages": [
            {"role": "user", "content": "Weather in SF?"},
            {"role": "assistant", "content": "I will check."},
        ],
    }
    renderer = create_renderer(llama3_tokenizer, Llama3RendererConfig())
    assert renderer.config.tools_in_user_message is True
    renderers_out = _render_with_llama3_renderers(
        sample, llama3_tokenizer, renderer=renderer,
    )
    with caplog.at_level(logging.WARNING, logger="titaninfra.sft.capabilities"):
        native_out = tokenize_sample(sample, llama3_tokenizer, family="llama3")
    assert any(
        "family=llama3" in record.message
        and "feature=tools_declaration" in record.message
        for record in caplog.records
    )

    native_text = llama3_tokenizer.decode(native_out.input_ids, skip_special_tokens=False)
    renderers_text = llama3_tokenizer.decode(renderers_out.input_ids, skip_special_tokens=False)
    assert "get_weather" not in native_text
    assert "Given the following functions" not in native_text
    first_user = renderers_text.split(
        "<|start_header_id|>user<|end_header_id|>\n\n", 1,
    )[1].split("<|eot_id|>", 1)[0]
    assert "Given the following functions" in first_user
    assert "get_weather" in first_user
    assert native_out.input_ids != renderers_out.input_ids
