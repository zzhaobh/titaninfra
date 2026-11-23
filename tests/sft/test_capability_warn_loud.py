"""Capability mismatch warn-loud contract tests.

Covers the three-state contract from
``docs/titaninfra_message_semantic_extraction.md`` §5 and
[[feedback_capability_mismatch_must_fail_loud]]:

* **PARTIAL-WITH-WARN** — sample carries blocks unsupported by the
  target family; dispatch logs a structured warning per skipped block
  and the encoder produces a valid EncodedSample from the supported
  remainder.
* **EMPTY-RAISE** — every assistant block was unsupported (or pruned
  empty); dispatch raises ``EmptyAssistantError`` because the sample
  contributes no learnable content.
* **PASS** (legacy) — encoder module without ``SUPPORTED_BLOCKS``
  declaration is treated as supports-everything; capability check
  is skipped and the encoder runs unchanged.

These tests complement the per-family parity tests in
``tests/oracle/`` — those verify wire correctness for supported
combinations; these verify the contract for unsupported combinations.
"""

from __future__ import annotations

import logging

import pytest

from titaninfra.data.sft.capabilities import EmptyAssistantError
from titaninfra.data.sft.dispatch import tokenize_sample

_CAPS_LOGGER = "titaninfra.sft.capabilities"


# ---------------------------------------------------------------------------
# PARTIAL-WITH-WARN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "family",
    ["qwen2_5", "llama3", "deepseek_v3"],
)
def test_partial_thinking_skipped_with_warning(family, request, caplog):
    """Families with no ThinkingBlock support warn-and-skip + still encode."""
    tokenizer = request.getfixturevalue(f"{family}_tokenizer")
    sample = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "blocks": [
                    {"type": "thinking", "text": "internal CoT"},
                    {"type": "text", "text": "hello"},
                ],
            },
        ]
    }
    with caplog.at_level(logging.WARNING, logger=_CAPS_LOGGER):
        encoded = tokenize_sample(sample, tokenizer, family=family, max_length=128)

    skip_msgs = [
        r.message for r in caplog.records
        if f"family={family}" in r.message and "block=ThinkingBlock" in r.message
    ]
    assert skip_msgs, (
        f"expected at least one ThinkingBlock skip warning for {family}, "
        f"got records: {[r.message for r in caplog.records]}"
    )
    assert "skipped=True" in skip_msgs[0]
    assert "message_idx=1" in skip_msgs[0]
    assert "block_idx=0" in skip_msgs[0]
    assert len(encoded.input_ids) > 0


def test_partial_tool_blocks_skipped_with_warning_on_text_only_family(
    llama3_tokenizer, caplog,
):
    """Llama3 supports only TextBlock — tool_use & tool_result both skip."""
    sample = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "blocks": [
                    {"type": "text", "text": "let me check"},
                    {"type": "tool_use", "id": "c0", "name": "wx", "input": {"city": "SF"}},
                ],
            },
            {
                "role": "tool",
                "blocks": [
                    {"type": "tool_result", "tool_use_id": "c0", "text": "sunny"},
                ],
            },
            {
                "role": "assistant",
                "blocks": [
                    {"type": "text", "text": "It's sunny."},
                ],
            },
        ]
    }
    with caplog.at_level(logging.WARNING, logger=_CAPS_LOGGER):
        encoded = tokenize_sample(sample, llama3_tokenizer, family="llama3", max_length=256)

    skipped_kinds = {
        b for r in caplog.records for b in ("ToolUseBlock", "ToolResultBlock")
        if f"block={b}" in r.message and "family=llama3" in r.message
    }
    assert skipped_kinds == {"ToolUseBlock", "ToolResultBlock"}, (
        f"expected both tool block types to warn-skip, got {skipped_kinds}"
    )
    assert len(encoded.input_ids) > 0


# ---------------------------------------------------------------------------
# EMPTY-RAISE
# ---------------------------------------------------------------------------
def test_empty_raise_when_only_assistant_block_is_unsupported(qwen2_5_tokenizer):
    """All assistant content is ThinkingBlock → qwen2_5 has nothing to learn."""
    sample = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "blocks": [
                    {"type": "thinking", "text": "only thinking, no text"},
                ],
            },
        ]
    }
    with pytest.raises(EmptyAssistantError) as exc_info:
        tokenize_sample(sample, qwen2_5_tokenizer, family="qwen2_5", max_length=128)
    err = exc_info.value
    assert err.family == "qwen2_5"
    assert "ThinkingBlock" in err.skipped_block_types
    assert "qwen2_5Encoder" in str(err) or "qwen2_5" in str(err)


def test_empty_raise_when_assistant_has_multiple_unsupported_blocks(llama3_tokenizer):
    """Multiple unsupported block types in one assistant turn all reported."""
    sample = {
        "messages": [
            {"role": "user", "content": "ping"},
            {
                "role": "assistant",
                "blocks": [
                    {"type": "thinking", "text": "..."},
                    {"type": "tool_use", "id": "c0", "name": "x", "input": {}},
                ],
            },
        ]
    }
    with pytest.raises(EmptyAssistantError) as exc_info:
        tokenize_sample(sample, llama3_tokenizer, family="llama3", max_length=128)
    skipped = set(exc_info.value.skipped_block_types)
    assert skipped == {"ThinkingBlock", "ToolUseBlock"}


def test_no_raise_when_one_of_multiple_assistant_turns_still_has_content(
    qwen2_5_tokenizer, caplog,
):
    """If at least one assistant turn keeps supported content, encode proceeds."""
    sample = {
        "messages": [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "blocks": [{"type": "thinking", "text": "only thought"}],
            },
            {"role": "user", "content": "Q2"},
            {
                "role": "assistant",
                "blocks": [{"type": "text", "text": "A2"}],
            },
        ]
    }
    with caplog.at_level(logging.WARNING, logger=_CAPS_LOGGER):
        encoded = tokenize_sample(sample, qwen2_5_tokenizer, family="qwen2_5", max_length=128)
    assert len(encoded.input_ids) > 0
    # The first assistant turn went empty but the second one carries
    # text — overall sample is not empty, so no raise.


# ---------------------------------------------------------------------------
# PASS (no capability constraint when SUPPORTED_BLOCKS absent)
# ---------------------------------------------------------------------------
def test_legacy_encoder_without_supported_blocks_passes_through(qwen3_tokenizer):
    """Encoders missing SUPPORTED_BLOCKS declaration get no capability gate.

    Backward-compat path: a family that hasn't opted into the
    capability registry yet should keep working without warnings or
    structural changes to the sample. The migration is to add the
    declaration; until then dispatch treats it as supports-everything.
    """
    import sys

    # Use an existing module (qwen3) as a starting point and mask
    # SUPPORTED_BLOCKS to simulate a legacy encoder. Import explicitly
    # so this test works under pytest-xdist where the worker process
    # may not have loaded the module yet.
    import titaninfra.data.sft.encoders.qwen3  # noqa: F401

    module = sys.modules.get("titaninfra.data.sft.encoders.qwen3")
    assert module is not None
    saved = module.SUPPORTED_BLOCKS
    delattr(module, "SUPPORTED_BLOCKS")
    try:
        sample = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "blocks": [
                        {"type": "thinking", "text": "should pass through"},
                        {"type": "text", "text": "ok"},
                    ],
                },
            ]
        }
        # Should not raise EmptyAssistantError nor strip the thinking
        # block — qwen3 actually supports thinking, so this just goes
        # through the normal encode path.
        encoded = tokenize_sample(sample, qwen3_tokenizer, family="qwen3", max_length=256)
        assert len(encoded.input_ids) > 0
    finally:
        module.SUPPORTED_BLOCKS = saved


# ---------------------------------------------------------------------------
# Idempotency: capability check must not mutate caller's sample dict
# ---------------------------------------------------------------------------
def test_capability_check_does_not_mutate_input(qwen2_5_tokenizer):
    """``sample_json`` passed in must not have its blocks list modified."""
    sample = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "blocks": [
                    {"type": "thinking", "text": "internal"},
                    {"type": "text", "text": "answer"},
                ],
            },
        ]
    }
    before = len(sample["messages"][1]["blocks"])
    tokenize_sample(sample, qwen2_5_tokenizer, family="qwen2_5", max_length=128)
    after = len(sample["messages"][1]["blocks"])
    assert before == after == 2, (
        f"dispatch must deepcopy before pruning; "
        f"input blocks count went {before} → {after}"
    )
