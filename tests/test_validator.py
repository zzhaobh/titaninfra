"""Unit tests for ``titaninfra.data.validator``.

Executable mirror of ``docs/titaninfra_message_format.md`` §2. Each test
corresponds to one rule from the spec — if the test name doesn't map
to a doc rule, either the test or the doc is wrong.

The committed fixture pool ``tests/fixtures/titaninfra_messages/`` is also
swept: every fixture must validate clean (it represents a canonical
shape of the format). A fixture that begins failing this sweep means
either the validator rule list is too strict OR the fixture violates
the format — either way, surfacing it loud is the point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from titaninfra.data.validator import (
    CODE_BLOCK_ROLE_MISMATCH,
    CODE_ROLE_INVALID,
    CODE_SCHEMA_DECODE,
    CODE_STRUCT_BLOCKS_EMPTY,
    CODE_STRUCT_MESSAGES_EMPTY,
    CODE_STRUCT_SYSTEM_DUPLICATE,
    CODE_THINKING_INTENT_CONFLICT,
    CODE_TOOL_RESULT_MSG_SPLIT,
    CODE_TOOL_RESULT_UNMATCHED,
    CODE_TOOL_USE_ID_DUPLICATE,
    CODE_TOOL_USE_UNMATCHED,
    CODE_TYPE_IMAGE_SOURCE,
    CODE_TYPE_WEIGHT_RANGE,
    InvalidTitanInfraMessageError,
    validate,
    validate_or_raise,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "titaninfra_messages"


def _codes(result) -> set[str]:
    """Helper — extract the set of issue codes from a ValidationResult."""
    return {i.code for i in result.issues}


# ---------------------------------------------------------------------------
# Happy path — fixture pool is canonical
# ---------------------------------------------------------------------------
def _all_fixture_files() -> list[Path]:
    return sorted(_FIXTURE_ROOT.glob("*/*.json"))


@pytest.mark.parametrize(
    "fixture_path",
    _all_fixture_files(),
    ids=lambda p: f"{p.parent.name}/{p.stem}",
)
def test_all_committed_fixtures_are_valid(fixture_path: Path):
    """Every fixture in the pool must round-trip through validate() clean.

    Fixtures are the canonical examples of the format; if one fails
    here, either the fixture is wrong (fix the JSON) or the validator
    is too strict (fix the rule). Both are caught loud.

    Exception: a fixture may opt out by setting ``_expects_validator_fail``
    to the issue code it is meant to trigger — used for adversarial
    fixtures that exist precisely to verify converter-side fail-loud
    (e.g. ``with_thinking_intent_off/adversarial.json`` for P5.2's
    THINKING_INTENT_CONFLICT). The sweep then asserts the *expected*
    code fires instead of demanding clean.
    """
    data = json.loads(fixture_path.read_text())
    sample = data["sample"]
    expected_fail = data.get("_expects_validator_fail")
    result = validate(sample)
    if expected_fail:
        assert not result.ok, (
            f"fixture {fixture_path.relative_to(_REPO_ROOT)} marked "
            f"_expects_validator_fail={expected_fail!r} but passed validation"
        )
        assert expected_fail in _codes(result), (
            f"fixture {fixture_path.relative_to(_REPO_ROOT)} expected "
            f"{expected_fail!r} but got codes {_codes(result)}"
        )
        return
    assert result.ok, (
        f"fixture {fixture_path.relative_to(_REPO_ROOT)} is supposed to be "
        f"canonical but failed validation:\n"
        + "\n".join(f"  {i.code} at {i.path!r}: {i.message}" for i in result.issues)
    )


# ---------------------------------------------------------------------------
# §2.1 structural
# ---------------------------------------------------------------------------
def test_messages_empty_fails():
    result = validate({"messages": []})
    assert not result.ok
    assert CODE_STRUCT_MESSAGES_EMPTY in _codes(result)


def test_messages_missing_fails():
    """Missing ``messages`` is caught at decode time (msgspec required field)."""
    result = validate({})
    assert not result.ok
    # msgspec rejects the missing required field before our structural
    # check runs — either code is acceptable for "messages absent".
    assert CODE_SCHEMA_DECODE in _codes(result) or CODE_STRUCT_MESSAGES_EMPTY in _codes(result)


def test_blocks_empty_fails():
    result = validate({
        "messages": [{"role": "user", "blocks": []}],
    })
    assert not result.ok
    assert CODE_STRUCT_BLOCKS_EMPTY in _codes(result)


def test_system_field_and_system_role_duplicate_fails():
    """§2.1 — top-level `system` and `messages[0].role=='system'` are mutually exclusive."""
    result = validate({
        "system": "you are helpful",
        "messages": [
            {"role": "system", "blocks": [{"type": "text", "text": "you are helpful"}]},
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "hello"}]},
        ],
    })
    assert CODE_STRUCT_SYSTEM_DUPLICATE in _codes(result)


def test_system_field_alone_ok():
    """Top-level system without role:system message is canonical."""
    result = validate({
        "system": "you are helpful",
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "hello"}]},
        ],
    })
    assert result.ok, [i.message for i in result.issues]


# ---------------------------------------------------------------------------
# Role + §2.2 block × role matrix
# ---------------------------------------------------------------------------
def test_unknown_role_fails():
    result = validate({
        "messages": [{"role": "wizard", "blocks": [{"type": "text", "text": "hi"}]}],
    })
    assert CODE_ROLE_INVALID in _codes(result)


@pytest.mark.parametrize("role,block_type", [
    ("system", "thinking"),
    ("user", "thinking"),
    ("tool", "thinking"),
    ("system", "tool_use"),
    ("user", "tool_use"),
    ("system", "tool_result"),
    ("user", "tool_result"),
    ("assistant", "tool_result"),
    ("system", "image"),
    ("assistant", "image"),
    ("tool", "image"),
    ("tool", "text"),
])
def test_block_role_matrix_rejects_invalid(role, block_type):
    """§2.2 — every illegal (role, block) combination must report mismatch."""
    block = _minimal_block(block_type)
    sample = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            # Wrap the illegal block in a structurally complete message
            # so we test ONLY the role-matrix rule.
            {"role": role, "blocks": [block]},
        ],
    }
    # Augment with required pairings so we don't trip OTHER rules.
    if block_type == "tool_use":
        sample["messages"].append({"role": "tool", "blocks": [
            {"type": "tool_result", "tool_use_id": "x", "text": "ok"},
        ]})
    result = validate(sample)
    assert CODE_BLOCK_ROLE_MISMATCH in _codes(result), (
        f"({role}, {block_type}) should be illegal per doc §2.2 "
        f"but validator accepted it"
    )


def _minimal_block(tag: str) -> dict:
    """Return a minimal valid block of the given type for matrix tests."""
    if tag == "text":
        return {"type": "text", "text": "x"}
    if tag == "thinking":
        return {"type": "thinking", "text": "x"}
    if tag == "tool_use":
        return {"type": "tool_use", "id": "x", "name": "n", "input": {}}
    if tag == "tool_result":
        return {"type": "tool_result", "tool_use_id": "x", "text": "ok"}
    if tag == "image":
        return {"type": "image", "source": {"type": "path", "data": "/x.jpg"}}
    raise ValueError(tag)


# ---------------------------------------------------------------------------
# §2.3 tool reference
# ---------------------------------------------------------------------------
def test_duplicate_tool_use_id_fails():
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "do two things"}]},
            {"role": "assistant", "blocks": [
                {"type": "tool_use", "id": "c1", "name": "n", "input": {}},
                {"type": "tool_use", "id": "c1", "name": "n", "input": {}},
            ]},
            {"role": "tool", "blocks": [
                {"type": "tool_result", "tool_use_id": "c1", "text": "ok"},
                {"type": "tool_result", "tool_use_id": "c1", "text": "ok"},
            ]},
        ],
    })
    assert CODE_TOOL_USE_ID_DUPLICATE in _codes(result)


def test_tool_use_unmatched_fails():
    """A tool_use NOT in the terminal assistant message must be matched."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "call it"}]},
            {"role": "assistant", "blocks": [
                # First assistant turn — unmatched tool_use is NOT
                # terminal here, so it must have a tool_result.
                {"type": "tool_use", "id": "c1", "name": "n", "input": {}},
            ]},
            {"role": "user", "blocks": [{"type": "text", "text": "??"}]},
            # Terminal assistant turn carries plain text (no tool_use).
            {"role": "assistant", "blocks": [{"type": "text", "text": "ok"}]},
        ],
    })
    assert CODE_TOOL_USE_UNMATCHED in _codes(result)


def test_terminal_tool_use_without_result_is_canonical():
    """§2.3 carve-out: tool_use in the LAST assistant message is allowed
    to be unmatched — call-only training (ToolBench / Gorilla style)."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "blocks": [
                {"type": "tool_use", "id": "c1", "name": "get_weather",
                 "input": {"city": "Beijing"}},
            ]},
        ],
    })
    assert result.ok, [i.message for i in result.issues]


def test_terminal_parallel_tool_use_without_result_is_canonical():
    """§2.3 carve-out also applies to parallel call-only."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "do both"}]},
            {"role": "assistant", "blocks": [
                {"type": "tool_use", "id": "c1", "name": "n", "input": {}},
                {"type": "tool_use", "id": "c2", "name": "n", "input": {}},
            ]},
        ],
    })
    assert result.ok, [i.message for i in result.issues]


def test_tool_result_unmatched_fails():
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [
                {"type": "tool_use", "id": "c1", "name": "n", "input": {}},
            ]},
            {"role": "tool", "blocks": [
                # References a non-existent tool_use id.
                {"type": "tool_result", "tool_use_id": "ghost", "text": "?"},
            ]},
        ],
    })
    assert CODE_TOOL_RESULT_UNMATCHED in _codes(result)


def test_parallel_tool_split_across_messages_fails():
    """§2.3 — parallel tool_use must be resolved in ONE tool message.

    Constructed so the offending assistant turn is NOT terminal (a
    later assistant text message follows), otherwise the call-only
    carve-out would absorb the violation.
    """
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "do two"}]},
            {"role": "assistant", "blocks": [
                {"type": "tool_use", "id": "c1", "name": "n", "input": {}},
                {"type": "tool_use", "id": "c2", "name": "n", "input": {}},
            ]},
            # Only ONE of the two results — split across messages.
            {"role": "tool", "blocks": [
                {"type": "tool_result", "tool_use_id": "c1", "text": "ok"},
            ]},
            {"role": "tool", "blocks": [
                {"type": "tool_result", "tool_use_id": "c2", "text": "ok"},
            ]},
            # Final assistant turn makes the parallel-call turn
            # non-terminal so the split rule applies.
            {"role": "assistant", "blocks": [{"type": "text", "text": "done"}]},
        ],
    })
    assert CODE_TOOL_RESULT_MSG_SPLIT in _codes(result)


# ---------------------------------------------------------------------------
# §2.4 type constraints
# ---------------------------------------------------------------------------
def test_image_source_missing_keys_fails():
    result = validate({
        "messages": [
            {"role": "user", "blocks": [
                {"type": "image", "source": {"type": "path"}},  # data missing
                {"type": "text", "text": "?"},
            ]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "ok"}]},
        ],
    })
    assert CODE_TYPE_IMAGE_SOURCE in _codes(result)


@pytest.mark.parametrize("weight", [-0.1, 1.5, 2.0])
def test_weight_out_of_range_fails(weight):
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [
                {"type": "text", "text": "hello", "weight": weight},
            ]},
        ],
    })
    assert CODE_TYPE_WEIGHT_RANGE in _codes(result)


def test_weight_in_loss_config_out_of_range_fails():
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "ok"}]},
        ],
        "loss_config": {"type_overrides": {"thinking": 1.5}},
    })
    assert CODE_TYPE_WEIGHT_RANGE in _codes(result)


# ---------------------------------------------------------------------------
# Decode-layer errors
# ---------------------------------------------------------------------------
def test_unknown_block_type_fails_at_decode():
    """msgspec discriminator rejects unknown tag → SCHEMA_DECODE."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "video", "data": "?"}]},
        ],
    })
    assert CODE_SCHEMA_DECODE in _codes(result)
    assert result.sample is None


def test_malformed_json_bytes_fails_at_decode():
    result = validate(b"{not json")
    assert CODE_SCHEMA_DECODE in _codes(result)
    assert result.sample is None


# ---------------------------------------------------------------------------
# validate_or_raise (production entry — strict by default)
# ---------------------------------------------------------------------------
def test_or_raise_returns_sample_on_clean_input():
    sample = validate_or_raise({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "ok"}]},
        ],
    })
    assert sample.messages[0].role == "user"


def test_or_raise_raises_with_source_attribution():
    """Source identifier must appear in the raised exception message."""
    with pytest.raises(InvalidTitanInfraMessageError) as exc_info:
        validate_or_raise({"messages": []}, source="train.jsonl:42")
    assert exc_info.value.source == "train.jsonl:42"
    assert "train.jsonl:42" in str(exc_info.value)
    assert exc_info.value.issues


def test_or_raise_carries_full_issue_list():
    """A sample with multiple violations reports ALL of them on the exception."""
    with pytest.raises(InvalidTitanInfraMessageError) as exc_info:
        validate_or_raise({
            "messages": [
                # Wrong role + block-role mismatch + missing tool_result
                {"role": "wizard", "blocks": [
                    {"type": "tool_use", "id": "c1", "name": "n", "input": {}},
                ]},
            ],
        }, source="ad-hoc")
    codes = {i.code for i in exc_info.value.issues}
    assert CODE_ROLE_INVALID in codes
    assert CODE_TOOL_USE_UNMATCHED in codes


# ---------------------------------------------------------------------------
# Sanity — multi-issue collection
# ---------------------------------------------------------------------------
def test_multiple_issues_collected_in_one_pass():
    """validate() does NOT short-circuit on first issue — it surfaces them all."""
    result = validate({
        "messages": [
            # blocks empty + role invalid
            {"role": "ghost", "blocks": []},
        ],
    })
    assert not result.ok
    codes = _codes(result)
    assert CODE_STRUCT_BLOCKS_EMPTY in codes
    assert CODE_ROLE_INVALID in codes


# ---------------------------------------------------------------------------
# §1.6 — ThinkingIntent (family-agnostic sample-level thinking control)
# ---------------------------------------------------------------------------
# msgspec ``Literal[...]`` already enforces value-set membership at decode
# time, so the validator does not duplicate the check. These tests pin
# down that contract: invalid enum → CODE_SCHEMA_DECODE, valid (or absent)
# intent → clean validation.
def test_thinking_intent_absent_is_valid():
    """``thinking_intent`` is optional — samples without it are canonical."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "ok"}]},
        ],
    })
    assert result.ok, f"unexpected issues: {result.issues}"
    assert result.sample.thinking_intent is None


def test_thinking_intent_valid_values_pass():
    """All combinations of valid (mode, effort) values validate clean."""
    for mode in ["on", "off", None]:
        for effort in ["low", "medium", "high", None]:
            intent = {"mode": mode, "effort": effort}
            payload = {
                "messages": [
                    {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
                    {"role": "assistant", "blocks": [{"type": "text", "text": "ok"}]},
                ],
                "thinking_intent": intent,
            }
            result = validate(payload)
            assert result.ok, (
                f"valid intent {intent!r} unexpectedly failed: {result.issues}"
            )
            assert result.sample.thinking_intent.mode == mode
            assert result.sample.thinking_intent.effort == effort


def test_thinking_intent_invalid_mode_rejected_at_decode():
    """Bad ``mode`` value is caught by msgspec Literal — CODE_SCHEMA_DECODE."""
    payload = json.dumps({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
        ],
        "thinking_intent": {"mode": "sometimes"},
    }).encode()
    result = validate(payload)
    assert not result.ok
    assert CODE_SCHEMA_DECODE in _codes(result)
    # Decoder failure short-circuits → no sample object is built.
    assert result.sample is None


def test_thinking_intent_invalid_effort_rejected_at_decode():
    """Bad ``effort`` value is caught by msgspec Literal — CODE_SCHEMA_DECODE."""
    payload = json.dumps({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
        ],
        "thinking_intent": {"mode": "on", "effort": "extreme"},
    }).encode()
    result = validate(payload)
    assert not result.ok
    assert CODE_SCHEMA_DECODE in _codes(result)
    assert result.sample is None


# ---------------------------------------------------------------------------
# §2.5 thinking_intent semantics — P5.2 fail-loud on mode/block contradiction
# ---------------------------------------------------------------------------
def test_thinking_intent_off_with_thinking_block_fails():
    """mode=off + non-empty assistant ThinkingBlock → CODE_THINKING_INTENT_CONFLICT."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "2+2?"}]},
            {"role": "assistant", "blocks": [
                {"type": "thinking", "text": "Let me compute..."},
                {"type": "text", "text": "4"},
            ]},
        ],
        "thinking_intent": {"mode": "off"},
    })
    assert not result.ok
    assert CODE_THINKING_INTENT_CONFLICT in _codes(result)
    # Path points to the offending block so the data producer can locate it.
    conflict = next(i for i in result.issues if i.code == CODE_THINKING_INTENT_CONFLICT)
    assert conflict.path == "messages[1].blocks[0]"


def test_thinking_intent_off_with_empty_thinking_block_ok():
    """Empty-text ThinkingBlock alongside mode=off is not a contradiction.

    The qwen3 mixed-thinking recipe injects empty ``<think>\\n\\n</think>``
    envelopes deliberately; an empty assistant ThinkingBlock semantically
    means "model declined to think" and is consistent with mode=off.
    """
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "2+2?"}]},
            {"role": "assistant", "blocks": [
                {"type": "thinking", "text": ""},
                {"type": "text", "text": "4"},
            ]},
        ],
        "thinking_intent": {"mode": "off"},
    })
    assert result.ok, [i.message for i in result.issues]


def test_thinking_intent_on_with_thinking_block_ok():
    """mode=on + ThinkingBlock is canonical (the whole point of mode=on)."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "2+2?"}]},
            {"role": "assistant", "blocks": [
                {"type": "thinking", "text": "trivial"},
                {"type": "text", "text": "4"},
            ]},
        ],
        "thinking_intent": {"mode": "on", "effort": "low"},
    })
    assert result.ok, [i.message for i in result.issues]


def test_thinking_intent_on_without_thinking_block_ok():
    """mode=on without ThinkingBlock is legitimate ("model decides this turn")."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "2+2?"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "4"}]},
        ],
        "thinking_intent": {"mode": "on"},
    })
    assert result.ok, [i.message for i in result.issues]


def test_thinking_intent_off_multi_turn_flags_each_contradicting_block():
    """All contradicting assistant ThinkingBlocks across turns are surfaced."""
    result = validate({
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "q1"}]},
            {"role": "assistant", "blocks": [
                {"type": "thinking", "text": "t1"},
                {"type": "text", "text": "a1"},
            ]},
            {"role": "user", "blocks": [{"type": "text", "text": "q2"}]},
            {"role": "assistant", "blocks": [
                {"type": "thinking", "text": "t2"},
                {"type": "text", "text": "a2"},
            ]},
        ],
        "thinking_intent": {"mode": "off"},
    })
    assert not result.ok
    conflicts = [i for i in result.issues if i.code == CODE_THINKING_INTENT_CONFLICT]
    assert len(conflicts) == 2
    assert {c.path for c in conflicts} == {
        "messages[1].blocks[0]",
        "messages[3].blocks[0]",
    }


def test_thinking_intent_off_validate_or_raise():
    """``validate_or_raise`` propagates the conflict as InvalidTitanInfraMessageError."""
    payload = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "blocks": [
                {"type": "thinking", "text": "cot"},
                {"type": "text", "text": "a"},
            ]},
        ],
        "thinking_intent": {"mode": "off"},
    }
    with pytest.raises(InvalidTitanInfraMessageError) as excinfo:
        validate_or_raise(payload, source="unit-test")
    assert any(
        i.code == CODE_THINKING_INTENT_CONFLICT for i in excinfo.value.issues
    )


def test_thinking_intent_off_adversarial_fixture_triggers_conflict():
    """The committed adversarial fixture is wired to land on this exact rule."""
    fixture = _FIXTURE_ROOT / "with_thinking_intent_off" / "adversarial.json"
    data = json.loads(fixture.read_text())
    assert data.get("_expects_validator_fail") == "THINKING_INTENT_CONFLICT"
    result = validate(data["sample"])
    assert not result.ok
    assert CODE_THINKING_INTENT_CONFLICT in _codes(result)
