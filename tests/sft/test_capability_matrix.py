"""Capability matrix — every fixture × every family.

Cartesian product over the 42 ``tests/fixtures/titaninfra_messages/`` cases
and the 7 registered encoder families (``qwen3``, ``qwen2_5``, ``llama3``,
``deepseek_v3``, ``deepseek_v4``, ``gpt_oss``, ``gemma4``). 294
combinations total.

The expected outcome for each cell is derived from the fixture's metadata
together with the family's ``SUPPORTED_BLOCKS`` declaration; no per-cell
hand-authored expectations live in this file. Adding a new fixture or
flipping a family's capability set re-derives all affected cells
automatically.

Three-state contract (see [[feedback_capability_mismatch_must_fail_loud]]
and ``docs/titaninfra_message_semantic_extraction.md`` §5):

* **EMPTY-RAISE** — family appears in the fixture's
  ``_expects_empty_for_families`` list. Dispatch must raise
  ``EmptyAssistantError``; the sample carries no learnable content for
  this family.
* **PARTIAL-WITH-WARN** — fixture's ``_uses_blocks`` carries at least
  one block type NOT in the family's ``SUPPORTED_BLOCKS`` AND the
  sample is not EMPTY-RAISE. Dispatch must log at least one
  ``capability_mismatch family=<f> block=<B> ... skipped=True``
  warning and still emit a valid ``EncodedSample``.
* **PASS** — every block type used by the fixture is in the family's
  ``SUPPORTED_BLOCKS``. Dispatch must emit a valid ``EncodedSample``
  with no skip warnings.

Complementary to ``tests/oracle/`` (wire correctness per family) and
``test_capability_warn_loud.py`` (hand-authored edge cases). This file
guards against silent capability drift across the full fixture pool.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path

import pytest

from titaninfra.data.sft.capabilities import EmptyAssistantError
from titaninfra.data.sft.dispatch import tokenize_sample

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "titaninfra_messages"
_CAPS_LOGGER = "titaninfra.sft.capabilities"

# Single source of truth for the family axis. Order is incidental but
# kept consistent with how encoders are listed in CLAUDE.md.
_FAMILIES: tuple[str, ...] = (
    "qwen3",
    "qwen2_5",
    "llama3",
    "deepseek_v3",
    "deepseek_v4",
    "gpt_oss",
    "gemma4",
)

# Feature → block class name that carries it. A feature can only fire
# a warning if at least one carrier block survived block-level prune;
# otherwise dispatch's _detect_features (which runs AFTER prune) won't
# see it. Features with no carrier (system, multi_turn,
# tools_declaration, loss_weight_*, type_overrides) are unconditional.
_FEATURE_CARRIER: dict[str, str] = {
    "tool_call": "ToolUseBlock",
    "parallel_tool_calls": "ToolUseBlock",
    "tool_result": "ToolResultBlock",
    "thinking_current_turn": "ThinkingBlock",
    "thinking_history": "ThinkingBlock",
    "image": "ImageBlock",
}


def _supported_block_names(family: str) -> frozenset[str]:
    """Return the family encoder's SUPPORTED_BLOCKS as class-name strings.

    Fixtures declare ``_uses_blocks`` as a list of class-name strings
    (e.g. ``["TextBlock", "ThinkingBlock"]``) for easy JSON authoring;
    encoders declare ``SUPPORTED_BLOCKS`` as a frozenset of Block
    classes. This helper bridges the two so the matrix check is a
    plain string-set comparison.
    """
    module = importlib.import_module(f"titaninfra.data.sft.encoders.{family}")
    blocks = getattr(module, "SUPPORTED_BLOCKS", None)
    if blocks is None:
        # Legacy encoder without capability declaration — treat as
        # supporting every block type used in any fixture.
        return frozenset({"TextBlock", "ThinkingBlock", "ToolUseBlock", "ToolResultBlock", "ImageBlock"})
    return frozenset(cls.__name__ for cls in blocks)


def _supported_features(family: str) -> frozenset[str]:
    """Return the family encoder's SUPPORTED_FEATURES.

    Empty frozenset for legacy encoders without the declaration — same
    semantics as ``_supported_block_names`` legacy fallback. Encoders
    must declare both constants (all 7 current families do).
    """
    module = importlib.import_module(f"titaninfra.data.sft.encoders.{family}")
    return getattr(module, "SUPPORTED_FEATURES", frozenset())


def _expected_feature_warnings(fixture: dict, family: str) -> frozenset[str]:
    """Predict which feature warnings dispatch should emit for this cell.

    A feature warning is expected when:
    1. The fixture's ``_uses_features`` lists the feature.
    2. The family's ``SUPPORTED_FEATURES`` does NOT contain it.
    3. The feature still has a "live" source after block prune. For
       block-tagged features (``tool_call`` etc.) this is the carrier
       block class surviving in ``SUPPORTED_BLOCKS``. For
       ``loss_weight_per_block`` it requires at least one surviving
       block to actually carry a ``weight`` key in the fixture data —
       since a per-block weight on a pruned-out block disappears with
       its carrier.

    The ``_uses_features`` list is the spec; this predicate filters it
    down to what the post-prune detector will actually see. The runtime
    is the SoT; this helper just keeps the test in sync with it.
    """
    supported_features = _supported_features(family)
    supported_blocks = _supported_block_names(family)
    used = set(fixture.get("_uses_features", []))
    sample = fixture.get("sample") or {}
    expected: set[str] = set()
    for feature in used:
        if feature in supported_features:
            continue
        carrier = _FEATURE_CARRIER.get(feature)
        if carrier is not None and carrier not in supported_blocks:
            # Block prune already removed the carrier → no surviving
            # source for the detector to fire on.
            continue
        if feature == "loss_weight_per_block":
            # Only fires if a SURVIVING block (its type ∈ SUPPORTED_BLOCKS)
            # actually carries a ``weight`` key. A weight on a pruned
            # block disappears with the block.
            if not _any_surviving_block_has_weight(sample, supported_blocks):
                continue
        expected.add(feature)
    return frozenset(expected)


_TAG_TO_CLASS_NAME = {
    "text": "TextBlock",
    "thinking": "ThinkingBlock",
    "tool_use": "ToolUseBlock",
    "tool_result": "ToolResultBlock",
    "image": "ImageBlock",
}


def _any_surviving_block_has_weight(sample: dict, supported_blocks: frozenset[str]) -> bool:
    """True iff any block whose type maps to a surviving block class
    has a numeric ``weight`` key set in the fixture.

    Used to decide whether ``loss_weight_per_block`` should still fire
    after block-level prune.
    """
    for msg in sample.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        blocks = msg.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            cls_name = _TAG_TO_CLASS_NAME.get(block.get("type"))
            if cls_name is None or cls_name not in supported_blocks:
                continue
            if isinstance(block.get("weight"), (int, float)):
                return True
    return False


def _collect_fixtures() -> list[tuple[str, dict]]:
    """Load all 42 fixtures and return ``(label, data)`` pairs.

    Label format ``<scenario>/<variant>`` keeps parametrize ids
    human-readable in pytest -v output (e.g. ``single_turn_plain/minimal``).
    Sorted so parametrize ordering is deterministic.
    """
    out: list[tuple[str, dict]] = []
    for path in sorted(_FIXTURE_ROOT.glob("*/*.json")):
        with path.open("r", encoding="utf-8") as f:
            out.append((f"{path.parent.name}/{path.stem}", json.load(f)))
    return out


_FIXTURES: list[tuple[str, dict]] = _collect_fixtures()


def _classify(fixture: dict, family: str) -> str:
    """Derive expected state for one (fixture, family) cell.

    Returns one of: ``"EMPTY"``, ``"PARTIAL"``, ``"PASS"``.

    Precedence (matches dispatch.py runtime semantics):
    1. EMPTY beats PARTIAL — if the assistant has only unsupported
       blocks, dispatch raises before encoding starts.
    2. PARTIAL beats PASS — at least one warning will fire even if the
       rest of the sample is supported.
    """
    if family in fixture.get("_expects_empty_for_families", []):
        return "EMPTY"
    used = set(fixture.get("_uses_blocks", []))
    if used - _supported_block_names(family):
        return "PARTIAL"
    return "PASS"


def _has_skip_warning(records: list[logging.LogRecord], family: str) -> bool:
    """True iff caplog captured at least one capability_mismatch skip
    for this family. Match on the structured message format emitted by
    ``capabilities.warn_unsupported_block``.
    """
    needle_family = f"family={family}"
    return any(
        "capability_mismatch" in r.message
        and needle_family in r.message
        and "skipped=True" in r.message
        for r in records
    )


def _feature_warnings_in_caplog(records: list[logging.LogRecord], family: str) -> set[str]:
    """Return the set of feature names that fired ``ignored=True`` warnings.

    The matcher parses the structured format
    ``capability_mismatch family=<f> feature=<name> ignored=True ...`` —
    same shape as ``warn_unsupported_feature`` emits.
    """
    needle_family = f"family={family}"
    out: set[str] = set()
    for r in records:
        if "capability_mismatch" not in r.message:
            continue
        if needle_family not in r.message or "ignored=True" not in r.message:
            continue
        # Pull "feature=NAME " token out of the message.
        for chunk in r.message.split():
            if chunk.startswith("feature="):
                out.add(chunk.split("=", 1)[1])
                break
    return out


_FIXTURE_LABELS = [label for label, _ in _FIXTURES]
_FIXTURE_DATA = {label: data for label, data in _FIXTURES}


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("fixture_label", _FIXTURE_LABELS)
def test_capability_matrix(fixture_label, family, request, caplog):
    """One assertion per (fixture, family) cell. 42 × 7 = 294 cells."""
    fixture = _FIXTURE_DATA[fixture_label]
    sample = fixture["sample"]
    tokenizer = request.getfixturevalue(f"{family}_tokenizer")
    expected = _classify(fixture, family)
    # Fixtures with tools_declaration render long JSON schema headers
    # (Qwen3 puts the schema inline in the system tail). 256 falls
    # mid-segment for several encoders → use 2048 so assemble can
    # commit whole segments. The matrix tests capability semantics,
    # not length-trimming behaviour.
    max_length = 2048

    if expected == "EMPTY":
        with caplog.at_level(logging.WARNING, logger=_CAPS_LOGGER):
            with pytest.raises(EmptyAssistantError) as exc_info:
                tokenize_sample(sample, tokenizer, family=family, max_length=max_length)
        # Exception forensics carry the family name and at least one
        # skipped type — these are the structured handles operators
        # grep for when a dataset starts emitting EMPTY-RAISE.
        assert exc_info.value.family == family
        assert exc_info.value.skipped_block_types, (
            f"EmptyAssistantError must report skipped block types, got: "
            f"{exc_info.value.skipped_block_types!r}"
        )
        return

    expected_features = _expected_feature_warnings(fixture, family)

    if expected == "PARTIAL":
        with caplog.at_level(logging.WARNING, logger=_CAPS_LOGGER):
            encoded = tokenize_sample(sample, tokenizer, family=family, max_length=max_length)
        assert encoded.input_ids, (
            f"PARTIAL cell ({fixture_label}, {family}) must still produce "
            f"a non-empty EncodedSample after pruning unsupported blocks."
        )
        assert _has_skip_warning(caplog.records, family), (
            f"PARTIAL cell ({fixture_label}, {family}) must emit at least "
            f"one capability_mismatch warning. caplog records: "
            f"{[r.message for r in caplog.records]}"
        )
        _assert_feature_warnings_match(caplog.records, family, expected_features, fixture_label)
        return

    # PASS — encode succeeds with no block-level capability warnings.
    # Feature warnings MAY still fire (e.g. a text-only multi-turn sample
    # passes for qwen2_5 block-wise but qwen2_5 supports multi_turn so
    # no feature warning either; whereas a fixture with type_overrides
    # against a family that doesn't support type_overrides would PASS
    # block-wise but still warn on the feature).
    with caplog.at_level(logging.WARNING, logger=_CAPS_LOGGER):
        encoded = tokenize_sample(sample, tokenizer, family=family, max_length=max_length)
    assert encoded.input_ids, (
        f"PASS cell ({fixture_label}, {family}) must produce a non-empty "
        f"EncodedSample."
    )
    assert not _has_skip_warning(caplog.records, family), (
        f"PASS cell ({fixture_label}, {family}) must NOT emit block-skip "
        f"warnings — fixture _uses_blocks is a subset of family "
        f"SUPPORTED_BLOCKS. Got: "
        f"{[r.message for r in caplog.records if 'capability_mismatch' in r.message]}"
    )
    _assert_feature_warnings_match(caplog.records, family, expected_features, fixture_label)


def _assert_feature_warnings_match(
    records: list[logging.LogRecord],
    family: str,
    expected: frozenset[str],
    fixture_label: str,
) -> None:
    """Assert dispatch emitted exactly the feature warnings we predicted.

    Equality (not subset) — extra warnings would indicate the detector
    is over-firing; missing warnings would indicate under-firing. Both
    are bugs in either the encoder declaration or the detection rules.
    """
    actual = _feature_warnings_in_caplog(records, family)
    assert actual == expected, (
        f"feature warnings mismatch for ({fixture_label}, {family}): "
        f"expected={sorted(expected)} actual={sorted(actual)}"
    )


def test_matrix_size_is_seven_times_fortyeight():
    """Sanity check the matrix shape — keeps the 7×48 contract honest.

    If a fixture is added/removed without updating downstream
    expectations, this fails first with a clear count instead of
    silently changing the parametrize cardinality of the test above.

    Grew from 42 → 48 on 2026-05-31 with the two thinking_intent
    fixture scenarios (off + on_high, three variants each).
    """
    assert len(_FIXTURES) == 48, f"expected 48 fixtures, found {len(_FIXTURES)}"
    assert len(_FAMILIES) == 7, f"expected 7 families, found {len(_FAMILIES)}"


def test_all_families_declare_loss_weight_per_message():
    """Audit #9 (2026-06-06): every family honors per-message weight via the
    boundary builder (resolve_weights collapses msg.weight into nm.weight;
    manual mode passes it verbatim, auto mode always respects weight==0).
    So all 7 must DECLARE FEATURE_LOSS_WEIGHT_PER_MESSAGE — otherwise dispatch
    emits a false 'unsupported' warning that can mislead a user into stripping
    weight annotations (silent full-masking regression)."""
    import importlib

    from titaninfra.data.sft.capabilities import FEATURE_LOSS_WEIGHT_PER_MESSAGE
    for fam in ("qwen3", "qwen2_5", "llama3", "deepseek_v3",
                "deepseek_v4", "gpt_oss", "gemma4"):
        mod = importlib.import_module(f"titaninfra.data.sft.encoders.{fam}")
        assert FEATURE_LOSS_WEIGHT_PER_MESSAGE in mod.SUPPORTED_FEATURES, fam
