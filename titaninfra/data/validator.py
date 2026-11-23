"""Family-agnostic schema + invariant validator for titaninfra message format.

This module is the **executable mirror of ``docs/titaninfra_message_format.md`` §2**.
Every rule in doc §2 maps to exactly one ``ValidationIssue.code`` here;
adding a rule to the doc means adding a check here (and vice versa).

Design principles (per [[feedback_validate_at_converter_not_runtime]]):

* **Family-agnostic.** This validator does NOT know about qwen3 /
  llama3 / etc. It checks whether a sample is a valid titaninfra message
  per the global schema. Family-specific render concerns (capability
  prune) belong to ``dispatch.tokenize_sample``.
* **Offline only.** Callers are converters and CI lints; the SFT
  runtime trusts MDS contents and never invokes this.
* **Strict by default.** ``validate_or_raise`` is the production entry
  point — it raises ``InvalidTitanInfraMessageError`` on the first bad
  sample, forcing the data producer to fix the source. ``validate``
  is the inspection entry point — it collects all issues without
  raising, used by tooling that wants to surface multiple problems
  in one pass.

Cross-references:
* doc spec: ``docs/titaninfra_message_format.md`` §2
* schema struct definitions: ``titaninfra/data/schema.py``
* converter wire point: ``titaninfra/data/converter.py``
  (``convert_messages_to_mds``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import msgspec

from titaninfra.data.schema import (
    ImageBlock,
    SFTSample,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    decode_sample,
)

# ---------------------------------------------------------------------------
# Issue code taxonomy — one entry per rule in doc §2
# ---------------------------------------------------------------------------
# Codes are prefixed by category so callers can grep / aggregate
# (e.g. ``code.startswith("TOOL_")`` matches all tool-use problems).
# Renaming a code is a breaking change for downstream report readers —
# do not rename; deprecate and add a new code if semantics change.

# Decoder layer (msgspec failures)
CODE_SCHEMA_DECODE = "SCHEMA_DECODE"

# §2.1 structural
CODE_STRUCT_MESSAGES_EMPTY = "STRUCT_MESSAGES_EMPTY"
CODE_STRUCT_BLOCKS_EMPTY = "STRUCT_BLOCKS_EMPTY"
CODE_STRUCT_SYSTEM_DUPLICATE = "STRUCT_SYSTEM_DUPLICATE"

# Role layer
CODE_ROLE_INVALID = "ROLE_INVALID"

# §2.2 block × role matrix
CODE_BLOCK_ROLE_MISMATCH = "BLOCK_ROLE_MISMATCH"

# §2.3 tool reference
CODE_TOOL_USE_ID_DUPLICATE = "TOOL_USE_ID_DUPLICATE"
CODE_TOOL_USE_UNMATCHED = "TOOL_USE_UNMATCHED"
CODE_TOOL_RESULT_UNMATCHED = "TOOL_RESULT_UNMATCHED"
CODE_TOOL_RESULT_MSG_SPLIT = "TOOL_RESULT_MSG_SPLIT"

# §2.4 type constraints
CODE_TYPE_TOOL_USE_INPUT = "TYPE_TOOL_USE_INPUT"
CODE_TYPE_TOOL_RESULT_TEXT = "TYPE_TOOL_RESULT_TEXT"
CODE_TYPE_IMAGE_SOURCE = "TYPE_IMAGE_SOURCE"
CODE_TYPE_WEIGHT_RANGE = "TYPE_WEIGHT_RANGE"

# §2.5 thinking_intent semantics (P5.2 — 2026-05-31)
# Top-level ``mode=off`` declares the whole sample is non-thinking; an
# assistant ThinkingBlock alongside it is a literal contradiction and
# any silent resolution corrupts training data (see v2 next-session
# doc, Decision 2). The validator fails this loud at the converter
# stage so it never lands in MDS.
CODE_THINKING_INTENT_CONFLICT = "THINKING_INTENT_CONFLICT"

# Set of all roles allowed in a titaninfra message. Adding a role is a
# schema-level decision documented in the format doc, not done here.
_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})

# Block × role legality matrix (doc §2.2). Lookup key is the JSON
# ``type`` tag, value is the set of roles that may carry that block.
_BLOCK_ROLE_MATRIX: dict[str, frozenset[str]] = {
    "text": frozenset({"system", "user", "assistant"}),
    "thinking": frozenset({"assistant"}),
    "tool_use": frozenset({"assistant"}),
    "tool_result": frozenset({"tool"}),
    "image": frozenset({"user"}),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationIssue:
    """One specific rule violation found in a sample.

    ``path`` follows the convention ``messages[2].blocks[0].input`` —
    bracketed integer indices, dot-separated field names. This matches
    how Pydantic / jsonschema report locations and lets a human jump
    to the offending node without re-parsing.
    """

    code: str
    message: str
    path: str = ""


@dataclass
class ValidationResult:
    """Outcome of validating one sample.

    ``ok`` is true iff ``issues`` is empty. ``sample`` is the decoded
    ``SFTSample`` when decoding succeeded (even if higher-level checks
    found issues); ``None`` only when ``SCHEMA_DECODE`` itself fails.
    Downstream tools can therefore still inspect a struct that had
    semantic violations.
    """

    ok: bool
    sample: SFTSample | None
    issues: list[ValidationIssue] = field(default_factory=list)


class InvalidTitanInfraMessageError(ValueError):
    """Raised by ``validate_or_raise`` on the first invalid sample.

    Attributes are designed for converter log forensics: ``issues``
    is the full list (so a single message can carry multiple
    violations), ``source`` is a short identifier of where the bad
    sample came from (file path + line number for jsonl, row index
    for parquet) for the human to grep.
    """

    def __init__(self, issues: list[ValidationIssue], source: str = "") -> None:
        self.issues = list(issues)
        self.source = source
        head = issues[0] if issues else ValidationIssue(code="UNKNOWN", message="")
        super().__init__(
            f"invalid titaninfra message at {source!r}: "
            f"{head.code} at {head.path!r}: {head.message} "
            f"(+ {len(issues) - 1} more issue(s))"
        )


# ---------------------------------------------------------------------------
# Internal helpers — each implements one rule from doc §2
# ---------------------------------------------------------------------------
def _check_structural(raw: dict, issues: list[ValidationIssue]) -> None:
    """§2.1 — structural invariants on the raw dict (pre-decode).

    Runs on the raw dict because msgspec ``decode_sample`` is happy
    with ``messages=[]`` — emptiness is a titaninfra invariant, not a
    schema constraint.
    """
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        issues.append(ValidationIssue(
            code=CODE_STRUCT_MESSAGES_EMPTY,
            message="SFTSample.messages must be a non-empty list",
            path="messages",
        ))
        return

    has_top_system = bool(raw.get("system"))
    first_is_system_msg = (
        isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    )
    if has_top_system and first_is_system_msg:
        issues.append(ValidationIssue(
            code=CODE_STRUCT_SYSTEM_DUPLICATE,
            message=(
                "top-level 'system' and messages[0].role=='system' are "
                "mutually exclusive (pick exactly one canonical place)"
            ),
            path="system",
        ))

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        blocks = msg.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            issues.append(ValidationIssue(
                code=CODE_STRUCT_BLOCKS_EMPTY,
                message="Message.blocks must be a non-empty list",
                path=f"messages[{idx}].blocks",
            ))


def _check_roles_and_matrix(sample: SFTSample, issues: list[ValidationIssue]) -> None:
    """Validate role names + §2.2 Block × Role legality matrix."""
    for m_idx, msg in enumerate(sample.messages):
        if msg.role not in _VALID_ROLES:
            issues.append(ValidationIssue(
                code=CODE_ROLE_INVALID,
                message=(
                    f"role={msg.role!r} not in {sorted(_VALID_ROLES)} "
                    f"(see doc §1.2)"
                ),
                path=f"messages[{m_idx}].role",
            ))
            continue  # downstream checks assume known role
        for b_idx, block in enumerate(msg.blocks):
            btag = _block_tag(block)
            allowed = _BLOCK_ROLE_MATRIX.get(btag)
            if allowed is None:
                # Should not happen — msgspec discriminator would have
                # rejected unknown tags. Defensive only.
                continue
            if msg.role not in allowed:
                issues.append(ValidationIssue(
                    code=CODE_BLOCK_ROLE_MISMATCH,
                    message=(
                        f"block type {btag!r} not allowed in role "
                        f"{msg.role!r} (see doc §2.2 matrix)"
                    ),
                    path=f"messages[{m_idx}].blocks[{b_idx}]",
                ))


def _check_tool_references(sample: SFTSample, issues: list[ValidationIssue]) -> None:
    """§2.3 — tool_use ↔ tool_result pairing.

    Builds two indices:
    * ``tool_use_ids``: ordered list of (msg_idx, block_idx, id) tuples
      across the whole sample.
    * ``tool_result_ids``: ordered list of (msg_idx, block_idx, tool_use_id)
      from tool-role messages.

    Then checks: (a) id uniqueness across tool_use; (b) every tool_use
    is matched OR is in the sample's terminal assistant message
    (call-only training case, doc §2.3); (c) every tool_result
    references a known tool_use; (d) parallel tool calls in one
    assistant turn must be resolved in exactly one immediately-following
    tool message.

    The "terminal call-only" carve-out matches industry datasets
    (ToolBench / Gorilla / Hermes-Tool-Use) where the model is
    trained to *emit* a tool call without seeing the result inline —
    downstream systems execute the call and handle continuation.
    """
    # Locate the LAST assistant message index — tool_use blocks here
    # are allowed to be unmatched (terminal call-only training).
    last_assistant_idx = -1
    for m_idx, msg in enumerate(sample.messages):
        if msg.role == "assistant":
            last_assistant_idx = m_idx

    tool_use_index: list[tuple[int, int, str]] = []
    seen_ids: dict[str, tuple[int, int]] = {}
    for m_idx, msg in enumerate(sample.messages):
        for b_idx, block in enumerate(msg.blocks):
            if _block_tag(block) != "tool_use":
                continue
            use_id = block.id  # type: ignore[attr-defined]
            if use_id in seen_ids:
                prev = seen_ids[use_id]
                issues.append(ValidationIssue(
                    code=CODE_TOOL_USE_ID_DUPLICATE,
                    message=(
                        f"duplicate tool_use.id={use_id!r}; first seen at "
                        f"messages[{prev[0]}].blocks[{prev[1]}]"
                    ),
                    path=f"messages[{m_idx}].blocks[{b_idx}].id",
                ))
            else:
                seen_ids[use_id] = (m_idx, b_idx)
            tool_use_index.append((m_idx, b_idx, use_id))

    matched_ids: set[str] = set()
    for m_idx, msg in enumerate(sample.messages):
        if msg.role != "tool":
            continue
        for b_idx, block in enumerate(msg.blocks):
            if _block_tag(block) != "tool_result":
                continue
            ref_id = block.tool_use_id  # type: ignore[attr-defined]
            if ref_id not in seen_ids:
                issues.append(ValidationIssue(
                    code=CODE_TOOL_RESULT_UNMATCHED,
                    message=(
                        f"tool_result.tool_use_id={ref_id!r} has no "
                        f"matching tool_use in this sample"
                    ),
                    path=f"messages[{m_idx}].blocks[{b_idx}].tool_use_id",
                ))
            else:
                matched_ids.add(ref_id)

    for m_idx, b_idx, use_id in tool_use_index:
        if use_id in matched_ids:
            continue
        # Terminal call-only carve-out: a tool_use in the last
        # assistant message without a tool_result is canonical
        # (model learns to emit the call itself).
        if m_idx == last_assistant_idx:
            continue
        issues.append(ValidationIssue(
            code=CODE_TOOL_USE_UNMATCHED,
            message=(
                f"tool_use.id={use_id!r} has no matching tool_result "
                f"(and is not in the terminal assistant message — "
                f"see doc §2.3 'terminal call-only' carve-out)"
            ),
            path=f"messages[{m_idx}].blocks[{b_idx}].id",
        ))

    _check_parallel_tool_split(sample, issues, last_assistant_idx)


def _check_parallel_tool_split(
    sample: SFTSample,
    issues: list[ValidationIssue],
    last_assistant_idx: int,
) -> None:
    """§2.3 parallel-call rule.

    If an assistant message contains N >= 2 tool_use blocks, the
    immediately following message must be a single ``role: "tool"``
    message containing exactly N tool_result blocks (in any order;
    matching is by id). Splitting parallel results across multiple
    tool messages is forbidden so encoders can rely on the 1:1
    assistant↔tool pairing structure.

    Terminal call-only carve-out (§2.3): if the parallel tool_use
    block lives in the SAMPLE'S LAST assistant message, no following
    tool message is required — this matches industry datasets that
    train the model to emit parallel calls without seeing results.
    """
    msgs = sample.messages
    for m_idx, msg in enumerate(msgs):
        if msg.role != "assistant":
            continue
        use_ids = [
            b.id  # type: ignore[attr-defined]
            for b in msg.blocks if _block_tag(b) == "tool_use"
        ]
        if len(use_ids) < 2:
            continue
        # Terminal parallel call-only: no following tool message
        # required.
        if m_idx == last_assistant_idx:
            continue
        next_idx = m_idx + 1
        if next_idx >= len(msgs) or msgs[next_idx].role != "tool":
            # No following tool message — pairing was already flagged
            # as UNMATCHED above; do not double-report here.
            continue
        result_ids = [
            b.tool_use_id  # type: ignore[attr-defined]
            for b in msgs[next_idx].blocks if _block_tag(b) == "tool_result"
        ]
        if set(result_ids) != set(use_ids):
            issues.append(ValidationIssue(
                code=CODE_TOOL_RESULT_MSG_SPLIT,
                message=(
                    f"parallel tool_use ids {sorted(use_ids)} must be resolved "
                    f"in exactly one following tool message; "
                    f"got tool_result ids {sorted(result_ids)} "
                    f"(see doc §2.3)"
                ),
                path=f"messages[{m_idx + 1}].blocks",
            ))


def _check_thinking_intent(sample: SFTSample, issues: list[ValidationIssue]) -> None:
    """§2.5 — sample-level thinking_intent must not contradict per-message thinking.

    Specifically, ``thinking_intent.mode == "off"`` declares the entire
    sample non-thinking. Any assistant message carrying a non-empty
    ``ThinkingBlock`` is a literal contradiction: trusting the mode
    drops user-written CoT, trusting the block trains the model to
    emit thinking despite the user pinning ``enable_thinking=False``.
    Both silent resolutions are data poisoning. See
    [[feedback_validate_at_converter_not_runtime]] and 2026-05-31
    next-session-v2 Decision 2.

    No symmetric check on ``mode == "on"`` without ThinkingBlock —
    that is a legitimate "model decides whether to think this turn"
    pattern (qwen3 mixed-thinking recipe). Only the off+block pair
    is structurally impossible.
    """
    ti = sample.thinking_intent
    if ti is None or ti.mode != "off":
        return
    for m_idx, msg in enumerate(sample.messages):
        if msg.role != "assistant":
            continue
        for b_idx, block in enumerate(msg.blocks):
            if _block_tag(block) != "thinking":
                continue
            text = getattr(block, "text", "") or ""
            if text.strip():
                issues.append(ValidationIssue(
                    code=CODE_THINKING_INTENT_CONFLICT,
                    message=(
                        "thinking_intent.mode=='off' contradicts non-empty "
                        "ThinkingBlock at this position (see doc §2.5). "
                        "Resolve at the data source: either drop the "
                        "ThinkingBlock or set mode='on'/None."
                    ),
                    path=f"messages[{m_idx}].blocks[{b_idx}]",
                ))


def _check_types(sample: SFTSample, issues: list[ValidationIssue]) -> None:
    """§2.4 type constraints not caught by msgspec discrimination.

    msgspec already enforces top-level field types (``input: dict``,
    ``text: str``) — but it does not enforce semantic types (image
    source shape, weight range). Those land here.
    """
    _check_weight_range(sample.loss_config, issues, "loss_config")
    for m_idx, msg in enumerate(sample.messages):
        _check_weight(msg.weight, issues, f"messages[{m_idx}].weight")
        for b_idx, block in enumerate(msg.blocks):
            tag = _block_tag(block)
            base = f"messages[{m_idx}].blocks[{b_idx}]"
            _check_weight(getattr(block, "weight", None), issues, f"{base}.weight")
            if tag == "image":
                src = block.source  # type: ignore[attr-defined]
                if not isinstance(src, dict) or "type" not in src or "data" not in src:
                    issues.append(ValidationIssue(
                        code=CODE_TYPE_IMAGE_SOURCE,
                        message=(
                            "image.source must be a dict with 'type' and "
                            "'data' keys (see doc §2.4)"
                        ),
                        path=f"{base}.source",
                    ))


def _check_weight(w: float | None, issues: list[ValidationIssue], path: str) -> None:
    """Per-field weight range check ([0,1] or None)."""
    if w is None:
        return
    if not isinstance(w, (int, float)) or isinstance(w, bool):
        issues.append(ValidationIssue(
            code=CODE_TYPE_WEIGHT_RANGE,
            message=f"weight={w!r} must be a numeric value in [0, 1] or None",
            path=path,
        ))
        return
    if w < 0.0 or w > 1.0:
        issues.append(ValidationIssue(
            code=CODE_TYPE_WEIGHT_RANGE,
            message=f"weight={w} out of range [0, 1]",
            path=path,
        ))


def _check_weight_range(loss_config, issues: list[ValidationIssue], path: str) -> None:
    """Check LossConfig.type_overrides values are all in [0, 1]."""
    overrides = loss_config.type_overrides if loss_config is not None else {}
    for tag, val in overrides.items():
        _check_weight(val, issues, f"{path}.type_overrides[{tag!r}]")


def _block_tag(block) -> str:
    """Return the JSON discriminator tag of a Block struct instance.

    msgspec uses Python class identity for its decoded types rather
    than exposing the ``tag=`` argument as an attribute, so we map
    via the class table below. The mapping mirrors the ``tag=...``
    arguments in ``titaninfra/data/schema.py`` exactly; if a new Block
    class lands there, add it here too (the test pool will catch
    drift the next time fixtures use the new tag).
    """
    return _BLOCK_CLASS_TO_TAG.get(type(block), "")


_BLOCK_CLASS_TO_TAG: dict[type, str] = {
    TextBlock: "text",
    ThinkingBlock: "thinking",
    ToolUseBlock: "tool_use",
    ToolResultBlock: "tool_result",
    ImageBlock: "image",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate(raw: dict | bytes | str) -> ValidationResult:
    """Validate one sample and return all issues found, no raise.

    Use this from inspection tooling (lint reports, ``inspect_mds.py``,
    test fixtures). For converter pipelines see ``validate_or_raise``.

    The input may be a Python dict (already parsed) or raw JSON
    bytes/str (will be decoded). Decode failure produces an issue
    with ``code=SCHEMA_DECODE`` and ``sample=None``.
    """
    issues: list[ValidationIssue] = []

    if isinstance(raw, (bytes, str)):
        try:
            sample = decode_sample(raw)
        except msgspec.DecodeError as exc:
            issues.append(ValidationIssue(
                code=CODE_SCHEMA_DECODE,
                message=f"msgspec decode failed: {exc}",
                path="",
            ))
            return ValidationResult(ok=False, sample=None, issues=issues)
        raw_dict = json.loads(raw)
    else:
        raw_dict = raw
        try:
            sample = decode_sample(json.dumps(raw).encode())
        except msgspec.DecodeError as exc:
            issues.append(ValidationIssue(
                code=CODE_SCHEMA_DECODE,
                message=f"msgspec decode failed: {exc}",
                path="",
            ))
            return ValidationResult(ok=False, sample=None, issues=issues)

    # §2.1 needs the raw dict (msgspec accepts empty messages list).
    _check_structural(raw_dict, issues)
    # Higher-level checks operate on the decoded SFTSample for type
    # safety. They still run even if §2.1 surfaced issues — surfacing
    # more issues in one pass is cheaper for the human fixer.
    _check_roles_and_matrix(sample, issues)
    _check_tool_references(sample, issues)
    _check_types(sample, issues)
    _check_thinking_intent(sample, issues)

    return ValidationResult(ok=not issues, sample=sample, issues=issues)


def validate_or_raise(raw: dict | bytes | str, *, source: str = "") -> SFTSample:
    """Validate; return the decoded ``SFTSample`` or raise on the first failure.

    This is the **production entry point** for converters. Per
    [[feedback_validate_at_converter_not_runtime]], offline validation
    is strict by default — the data producer fixes the source rather
    than the runtime tolerating drift.

    ``source`` is a short identifier (file path + line number, or row
    index) embedded in the raised exception message so operators can
    immediately locate the offending sample. Pass it when you have it;
    leave default ``""`` for ad-hoc validation.
    """
    result = validate(raw)
    if not result.ok or result.sample is None:
        raise InvalidTitanInfraMessageError(result.issues, source=source)
    return result.sample


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "InvalidTitanInfraMessageError",
    "validate",
    "validate_or_raise",
    # Issue codes — exported for downstream report consumers / tests
    # that pattern-match by code.
    "CODE_SCHEMA_DECODE",
    "CODE_STRUCT_MESSAGES_EMPTY",
    "CODE_STRUCT_BLOCKS_EMPTY",
    "CODE_STRUCT_SYSTEM_DUPLICATE",
    "CODE_ROLE_INVALID",
    "CODE_BLOCK_ROLE_MISMATCH",
    "CODE_TOOL_USE_ID_DUPLICATE",
    "CODE_TOOL_USE_UNMATCHED",
    "CODE_TOOL_RESULT_UNMATCHED",
    "CODE_TOOL_RESULT_MSG_SPLIT",
    "CODE_TYPE_TOOL_USE_INPUT",
    "CODE_TYPE_TOOL_RESULT_TEXT",
    "CODE_TYPE_IMAGE_SOURCE",
    "CODE_TYPE_WEIGHT_RANGE",
    "CODE_THINKING_INTENT_CONFLICT",
]
