"""Per-family encoder capability declarations.

Each encoder in ``titaninfra/data/sft/encoders/<family>.py`` declares two
module-level constants that describe what subset of the titaninfra message
IR it actually implements:

    SUPPORTED_BLOCKS: frozenset[type[Block]]
        Block classes the encoder will emit tokens for. Any block whose
        type is *not* in this set gets warn-and-skipped at dispatch
        time (see ``feedback_capability_mismatch_must_fail_loud``).

    SUPPORTED_FEATURES: frozenset[str]
        Cross-cutting capability flags that aren't tied to a single
        Block class — e.g. multi-turn, tools declaration, top-level
        system, future ``thinking_intent`` field. Names are defined
        as ``FEATURE_*`` constants below to prevent typo drift across
        encoders.

Why this lives in its own module (not on ``SFTEncoder`` Protocol):

* SUPPORTED_* are **data**, not behaviour. The dispatch layer reads
  them; encoders never read their own set. Putting them on the class
  body keeps them greppable and lets ``dispatch.tokenize_sample``
  inspect them without instantiating the encoder.
* New families add one entry; no changes to this module unless a
  genuinely novel capability appears.

The semantic-extraction paradigm doc (``docs/titaninfra_message_semantic_extraction.md``)
governs which features are allowed to enter the IR at all. This module
is the runtime-enforcement layer for that paradigm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Feature flags — string constants, NOT Enum
# ---------------------------------------------------------------------------
# Rationale: encoders declare ``SUPPORTED_FEATURES = frozenset({FEATURE_TOOLS,
# FEATURE_MULTI_TURN})``. String constants keep the declaration site readable
# and let new features land as one-line additions here without forcing every
# encoder to import a new Enum member.
#
# When adding a feature, document its semantic intent in the comment line —
# encoder authors will grep this file to decide whether to declare support.

FEATURE_SYSTEM = "system"
"""Top-level ``SFTSample.system`` prompt rendering."""

FEATURE_MULTI_TURN = "multi_turn"
"""More than one user/assistant turn pair in a single sample."""

FEATURE_TOOLS_DECLARATION = "tools_declaration"
"""Top-level ``SFTSample.tools`` rendered as function signatures somewhere
in the prompt (location depends on family)."""

FEATURE_TOOL_CALL = "tool_call"
"""``ToolUseBlock`` inside an assistant message — the model emits a call."""

FEATURE_TOOL_RESULT = "tool_result"
"""``ToolResultBlock`` inside a tool-role message — call output fed back."""

FEATURE_PARALLEL_TOOL_CALLS = "parallel_tool_calls"
"""Two or more ``ToolUseBlock`` instances inside one assistant message."""

FEATURE_THINKING_CURRENT_TURN = "thinking_current_turn"
"""``ThinkingBlock`` on the LAST assistant turn (preserved by all
thinking-capable families)."""

FEATURE_THINKING_HISTORY = "thinking_history"
"""``ThinkingBlock`` on assistant turns that are NOT the last turn.
Most families drop these at inference time; supervising them is a
training-recipe choice that some families preserve (qwen3, dsv4, gpt_oss)
and others don't."""

FEATURE_THINKING_INTENT_MODE = "thinking_intent_mode"
"""Honors top-level ``SFTSample.thinking_intent.mode`` (on/off switch for
the thinking branch). Families declaring this must implement the wire-level
projection (qwen3 / gemma4: inject empty ``<think>`` markers; dsv4: force
``nm.thinking=None``; gpt_oss: ``Reasoning: none`` line). Families without
support (qwen2_5, llama3, deepseek_v3) declare absence so dispatch warn-
loud surfaces a configuration drift before training silently ignores
the user's intent."""

FEATURE_THINKING_INTENT_EFFORT = "thinking_intent_effort"
"""Honors top-level ``SFTSample.thinking_intent.effort`` (low/medium/high
reasoning depth dial). Currently ONLY gpt_oss has a wire-level field for
this (the Harmony ``Reasoning: <effort>`` line); qwen3 / dsv4 / gemma4
chat templates carry no equivalent. See the 2026-05-31 wire survey in
``docs/superpowers/research/2026-06-thinking-intent-next-session.md`` —
declaring absence here means a sample with ``effort`` set against e.g.
dsv4 emits one warn-loud log line and the encoder ignores it."""

FEATURE_IMAGE = "image"
"""``ImageBlock`` content — VLM-only."""

FEATURE_LOSS_WEIGHT_PER_BLOCK = "loss_weight_per_block"
"""Honor ``Block.weight`` for per-token loss scaling."""

FEATURE_LOSS_WEIGHT_PER_MESSAGE = "loss_weight_per_message"
"""Honor ``Message.weight`` for per-token loss scaling."""

FEATURE_TYPE_OVERRIDES = "type_overrides"
"""Honor ``LossConfig.type_overrides`` block-type-keyed multipliers."""


# ---------------------------------------------------------------------------
# HistoryPolicy capability flags (Phase 1.3 / spec §3.3)
# ---------------------------------------------------------------------------
# Per-family declaration of HistoryPolicy support. A family may honor
# some actions and not others (e.g. all families can ``drop``, but only
# DSv4 has a wire-level slot for ``latest_reminder``). Unsupported styles
# warn-loud and fall back to ``system_prefix`` (spec §3.3 footnote).

FEATURE_HISTORY_POLICY_DROP = "history_policy_drop"
"""Honors ``HistoryAction(action='drop')`` for historical-turn blocks."""

FEATURE_HISTORY_POLICY_SUMMARIZE = "history_policy_summarize"
"""Honors ``HistoryAction(action='summarize')`` — replaces block with a
TextBlock rendered from ``summary_template``."""

FEATURE_HISTORY_SUMMARY_BLOCK = "history_summary_block"
"""Renders ``HistorySummaryBlock`` in some family-specific location.
Required if the family wants to honor ``HistoryPolicy.insert_summary``."""

FEATURE_HISTORY_SUMMARY_LATEST_REMINDER = "history_summary_latest_reminder"
"""Renders a ``HistorySummaryBlock(style='latest_reminder')`` using a
DSv4-style ``<｜latest_reminder｜>...<｜/latest_reminder｜>`` wrapping.
DSv4-only at time of writing."""

FEATURE_HISTORY_SUMMARY_SYSTEM_PREFIX = "history_summary_system_prefix"
"""Renders a ``HistorySummaryBlock(style='system_prefix')`` by prepending
the summary text to the system prompt. Universal fallback; every family
declares support."""

FEATURE_HISTORY_SUMMARY_EPHEMERAL_BLOCK = "history_summary_ephemeral_block"
"""Renders a ``HistorySummaryBlock(style='ephemeral_block')`` as a
standalone protocol turn (Gemma4 ``<|turn>system`` block)."""


# ---------------------------------------------------------------------------
# Spec dataclass — optional bundling for encoders that want it
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CapabilitySpec:
    """Optional convenience bundle for an encoder's full capability profile.

    Encoders MAY declare a single ``CAPABILITY: CapabilitySpec = ...``
    instead of two separate ``SUPPORTED_BLOCKS`` / ``SUPPORTED_FEATURES``
    constants. Dispatch accepts either form.

    The ``notes`` field is a free-form per-feature reason map used by the
    encoder author to record WHY a feature is unsupported — surfaced in
    the warning message so users can grep root cause.
    """

    blocks: frozenset[type] = field(default_factory=frozenset)
    features: frozenset[str] = field(default_factory=frozenset)
    notes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Warning helper
# ---------------------------------------------------------------------------
_logger = logging.getLogger("titaninfra.sft.capabilities")


def warn_unsupported_block(
    family: str,
    block_type_name: str,
    message_index: int,
    block_index: int,
    reason: str | None = None,
) -> None:
    """Emit a warn-loud log entry for one skipped block.

    The format is intentionally structured (key=value pairs) so CI greps
    and caplog asserts can target specific fields without parsing prose.

    Caller is responsible for actually skipping the block in its encode
    loop — this helper only logs.
    """
    reason_suffix = f" reason={reason!r}" if reason else ""
    _logger.warning(
        "capability_mismatch family=%s block=%s message_idx=%d block_idx=%d skipped=True%s",
        family,
        block_type_name,
        message_index,
        block_index,
        reason_suffix,
    )


def warn_unsupported_feature(
    family: str,
    feature: str,
    reason: str | None = None,
) -> None:
    """Emit a warn-loud log entry for one ignored IR-level feature.

    Used when a sample carries a feature flag (e.g. ``thinking_intent``)
    that the target family does not honor. Logged once per
    ``tokenize_sample`` call, not once per block.
    """
    reason_suffix = f" reason={reason!r}" if reason else ""
    _logger.warning(
        "capability_mismatch family=%s feature=%s ignored=True%s",
        family,
        feature,
        reason_suffix,
    )


def warn_history_thinking_trained(family: str, n_history_thinking_turns: int) -> None:
    """Warn that historical-turn thinking will be supervised (trained on).

    Emitted when a multi-turn sample carries ``ThinkingBlock``s on
    historical assistant turns (``turn_role == "history"``) AND the
    sample provided no ``history_policy`` to drop/summarize them. By
    default titaninfra supervises historical thinking verbatim — the
    opposite of some families' shipped inference recipe (e.g. DeepSeek's
    reference defaults ``drop_thinking=True``, rendering past turns as a
    bare close marker).

    This is a deliberate titaninfra design choice (the training engine owns
    the masking policy, not the model's chat_template — see memory
    ``feedback_jinja_is_recipe_projection_not_protocol``), surfaced loud
    so the user can opt into dropping via ``StreamConfig.history_policy``
    (``per_type={"thinking": HistoryAction(action="drop")}``) when they
    want train/inference alignment with the family's default recipe.
    """
    _logger.warning(
        "history_thinking_trained family=%s history_thinking_turns=%d "
        "history_policy=None hint='set StreamConfig.history_policy per_type "
        "thinking=drop to align with the family inference recipe'",
        family,
        n_history_thinking_turns,
    )


# ---------------------------------------------------------------------------
# EMPTY-RAISE exception
# ---------------------------------------------------------------------------
class EmptyAssistantError(ValueError):
    """All assistant content was skipped due to capability mismatch.

    Raised by ``dispatch.tokenize_sample`` (NOT by encoders) when post-
    skip there is no learnable assistant content left. This is the only
    case where capability mismatch escalates from warn to raise — the
    sample contributes nothing to training and silently emitting a
    zero-loss EncodedSample would mask the bug.

    Attributes are designed for test assertions and structured logging:
    ``family`` identifies the encoder, ``skipped_block_types`` enumerates
    what got dropped, ``sample_repr`` is a short repr of the input for
    log forensics (NOT the full sample — could be PII / huge).
    """

    def __init__(
        self,
        family: str,
        skipped_block_types: list[str],
        sample_repr: str = "",
    ) -> None:
        self.family = family
        self.skipped_block_types = list(skipped_block_types)
        self.sample_repr = sample_repr
        super().__init__(
            f"All assistant blocks were skipped by {family}Encoder due to "
            f"capability mismatch (skipped types: {sorted(set(skipped_block_types))}). "
            f"Sample contributes no learnable content; refusing to emit empty EncodedSample. "
            f"Either pick a family that supports these blocks or remove them from the sample."
        )


__all__ = [
    "FEATURE_SYSTEM",
    "FEATURE_MULTI_TURN",
    "FEATURE_TOOLS_DECLARATION",
    "FEATURE_TOOL_CALL",
    "FEATURE_TOOL_RESULT",
    "FEATURE_PARALLEL_TOOL_CALLS",
    "FEATURE_THINKING_CURRENT_TURN",
    "FEATURE_THINKING_HISTORY",
    "FEATURE_THINKING_INTENT_MODE",
    "FEATURE_THINKING_INTENT_EFFORT",
    "FEATURE_IMAGE",
    "FEATURE_LOSS_WEIGHT_PER_BLOCK",
    "FEATURE_LOSS_WEIGHT_PER_MESSAGE",
    "FEATURE_TYPE_OVERRIDES",
    "FEATURE_HISTORY_POLICY_DROP",
    "FEATURE_HISTORY_POLICY_SUMMARIZE",
    "FEATURE_HISTORY_SUMMARY_BLOCK",
    "FEATURE_HISTORY_SUMMARY_LATEST_REMINDER",
    "FEATURE_HISTORY_SUMMARY_SYSTEM_PREFIX",
    "FEATURE_HISTORY_SUMMARY_EPHEMERAL_BLOCK",
    "CapabilitySpec",
    "warn_unsupported_block",
    "warn_unsupported_feature",
    "warn_history_thinking_trained",
    "EmptyAssistantError",
]
