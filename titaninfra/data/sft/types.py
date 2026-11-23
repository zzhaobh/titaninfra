"""
Token-first SFT encoder data types.

These dataclasses are the contract between the four layers of the new
SFT pipeline:

    raw SFTSample
        │
        ▼  normalize.py
    NormalizedSample        ── canonical, family-agnostic
        │
        ▼  encoders/<family>.py
    EncodedSample           ── (input_ids, labels, positions)
        │
        ▼  packer / dataloader
    training tensors

All structures are frozen. Invariants are asserted at construction time
so a bad encoder fails at the boundary, not three layers downstream.

The encoder contract terminates at EncodedSample. The FlexInnerAttention
BlockMask is constructed downstream in ``Model.get_attention_masks()``
from ``positions == 0`` document-boundary signals; the encoder owns
the positions, not the mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from titaninfra.data.schema import HistoryPolicy

IGNORE_INDEX: int = -100

# DSv4 adds developer and latest_reminder roles for tool declarations and
# session headers. Other families still reject roles they do not model.
Role = Literal["system", "user", "assistant", "tool", "developer", "latest_reminder"]


@dataclass(frozen=True)
class NormalizedMessage:
    """Canonical, family-agnostic message.

    Phase 2 (encoder-layer-refactor) ships this in a *transitional*
    shape: the canonical-going-forward field is ``blocks`` — encoders
    walk the structured Block tuple directly. The legacy fields
    (``text`` / ``thinking`` / ``tool_calls`` / ``tool_call_id``) are
    kept populated by ``normalize.py`` so existing encoder callsites
    compile unchanged until each family is migrated. After every
    encoder reads from ``blocks`` only, the legacy fields will be
    removed (Phase 5).

    Fields (canonical)
    ------------------
    role:
        One of ``system`` / ``user`` / ``assistant`` / ``tool`` /
        ``developer`` / ``latest_reminder``.
    blocks:
        Tuple of structured Block instances (see ``titaninfra.data.schema``)
        from the source SFTSample. Encoders walk this directly; no
        reverse-parsing of a pre-rendered string. Empty tuple is legal
        for purely-marker turns synthesized by normalize itself.
    weight:
        Post-cascade weight in ``[0.0, 1.0]``. ``0.0`` → mask out;
        ``> 0.0`` → contribute to loss.
    turn_role:
        One of ``current`` / ``history`` / ``summary``. ``current`` is
        the default; ``history`` marks turns strictly before the last
        user turn (subject to HistoryPolicy); ``summary`` marks the
        synthesized turn that carries a ``HistorySummaryBlock`` when
        ``HistoryPolicy.insert_summary`` is True.
    task:
        DSv4-specific per-message ``task`` tag (e.g. ``"action"`` for
        the task-transition token).
    """

    role: Role
    weight: float = 0.0
    task: str | None = None
    blocks: tuple = ()  # type: ignore[type-arg]
    turn_role: Literal["current", "history", "summary"] = "current"

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(
                f"NormalizedMessage.weight must be in [0.0, 1.0], got {self.weight!r}"
            )


@dataclass(frozen=True)
class NormalizedThinkingIntent:
    """Family-agnostic thinking-mode intent hoisted to ``NormalizedSample``.

    Mirrors the schema-level ``ThinkingIntent`` msgspec struct but in
    plain-dataclass form so the normalize→encoder hop stays inside the
    SFT pipeline's frozen-dataclass world. Encoders read this directly;
    see encoder per-family docstrings for projection rules.

    Both sub-fields are independently optional. ``mode`` is the on/off
    switch (all thinking-capable families honor it); ``effort`` is the
    depth dial (only gpt_oss has a wire-level field for it — other
    families warn-loud and ignore via the dispatch capability check).
    """

    mode: str | None = None
    effort: str | None = None


@dataclass(frozen=True)
class NormalizedSample:
    """Canonical, family-agnostic training sample.

    System prompt is hoisted to a top-level field rather than living
    in the first message — encoders that need to splice it into the
    template prologue (Qwen3, Llama3) read it directly; encoders that
    render it as a normal turn (ChatML) ignore the field and consume
    the synthesized system message in ``messages``.

    Tools likewise sit at the top level — each family renders them
    where its protocol demands (Qwen3 system tail, GPT-OSS developer
    block, Gemma4 standalone block, Llama4 pre-first-user).

    ``thinking_intent`` carries the optional sample-level mode/effort
    intent from ``SFTSample.thinking_intent``. Per-family projection
    rules live in the encoder modules; the dispatch capability check
    surfaces samples that set fields a family does not honor.

    ``history_policy`` (Phase 2 / spec §3.2): per-Block-type strategy
    for historical-turn handling. ``None`` reproduces legacy behavior
    (no compression). Encoders that declare any
    ``FEATURE_HISTORY_POLICY_*`` flag read this directly; encoders
    that don't declare support warn-loud at dispatch time and fall
    back to ``turn_role="current"`` semantics.
    """

    messages: tuple[NormalizedMessage, ...]
    tools: tuple[dict, ...] = ()
    system: str | None = None
    thinking_intent: NormalizedThinkingIntent | None = None
    history_policy: HistoryPolicy | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("NormalizedSample must contain at least one message")


@dataclass(frozen=True)
class EncodedSample:
    """Per-family encoder output. The contract terminates here.

    The parallel arrays are aligned 1:1 by token position:
        input_ids[i]    — the token id
        labels[i]       — token id when contributing to loss, else IGNORE_INDEX
        positions[i]    — 0 at each document start, 1..N within document;
                          the FlexInnerAttention mask is built downstream from
                          ``positions == 0`` so document boundaries flow
                          from the encoder, not the packer.
        loss_weights[i] — optional per-token loss multiplier in [0.0, +∞).
                          ``None`` means "no per-token weighting" (downstream
                          consumers treat absent weights as a uniform 1.0).
                          Populated by the Jinja path when ``loss_config.
                          type_overrides`` or ``StreamConfig.thinking_scale``
                          is active; native encoders do not yet emit weights
                          (M2.0 wiring only surfaces what Jinja produces).

    Padding is NOT applied here — the packer / dataloader owns padding
    and uses ``-1`` for pad positions (so ``positions == 0`` remains
    an unambiguous boundary signal).
    """

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    positions: tuple[int, ...]
    loss_weights: tuple[float, ...] | None = None
    # Optional VLM tensors: flattened image patches and per-image grid shapes.
    # They are not token-aligned, so length assertions exclude them.
    pixel_values: object | None = None
    grid_thw: list[tuple[int, int, int]] | None = None
    debug: dict = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        n = len(self.input_ids)
        if len(self.labels) != n:
            raise AssertionError(
                f"EncodedSample: labels length {len(self.labels)} != input_ids length {n}"
            )
        if len(self.positions) != n:
            raise AssertionError(
                f"EncodedSample: positions length {len(self.positions)} != input_ids length {n}"
            )
        if self.loss_weights is not None and len(self.loss_weights) != n:
            raise AssertionError(
                f"EncodedSample: loss_weights length {len(self.loss_weights)} != input_ids length {n}"
            )
        if n == 0:
            raise AssertionError("EncodedSample must be non-empty")
