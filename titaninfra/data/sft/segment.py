"""Token-first cross-layer payload for the segment-encoder pipeline.

A ``Segment`` is one bit of the on-the-wire stream that Layer 2 (the
per-family ``FamilyEncoder``) emits and Layer 3 (the ``Assembler``)
consumes. Encoders never tokenize and never read training policy --
they just describe the wire-level layout one segment at a time.

The spec for the new pipeline lives at
``docs/superpowers/specs/2026-05-28-segment-encoder-interface-draft.md``.
M-alpha only: this module is additive; no existing callsite imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Role = Literal["system", "user", "assistant", "tool", "developer", "latest_reminder"]


class SegKind(str, Enum):
    """What this segment means to the protocol. Drives Assembler routing.

    MARKER          role tag / bos / framework turn separator -- never has loss
                    (the inference framework injects it identically; the model
                    does NOT autonomously emit it).
    STOP_MARKER     the model's OWN autonomous turn-terminator -- the token the
                    model decides to emit to end its turn (qwen3/qwen2_5
                    assistant ``<|im_end|>``, llama3 ``<|eot_id|>``, dsv3
                    ``<｜end▁of▁sentence｜>``, gpt_oss ``<|return|>`` final /
                    ``<|call|>`` tool-call). At inference vLLM/SGLang stop when
                    the model emits it, so it IS trained (class-B). The ONLY
                    MARKER-family kind permitted to carry loss.
    CONTENT         user / assistant plain text
    THINKING        assistant <think>...</think> body -- may be scaled
    TOOL_CALL       assistant <tool_call>...</tool_call> body
    TOOL_RESULT     tool-side response merged into user shell
    SYSTEM          system prompt body
    TOOLS_HEADER    the once-per-conversation tools schema block
    IMAGE           VLM image placeholder -- carries image_source, no text
    """

    MARKER = "marker"
    STOP_MARKER = "stop_marker"
    CONTENT = "content"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    TOOLS_HEADER = "tools_header"
    IMAGE = "image"


class TokenClass(str, Enum):
    """A/B/C token classification — see ``feedback_token_classification_abc_rule``.

    Determines whether each token survives in the inference prompt:

      PROTOCOL — framework injects identically at inference. KEEP.
                 (``<|im_start|>``, opening role markers, system / user
                 closing ``<|im_end|>``, tool framing tokens.)
      DECODED  — LLM autonomously emits. KEEP. Trains with loss.
                 (assistant body, ``<think>`` inner, tool_call JSON
                 payload, assistant turn's closing ``<|im_end|>`` — the
                 model's own stop signal.)
      METADATA — present in training, MUST BE DELETED in the inference
                 prompt builder. (GPT-OSS ``Current date:``, default
                 ``Reasoning:`` line, Llama3 trailing assistant header,
                 jinja-only blank lines.)

    See ``titaninfra/CLAUDE.md`` → "Token Three-Way Classification" for the
    full taxonomy and the inference-prompt-builder rationale.
    """

    PROTOCOL = "protocol"
    DECODED = "decoded"
    METADATA = "metadata"


def _base_mask(kind: SegKind, role: Role) -> float:
    """Structural half of loss_scale — does this (kind, role) ever carry loss?

    Returns 1.0 iff the token is one the MODEL autonomously emits at
    inference (so training on it is correct), else 0.0 for anything the
    inference framework injects identically (training on it teaches a
    phantom prefix → train/serve drift). This is the spec §3.2 table;
    every Segment's final ``loss_scale`` is ``_base_mask × scale_factor``.

    Rule:
      * role != "assistant"                  → 0.0 (framework-authored turn)
      * assistant + STOP_MARKER/CONTENT/
        THINKING/TOOL_CALL                   → 1.0 (model decodes it)
      * assistant + MARKER/SYSTEM/
        TOOLS_HEADER/TOOL_RESULT             → 0.0 (framework-injected)
      * IMAGE (any role)                     → 0.0 (vision input, not generated)
    """
    if role != "assistant":
        return 0.0
    if kind in (
        SegKind.STOP_MARKER,
        SegKind.CONTENT,
        SegKind.THINKING,
        SegKind.TOOL_CALL,
    ):
        return 1.0
    return 0.0


@dataclass(frozen=True)
class Segment:
    """One bit of the on-the-wire stream, with everything Assembler needs.

    Exactly one of ``text`` / ``token_ids`` is set. ``text`` is the common
    case; ``token_ids`` exists for pre-encoded special-token sequences
    (e.g. ``<|im_start|>`` in a family where the tokenizer splits it
    inconsistently).

    ``loss_scale`` is the FINAL per-token multiplier. The boundary
    builder has already collapsed (sft_mask_mode x msg.weight x
    type_overrides) into this single float, so Assembler just writes it
    out -- no policy logic in Layer 3.

      0.0  -> labels = IGNORE_INDEX for every token in this segment
      1.0  -> labels = token_id, loss_weights unset (binary fast path)
      else -> labels = token_id, loss_weights[t] = scale

    ``is_generation`` tells the RL Assembler where the prompt ends.
    First segment with ``is_generation=True`` starts the response;
    everything before is prompt.
    """

    text: str | None
    token_ids: tuple[int, ...] | None
    kind: SegKind
    role: Role
    is_generation: bool
    # token_class is an explicit A/B/C label for inference-prompt consumers.
    # It cannot be derived from kind and role alone: a dangling assistant
    # header may be MARKER+assistant but must be METADATA, not PROTOCOL.
    token_class: TokenClass
    # Optional source NormalizedMessage index.
    msg_index: int | None = None
    # scale_factor is the multiplicative policy term. __post_init__ computes
    # loss_scale = _base_mask(kind, role) * (scale_factor or 1.0).
    scale_factor: float | None = None
    # image_source is valid only for IMAGE segments and stores the source
    # reference. Training consumes predecoded tensors from ImageDecodePool.
    image_source: dict | None = None
    # loss_scale is derived in __post_init__, never accepted as input.
    loss_scale: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.kind is SegKind.IMAGE:
            if self.image_source is None:
                raise ValueError(
                    "Segment(kind=IMAGE) requires image_source (dict), got None"
                )
            if self.text is not None or self.token_ids is not None:
                raise ValueError(
                    "Segment(kind=IMAGE): text and token_ids must both be None "
                    f"(text={self.text!r}, token_ids={self.token_ids!r})"
                )
        else:
            if self.image_source is not None:
                raise ValueError(
                    "Segment.image_source is only valid for kind=IMAGE "
                    f"(got kind={self.kind!r})"
                )
            if (self.text is None) == (self.token_ids is None):
                raise ValueError(
                    "Segment: exactly one of text / token_ids must be set "
                    f"(text={self.text!r}, token_ids={self.token_ids!r})"
                )

        base = _base_mask(self.kind, self.role)
        factor = 1.0 if self.scale_factor is None else float(self.scale_factor)
        if factor < 0.0:
            raise ValueError(
                f"Segment.scale_factor must be non-negative, got {self.scale_factor!r}"
            )
        object.__setattr__(self, "loss_scale", base * factor)

        # Validate token-class invariants against the final loss_scale.
        final = self.loss_scale
        if self.token_class is TokenClass.METADATA and final != 0.0:
            raise ValueError(
                "Segment with token_class=METADATA must have loss_scale==0.0 "
                f"(got loss_scale={final!r})"
            )
        if (
            self.token_class is TokenClass.PROTOCOL
            and final != 0.0
            and not (self.kind is SegKind.STOP_MARKER and self.role == "assistant")
        ):
            raise ValueError(
                "Segment with token_class=PROTOCOL must have loss_scale==0.0 "
                "unless it's an assistant turn's STOP_MARKER (the model's own "
                "autonomous turn terminator) "
                f"(got kind={self.kind!r}, role={self.role!r}, "
                f"is_generation={self.is_generation!r}, loss_scale={final!r})"
            )
        if (
            final > 0.0
            and self.token_class is not TokenClass.DECODED
            and not (
                self.token_class is TokenClass.PROTOCOL
                and self.kind is SegKind.STOP_MARKER
                and self.role == "assistant"
            )
        ):
            raise ValueError(
                "Segment with loss_scale>0 must be DECODED "
                f"(got token_class={self.token_class!r}, loss_scale={final!r})"
            )


__all__ = ["SegKind", "Segment", "TokenClass", "Role", "_base_mask"]
