"""Qwen2.5 family SFT encoder (segment path, blocks-only).

Pure ChatML, one Qwen2.5-specific quirk:

* **Default-system auto-injection**. When the sample carries no system
  message, the Jinja template emits ``You are Qwen, created by Alibaba
  Cloud. You are a helpful assistant.`` before the per-message loop.
  Encoder mirrors this.

Out of scope: thinking blocks, tool_calls, tool-role messages. Qwen2.5
has no native protocol for any of these — training jobs that need them
pick the qwen3 family. Capability dispatch warns-and-skips offending
blocks before encode_segments runs; this encoder asserts on roles it
does not model and ignores Block subclasses outside SUPPORTED_BLOCKS.

Phase 3.1 (encoder-layer-refactor): walks ``nm.blocks`` directly.
``_build_hf_messages`` and the ``nm.text`` reverse-engineered string
view are gone. Every Segment carries an explicit ``token_class``.
"""

from __future__ import annotations

from titaninfra.data.schema import TextBlock
from titaninfra.data.sft.capabilities import (
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
    FEATURE_SYSTEM,
)
from titaninfra.data.sft.encoders import _blocks, register
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass
from titaninfra.data.sft.types import EncodedSample, NormalizedMessage, NormalizedSample

# Capability declaration — see titaninfra/data/sft/capabilities.py.
# Qwen2.5 wire is pure ChatML text turns only. Thinking, tool_call,
# tool_result, parallel_tool_calls, tools_declaration, thinking_history,
# thinking_intent are all UNSUPPORTED — Qwen2.5 vocab has no <think>
# tokens nor Hermes <tool_call> protocol; if a sample carries those
# blocks dispatch will warn-and-skip them. Pick the qwen3 family
# for thinking / tool training.
SUPPORTED_BLOCKS = frozenset({TextBlock})
SUPPORTED_FEATURES = frozenset({
    FEATURE_SYSTEM,
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
})

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_DEFAULT_SYSTEM = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)


@register("qwen2_5")
class Qwen2_5Encoder:
    """Token-first encoder for the Qwen2.5 family."""

    name = "qwen2_5"

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(
        self,
        sample: NormalizedSample,
        *,
        max_length: int | None = None,
        loss_config: dict | None = None,
        truncate_at_last_segment: bool = False,
    ) -> EncodedSample:
        """Delegate to encode_segments + assemble."""
        from titaninfra.data.sft.assembler import assemble

        del loss_config  # qwen2.5 has no scalable inner segments
        segments = self.encode_segments(sample)
        return assemble(
            segments, self.tokenizer, mode="sft", max_length=max_length,
            truncate_at_last_segment=truncate_at_last_segment,
        )

    def encode_segments(
        self,
        sample: NormalizedSample,
        *,
        type_overrides: dict[str, float] | None = None,
    ) -> list[Segment]:
        del type_overrides  # qwen2.5 has no thinking / tool_call blocks
        segs: list[Segment] = []

        # 1. System prologue:
        #   - explicit system message → render as system turn
        #   - top-level SFTSample.system (no system message) → render
        #   - neither → inject default-system (Jinja parity)
        first_msg = sample.messages[0] if sample.messages else None
        has_explicit_system_msg = first_msg is not None and first_msg.role == "system"

        if has_explicit_system_msg:
            # Will be rendered in the per-message loop below; nothing
            # to inject here.
            pass
        elif sample.system:
            segs.extend(self._render_system_turn(sample.system, msg_index=None))
        else:
            segs.extend(self._render_system_turn(_DEFAULT_SYSTEM, msg_index=None))

        # 2. Per-message loop.
        for i, msg in enumerate(sample.messages):
            self._assert_role_supported(msg)
            if msg.role == "system":
                segs.extend(self._render_system_from_msg(msg, msg_index=i))
            elif msg.role == "user":
                segs.extend(self._render_user_turn(msg, msg_index=i))
            elif msg.role == "assistant":
                segs.extend(self._render_assistant_turn(msg, msg_index=i))
            else:  # pragma: no cover — guarded by _assert_role_supported
                raise NotImplementedError(
                    f"Qwen2_5Encoder: role={msg.role!r} not supported"
                )

        return segs

    # ------------------------------------------------------------------
    # Per-role / per-block emitters
    # ------------------------------------------------------------------
    @staticmethod
    def _assert_role_supported(msg: NormalizedMessage) -> None:
        if msg.role in ("system", "user", "assistant"):
            return
        raise NotImplementedError(
            f"Qwen2_5Encoder does not handle role={msg.role!r}. "
            "Qwen2.5 has no native tool / thinking protocol; pick the qwen3 family."
        )

    @staticmethod
    def _block_text(blocks) -> str:
        """Concatenate the ``text`` field of every TextBlock; ignore others.

        Capability dispatch already warned about non-TextBlock blocks
        before reaching here, so silently skipping them mirrors the
        warn-loud-then-skip protocol. An assistant turn with zero
        TextBlocks reaches assemble() with zero CONTENT-segment text,
        which assemble rejects as a 0-token segment — that is the
        right failure mode (the user sees that the encoder was given
        nothing to learn).
        """
        return _blocks.block_text(blocks)

    def _render_system_turn(
        self, system_text: str, *, msg_index: int | None
    ) -> list[Segment]:
        """Render an injected/standalone system block (one MARKER + body wrapper)."""
        return [
            Segment(
                text=f"{_IM_START}system\n{system_text}{_IM_END}\n",
                token_ids=None,
                kind=SegKind.SYSTEM,
                role="system",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=msg_index,
            )
        ]

    def _render_system_from_msg(
        self, msg: NormalizedMessage, *, msg_index: int
    ) -> list[Segment]:
        text = self._block_text(msg.blocks)
        return self._render_system_turn(text, msg_index=msg_index)

    def _render_user_turn(
        self, msg: NormalizedMessage, *, msg_index: int
    ) -> list[Segment]:
        text = self._block_text(msg.blocks)
        return [
            Segment(
                text=f"{_IM_START}user\n{text}{_IM_END}\n",
                token_ids=None,
                kind=SegKind.CONTENT,
                role="user",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=msg_index,
            )
        ]

    def _render_assistant_turn(
        self, msg: NormalizedMessage, *, msg_index: int
    ) -> list[Segment]:
        # Split into:
        #  * MARKER (PROTOCOL, loss=0) — the opening role tag the
        #    inference framework injects identically.
        #  * CONTENT (DECODED, loss=msg.weight) — body + closing
        #    <|im_end|>\n. The closing marker is class-B (model's
        #    autonomous stop signal); keeping it inside the body
        #    segment lets the model learn the turn EOS.
        body_text = self._block_text(msg.blocks)
        if not body_text:
            # An assistant turn with zero text blocks contributes no
            # learnable content. This happens when the sample carries
            # only unsupported blocks (e.g. HF-standard tool_calls) on
            # an assistant turn and capability dispatch warned-and-
            # skipped them. Fail loud — silently training on a bare
            # <|im_end|> would mask the data error.
            raise ValueError(
                f"Qwen2_5Encoder: assistant turn at msg_index={msg_index} "
                "has no learnable text content (zero TextBlock-derived text). "
                "If the sample carries tool_calls / thinking, switch to "
                "the qwen3 family; otherwise check the sample's blocks/content fields."
            )
        # BPE-stable assistant emit:
        #
        # * Clean body (no leading ``\n``): MARKER ends with ``\n``,
        #   CONTENT carries the body. The MARKER ``\n`` is its own token,
        #   masked. This is the historical layout and what the oracle
        #   loss-mask baseline expects.
        # * Leading ``\n`` body: relocate the trailing ``\n`` from MARKER
        #   into CONTENT head so joint-render BPE can merge ``\n\n`` into
        #   one token (id 271 in qwen2.5 vocab). Without relocation,
        #   two-piece tokenize produces 2 separate ``\n`` tokens — 1
        #   phantom token of drift vs jinja. The merged ``\n\n`` falls
        #   inside CONTENT and earns ``loss_scale=msg.weight``; this
        #   matches the HF Trainer-style SFT convention where the
        #   ``apply_chat_template`` output is tokenized once and the
        #   loss mask is computed from response-prefix offsets.
        # F1: split the closing ``<|im_end|>\n`` (model's OWN autonomous
        # stop signal) into a dedicated STOP_MARKER segment instead of
        # folding it into CONTENT — uniform stop-token shape across families
        # (matches qwen3). The leading-``\n`` relocation still applies to the
        # CONTENT head only; the trailing close marker now lives in its own
        # segment. <|im_end|> is a special token so CONTENT|STOP_MARKER is
        # BPE-stable, and the STOP_MARKER's leading char is <|im_end|> so
        # CONTENT's body tail cannot merge across the seam.
        if body_text.startswith("\n"):
            marker_text = _IM_START + "assistant"
            content_text = f"\n{body_text}"
        else:
            marker_text = f"{_IM_START}assistant\n"
            content_text = body_text
        return [
            Segment(
                text=marker_text,
                token_ids=None,
                kind=SegKind.MARKER,
                role="assistant",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=msg_index,
            ),
            Segment(
                text=content_text,
                token_ids=None,
                kind=SegKind.CONTENT,
                role="assistant",
                scale_factor=float(msg.weight),
                is_generation=False,
                token_class=TokenClass.DECODED,
                msg_index=msg_index,
            ),
            Segment(
                text=f"{_IM_END}\n",
                token_ids=None,
                kind=SegKind.STOP_MARKER,
                role="assistant",
                scale_factor=float(msg.weight),
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=msg_index,
            ),
        ]


__all__ = ["Qwen2_5Encoder"]
