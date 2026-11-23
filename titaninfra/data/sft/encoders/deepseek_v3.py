"""DeepSeek-V3 / R1 family SFT encoder (segment path, blocks-only).

Wire format -- mirrors the byte sequence the official DSv3 jinja
``chat_template.jinja`` produces. Anchors:

* ``<｜begin▁of▁sentence｜>`` BOS (always emitted)
* System messages concatenated raw between BOS and the per-message
  loop, multiple systems joined by ``\\n\\n``. **No marker prefix /
  suffix.** This is unique to DSv3 (vs ChatML which wraps everything
  with role markers).
* ``user``  -> ``<｜User｜>{content}`` (no suffix; next marker
  immediately follows)
* ``assistant`` text turn (no preceding tool message) ->
  ``<｜Assistant｜>{content}<｜end▁of▁sentence｜>``
* ``assistant`` text after tool messages ->
  ``<｜tool▁outputs▁end｜>{content}<｜end▁of▁sentence｜>``
  (the ``<｜Assistant｜>`` marker is SUPPRESSED -- the model continues
  the prior tool-output frame)
* ``assistant`` with tool_calls and content empty ->
  ``<｜Assistant｜><｜tool▁calls▁begin｜>{call_0}{call_n>=1...}``
  Each subsequent call is prefixed with ``\\n`` and suffixed with
  ``<｜tool▁calls▁end｜><｜end▁of▁sentence｜>``. **Yes, this is a jinja
  bug** -- the closing tags fire on every iteration starting at index 1.
* ``tool`` first-of-chain ->
  ``<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{content}``
  ``<｜tool▁output▁end｜>`` ; subsequent tool in chain prepends ``\\n``.
* End-of-conversation while still ``is_tool`` -> append
  ``<｜tool▁outputs▁end｜>``.

Phase 3.3 (encoder-layer-refactor): walks ``nm.blocks`` directly.
``_build_hf_messages`` and ``nm.text``/``nm.thinking``/``nm.tool_calls``
reads are gone. Every Segment carries an explicit ``token_class``.
"""

from __future__ import annotations

import json

from titaninfra.data.schema import TextBlock, ToolResultBlock, ToolUseBlock
from titaninfra.data.sft.capabilities import (
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
    FEATURE_PARALLEL_TOOL_CALLS,
    FEATURE_SYSTEM,
    FEATURE_TOOL_CALL,
    FEATURE_TOOL_RESULT,
    FEATURE_TOOLS_DECLARATION,
)
from titaninfra.data.sft.encoders import _blocks, register
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass
from titaninfra.data.sft.types import EncodedSample, NormalizedMessage, NormalizedSample

# Capability declaration — see titaninfra/data/sft/capabilities.py.
SUPPORTED_BLOCKS = frozenset({TextBlock, ToolUseBlock, ToolResultBlock})
SUPPORTED_FEATURES = frozenset({
    FEATURE_SYSTEM,
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
    FEATURE_TOOLS_DECLARATION,
    FEATURE_TOOL_CALL,
    FEATURE_TOOL_RESULT,
    FEATURE_PARALLEL_TOOL_CALLS,
})

_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
_TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_TOOL_CALL_END = "<｜tool▁call▁end｜>"
_TOOL_SEP = "<｜tool▁sep｜>"
_TOOL_OUTPUTS_BEGIN = "<｜tool▁outputs▁begin｜>"
_TOOL_OUTPUTS_END = "<｜tool▁outputs▁end｜>"
_TOOL_OUTPUT_BEGIN = "<｜tool▁output▁begin｜>"
_TOOL_OUTPUT_END = "<｜tool▁output▁end｜>"


@register("deepseek_v3")
class DeepSeekV3Encoder:
    """Token-first encoder for the DeepSeek-V3 / R1 family."""

    name = "deepseek_v3"

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
        from titaninfra.data.sft.assembler import assemble

        del loss_config
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
        del type_overrides

        # Build the effective message list with optional synthesized system.
        messages: list[tuple[int | None, NormalizedMessage]] = []
        first = sample.messages[0] if sample.messages else None
        if sample.system and (first is None or first.role != "system"):
            messages.append((None, NormalizedMessage(
                role="system",
                blocks=(TextBlock(text=sample.system),),
                weight=0.0,
                turn_role="current",
            )))
        for i, m in enumerate(sample.messages):
            messages.append((i, m))

        # System prologue: BOS + concatenated system content (no markers).
        # Multiple systems joined by "\n\n"; if none, prologue is just BOS.
        system_msgs = [(idx, m) for idx, m in messages if m.role == "system"]
        prologue_text = _BOS + "\n\n".join(
            _block_text(m.blocks) for _, m in system_msgs
        )
        segs: list[Segment] = [Segment(
            text=prologue_text,
            token_ids=None,
            kind=SegKind.SYSTEM,
            role="system",
            is_generation=False,
            token_class=TokenClass.PROTOCOL,
            msg_index=system_msgs[0][0] if system_msgs else None,
        )]

        # is_tool state machine: tracks whether we are currently inside a
        # <｜tool▁outputs▁begin｜>...<｜tool▁outputs▁end｜> block.
        is_tool = False
        for idx, (msg_index, msg) in enumerate(messages):
            if msg.role == "system":
                continue  # already emitted in prologue
            if msg.role == "user":
                is_tool = False
                segs.append(Segment(
                    text=f"{_USER}{_block_text(msg.blocks)}",
                    token_ids=None,
                    kind=SegKind.CONTENT,
                    role="user",
                    is_generation=False,
                    token_class=TokenClass.PROTOCOL,
                    msg_index=msg_index,
                ))
                continue
            if msg.role == "assistant":
                next_role = (
                    messages[idx + 1][1].role if idx + 1 < len(messages) else None
                )
                segs.extend(_assistant_segments(
                    msg, msg_index,
                    is_tool=is_tool,
                    next_is_tool=(next_role == "tool"),
                ))
                is_tool = False
                continue
            if msg.role == "tool":
                segs.append(_tool_segment(msg, msg_index, opens_block=not is_tool))
                is_tool = True
                continue
            raise NotImplementedError(
                f"DeepSeekV3Encoder.encode_segments: unsupported role "
                f"{msg.role!r} at msg_index={msg_index}"
            )

        # Conversation ended while inside a tool-output block: append the
        # closing tag (no message owns it, so it stays masked).
        if is_tool:
            segs.append(Segment(
                text=_TOOL_OUTPUTS_END,
                token_ids=None,
                kind=SegKind.MARKER,
                role="tool",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=None,
            ))
        return segs


def _block_text(blocks) -> str:
    """Concatenate every TextBlock's text; ignore others."""
    return _blocks.block_text(blocks)


def _extract_tool_calls(blocks) -> list[ToolUseBlock]:
    return _blocks.extract_tool_calls(blocks)


def _format_tool_call(tc: ToolUseBlock) -> str:
    """One ``<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>`` body.

    Renders ``function<｜tool▁sep｜>{name}\\n```json\\n{args}\\n````` between
    the begin/end markers. The DSv3 jinja always uses ``type='function'``;
    we hardcode ``function`` to match.
    """
    args_text = json.dumps(tc.input, ensure_ascii=False)
    return (
        f"{_TOOL_CALL_BEGIN}function{_TOOL_SEP}{tc.name}\n"
        f"```json\n{args_text}\n```{_TOOL_CALL_END}"
    )


def _assistant_segments(
    msg: NormalizedMessage,
    msg_index: int | None,
    *,
    is_tool: bool,
    next_is_tool: bool,
) -> list[Segment]:
    """Emit MARKER + body + closing tags for one assistant turn."""
    text_body = _block_text(msg.blocks)
    tool_calls = _extract_tool_calls(msg.blocks)
    msg_weight = float(msg.weight)

    if not text_body and tool_calls:
        segs = _assistant_tool_call_segments(msg_index, tool_calls, msg_weight=msg_weight)
    else:
        marker_text = _TOOL_OUTPUTS_END if is_tool else _ASSISTANT
        # F1: split the closing ``<｜end▁of▁sentence｜>`` (model's OWN
        # autonomous stop signal) into a dedicated STOP_MARKER segment
        # instead of folding it into CONTENT — uniform stop-token shape
        # across families. _EOS is a special token so the seam is BPE-stable.
        segs = [
            Segment(
                text=marker_text,
                token_ids=None,
                kind=SegKind.MARKER,
                role="assistant",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=msg_index,
            ),
        ]
        if text_body:
            segs.append(Segment(
                text=text_body,
                token_ids=None,
                kind=SegKind.CONTENT,
                role="assistant",
                scale_factor=msg_weight,
                is_generation=False,
                token_class=TokenClass.DECODED,
                msg_index=msg_index,
            ))
        segs.append(Segment(
            text=_EOS,
            token_ids=None,
            kind=SegKind.STOP_MARKER,
            role="assistant",
            scale_factor=msg_weight,
            is_generation=False,
            token_class=TokenClass.PROTOCOL,
            msg_index=msg_index,
        ))
    if next_is_tool:
        # <｜tool▁outputs▁begin｜> is a framework-injected structural
        # delimiter that introduces the tool's output — the model emits
        # only its tool-call body, never this frame. Class-A PROTOCOL,
        # loss_scale=0.0 (audit #3, 2026-06-06). This is a DELIBERATE
        # divergence from the jinja oracle's crude "whole assistant span =
        # loss" mask, which wrongly trains this delimiter.
        segs.append(Segment(
            text=_TOOL_OUTPUTS_BEGIN,
            token_ids=None,
            kind=SegKind.MARKER,
            role="assistant",
            is_generation=False,
            token_class=TokenClass.PROTOCOL,
            msg_index=msg_index,
        ))
    return segs


def _assistant_tool_call_segments(
    msg_index: int | None,
    tool_calls: list[ToolUseBlock],
    *,
    msg_weight: float,
) -> list[Segment]:
    """Emit the tool_call branch for one assistant turn.

    Layout mirrors jinja byte-for-byte including the per-iteration
    closing-tag bug:

    First call: ``<｜Assistant｜><｜tool▁calls▁begin｜>{call_body_0}``
    Each next call: ``\\n{call_body_k}<｜tool▁calls▁end｜><｜end▁of▁sentence｜>``

    Total closing tags = max(0, len(tool_calls) - 1).
    """
    out: list[Segment] = [
        Segment(
            text=_ASSISTANT,
            token_ids=None,
            kind=SegKind.MARKER,
            role="assistant",
            is_generation=False,
            token_class=TokenClass.PROTOCOL,
            msg_index=msg_index,
        ),
        Segment(
            text=_TOOL_CALLS_BEGIN + _format_tool_call(tool_calls[0]),
            token_ids=None,
            kind=SegKind.TOOL_CALL,
            role="assistant",
            scale_factor=msg_weight,
            is_generation=False,
            token_class=TokenClass.DECODED,
            msg_index=msg_index,
        ),
    ]
    for k in range(1, len(tool_calls)):
        out.append(Segment(
            text="\n" + _format_tool_call(tool_calls[k]) + _TOOL_CALLS_END + _EOS,
            token_ids=None,
            kind=SegKind.TOOL_CALL,
            role="assistant",
            scale_factor=msg_weight,
            is_generation=False,
            token_class=TokenClass.DECODED,
            msg_index=msg_index,
        ))
    return out


def _tool_segment(
    msg: NormalizedMessage,
    msg_index: int | None,
    *,
    opens_block: bool,
) -> Segment:
    """One tool message -> one segment, always loss_scale=0 (PROTOCOL)."""
    content_parts: list[str] = []
    for b in msg.blocks:
        if isinstance(b, ToolResultBlock):
            content_parts.append(b.text)
        elif isinstance(b, TextBlock):
            content_parts.append(b.text)
    content = "".join(content_parts)
    prefix = "" if opens_block else "\n"
    text = f"{prefix}{_TOOL_OUTPUT_BEGIN}{content}{_TOOL_OUTPUT_END}"
    return Segment(
        text=text,
        token_ids=None,
        kind=SegKind.TOOL_RESULT,
        role="tool",
        is_generation=False,
        token_class=TokenClass.PROTOCOL,
        msg_index=msg_index,
    )


__all__ = ["DeepSeekV3Encoder"]
