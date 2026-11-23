"""Qwen3 / Qwen3.5 family SFT encoder (segment path, blocks-only).

Scope:

* user + assistant **plain text** turns
* optional ``system`` (top-level on ``NormalizedSample``)
* assistant **thinking** blocks (preserved on BOTH last AND historical
  assistant turns — Quirk B)
* Quirk A: last assistant without ``<think>`` block gets injected
  empty ``<think>\\n\\n</think>\\n\\n`` envelope (masked)
* optional ``max_length`` right-truncation
* optional per-token ``loss_weights`` via ``loss_config.type_overrides
  .thinking``
* tools_declaration + tool_use + tool_result (Hermes <tool_call> /
  <tool_response> protocol)
* thinking_intent.mode=off → empty-think envelope rendered on the LAST
  post-query assistant turn only (mirrors jinja add_generation_prompt;
  audit #5 corrected an earlier over-injection onto every post-query turn)

Phase 3.4 (encoder-layer-refactor): walks ``nm.blocks`` directly. The
fixture-09 bug class (3-vs-2 \\n between </think> and <tool_call>) is
structurally impossible:

* Per-block emitters produce complete logical units (thinking is one
  segment with its full ``<think>\\n...\\n</think>\\n\\n`` envelope).
* Tool calls are separate ``SegKind.TOOL_CALL`` segments; the merge
  pass in Assembler keeps them separate (different kind), and the
  natural concatenation produces ``...</think>\\n\\n<tool_call>...``
  byte-identically to jinja — no encoder-planted linker required.
"""

from __future__ import annotations

import json

from titaninfra.data.schema import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from titaninfra.data.sft.capabilities import (
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
    FEATURE_PARALLEL_TOOL_CALLS,
    FEATURE_SYSTEM,
    FEATURE_THINKING_CURRENT_TURN,
    FEATURE_THINKING_HISTORY,
    FEATURE_THINKING_INTENT_MODE,
    FEATURE_TOOL_CALL,
    FEATURE_TOOL_RESULT,
    FEATURE_TOOLS_DECLARATION,
    FEATURE_TYPE_OVERRIDES,
)
from titaninfra.data.sft.encoders import _blocks, register
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass
from titaninfra.data.sft.types import EncodedSample, NormalizedMessage, NormalizedSample

SUPPORTED_BLOCKS = frozenset({TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock})
SUPPORTED_FEATURES = frozenset({
    FEATURE_SYSTEM,
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
    FEATURE_TOOLS_DECLARATION,
    FEATURE_TOOL_CALL,
    FEATURE_TOOL_RESULT,
    FEATURE_PARALLEL_TOOL_CALLS,
    FEATURE_THINKING_CURRENT_TURN,
    FEATURE_THINKING_HISTORY,
    FEATURE_THINKING_INTENT_MODE,
    FEATURE_TYPE_OVERRIDES,
})

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_TOOLS_HEADER_PREFIX = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
_TOOLS_HEADER_SUFFIX = (
    "\n</tools>\n\nFor each function call, return a json object with function name "
    "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


@register("qwen3")
class Qwen3Encoder:
    """Token-first encoder for the Qwen3 / Qwen3.5 family."""

    name = "qwen3"

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(
        self,
        sample: NormalizedSample,
        *,
        max_length: int | None = None,
        loss_config: dict | None = None,
        truncate_at_last_segment: bool = False,
        spatial_merge_size: int | None = None,
        decoded_images: list | None = None,
    ) -> EncodedSample:
        from titaninfra.data.sft.assembler import assemble

        type_overrides = None
        if isinstance(loss_config, dict):
            type_overrides = loss_config.get("type_overrides")
        segments = self.encode_segments(sample, type_overrides=type_overrides)
        return assemble(
            segments, self.tokenizer, mode="sft", max_length=max_length,
            truncate_at_last_segment=truncate_at_last_segment,
            spatial_merge_size=spatial_merge_size,
            decoded_images=decoded_images,
        )

    def encode_segments(
        self,
        sample: NormalizedSample,
        *,
        type_overrides: dict[str, float] | None = None,
    ) -> list[Segment]:
        thinking_scale = (type_overrides or {}).get("thinking", 1.0)
        segs: list[Segment] = []

        # Build effective message list with optional synthesized system at idx None.
        messages: list[tuple[int | None, NormalizedMessage]] = []
        first = sample.messages[0] if sample.messages else None
        first_is_system = first is not None and first.role == "system"
        if sample.system and not first_is_system:
            messages.append((None, NormalizedMessage(
                role="system",
                blocks=(TextBlock(text=sample.system),),
                weight=0.0,
                turn_role="current",
            )))
        for i, m in enumerate(sample.messages):
            messages.append((i, m))

        last_query_index = _last_query_index_blocks(messages)
        n = len(messages)

        ti = sample.thinking_intent
        mode_off = ti is not None and ti.mode == "off"

        # Prelude: tools-header (with optional system text), or first-system
        # block, or nothing. Loss is always 0 (PROTOCOL).
        eff_first_msg = messages[0][1] if messages else None
        eff_first_is_system = eff_first_msg is not None and eff_first_msg.role == "system"
        eff_first_msg_index = messages[0][0] if eff_first_is_system else None
        if sample.tools:
            system_text = _block_text(eff_first_msg.blocks) if eff_first_is_system else None
            segs.append(Segment(
                text=_render_tools_header(sample.tools, system_text),
                token_ids=None,
                kind=SegKind.TOOLS_HEADER,
                role="system",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=eff_first_msg_index,
            ))
        elif eff_first_is_system:
            segs.append(Segment(
                text=f"{_IM_START}system\n{_block_text(eff_first_msg.blocks)}{_IM_END}\n",
                token_ids=None,
                kind=SegKind.SYSTEM,
                role="system",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=eff_first_msg_index,
            ))

        for idx, (msg_index, msg) in enumerate(messages):
            if idx == 0 and msg.role == "system":
                continue  # already in prelude
            if msg.role == "system":
                segs.append(Segment(
                    text=f"{_IM_START}system\n{_block_text(msg.blocks)}{_IM_END}\n",
                    token_ids=None,
                    kind=SegKind.SYSTEM,
                    role="system",
                    is_generation=False,
                    token_class=TokenClass.PROTOCOL,
                    msg_index=msg_index,
                ))
                continue
            if msg.role == "user":
                segs.extend(self._user_segments(msg, msg_index))
                continue
            if msg.role == "assistant":
                segs.extend(_assistant_segments(
                    msg, msg_index, idx, n, last_query_index,
                    thinking_scale=thinking_scale,
                    mode_off=mode_off,
                ))
                continue
            if msg.role == "tool":
                segs.extend(_tool_segments(messages, idx, msg_index=msg_index))
                continue
            raise NotImplementedError(
                f"Qwen3Encoder.encode_segments: unsupported role {msg.role!r} at msg_index={msg_index}"
            )
        return segs

    def _user_segments(self, msg: NormalizedMessage, msg_index: int | None) -> list[Segment]:
        """Text-only user turn → single CONTENT segment (byte-identical to pre-VLM qwen3).

        qwen3_5 overrides this to interleave IMAGE segments before user text.
        qwen3 itself has no vision support.
        """
        text = _block_text(msg.blocks)
        return [Segment(
            text=f"{_IM_START}user\n{text}{_IM_END}\n", token_ids=None,
            kind=SegKind.CONTENT, role="user", is_generation=False,
            token_class=TokenClass.PROTOCOL, msg_index=msg_index,
        )]


def _block_text(blocks) -> str:
    """Concatenate every TextBlock's text; ignore others."""
    return _blocks.block_text(blocks)


def _extract_thinking(blocks) -> str | None:
    return _blocks.extract_thinking(blocks)


def _extract_tool_calls(blocks) -> list[ToolUseBlock]:
    return _blocks.extract_tool_calls(blocks)


def _last_query_index_blocks(
    messages: list[tuple[int | None, NormalizedMessage]],
) -> int:
    """Mirror Qwen3 Jinja ``ns.last_query_index`` for the blocks-only path."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i][1]
        if m.role != "user":
            continue
        content = _block_text(m.blocks)
        if (
            content.startswith("<tool_response>")
            and content.endswith("</tool_response>")
        ):
            continue
        return i
    return -1


def _render_tools_header(tools, system_content: str | None) -> str:
    """Build the tools + (optional) system header as one <|im_start|>system block.

    Transformers ``tojson`` filter defaults: ``ensure_ascii=False,
    sort_keys=False, indent=None``. Mirror exactly.
    """
    out = f"{_IM_START}system\n"
    if system_content is not None:
        out += system_content + "\n\n"
    out += _TOOLS_HEADER_PREFIX
    for tool in tools:
        if hasattr(tool, "__struct_fields__"):  # msgspec.Struct
            tool_dict = {f: getattr(tool, f) for f in tool.__struct_fields__}
        else:
            tool_dict = dict(tool)
        out += "\n" + json.dumps(tool_dict, ensure_ascii=False)
    out += _TOOLS_HEADER_SUFFIX
    out += f"{_IM_END}\n"
    return out


def _format_tool_call(tc: ToolUseBlock) -> str:
    """Render one ToolUseBlock as ``<tool_call>\\n{...}\\n</tool_call>``."""
    args_text = json.dumps(tc.input, ensure_ascii=False)
    return f'<tool_call>\n{{"name": "{tc.name}", "arguments": {args_text}}}\n</tool_call>'


def _assistant_segments(
    msg: NormalizedMessage,
    msg_index: int | None,
    idx: int,
    n: int,
    last_query_index: int,
    *,
    thinking_scale: float,
    mode_off: bool,
) -> list[Segment]:
    """Emit MARKER + (THINKING)? + CONTENT + (TOOL_CALL)* + MARKER for one assistant turn.

    Phase 4 merge pass collapses adjacent same-metadata text segments,
    so we emit complete logical units. fixture-09 linker-\\n drift is
    structurally impossible: THINKING/CONTENT/TOOL_CALL have different
    ``kind`` and stay separate; concatenation produces the natural
    ``...</think>\\n\\n<tool_call>...`` form.
    """
    msg_weight = float(msg.weight)
    text_body = _block_text(msg.blocks)
    thinking = _extract_thinking(msg.blocks)
    tool_calls = _extract_tool_calls(msg.blocks)
    after_query = idx > last_query_index
    is_last = idx == (n - 1)
    has_think = thinking is not None
    # quirk_a: inject the empty ``<think>\n\n</think>`` envelope. Gated on
    # is_last ONLY (audit #5, 2026-06-06). The last post-query non-thinking
    # assistant turn mirrors jinja's add_generation_prompt envelope — this is
    # the mixed-thinking recipe (mode=None) AND the mode='off' projection
    # (both render the same last-turn envelope). It must NOT fire on a
    # non-last consecutive post-query assistant turn (agentic user->asst->asst):
    # jinja injects the envelope only on the FINAL turn, so generalizing via
    # mode_off added a phantom envelope + trained ``\n\n`` jinja never emits.
    quirk_a = after_query and not has_think and is_last

    out: list[Segment] = []

    # Two layouts for the opening role marker, governed by whether this
    # turn falls inside jinja's L45 ``content.lstrip('\n')`` branch:
    #
    #   marker_lstrips := has_think or quirk_a
    #     ≡ after_query and (is_last or has_think or mode_off)
    #     ≡ jinja L45 trigger + titaninfra's mode_off extension
    #
    # When ``marker_lstrips`` holds (current branch matches jinja L45):
    #   * MARKER = ``<|im_start|>assistant\n``
    #   * Body's leading ``\n`` is stripped (mirrors jinja
    #     ``content.lstrip('\n')``).
    #
    # When it does NOT hold (jinja L47 or L50 — historical turn or
    # post-query non-last non-thinking turn): jinja emits the body as-is
    # without lstripping. We must preserve the leading ``\n``. Naive
    # MARKER ``...assistant\n`` + CONTENT ``\nSubject...`` produces TWO
    # ``\n`` tokens (id 198) but jinja's joint render BPE-merges into
    # one ``\n\n`` token (id 271) — 1 phantom token of drift.
    #
    # Conditional Option C: when body starts with ``\n``, relocate
    # MARKER's trailing ``\n`` into CONTENT head so the seam BPE-merges.
    # Wire string stays identical to jinja; the merged ``\n\n`` lives
    # in CONTENT and earns ``loss_scale=msg_weight``, matching HF
    # Trainer tokenize-then-mask convention. Mirrors the qwen2_5 fix
    # in titaninfra/data/sft/encoders/qwen2_5.py.
    marker_lstrips = has_think or quirk_a
    body_leading_nl = bool(text_body) and text_body.startswith("\n")
    relocate_nl_to_content = (not marker_lstrips) and body_leading_nl

    marker_text = (
        f"{_IM_START}assistant"
        if relocate_nl_to_content
        else f"{_IM_START}assistant\n"
    )
    out.append(Segment(
        text=marker_text,
        token_ids=None,
        kind=SegKind.MARKER,
        role="assistant",
        is_generation=after_query,
        token_class=TokenClass.PROTOCOL,
        msg_index=msg_index,
    ))

    if has_think:
        inner_scale = msg_weight * thinking_scale
        if thinking_scale == 1.0:
            out.append(Segment(
                text=f"{_THINK_OPEN}\n{thinking}\n{_THINK_CLOSE}\n\n",
                token_ids=None,
                kind=SegKind.THINKING,
                role="assistant",
                scale_factor=msg_weight,
                is_generation=after_query,
                token_class=TokenClass.DECODED,
                msg_index=msg_index,
            ))
        else:
            # Split into open marker + scaled inner + close marker; merge
            # pass keeps them separate (different loss_scale).
            #
            # Oracle scope: tokens AFTER the `<think>` open marker through
            # tokens BEFORE the `</think>` close marker get the inner
            # scale. So the leading `\n` inside `<think>\n...` and the
            # trailing `\n` before `\n</think>` BOTH belong to the inner
            # range. The `</think>\n\n` close stays at msg_weight.
            out.append(Segment(
                text=_THINK_OPEN,
                token_ids=None,
                kind=SegKind.THINKING,
                role="assistant",
                scale_factor=msg_weight,
                is_generation=after_query,
                token_class=TokenClass.DECODED,
                msg_index=msg_index,
            ))
            out.append(Segment(
                text=f"\n{thinking}\n",
                token_ids=None,
                kind=SegKind.THINKING,
                role="assistant",
                scale_factor=inner_scale,
                is_generation=after_query,
                token_class=TokenClass.DECODED,
                msg_index=msg_index,
            ))
            out.append(Segment(
                text=f"{_THINK_CLOSE}\n\n",
                token_ids=None,
                kind=SegKind.THINKING,
                role="assistant",
                scale_factor=msg_weight,
                is_generation=after_query,
                token_class=TokenClass.DECODED,
                msg_index=msg_index,
            ))
    elif quirk_a:
        # Injected empty envelope: masked (loss=0), plus trailing \n\n at
        # msg_weight (legacy _mask_empty_thinking semantics).
        #
        # token_class=PROTOCOL (class A), NOT METADATA: the inference
        # framework injects this exact envelope. Verified empirically:
        # apply_chat_template(add_generation_prompt=True, enable_thinking=
        # False) emits ``...assistant\n<think>\n\n</think>\n\n``. So vLLM/
        # SGLang produce these same tokens at this same position at
        # inference → it is framework-injected protocol, not dynamic
        # metadata. Mislabeling it METADATA would make a future RL prompt
        # builder (which strips METADATA) wrongly delete an envelope that
        # inference actually injects → reverse train/inference drift.
        # CLAUDE.md "Token Three-Way Classification" lists <think>/</think>
        # markers as class A. Empty-think is a deliberate mixed-thinking
        # training recipe (kept, masked) — see
        # memory feedback_jinja_is_recipe_projection_not_protocol.
        out.append(Segment(
            text=f"{_THINK_OPEN}\n\n{_THINK_CLOSE}",
            token_ids=None,
            kind=SegKind.THINKING,
            role="assistant",
            scale_factor=0.0,
            is_generation=after_query,
            token_class=TokenClass.PROTOCOL,
            msg_index=msg_index,
        ))
        out.append(Segment(
            text="\n\n",
            token_ids=None,
            kind=SegKind.CONTENT,
            role="assistant",
            scale_factor=msg_weight,
            is_generation=after_query,
            token_class=TokenClass.DECODED,
            msg_index=msg_index,
        ))

    if text_body:
        # Mirror jinja for the assistant text body:
        #
        # * ``marker_lstrips`` branch (jinja L45): ``content.lstrip('\n')``
        #   — assistant body cannot start with ``\n``. Two distinct seams
        #   break otherwise:
        #     1. Standard path: MARKER ``...assistant\n`` directly followed
        #        by CONTENT ``\nSubject...``. Different kind → merge pass
        #        keeps them split → joint-tokenize merges ``\n\n`` more
        #        tightly than two-piece tokenize (1-token BPE drift).
        #     2. Quirk-A path: injected ``\n\n`` CONTENT segment then
        #        merges with this body (same kind+loss). Pre-strip ⇒
        #        jinja-aligned ``\n\n`` + body; without strip ⇒ ``\n\n\n``
        #        + body (string mismatch, not just BPE drift).
        #
        # * Non-lstrip branch (jinja L47 / L50 — historical or post-query
        #   non-last non-thinking): preserve the leading ``\n``. When
        #   ``relocate_nl_to_content`` already moved MARKER's trailing
        #   ``\n`` into CONTENT head (above), prepend that ``\n`` here so
        #   the wire becomes ``...assistant`` + ``\n`` + body — joint-
        #   tokenize merges ``\n\n`` into one token; two-piece path
        #   merges within CONTENT (same kind, same loss) and matches.
        if marker_lstrips:
            text_body = text_body.lstrip("\n")
        elif relocate_nl_to_content:
            text_body = "\n" + text_body
    if text_body:
        # When tool_calls follow, attach the linker \n to the END of the
        # body segment so the BPE boundary lands on `\n<tool_call>` (the
        # `.\n` merge produces a single token, matching jinja's
        # ``{content}\n<tool_call>`` form).
        body_text_eff = text_body + "\n" if tool_calls else text_body
        out.append(Segment(
            text=body_text_eff,
            token_ids=None,
            kind=SegKind.CONTENT,
            role="assistant",
            scale_factor=msg_weight,
            is_generation=after_query,
            token_class=TokenClass.DECODED,
            msg_index=msg_index,
        ))

    # Tool calls. Body's trailing \n (above) handles the body→first-call
    # seam. Between calls: \n.
    for j, tc in enumerate(tool_calls):
        prefix = "\n" if j > 0 else ""
        out.append(Segment(
            text=prefix + _format_tool_call(tc),
            token_ids=None,
            kind=SegKind.TOOL_CALL,
            role="assistant",
            scale_factor=msg_weight,
            is_generation=after_query,
            token_class=TokenClass.DECODED,
            msg_index=msg_index,
        ))

    # Closing marker — assistant STOP_MARKER exception applies:
    # PROTOCOL + loss_scale=msg_weight is legal because <|im_end|> is the
    # model's own autonomous turn terminator (see Segment.__post_init__).
    out.append(Segment(
        text=f"{_IM_END}\n",
        token_ids=None,
        kind=SegKind.STOP_MARKER,
        role="assistant",
        scale_factor=msg_weight,
        is_generation=after_query,
        token_class=TokenClass.PROTOCOL,
        msg_index=msg_index,
    ))
    return out


def _tool_segments(
    messages: list[tuple[int | None, NormalizedMessage]],
    idx: int,
    *,
    msg_index: int | None,
) -> list[Segment]:
    """One tool message → fragment of the merged ``<|im_start|>user`` chain.

    Open marker only when prev is not tool; close marker only when next
    is not tool. Tool segments never carry loss (PROTOCOL).
    """
    msg = messages[idx][1]
    prev_role = messages[idx - 1][1].role if idx > 0 else None
    next_role = messages[idx + 1][1].role if idx + 1 < len(messages) else None

    content_parts: list[str] = []
    for b in msg.blocks:
        if isinstance(b, ToolResultBlock):
            content_parts.append(b.text)
        elif isinstance(b, TextBlock):
            content_parts.append(b.text)
    content = "".join(content_parts)

    text = ""
    if prev_role != "tool":
        text += f"{_IM_START}user"
    text += f"\n<tool_response>\n{content}\n</tool_response>"
    if next_role != "tool":
        text += f"{_IM_END}\n"

    return [Segment(
        text=text,
        token_ids=None,
        kind=SegKind.TOOL_RESULT,
        role="tool",
        is_generation=False,
        token_class=TokenClass.PROTOCOL,
        msg_index=msg_index,
    )]


__all__ = ["Qwen3Encoder"]
