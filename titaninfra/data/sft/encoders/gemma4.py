"""Gemma4 family SFT encoder — native Segment emission (7-family unification).

Emits ``list[Segment]`` like the other 6 families; loss is derived by
``Segment.__post_init__`` via ``base_mask(kind, role) × scale_factor``
(Plan 1). The old jinja-render + offset-masking path is gone — gemma4 no
longer round-trips through a flat string to recover loss boundaries, so
the F3 content-collision bug (user ``<|turn>model`` bytes shifting the
mask) is structurally impossible.

Scope:

* user / assistant / system / tool roles
* assistant thinking blocks (preserved on current AND historical turns)
* tool_use blocks (native wire, no jinja)
* continue_same_model_turn quirk (post-tool assistant text lives inside the
  previous model turn — a structural emit decision, not a char-scan)
* thinking_intent.mode=on -> ``<|think|>`` injection (class A / PROTOCOL)
* type_overrides["thinking"] down-weight via THINKING segment scale_factor
* RL-ready: is_generation marks the last assistant turn

The wire formatters (``_format_*``, ``_strip_thinking``) are unchanged from
the prior implementation — they were already byte-exact vs the gemma4 chat
template; only the loss-mask mechanism changed from offset-measurement to
structural Segment declaration.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence

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
from titaninfra.data.sft.encoders import register
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass
from titaninfra.data.sft.types import EncodedSample, NormalizedSample

logger = logging.getLogger(__name__)

# Capability declaration — see titaninfra/data/sft/capabilities.py.
# Gemma4 wire: <|turn>role tokens, continue_same_model_turn quirk for
# assistant text following tool_response, thinking blocks preserved on
# both current and history turns. thinking_intent.mode declared supported
# here (mode=on injects <|think|> markers; mode=off keeps current default).
# effort is NOT declared — Gemma4 wire has no reasoning_effort dial, so
# a sample with effort set produces one warn-loud line and the field is
# ignored.
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


# ---------------------------------------------------------------------------
# Wire-format helpers — byte-exact ports of the gemma4 chat_template.jinja
# macros. UNCHANGED from the prior implementation; only the loss-mask
# mechanism (offset-measurement -> Segment emission) was rewritten. The
# Jinja file is kept only as a test oracle (tests/oracle/_jinja_oracle/).
#
# Jinja semantics watch-outs ported faithfully:
#   * ``is boolean`` — must check ``type(arg) is bool`` BEFORE ``int``
#     because ``isinstance(True, int)`` is True in Python.
#   * ``| dictsort`` — sort by key (Jinja default ordering).
#   * ``| trim`` — Python ``str.strip()`` matches Jinja whitespace set.
# ---------------------------------------------------------------------------
def _format_argument(arg, escape_keys: bool = True) -> str:
    """Mirror of the jinja ``format_argument(arg, escape_keys=True)`` macro."""
    if isinstance(arg, str):
        return f'<|"|>{arg}<|"|>'
    if type(arg) is bool:
        return "true" if arg else "false"
    if isinstance(arg, Mapping):
        items = sorted(arg.items(), key=lambda kv: kv[0])
        parts = []
        for k, v in items:
            key_repr = f'<|"|>{k}<|"|>' if escape_keys else str(k)
            parts.append(f"{key_repr}:{_format_argument(v, escape_keys=escape_keys)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
        return "[" + ",".join(_format_argument(item, escape_keys=escape_keys) for item in arg) + "]"
    return str(arg)


def _format_parameters(properties: Mapping, required, filter_keys: bool = False) -> str:
    """Mirror of the jinja ``format_parameters`` macro.

    Recursive on object items. The trailing ``,type:<|"|>...<|"|>}`` is
    ALWAYS emitted (the macro hardcodes ``add_comma`` to True before it).
    """
    standard_keys = {"description", "type", "properties", "required", "nullable"}
    out_parts: list[str] = []
    for key, value in sorted(properties.items(), key=lambda kv: kv[0]):
        if filter_keys and key in standard_keys:
            continue
        segments: list[str] = []
        if value.get("description"):
            segments.append(f'description:<|"|>{value["description"]}<|"|>')
        vtype = str(value.get("type") or "").upper()
        if vtype == "STRING":
            if value.get("enum"):
                segments.append(f"enum:{_format_argument(value['enum'])}")
        elif vtype == "ARRAY":
            items_val = value.get("items")
            if isinstance(items_val, Mapping) and items_val:
                inner: list[str] = []
                for item_key, item_value in sorted(items_val.items(), key=lambda kv: kv[0]):
                    if item_value is None:
                        continue
                    if item_key == "properties":
                        nested = ""
                        if isinstance(item_value, Mapping):
                            nested = _format_parameters(
                                item_value, items_val.get("required") or [],
                            )
                        inner.append(f"properties:{{{nested}}}")
                    elif item_key == "required":
                        req_repr = ",".join(f'<|"|>{r}<|"|>' for r in item_value)
                        inner.append(f"required:[{req_repr}]")
                    elif item_key == "type":
                        if isinstance(item_value, str):
                            inner.append(f"type:{_format_argument(item_value.upper())}")
                        else:
                            inner.append(f"type:{_format_argument([str(x).upper() for x in item_value])}")
                    else:
                        inner.append(f"{item_key}:{_format_argument(item_value)}")
                segments.append("items:{" + ",".join(inner) + "}")
        if value.get("nullable"):
            segments.append("nullable:true")
        if vtype == "OBJECT":
            if "properties" in value and isinstance(value["properties"], Mapping):
                nested = _format_parameters(value["properties"], value.get("required") or [])
                segments.append("properties:{" + nested + "}")
            elif isinstance(value, Mapping):
                nested = _format_parameters(
                    value, value.get("required") or [], filter_keys=True,
                )
                segments.append("properties:{" + nested + "}")
            if value.get("required"):
                req_repr = ",".join(f'<|"|>{r}<|"|>' for r in value["required"])
                segments.append(f"required:[{req_repr}]")
        segments.append(f'type:<|"|>{vtype}<|"|>')
        out_parts.append(f"{key}:{{{','.join(segments)}}}")
    return ",".join(out_parts)


def _format_function_declaration(tool_data: Mapping) -> str:
    """Mirror of the jinja ``format_function_declaration`` macro.

    Supports both titaninfra flat Tool schema {name, description, input_schema}
    and OpenAI nested schema {function: {name, description, parameters}}.
    """
    if "function" in tool_data:
        # OpenAI / HF nested format
        fn = tool_data["function"]
        params_key = "parameters"
    else:
        # titaninfra IR flat format (Tool schema in schema.py)
        fn = tool_data
        params_key = "input_schema"
    name = fn.get("name", "")
    description = fn.get("description", "")
    out = f'declaration:{name}{{description:<|"|>{description}<|"|>'
    params = fn.get(params_key)
    if params:
        out += ",parameters:{"
        if params.get("properties"):
            out += "properties:{" + _format_parameters(
                params["properties"], params.get("required") or [],
            ) + "},"
        if params.get("required"):
            req_repr = ",".join(f'<|"|>{r}<|"|>' for r in params["required"])
            out += f"required:[{req_repr}],"
        ptype = str(params.get("type") or "").upper()
        if ptype:
            out += f'type:<|"|>{ptype}<|"|>}}'
    if "response" in fn:
        resp = fn["response"]
        out += ",response:{"
        if resp.get("description"):
            out += f'description:<|"|>{resp["description"]}<|"|>,'
        rtype = str(resp.get("type") or "").upper()
        if rtype == "OBJECT":
            out += f'type:<|"|>{rtype}<|"|>}}'
    out += "}"
    return out


def _format_tool_response_block(tool_name: str, response) -> str:
    """Mirror of the jinja ``format_tool_response_block`` macro."""
    out = "<|tool_response>"
    if isinstance(response, Mapping):
        out += f"response:{tool_name}{{"
        items = sorted(response.items(), key=lambda kv: kv[0])
        parts = [
            f"{k}:{_format_argument(v, escape_keys=False)}" for k, v in items
        ]
        out += ",".join(parts) + "}"
    else:
        out += (
            f"response:{tool_name}{{value:"
            f"{_format_argument(response, escape_keys=False)}}}"
        )
    out += "<tool_response|>"
    return out


def _strip_thinking(text: str) -> str:
    """Mirror of the jinja ``strip_thinking`` macro.

    For each part separated by ``<channel|>``: if ``<|channel>`` is in the
    part, keep only what is BEFORE the first ``<|channel>``. Else keep
    the whole part. Concatenate, strip whitespace.

    Plain ``<think>...</think>`` envelopes (used by titaninfra's normalize
    layer) contain neither marker and pass through verbatim modulo the
    final strip.
    """
    result = ""
    for part in text.split("<channel|>"):
        if "<|channel>" in part:
            result += part.split("<|channel>")[0]
        else:
            result += part
    return result.strip()


# ---------------------------------------------------------------------------
# NormalizedSample -> HF-shape message dicts (UNCHANGED — Phase 3.6 blocks path)
# ---------------------------------------------------------------------------
def _build_hf_messages(sample: NormalizedSample) -> list[dict]:
    """NormalizedSample → list[{role, content, [tool_calls]}] with synth system.

    Walks ``nm.blocks`` directly (blocks-only). Gemma4's tool_calls are
    emitted in OpenAI-wrapper shape ``{"id", "type": "function",
    "function": {"name", "arguments"}}`` (the emit helpers read this shape).
    Tools (top-level) are already in OpenAI form per fixtures.
    """
    out: list[dict] = []
    if sample.system and (not sample.messages or sample.messages[0].role != "system"):
        out.append({"role": "system", "content": sample.system})

    # Pre-scan: last user index. The channel=thought block only fires for
    # assistant messages AFTER this index AND carrying tool_calls.
    last_user_idx = -1
    nm_messages = sample.messages
    for idx, nm in enumerate(nm_messages):
        if nm.role == "user":
            last_user_idx = idx

    for idx, nm in enumerate(nm_messages):
        text_body = "".join(b.text for b in nm.blocks if isinstance(b, TextBlock))
        thinking_parts = [b.text for b in nm.blocks if isinstance(b, ThinkingBlock) and b.text]
        thinking = "\n".join(thinking_parts) if thinking_parts else None
        tool_use_blocks = [b for b in nm.blocks if isinstance(b, ToolUseBlock)]
        tool_result_blocks = [b for b in nm.blocks if isinstance(b, ToolResultBlock)]

        emits_channel_thought = (
            nm.role == "assistant"
            and thinking is not None
            and bool(tool_use_blocks)
            and idx > last_user_idx
        )

        if nm.role == "assistant" and thinking is not None and not emits_channel_thought:
            # Embed <think>...</think> in content so the non-thought branch
            # renders it inline (matches fixtures 03/04 where assistant turns
            # without tool_calls carry thinking inline).
            content = (
                f"<think>\n{thinking}\n</think>\n\n{text_body}"
                if text_body
                else f"<think>\n{thinking}\n</think>"
            )
        elif nm.role == "tool":
            tr_parts = [b.text for b in tool_result_blocks]
            if not tr_parts:
                tr_parts = [text_body] if text_body else []
            content = "".join(tr_parts)
        else:
            content = text_body

        msg: dict = {"role": nm.role, "content": content}
        if nm.role == "assistant" and tool_use_blocks:
            msg["tool_calls"] = [_to_openai_tool_call_from_block(b) for b in tool_use_blocks]
            if thinking is not None:
                msg["reasoning_content"] = thinking
        if nm.role == "tool":
            if tool_result_blocks and tool_result_blocks[0].tool_use_id:
                msg["tool_call_id"] = tool_result_blocks[0].tool_use_id
        out.append(msg)
    return out


def _to_openai_tool_call_from_block(blk) -> dict:
    """ToolUseBlock → OpenAI wrapper shape the gemma4 emit path expects."""
    args_str = json.dumps(blk.input, ensure_ascii=False)
    return {
        "id": blk.id,
        "type": "function",
        "function": {"name": blk.name, "arguments": args_str},
    }


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
@register("gemma4")
class Gemma4Encoder:
    """Native Segment encoder for the Gemma4 family."""

    name = "gemma4"

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(
        self,
        sample: NormalizedSample,
        *,
        max_length: int | None = None,
        loss_config: dict | None = None,
        add_generation_prompt: bool = False,
        truncate_at_last_segment: bool = False,
    ) -> EncodedSample:
        from titaninfra.data.sft.assembler import assemble

        type_overrides = None
        if isinstance(loss_config, dict):
            type_overrides = loss_config.get("type_overrides")
        segments = self.encode_segments(sample, type_overrides=type_overrides)

        if add_generation_prompt:
            # vLLM-aligned: append the static generation-prompt trailer (the
            # last assistant turn's opening marker) so RL prompt construction
            # matches inference. Deliberately omit gemma4's <|channel>thought
            # preface (class C template formatting the model never decodes).
            # Appended as a final METADATA marker segment: base_mask(MARKER,
            # assistant)=0 -> loss_scale=0, and METADATA requires loss_scale==0.
            from titaninfra.data.sft.gen_prompt import generation_prompt_for

            trailer = generation_prompt_for("gemma4")
            if trailer:
                segments = segments + [Segment(
                    text=trailer, token_ids=None, kind=SegKind.MARKER,
                    role="assistant", is_generation=False,
                    token_class=TokenClass.METADATA, msg_index=None,
                )]

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
        thinking_scale = (type_overrides or {}).get("thinking", 1.0)
        hf_messages = _build_hf_messages(sample)

        ti = sample.thinking_intent
        enable_thinking = ti is not None and ti.mode == "on"

        # _build_hf_messages may prepend a synthetic system dict (no NM
        # backing) -> weight 0.0 (PROTOCOL anyway). msg_offset aligns
        # hf_messages[msg_offset + i] with sample.messages[i].
        msg_offset = len(hf_messages) - len(sample.messages)

        bos = self.tokenizer.bos_token or ""
        last_asst_idx = _last_model_index(hf_messages)
        return _emit_segments(
            hf_messages, sample, bos=bos, msg_offset=msg_offset,
            thinking_scale=thinking_scale, enable_thinking=enable_thinking,
            last_asst_idx=last_asst_idx,
        )


# ---------------------------------------------------------------------------
# Segment emitter — port of the prior _render_gemma4 control flow, emitting
# one Segment per wire piece instead of appending strings to one list. The
# concatenation of every Segment.text equals the old flat render by
# construction (verified by the 9 oracle fixtures + the seam test).
# ---------------------------------------------------------------------------
def _last_model_index(hf_messages: list[dict]) -> int:
    """Index of the last assistant/model message (for is_generation)."""
    for i in range(len(hf_messages) - 1, -1, -1):
        if hf_messages[i].get("role") == "assistant":
            return i
    return -1


def _msg_weight(sample: NormalizedSample, hf_idx: int, msg_offset: int) -> float:
    """Weight of the NormalizedMessage backing hf_messages[hf_idx], else 0.0."""
    nm_idx = hf_idx - msg_offset
    if 0 <= nm_idx < len(sample.messages):
        return float(sample.messages[nm_idx].weight)
    return 0.0


def _emit_segments(
    hf_messages: list[dict],
    sample: NormalizedSample,
    *,
    bos: str,
    msg_offset: int,
    thinking_scale: float,
    enable_thinking: bool,
    last_asst_idx: int,
) -> list[Segment]:
    """Emit Segments in _render_gemma4 byte order."""
    segs: list[Segment] = []

    # --- System / tools header (one PROTOCOL segment, carries <bos>) ---
    has_system = bool(hf_messages) and hf_messages[0].get("role") in ("system", "developer")
    header_text = _render_system_header(
        hf_messages, sample, bos=bos, has_system=has_system,
        enable_thinking=enable_thinking,
    )
    if header_text is not None:
        segs.append(Segment(
            text=header_text, token_ids=None,
            kind=SegKind.TOOLS_HEADER if sample.tools else SegKind.SYSTEM,
            role="system", is_generation=False,
            token_class=TokenClass.PROTOCOL,
            msg_index=0 if has_system else None,
        ))
        bos_pending = False
    else:
        bos_pending = True  # bos rides the first emitted segment

    loop_messages = hf_messages[1:] if has_system else hf_messages
    loop_offset = 1 if has_system else 0  # index shift into hf_messages

    for li, message in enumerate(loop_messages):
        hf_idx = li + loop_offset
        role_raw = message.get("role", "user")
        if role_raw == "tool":
            # Consumed inside the preceding assistant turn (forward-scan).
            continue

        is_gen = hf_idx == last_asst_idx
        role = "model" if role_raw == "assistant" else role_raw

        # continue_same_model_turn: prev non-tool message was assistant.
        prev_nt_role = None
        for j in range(li - 1, -1, -1):
            if loop_messages[j].get("role") != "tool":
                prev_nt_role = loop_messages[j].get("role")
                break
        continue_same_model_turn = (role == "model" and prev_nt_role == "assistant")

        bos_prefix = bos if bos_pending else ""
        if role_raw == "user":
            body = _captured_user_text(message)
            segs.append(Segment(
                text=f"{bos_prefix}<|turn>user\n{body}<turn|>\n",
                token_ids=None, kind=SegKind.CONTENT, role="user",
                is_generation=False, token_class=TokenClass.PROTOCOL,
                msg_index=hf_idx,
            ))
            bos_pending = False
            continue

        if role_raw == "assistant":
            segs.extend(_emit_assistant_turn(
                message, loop_messages, li, hf_idx,
                bos_prefix=bos_prefix,
                continue_same_model_turn=continue_same_model_turn,
                is_gen=is_gen,
                msg_weight=_msg_weight(sample, hf_idx, msg_offset),
                thinking_scale=thinking_scale,
            ))
            bos_pending = False
            continue

        raise NotImplementedError(
            f"Gemma4Encoder.encode_segments: unsupported role {role_raw!r} "
            f"at hf index {hf_idx}"
        )

    return segs


def _render_system_header(
    hf_messages: list[dict],
    sample: NormalizedSample,
    *,
    bos: str,
    has_system: bool,
    enable_thinking: bool,
) -> str | None:
    """Build the ``<bos><|turn>system\\n...<turn|>\\n`` header, or None.

    Emitted iff system content OR tools OR enable_thinking.
    """
    if not (has_system or sample.tools or enable_thinking):
        return None
    out: list[str] = [bos, "<|turn>system\n"]
    if enable_thinking:
        out.append("<|think|>\n")
    if has_system:
        sys_content = hf_messages[0].get("content")
        if isinstance(sys_content, str):
            out.append(sys_content.strip())
        elif isinstance(sys_content, Sequence) and not isinstance(sys_content, (str, bytes)):
            for item in sys_content:
                if isinstance(item, Mapping):
                    out.append(item.get("text", "").strip() + " ")
    if sample.tools:
        for tool in sample.tools:
            out.append("<|tool>")
            out.append(_format_function_declaration(tool).strip())
            out.append("<tool|>")
    out.append("<turn|>\n")
    return "".join(out)


def _captured_user_text(message: dict) -> str:
    """User content captured the same way _render_gemma4 did (no strip_thinking)."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        captured = ""
        for item in content:
            if not isinstance(item, Mapping):
                continue
            itype = item.get("type")
            if itype == "text":
                captured += item.get("text", "").strip()
            elif itype == "image":
                captured += "<|image|>"
            elif itype == "audio":
                captured += "<|audio|>"
            elif itype == "video":
                captured += "<|video|>"
        return captured
    return ""


def _emit_assistant_turn(
    message: dict,
    loop_messages: list[dict],
    li: int,
    hf_idx: int,
    *,
    bos_prefix: str,
    continue_same_model_turn: bool,
    is_gen: bool,
    msg_weight: float,
    thinking_scale: float,
) -> list[Segment]:
    """Emit one assistant turn's Segments in _render_gemma4 byte order:
    [opening MARKER]? (channel-thought)? (tool_call)* (tool_response)* CONTENT? closer.
    """
    segs: list[Segment] = []

    # Opening <|turn>model\n marker (suppressed on continue_same_model_turn).
    if not continue_same_model_turn:
        segs.append(Segment(
            text=f"{bos_prefix}<|turn>model\n", token_ids=None,
            kind=SegKind.MARKER, role="assistant", is_generation=is_gen,
            token_class=TokenClass.PROTOCOL, msg_index=hf_idx,
        ))

    # channel-thought block — only when tool_calls + reasoning present.
    # PROTOCOL marker + DECODED body + PROTOCOL closer.
    thinking_text = message.get("reasoning") or message.get("reasoning_content")
    tool_calls = message.get("tool_calls")
    if thinking_text and tool_calls:
        segs.append(Segment(
            text="<|channel>thought\n", token_ids=None, kind=SegKind.MARKER,
            role="assistant", is_generation=is_gen,
            token_class=TokenClass.PROTOCOL, msg_index=hf_idx,
        ))
        segs.append(Segment(
            text=thinking_text, token_ids=None, kind=SegKind.THINKING,
            role="assistant", scale_factor=msg_weight * thinking_scale,
            is_generation=is_gen, token_class=TokenClass.DECODED,
            msg_index=hf_idx,
        ))
        segs.append(Segment(
            text="\n<channel|>", token_ids=None, kind=SegKind.MARKER,
            role="assistant", is_generation=is_gen,
            token_class=TokenClass.PROTOCOL, msg_index=hf_idx,
        ))

    # Tool calls (TRAINED).
    has_tool_call = False
    if tool_calls:
        for tc in tool_calls:
            fn = tc["function"]
            opener = f"<|tool_call>call:{fn['name']}{{"
            args = fn.get("arguments")
            body = ""
            if isinstance(args, Mapping):
                items = sorted(args.items(), key=lambda kv: kv[0])
                body = ",".join(
                    f"{k}:{_format_argument(v, escape_keys=False)}" for k, v in items
                )
            elif isinstance(args, str):
                body = args
            segs.append(Segment(
                text=f"{opener}{body}}}<tool_call|>", token_ids=None,
                kind=SegKind.TOOL_CALL, role="assistant",
                scale_factor=msg_weight, is_generation=is_gen,
                token_class=TokenClass.DECODED, msg_index=hf_idx,
            ))
        has_tool_call = True

    # Tool responses (MASKED) — forward-scan role:tool messages (OpenAI shape).
    emitted_tr = False
    if tool_calls:
        for k in range(li + 1, len(loop_messages)):
            nxt = loop_messages[k]
            if nxt.get("role") != "tool":
                break
            tname = _resolve_tool_name(nxt, tool_calls)
            tr_text = _render_tool_response_text(nxt, tname)
            segs.append(Segment(
                text=tr_text, token_ids=None, kind=SegKind.TOOL_RESULT,
                role="tool", is_generation=False,
                token_class=TokenClass.PROTOCOL, msg_index=hf_idx,
            ))
            emitted_tr = True

    # Captured assistant content (with strip_thinking) — TRAINED.
    #
    # When the content carries a leading ``<think>...</think>`` envelope
    # (the no-tool-call thinking case fused by _build_hf_messages), split it
    # into a THINKING segment (scaled by thinking_scale) + a CONTENT segment
    # for the remaining final answer. This is what makes
    # ``type_overrides["thinking"]`` actually reach the reasoning tokens —
    # a single fused CONTENT segment would apply msg_weight uniformly and the
    # thinking down-weight would be a no-op. The <think>/</think> markers ride
    # inside the THINKING segment (model decodes them -> DECODED, as qwen3).
    captured = _captured_model_text(message)
    has_content = bool(captured.strip())
    if has_content:
        thinking_part, rest = _split_thinking_envelope(captured)
        if thinking_part is not None:
            segs.append(Segment(
                text=thinking_part, token_ids=None, kind=SegKind.THINKING,
                role="assistant", scale_factor=msg_weight * thinking_scale,
                is_generation=is_gen, token_class=TokenClass.DECODED,
                msg_index=hf_idx,
            ))
            if rest:
                segs.append(Segment(
                    text=rest, token_ids=None, kind=SegKind.CONTENT,
                    role="assistant", scale_factor=msg_weight,
                    is_generation=is_gen, token_class=TokenClass.DECODED,
                    msg_index=hf_idx,
                ))
        else:
            segs.append(Segment(
                text=captured, token_ids=None, kind=SegKind.CONTENT,
                role="assistant", scale_factor=msg_weight, is_generation=is_gen,
                token_class=TokenClass.DECODED, msg_index=hf_idx,
            ))

    # Closer — mirrors _render_gemma4 L455-459:
    #   if last emit was a tool_call with no tool_response -> "<|tool_response>"
    #   elif not (tool_response emitted AND no content) -> "<turn|>\n"
    #   else: tool_response with no content -> closer already emitted.
    #
    # Both closers are the model's OWN autonomous turn terminator (class B):
    # the assistant emits "<turn|>\n" to end a normal turn, and emits the
    # trailing "<|tool_response>" to end a tool-call turn (signalling it now
    # expects a tool result). vLLM/SGLang stop on either, so both ARE trained
    # via SegKind.STOP_MARKER (base_mask=1, PROTOCOL carve-out for the
    # assistant stop marker).
    if has_tool_call and not emitted_tr:
        segs.append(Segment(
            text="<|tool_response>", token_ids=None, kind=SegKind.STOP_MARKER,
            role="assistant", scale_factor=msg_weight, is_generation=is_gen,
            token_class=TokenClass.PROTOCOL, msg_index=hf_idx,
        ))
    elif not (emitted_tr and not has_content):
        segs.append(Segment(
            text="<turn|>\n", token_ids=None, kind=SegKind.STOP_MARKER,
            role="assistant", scale_factor=msg_weight, is_generation=is_gen,
            token_class=TokenClass.PROTOCOL, msg_index=hf_idx,
        ))

    return segs


def _captured_model_text(message: dict) -> str:
    """Assistant content captured with strip_thinking (mirrors _render_gemma4)."""
    content = message.get("content")
    if isinstance(content, str):
        return _strip_thinking(content)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        captured = ""
        for item in content:
            if not isinstance(item, Mapping):
                continue
            itype = item.get("type")
            if itype == "text":
                captured += _strip_thinking(item.get("text", ""))
            elif itype == "image":
                captured += "<|image|>"
            elif itype == "audio":
                captured += "<|audio|>"
            elif itype == "video":
                captured += "<|video|>"
        return captured
    return ""


def _split_thinking_envelope(captured: str) -> tuple[str | None, str]:
    """Split a leading ``<think>...</think>`` envelope from the final answer.

    ``_build_hf_messages`` fuses a no-tool-call assistant's ThinkingBlock into
    ``content`` as ``<think>\\n{thinking}\\n</think>\\n\\n{text}`` (or
    ``<think>\\n{thinking}\\n</think>`` when there is no final text). To let
    ``thinking_scale`` reach only the reasoning tokens, the encoder emits the
    envelope as a separate THINKING segment. Returns
    ``(thinking_segment_text, rest)`` where ``thinking_segment_text`` includes
    the ``<think>...`` markers and the trailing ``\\n\\n`` separator (so the
    CONTENT segment starts exactly at the final answer); ``(None, captured)``
    when there is no leading ``<think>`` envelope.
    """
    if not captured.startswith("<think>"):
        return None, captured
    close = captured.find("</think>")
    if close < 0:
        return None, captured
    end = close + len("</think>")
    # Absorb the separator that follows the envelope (jinja emits "\n\n") into
    # the THINKING segment so the CONTENT boundary lands on the answer text.
    rest = captured[end:]
    sep_len = len(rest) - len(rest.lstrip("\n"))
    thinking_part = captured[: end + sep_len]
    return thinking_part, captured[end + sep_len:]


def _resolve_tool_name(tool_msg: dict, tool_calls: list) -> str:
    """Match a tool result to its function name via tool_call_id."""
    tname = tool_msg.get("name") or "unknown"
    for tc in tool_calls:
        if tc.get("id") == tool_msg.get("tool_call_id"):
            return tc["function"]["name"]
    return tname


def _render_tool_response_text(tool_msg: dict, tname: str) -> str:
    """Build the ``<|tool_response>...<tool_response|>`` block."""
    body = tool_msg.get("content")
    if isinstance(body, str):
        return _format_tool_response_block(tname, body)
    if isinstance(body, Sequence) and not isinstance(body, (str, bytes)):
        text_concat = "".join(
            part.get("text", "")
            for part in body
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
        out = _format_tool_response_block(tname, text_concat)
        for part in body:
            if not isinstance(part, Mapping):
                continue
            ptype = part.get("type")
            if ptype == "image":
                out += "<|image|>"
            elif ptype == "audio":
                out += "<|audio|>"
            elif ptype == "video":
                out += "<|video|>"
        return out
    return _format_tool_response_block(tname, body)


__all__ = ["Gemma4Encoder"]
