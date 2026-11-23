"""GPT-OSS (Harmony) SFT encoder — M-γ.5.

Harmony is OpenAI's internal chat format. It is structurally different
from ChatML in three key ways:

  1. Turn markers  : ``<|start|>ROLE<|message|>`` / ``<|end|>``  (not im_start/im_end)
  2. Channels      : assistant output is split into ``analysis`` (thinking),
                     ``commentary`` (tool-call routing), and ``final`` (response)
  3. Dynamic system: Jinja injects ``Current date`` / ``Knowledge cutoff`` /
                     ``Reasoning`` into the built-in system block — all class C
                     tokens (stripped structurally per design decision §2.1).

Decisions locked in docs/superpowers/specs/2026-05-29-gpt-oss-encoder-design.md §2:

  * Strip all three dynamic metadata lines (Current date / Knowledge cutoff /
    Reasoning).  They are class C tokens — inference frameworks inject different
    values → train/serve drift.
  * Preserve ALL historical assistant thinking (decision #2). Unlike Jinja which
    drops pre-tool-call analysis when a later final message exists, titaninfra trains
    on the full reasoning trace.
  * ``<|return|>`` only on the last assistant turn's final-channel closer;
    all other assistant closers use ``<|end|>`` (decision #3).
  * Golden-file test strategy (no Jinja oracle) because we diverge on the three
    axes above.
"""

from __future__ import annotations

import json
import logging

from titaninfra.data.schema import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from titaninfra.data.sft.capabilities import (
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
    FEATURE_PARALLEL_TOOL_CALLS,
    FEATURE_SYSTEM,
    FEATURE_THINKING_CURRENT_TURN,
    FEATURE_THINKING_HISTORY,
    FEATURE_THINKING_INTENT_EFFORT,
    FEATURE_THINKING_INTENT_MODE,
    FEATURE_TOOL_CALL,
    FEATURE_TOOL_RESULT,
    FEATURE_TOOLS_DECLARATION,
    FEATURE_TYPE_OVERRIDES,
)
from titaninfra.data.sft.encoders import _blocks, register
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass
from titaninfra.data.sft.types import EncodedSample, NormalizedMessage, NormalizedSample

_logger = logging.getLogger("titaninfra.sft.encoders.gpt_oss")

# Capability declaration — see titaninfra/data/sft/capabilities.py.
# GPT-OSS Harmony supports the widest IR subset: thinking (analysis
# channel), tool_call (commentary channel), tool_result, plus the
# thinking_intent field which maps to Reasoning: high/medium/low in
# the system header. gpt_oss is the ONLY family to declare both
# THINKING_INTENT_MODE and _EFFORT — its Harmony chat template has a
# wire-level ``Reasoning: <effort>`` line (jinja L196-206), so the
# effort dimension is genuinely actionable here. History thinking is
# preserved by titaninfra design choice (decision §2 of the GPT-OSS
# encoder spec).
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
    FEATURE_THINKING_INTENT_EFFORT,
    FEATURE_TYPE_OVERRIDES,
})

# ---------------------------------------------------------------------------
# Wire-format constants
# ---------------------------------------------------------------------------
_START = "<|start|>"
_MESSAGE = "<|message|>"
_END = "<|end|>"
_CHANNEL = "<|channel|>"
_CALL = "<|call|>"
_RETURN = "<|return|>"

_DEFAULT_IDENTITY = "You are ChatGPT, a large language model trained by OpenAI."
_VALID_CHANNELS_LINE = (
    "# Valid channels: analysis, commentary, final."
    " Channel must be included for every message."
)
_TOOLS_FOOTER = "\nCalls to these tools must go to the commentary channel: 'functions'."


# ---------------------------------------------------------------------------
# TypeScript namespace renderer  (spec §13 pitfall #4: Python recursion)
# ---------------------------------------------------------------------------

def _render_typescript_type(spec: dict) -> str:
    """Recursively render one JSON Schema property spec as a TypeScript type.

    Covers the subset of the Jinja render_typescript_type macro that appears
    in real tool schemas: string/number/integer/boolean/object/array, enum,
    nullable, oneOf, array-of-types list.  Unknown specs → "any".
    """
    if spec is None:
        return "any"

    # Array of type strings like ["object", "object"] (Union[dict, list])
    type_val = spec.get("type")
    if isinstance(type_val, list):
        if len(type_val) > 1:
            return " | ".join(type_val)
        return type_val[0] if type_val else "any"

    if "oneOf" in spec:
        variants = spec["oneOf"]
        has_object = any(v.get("type") == "object" for v in variants)
        if has_object and len(variants) > 1:
            return "any"
        parts = []
        for v in variants:
            parts.append(_render_typescript_type(v))
            if v.get("description"):
                parts[-1] += f"// {v['description']}"
            if "default" in v:
                parts[-1] += f"// default: {json.dumps(v['default'], ensure_ascii=False)}"
        return " | ".join(parts)

    if type_val == "array":
        items = spec.get("items")
        if not items:
            suffix = " | null" if spec.get("nullable") else ""
            return f"any[]{suffix}"
        inner_type = items.get("type")
        if inner_type == "string":
            ts = "string[]"
        elif inner_type in ("number", "integer"):
            ts = "number[]"
        elif inner_type == "boolean":
            ts = "boolean[]"
        else:
            rendered = _render_typescript_type(items)
            ts = "any[]" if (rendered == "object | object" or len(rendered) > 50) else rendered + "[]"
        if spec.get("nullable"):
            ts += " | null"
        return ts

    if type_val == "string":
        if spec.get("enum"):
            return '"' + '" | "'.join(spec["enum"]) + '"'
        suffix = " | null" if spec.get("nullable") else ""
        return f"string{suffix}"

    if type_val in ("number", "integer"):
        return "number"

    if type_val == "boolean":
        return "boolean"

    if type_val == "object":
        props = spec.get("properties")
        if not props:
            return "object"
        required = spec.get("required") or []
        lines = ["{\n"]
        prop_items = list(props.items())
        for idx, (prop_name, prop_spec) in enumerate(prop_items):
            opt = "" if prop_name in required else "?"
            ts_type = _render_typescript_type(prop_spec)
            lines.append(f"{prop_name}{opt}: {ts_type},\n")
        lines.append("}")
        return "".join(lines)

    return "any"


def _render_tool_namespace(tools: tuple[dict, ...]) -> str:
    """Render the TypeScript ``functions`` namespace for the developer block.

    Matches the Jinja ``render_tool_namespace("functions", tools)`` macro
    output byte-for-byte for the common schema patterns.
    """
    lines: list[str] = []
    lines.append("## functions\n\n")
    lines.append("namespace functions {\n\n")
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        description = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"// {description}\n")
        lines.append(f"type {name} = ")
        props = (params.get("properties") or {}) if params else {}
        if props:
            required = (params.get("required") or [])
            lines.append("(_: {\n")
            for prop_name, prop_spec in props.items():
                desc = prop_spec.get("description")
                if desc:
                    lines.append(f"// {desc}\n")
                opt = "" if prop_name in required else "?"
                ts_type = _render_typescript_type(prop_spec)
                default = prop_spec.get("default")
                if default is not None:
                    if prop_spec.get("enum"):
                        default_str = f", // default: {default}"
                    elif prop_spec.get("oneOf"):
                        default_str = f"// default: {default}"
                    else:
                        default_str = f", // default: {json.dumps(default, ensure_ascii=False)}"
                    lines.append(f"{prop_name}{opt}: {ts_type}{default_str},\n")
                else:
                    lines.append(f"{prop_name}{opt}: {ts_type},\n")
            lines.append("}) => any;\n\n")
        else:
            lines.append("() => any;\n\n")
    lines.append("} // namespace functions")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@register("gpt_oss")
class GptOssEncoder:
    """Segment-first encoder for the OpenAI Harmony (GPT-OSS) chat format."""

    name = "gpt_oss"

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

        type_overrides = None
        if isinstance(loss_config, dict):
            type_overrides = loss_config.get("type_overrides")
        segments = self.encode_segments(sample, type_overrides=type_overrides)
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
        sample = _promote_leading_system(sample)
        thinking_scale = (type_overrides or {}).get("thinking", 1.0)
        segs: list[Segment] = []

        # thinking_intent projection — Harmony wire has an actual
        # ``Reasoning: <effort>`` line (jinja L206) so user-pinned intent
        # upgrades that line from class C (dynamic metadata stripped by
        # default) to class A (deterministic injection, train/inference
        # consistent). mode=off additionally suppresses the analysis
        # channel so the model never emits or learns to emit thinking.
        ti = sample.thinking_intent
        mode_off = ti is not None and ti.mode == "off"

        # Built-in system block (always emitted, always loss=0)
        segs.append(_built_in_system_segment(sample))

        # Developer block (emitted iff system text or tools present)
        if sample.system or sample.tools:
            segs.append(_developer_segment(sample))

        # Message loop
        messages = sample.messages
        last_asst_idx = _last_assistant_index(messages)

        for i, nm in enumerate(messages):
            is_last_asst = (i == last_asst_idx)
            if nm.role == "user":
                segs.append(_user_segment(nm, i))
            elif nm.role == "assistant":
                segs.extend(_assistant_segments(
                    nm, i, is_last_asst=is_last_asst,
                    thinking_scale=thinking_scale,
                    mode_off=mode_off,
                ))
            elif nm.role == "tool":
                tool_name = _last_tool_name(
                    messages, i, tool_use_id=_tool_result_use_id(nm)
                )
                segs.append(_tool_result_segment(nm, i, tool_name=tool_name))
            else:
                raise NotImplementedError(
                    f"GptOssEncoder: unsupported role {nm.role!r} at index {i}"
                )

        return segs


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def _promote_leading_system(sample: NormalizedSample) -> NormalizedSample:
    """Fold leading role='system' messages into sample.system.

    Harmony's wire requires system text at top-level (built-in system /
    developer block); 6/7 families tolerate role='system' in messages,
    gpt_oss is the strict outlier. This bridges the de-facto contract.
    Conflict policy: if sample.system is already set AND a leading
    system message exists, raise — ambiguous source, fail loud at
    converter time per validate-at-converter-not-runtime contract.
    """
    from dataclasses import replace

    leading_texts: list[str] = []
    rest: list[NormalizedMessage] = []
    seen_non_system = False
    for nm in sample.messages:
        if nm.role == "system" and not seen_non_system:
            # Audit #10 (2026-06-06): only TextBlocks are promoted. Non-text
            # blocks in a leading system message would be silently dropped
            # here. The converter G1 validator (block x role matrix) already
            # rejects them for MDS data, but guard the non-validated in-memory
            # path (sft_utils etc.) with a warn so the drop is never silent.
            dropped = [
                type(b).__name__ for b in nm.blocks
                if not isinstance(b, TextBlock)
            ]
            if dropped:
                _logger.warning(
                    "gpt_oss _promote_leading_system: dropping non-TextBlock "
                    "content from a leading system message (%s). System role "
                    "supports text only; fix the source data.",
                    dropped,
                )
            text = "\n\n".join(
                b.text for b in nm.blocks
                if isinstance(b, TextBlock) and b.text
            )
            if text:
                leading_texts.append(text)
        else:
            seen_non_system = True
            rest.append(nm)
    if not leading_texts:
        return sample
    if sample.system:
        raise ValueError(
            "GptOssEncoder: sample.system is already set AND messages[0] "
            "has role='system' — ambiguous system source. Fix the source "
            "data to use exactly one of: top-level system field OR a "
            "leading system message."
        )
    return replace(sample, system="\n\n".join(leading_texts), messages=tuple(rest))


def _built_in_system_segment(sample: NormalizedSample) -> Segment:
    """Emit the fixed system block.

    Dynamic metadata lines (``Current date`` / ``Knowledge cutoff``) stay
    stripped — they remain class C tokens with no user-pinned value.

    When ``sample.thinking_intent`` is set, the ``Reasoning: <effort>``
    line (jinja L206) is injected as class A: user-pinned intent makes it
    deterministically reproducible at inference time, so the model can
    safely learn it. Mapping:
      mode=off          → ``Reasoning: none``
      mode=on, no effort → ``Reasoning: medium`` (Harmony default)
      mode=on, effort=X → ``Reasoning: X``
    When ``thinking_intent`` is None, the line is omitted entirely
    (preserves the design §2.1 behavior of treating it as class C).
    """
    tools_footer = _TOOLS_FOOTER if sample.tools else ""
    reasoning_line = _reasoning_line(sample.thinking_intent)
    text = (
        f"{_START}system{_MESSAGE}"
        f"{_DEFAULT_IDENTITY}\n"
        f"{reasoning_line}"
        f"{_VALID_CHANNELS_LINE}"
        f"{tools_footer}"
        f"{_END}"
    )
    return Segment(
        text=text,
        token_ids=None,
        kind=SegKind.SYSTEM,
        role="system",
        is_generation=False,
        token_class=TokenClass.PROTOCOL,
        msg_index=None,
    )


def _reasoning_line(ti) -> str:
    """Render the ``Reasoning: <effort>\\n\\n`` injection or empty string."""
    if ti is None:
        return ""
    if ti.mode == "off":
        return "Reasoning: none\n\n"
    # mode=on (explicit or implicit via effort-only): default to medium
    effort = ti.effort if ti.effort in ("low", "medium", "high") else "medium"
    return f"Reasoning: {effort}\n\n"


def _developer_segment(sample: NormalizedSample) -> Segment:
    """Emit developer block with optional instructions and/or tool namespace."""
    parts: list[str] = [f"{_START}developer{_MESSAGE}"]
    if sample.system:
        parts.append(f"# Instructions\n\n{sample.system}\n\n")
    if sample.tools:
        parts.append(f"# Tools\n\n{_render_tool_namespace(sample.tools)}")
    parts.append(_END)
    return Segment(
        text="".join(parts),
        token_ids=None,
        kind=SegKind.TOOLS_HEADER if sample.tools else SegKind.SYSTEM,
        role="developer",
        is_generation=False,
        token_class=TokenClass.PROTOCOL,
        msg_index=None,
    )


def _user_segment(nm, msg_index: int) -> Segment:
    return Segment(
        text=f"{_START}user{_MESSAGE}{_block_text(nm.blocks)}{_END}",
        token_ids=None,
        kind=SegKind.CONTENT,
        role="user",
        is_generation=False,
        token_class=TokenClass.PROTOCOL,
        msg_index=msg_index,
    )


# ---------------------------------------------------------------------------
# Block walkers — Phase 3.5: read nm.blocks directly, not legacy text/thinking fields
# ---------------------------------------------------------------------------
def _block_text(blocks) -> str:
    """Concatenate every TextBlock's text; ignore others."""
    return _blocks.block_text(blocks)


def _extract_thinking(blocks) -> str | None:
    return _blocks.extract_thinking(blocks)


def _extract_tool_calls(blocks) -> list:
    return _blocks.extract_tool_calls(blocks)


def _extract_tool_result_text(blocks) -> str:
    return _blocks.extract_tool_result_text(blocks)


def _assistant_segments(
    nm,
    msg_index: int,
    *,
    is_last_asst: bool,
    thinking_scale: float,
    mode_off: bool = False,
) -> list[Segment]:
    """Dispatch to Case A/B/C/D based on message content.

    ``mode_off`` (sample.thinking_intent.mode == "off") suppresses the
    analysis channel everywhere — the model must learn to skip
    ``<|channel|>analysis`` entirely. Case D collapses to empty.
    """
    tool_calls = _extract_tool_calls(nm.blocks)
    text_body = _block_text(nm.blocks)
    thinking = _extract_thinking(nm.blocks)
    has_tool_calls = bool(tool_calls)
    msg_weight = float(nm.weight)
    # Honor mode=off by hiding thinking from the case dispatchers.
    effective_thinking = None if mode_off else thinking

    if has_tool_calls:
        # Case A — tool call (possibly with thinking)
        return _case_a_segments(
            tool_calls, msg_index, is_gen=is_last_asst,
            msg_weight=msg_weight, thinking_scale=thinking_scale,
            thinking=effective_thinking,
        )
    elif is_last_asst:
        # Case B — last turn, final reply
        return _case_b_segments(
            text_body, msg_index, msg_weight=msg_weight,
            thinking_scale=thinking_scale, thinking=effective_thinking,
        )
    elif text_body or effective_thinking:
        # Case C — mid-turn, final reply (not last)
        return _case_c_segments(
            text_body, msg_index, msg_weight=msg_weight,
            thinking_scale=thinking_scale, thinking=effective_thinking,
        )
    else:
        # Case D — pure thinking, no content, no tool_calls (rare)
        return _case_d_segments(
            msg_index, msg_weight=msg_weight,
            thinking_scale=thinking_scale, thinking=effective_thinking,
        )


def _emit_analysis_block(
    msg_index: int, *, is_gen: bool, msg_weight: float, thinking_scale: float,
    thinking: str,
) -> list[Segment]:
    """Emit the ``<|channel|>analysis`` block for a message with thinking."""
    opener = f"{_START}assistant{_CHANNEL}analysis{_MESSAGE}"
    return [
        Segment(text=opener, token_ids=None, kind=SegKind.MARKER,
                role="assistant",
                is_generation=is_gen, token_class=TokenClass.PROTOCOL,
                msg_index=msg_index),
        Segment(text=thinking, token_ids=None, kind=SegKind.THINKING,
                role="assistant", scale_factor=msg_weight * thinking_scale,
                is_generation=is_gen, token_class=TokenClass.DECODED,
                msg_index=msg_index),
        Segment(text=_END, token_ids=None, kind=SegKind.MARKER,
                role="assistant",
                is_generation=is_gen, token_class=TokenClass.PROTOCOL,
                msg_index=msg_index),
    ]


def _case_a_segments(tool_calls, msg_index: int, *, is_gen: bool,
                     msg_weight: float, thinking_scale: float,
                     thinking: str | None) -> list[Segment]:
    """Case A: assistant with tool_call(s)."""
    segs: list[Segment] = []

    if thinking:
        segs.extend(_emit_analysis_block(msg_index, is_gen=is_gen,
                                         msg_weight=msg_weight,
                                         thinking_scale=thinking_scale,
                                         thinking=thinking))

    for tc in tool_calls:
        name = tc.name
        args = tc.input
        args_json = json.dumps(args, ensure_ascii=False, sort_keys=False)
        # ToolUseBlock has no content_type; default to "json" matching jinja.
        content_type = "json"
        opener = (
            f"{_START}assistant to=functions.{name}"
            f"{_CHANNEL}commentary {content_type}{_MESSAGE}"
        )
        segs.append(Segment(text=opener, token_ids=None, kind=SegKind.MARKER,
                            role="assistant",
                            is_generation=is_gen, token_class=TokenClass.PROTOCOL,
                            msg_index=msg_index))
        segs.append(Segment(text=args_json, token_ids=None, kind=SegKind.TOOL_CALL,
                            role="assistant", scale_factor=msg_weight,
                            is_generation=is_gen, token_class=TokenClass.DECODED,
                            msg_index=msg_index))
        segs.append(Segment(text=_CALL, token_ids=None, kind=SegKind.STOP_MARKER,
                            role="assistant", scale_factor=msg_weight,
                            is_generation=is_gen, token_class=TokenClass.PROTOCOL,
                            msg_index=msg_index))

    return segs


def _case_b_segments(text_body: str, msg_index: int, *, msg_weight: float,
                     thinking_scale: float,
                     thinking: str | None) -> list[Segment]:
    """Case B: last assistant turn — uses ``<|return|>`` as closer."""
    segs: list[Segment] = []

    if thinking:
        segs.extend(_emit_analysis_block(msg_index, is_gen=True,
                                         msg_weight=msg_weight,
                                         thinking_scale=thinking_scale,
                                         thinking=thinking))

    opener = f"{_START}assistant{_CHANNEL}final{_MESSAGE}"
    segs.append(Segment(text=opener, token_ids=None, kind=SegKind.MARKER,
                        role="assistant",
                        is_generation=True, token_class=TokenClass.PROTOCOL,
                        msg_index=msg_index))
    # Skip the CONTENT segment when body is empty — assemble() rejects
    # 0-token segments. This handles the post-capability-skip case where
    # all blocks were warn-and-skipped (e.g. only ImageBlock on a family
    # that doesn't support it). Marker + return still emits a learnable
    # closing signal.
    if text_body:
        segs.append(Segment(text=text_body, token_ids=None, kind=SegKind.CONTENT,
                            role="assistant", scale_factor=msg_weight,
                            is_generation=True, token_class=TokenClass.DECODED,
                            msg_index=msg_index))
    # <|return|> is the LLM's autonomous stop signal (class B token)
    segs.append(Segment(text=_RETURN, token_ids=None, kind=SegKind.STOP_MARKER,
                        role="assistant", scale_factor=msg_weight,
                        is_generation=True, token_class=TokenClass.PROTOCOL,
                        msg_index=msg_index))
    return segs


def _case_c_segments(text_body: str, msg_index: int, *, msg_weight: float,
                     thinking_scale: float,
                     thinking: str | None) -> list[Segment]:
    """Case C: non-last assistant turn — uses ``<|end|>`` as closer."""
    segs: list[Segment] = []

    if thinking:
        segs.extend(_emit_analysis_block(msg_index, is_gen=False,
                                         msg_weight=msg_weight,
                                         thinking_scale=thinking_scale,
                                         thinking=thinking))

    opener = f"{_START}assistant{_CHANNEL}final{_MESSAGE}"
    segs.append(Segment(text=opener, token_ids=None, kind=SegKind.MARKER,
                        role="assistant",
                        is_generation=False, token_class=TokenClass.PROTOCOL,
                        msg_index=msg_index))
    if text_body:
        segs.append(Segment(text=text_body, token_ids=None, kind=SegKind.CONTENT,
                            role="assistant", scale_factor=msg_weight,
                            is_generation=False, token_class=TokenClass.DECODED,
                            msg_index=msg_index))
    segs.append(Segment(text=_END, token_ids=None, kind=SegKind.MARKER,
                        role="assistant",
                        is_generation=False, token_class=TokenClass.PROTOCOL,
                        msg_index=msg_index))
    return segs


def _case_d_segments(msg_index: int, *, msg_weight: float,
                     thinking_scale: float,
                     thinking: str | None) -> list[Segment]:
    """Case D: pure thinking, no content, no tool_calls (rare edge case)."""
    if not thinking:
        return []
    return _emit_analysis_block(msg_index, is_gen=False,
                                msg_weight=msg_weight,
                                thinking_scale=thinking_scale,
                                thinking=thinking)


def _tool_result_segment(nm, msg_index: int, *, tool_name: str) -> Segment:
    """Emit a tool-result block.

    Renders the raw string content with no extra quoting/escaping, matching
    the official ``openai-harmony`` package (the same renderer vLLM's serving
    path uses) byte-for-byte. The stock HF jinja template instead pipes
    ``message.content|tojson``, which wraps a plain string result in a
    redundant pair of JSON quotes (e.g. ``"5"`` instead of ``5``,
    ``"299,792,458 m/s"`` instead of ``299,792,458 m/s``) -- that extra
    quoting is an artifact of the jinja template, not something the official
    Harmony renderer or vLLM's inference path ever produces or expects.
    Verified against a direct call into ``openai_harmony``'s own
    ``HarmonyEncoding.render`` (2026-11, SFT renderers-oracle investigation);
    see docs/renderers_oracle_gptoss_report.md Sec 5 L1-D for the discovery
    and docs/torchtitan_upgrade_plan_2026-11.md Sec 11 for the (zero) current
    production exposure at the time this was found.
    """
    content = _extract_tool_result_text(nm.blocks)
    text = (
        f"{_START}functions.{tool_name} to=assistant"
        f"{_CHANNEL}commentary{_MESSAGE}"
        f"{content}"
        f"{_END}"
    )
    return Segment(text=text, token_ids=None, kind=SegKind.TOOL_RESULT,
                   role="tool",
                   is_generation=False, token_class=TokenClass.PROTOCOL,
                   msg_index=msg_index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_assistant_index(messages) -> int:
    """Index of the last assistant turn in messages. -1 if none."""
    return _blocks.last_assistant_index(messages)


def _tool_result_use_id(nm) -> str | None:
    """Return the tool_use_id of the first ToolResultBlock in a tool message."""
    for b in nm.blocks:
        if isinstance(b, ToolResultBlock):
            return b.tool_use_id
    return None


def _last_tool_name(messages, tool_idx: int, *, tool_use_id: str | None = None) -> str:
    """Find the function name for the tool result at ``tool_idx``.

    Audit #7 (2026-06-06): when ``tool_use_id`` is given, match it against the
    preceding assistant turn's ``ToolUseBlock.id`` so PARALLEL tool calls map
    each result to the RIGHT function name. Falls back to the most recent
    assistant's first tool_call name only when no id match exists (legacy
    single-call data without ids).
    """
    for i in range(tool_idx - 1, -1, -1):
        nm = messages[i]
        if nm.role == "assistant":
            tcs = _extract_tool_calls(nm.blocks)
            if not tcs:
                continue
            if tool_use_id is not None:
                for tc in tcs:
                    if getattr(tc, "id", None) == tool_use_id:
                        return tc.name
            return tcs[0].name
    return "unknown"


__all__ = ["GptOssEncoder"]
