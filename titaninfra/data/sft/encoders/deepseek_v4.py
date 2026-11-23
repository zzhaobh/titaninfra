"""DeepSeek-V4 SFT Encoder (segment path + legacy fallback).

DSv4 has no Jinja chat_template; the wire format is built directly
from titaninfra-native ``SFTSample`` / ``Message`` / ``Block`` objects.
This module exposes two entry points:

* ``encode_segments(NormalizedSample, ...)`` -- the canonical segment
  pipeline used by ``dispatch.tokenize_sample``. Walks normalized
  messages and emits ``list[Segment]``; ``Assembler.assemble`` turns
  that into ``EncodedSample``.
* ``encode(messages_or_sample, *, tools, max_length, train_on_inputs,
  loss_config)`` -- preserves the pre-M0 API used by
  ``tests/test_dsv4_encoder.py`` and the legacy
  ``tests/oracle/_jinja_oracle/_jinja_renderer.py:template_name=
  'deepseek_v4'`` branch. Returns ``(input_ids, labels)``.

Wire protocol::

    BOS + system_text [+ "\\n\\n" + TOOLS_TEMPLATE if tools and no developer]
    [<｜latest_reminder｜>reminder_text]                   (IGNORE)
    [<｜User｜>developer_text [+ "\\n\\n" + TOOLS_TEMPLATE]] (IGNORE)
    <｜User｜>user_text                                    (IGNORE)
    [<｜Assistant｜></think><｜task_token｜>]              (IGNORE, only if user.task)
    <｜Assistant｜>                                        (IGNORE, skipped after task)
    [<think>thinking</think>] | </think>                   (assistant weight)
    text [+ DSML tool_calls] <｜end▁of▁sentence｜>         (assistant weight)
    <｜User｜><tool_result>...</tool_result>+              (IGNORE)
    ...

History thinking is a TITANINFRA-CONTROLLED training policy, not a wire
constant. By default DSv4 supervises every ThinkingBlock verbatim —
including historical turns — which is the OPPOSITE of DeepSeek's
reference encoder (``encoding_dsv4.py`` defaults ``drop_thinking=True``,
rendering past turns as a bare ``</think>``). To drop or summarize
historical thinking (e.g. to align train/inference with the reference
recipe), set ``StreamConfig.history_policy`` (``per_type={"thinking":
HistoryAction(action="drop")}``); ``normalize.py`` applies it before
the encoder runs. When historical thinking is present and no policy is
supplied, ``encode_segments`` emits a warn-loud (the training engine
owns the masking policy, not the chat_template — see memory
``feedback_jinja_is_recipe_projection_not_protocol``).
"""
# ruff: noqa: E501  -- _TOOLS_TEMPLATE contains protocol-critical lines that cannot be wrapped

from __future__ import annotations

import dataclasses
import json

from titaninfra.data.schema import (
    Message,
    TextBlock,
    ThinkingBlock,
    Tool,
    ToolResultBlock,
    ToolUseBlock,
)
from titaninfra.data.sft import normalize
from titaninfra.data.sft.capabilities import (
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
    FEATURE_SYSTEM,
    FEATURE_THINKING_CURRENT_TURN,
    FEATURE_THINKING_HISTORY,
    FEATURE_THINKING_INTENT_MODE,
    FEATURE_TOOL_CALL,
    FEATURE_TOOL_RESULT,
    FEATURE_TOOLS_DECLARATION,
    FEATURE_TYPE_OVERRIDES,
    EmptyAssistantError,
    warn_history_thinking_trained,
)
from titaninfra.data.sft.encoders import IGNORE_INDEX, _blocks, register
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass
from titaninfra.data.sft.types import NormalizedSample

# Capability declaration — see titaninfra/data/sft/capabilities.py.
# DSv4 is titaninfra-native (no jinja). It supervises ThinkingBlocks on
# the current turn by default, and supervises HISTORICAL thinking too
# UNLESS the sample's history_policy drops/summarizes it (titaninfra-
# controlled policy; warn-loud when historical thinking is trained with
# no policy — see _build_segments). Parallel tool_calls per assistant
# turn are currently NOT supported by the DSML <｜DSML｜> wire (single
# call only). thinking_intent.mode is declared supported here (projection
# in P3.3: mode=off forces ``nm.thinking=None`` so the wire emits the
# bare ``</think>`` close marker only). thinking_intent.effort is NOT
# declared because DSv4 wire has no reasoning_effort field — a sample
# with ``effort`` set produces one warn-loud line and the field is
# ignored. See 2026-05-31 wire survey in
# ``docs/superpowers/research/2026-06-thinking-intent-next-session.md``.
SUPPORTED_BLOCKS = frozenset({TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock})
SUPPORTED_FEATURES = frozenset({
    FEATURE_SYSTEM,
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
    FEATURE_TOOLS_DECLARATION,
    FEATURE_TOOL_CALL,
    FEATURE_TOOL_RESULT,
    FEATURE_THINKING_CURRENT_TURN,
    FEATURE_THINKING_HISTORY,
    FEATURE_THINKING_INTENT_MODE,
    FEATURE_TYPE_OVERRIDES,
})

# DSML format constants
_DSML = "｜DSML｜"
_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_THINK_S = "<think>"
_THINK_E = "</think>"
_LATEST_REMINDER = "<｜latest_reminder｜>"

_TOOLS_TEMPLATE = """\
## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

# Task-type special tokens
_DS_TASK_SP_TOKENS: dict[str, str] = {
    "action": "<｜action｜>",
}


def _render_tools_str(tools: list) -> str:
    """Render Tool structs (or dicts) into DSv4 TOOLS_TEMPLATE string."""
    tools_dicts: list[dict] = []
    for t in tools:
        if isinstance(t, Tool):
            tools_dicts.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            })
        elif isinstance(t, dict):
            params = t.get("parameters", t.get("input_schema", {}))
            tools_dicts.append({
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": params,
            })
    return _TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(json.dumps(d, ensure_ascii=False) for d in tools_dicts),
        dsml_token=_DSML,
        thinking_start_token=_THINK_S,
        thinking_end_token=_THINK_E,
    )


def _encode_dict_to_dsml_params(arguments: dict, dsml: str) -> str:
    tmpl = '<{dsml}parameter name="{k}" string="{s}">{v}</{dsml}parameter>'
    parts = []
    for k, v in arguments.items():
        is_str = isinstance(v, str)
        parts.append(tmpl.format(
            dsml=dsml,
            k=k,
            s="true" if is_str else "false",
            v=v if is_str else json.dumps(v, ensure_ascii=False),
        ))
    return "\n".join(parts)


def _render_dsml_tool_calls_text(tool_calls: list) -> str:
    """Build the ``\\n\\n<｜DSML｜tool_calls>...`` block as a string.

    Accepts both legacy dict-form ``[{"name", "arguments", "id"}]`` and
    Phase 3.7 block-form ``list[ToolUseBlock]``.

    Audit #6 (2026-06-06): the DSML wire is single-call-only. A parallel
    tool-call turn (>1 call) is rejected loud rather than silently rendered
    as one ``<｜DSML｜tool_calls>`` block with multiple ``<｜DSML｜invoke>``
    elements the model cannot reproduce at inference. dsv4 does not declare
    FEATURE_PARALLEL_TOOL_CALLS, so dispatch already warns; this drops the
    sample (ValueError → streaming_dataset skips) instead of emitting a
    malformed wire.
    """
    if not tool_calls:
        return ""
    if len(tool_calls) > 1:
        # EmptyAssistantError (not ValueError) keeps this consistent with the
        # capability contract: a parallel-call turn carries no DSv4-renderable
        # assistant content, so it is dropped like any unsupported-content
        # sample (streaming_dataset catches it; the capability matrix classifies
        # the cell EMPTY-RAISE).
        raise EmptyAssistantError(
            family="deepseek_v4",
            skipped_block_types=["ToolUseBlock"] * len(tool_calls),
            sample_repr=(
                f"parallel tool_calls unsupported — DSML wire is single-call "
                f"only, got {len(tool_calls)} calls in one assistant turn"
            ),
        )
    parts = []
    for tc in tool_calls:
        if isinstance(tc, ToolUseBlock):
            name = tc.name
            args = tc.input
        else:
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {}
            name = tc.get("name", "")
        params = _encode_dict_to_dsml_params(args, _DSML)
        parts.append(f'<{_DSML}invoke name="{name}">\n{params}\n</{_DSML}invoke>')
    return (
        f"\n\n<{_DSML}tool_calls>\n"
        + "\n".join(parts)
        + f"\n</{_DSML}tool_calls>"
    )


# ---------------------------------------------------------------------------
# Block walkers — Phase 3.7: read nm.blocks directly, not legacy text/thinking
# ---------------------------------------------------------------------------
def _block_text(blocks) -> str:
    """Concatenate every TextBlock's text; ignore others."""
    return _blocks.block_text(blocks)


def _extract_thinking(blocks) -> str | None:
    return _blocks.extract_thinking(blocks)


def _extract_tool_use_blocks(blocks) -> list:
    return _blocks.extract_tool_calls(blocks)


def _extract_tool_result_text(blocks) -> str:
    return _blocks.extract_tool_result_text(blocks)


def _merge_consecutive_tool_messages(messages):
    """Compatibility shim. The segment builder handles chain detection;
    callers no longer need to pre-merge. Kept exported so any external
    consumer that imported the helper before this refactor still works.
    """
    return list(messages)


@register("deepseek_v4")
class DeepSeekV4Encoder:
    """DSv4 SFT encoder. Segment path + legacy fallback."""

    name = "deepseek_v4"

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        # Lock boundary token ids to instance attrs so SFT loss-mask logic
        # (and the DSV4 thinking-mode contract test) can locate role/think
        # markers without re-tokenizing on every call.
        self._asst = tokenizer.convert_tokens_to_ids(_ASSISTANT)
        self._think_s = tokenizer.convert_tokens_to_ids(_THINK_S)
        self._think_e = tokenizer.convert_tokens_to_ids(_THINK_E)

    def encode_segments(
        self,
        sample: NormalizedSample,
        *,
        type_overrides: dict[str, float] | None = None,
    ) -> list[Segment]:
        segments = _build_segments(sample)
        if not type_overrides:
            return segments
        # DSv4 emits THINKING as standalone Segment (see _assistant_segments).
        # type_overrides["thinking"] multiplies into those segments' loss_scale;
        # 0.0 zeroes the loss, other values scale it (Assembler emits non-binary
        # loss_weights). Tool-call scaling is not yet supported -- DSv4 inlines
        # tool_calls into CONTENT body, not standalone segments.
        thinking_scale = type_overrides.get("thinking")
        if thinking_scale is None or thinking_scale == 1.0:
            return segments
        return [
            dataclasses.replace(
                seg,
                scale_factor=(1.0 if seg.scale_factor is None else seg.scale_factor)
                * thinking_scale,
            )
            if seg.kind == SegKind.THINKING
            else seg
            for seg in segments
        ]

    def encode(
        self,
        sample_or_messages,
        *,
        max_length: int | None = None,
        loss_config: dict | None = None,
        tools: list | None = None,
        train_on_inputs: bool = False,
        truncate_at_last_segment: bool = False,
    ):
        """Dual-shape entry.

        New shape::

            encode(NormalizedSample, *, max_length=None, loss_config=None)
                -> EncodedSample

        Legacy shape (still used by ``test_dsv4_encoder.py`` and the
        oracle harness's ``template_name='deepseek_v4'`` branch)::

            encode(list[Message], *, tools=..., max_length=...,
                   train_on_inputs=..., loss_config=...)
                -> (list[int], list[int])

        Dispatch on the type of the first positional argument. Returning
        different shapes from one method is unusual but the alternative
        (two methods) would break the SFTEncoder Protocol that
        ``dispatch.py`` and the legacy ``_jinja_renderer`` both rely on.
        """
        if isinstance(sample_or_messages, NormalizedSample):
            from titaninfra.data.sft.assembler import assemble

            if tools is not None:
                raise ValueError(
                    "DeepSeekV4Encoder.encode: when called with NormalizedSample, "
                    "tools must live on ``sample.tools`` -- positional ``tools=...`` "
                    "is reserved for the legacy ``encode(messages, tools=...)`` shape."
                )
            type_overrides = None
            if isinstance(loss_config, dict):
                type_overrides = loss_config.get("type_overrides")
            segments = self.encode_segments(
                sample_or_messages, type_overrides=type_overrides,
            )
            return assemble(
                segments, self.tokenizer, mode="sft", max_length=max_length,
                truncate_at_last_segment=truncate_at_last_segment,
            )

        # Legacy path discards loss_config -- non-binary scaling not yet
        # plumbed through ``encode(list[Message], ...)``.
        del loss_config

        # Legacy path: wrap the Message list into a NormalizedSample so
        # the segment pipeline does all the heavy lifting. We do NOT
        # pre-merge consecutive tool messages here -- the segment builder
        # handles chain detection per-message so each tool message stays
        # as a distinct ``<tool_result>...</tool_result>`` chunk under a
        # shared ``<｜User｜>`` opener.
        messages = sample_or_messages
        sample_json = _messages_and_tools_to_sample_json(messages, tools)
        normalized = normalize.from_sft_sample(sample_json)
        segments = _build_segments(normalized)
        return _segments_to_legacy_arrays(
            segments, self.tokenizer,
            train_on_inputs=train_on_inputs, max_length=max_length,
        )


def _messages_and_tools_to_sample_json(
    messages: list[Message], tools: list | None,
) -> dict:
    """Round-trip titaninfra ``Message`` list back into ``SFTSample`` dict shape.

    Used only by the legacy ``encode(list[Message], tools=...)`` path so
    ``normalize.from_sft_sample`` can produce a ``NormalizedSample``. We
    only round-trip the fields the segment builder reads -- ``role``,
    ``blocks``, ``weight``, ``task``.
    """
    msg_dicts: list[dict] = []
    for m in messages:
        block_dicts: list[dict] = []
        for b in m.blocks:
            if isinstance(b, TextBlock):
                block_dicts.append({"type": "text", "text": b.text})
            elif isinstance(b, ThinkingBlock):
                block_dicts.append({"type": "thinking", "text": b.text})
            elif isinstance(b, ToolUseBlock):
                block_dicts.append({
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.name,
                    "input": b.input,
                })
            elif isinstance(b, ToolResultBlock):
                block_dicts.append({
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id,
                    "text": b.text,
                })
            # ImageBlock and other types silently skipped (matches the
            # legacy text-only behavior).
        d: dict = {"role": m.role, "blocks": block_dicts}
        if m.weight is not None:
            d["weight"] = m.weight
        if m.task is not None:
            d["task"] = m.task
        msg_dicts.append(d)
    out: dict = {"messages": msg_dicts}
    if tools:
        tool_dicts: list[dict] = []
        for t in tools:
            if isinstance(t, Tool):
                tool_dicts.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            elif isinstance(t, dict):
                tool_dicts.append(dict(t))
        out["tools"] = tool_dicts
    return out


def _build_segments(sample: NormalizedSample) -> list[Segment]:
    """Walk normalized messages -> list[Segment] for DSv4."""
    msgs = list(sample.messages)
    tools = list(sample.tools) if sample.tools else []
    has_developer = any(m.role == "developer" for m in msgs)

    # History-thinking warn-loud: titaninfra supervises historical-turn
    # thinking by default (opposite of DeepSeek's reference
    # drop_thinking=True). Surface it so the user can opt into dropping
    # via StreamConfig.history_policy. Only fires when there is genuine
    # historical thinking AND no policy was supplied — a policy means the
    # user already made a deliberate choice (normalize.py applied it).
    if sample.history_policy is None:
        _n_hist_think = sum(
            1
            for nm in msgs
            if nm.turn_role == "history"
            and nm.role == "assistant"
            and any(isinstance(b, ThinkingBlock) for b in nm.blocks)
        )
        if _n_hist_think:
            warn_history_thinking_trained("deepseek_v4", _n_hist_think)

    # thinking_intent.mode=off projection — force every assistant turn
    # to render with bare ``</think>`` (no thinking content) by rebuilding
    # the message list with all ThinkingBlocks filtered out of blocks.
    # DSv4 wire always emits the ``</think>`` close marker regardless;
    # mode=off just strips the ``<think>{...}</think>`` body. effort is
    # not honored at the wire level (no reasoning_effort field); dispatch
    # warn-loud surfaces it.
    ti = sample.thinking_intent
    if ti is not None and ti.mode == "off":
        from dataclasses import replace as _dc_replace

        def _strip_thinking(nm):
            if nm.role != "assistant":
                return nm
            new_blocks = tuple(b for b in nm.blocks if not isinstance(b, ThinkingBlock))
            return _dc_replace(nm, blocks=new_blocks)

        msgs = [_strip_thinking(nm) for nm in msgs]

    segs: list[Segment] = []

    # BOS prelude is always emitted, even when no system message exists.
    segs.append(Segment(
        text=_BOS,
        token_ids=None,
        kind=SegKind.MARKER,
        role="system",
        is_generation=False,
        token_class=TokenClass.PROTOCOL,
        msg_index=None,
    ))

    prev_task: str | None = None
    for i, nm in enumerate(msgs):
        role = nm.role

        if role == "system":
            text = _block_text(nm.blocks)
            if tools and not has_developer:
                text = text + "\n\n" + _render_tools_str(tools)
            segs.append(Segment(
                text=text,
                token_ids=None,
                kind=SegKind.SYSTEM,
                role="system",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=i,
            ))
            prev_task = None
            continue

        if role == "latest_reminder":
            segs.append(Segment(
                text=f"{_LATEST_REMINDER}{_block_text(nm.blocks)}",
                token_ids=None,
                kind=SegKind.MARKER,
                role="latest_reminder",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=i,
            ))
            prev_task = None
            continue

        if role == "developer":
            text = f"{_USER}{_block_text(nm.blocks)}"
            if tools:
                text += "\n\n" + _render_tools_str(tools)
            segs.append(Segment(
                text=text,
                token_ids=None,
                kind=SegKind.CONTENT,
                role="developer",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=i,
            ))
            prev_task = None
            continue

        if role == "user":
            segs.append(Segment(
                text=f"{_USER}{_block_text(nm.blocks)}",
                token_ids=None,
                kind=SegKind.CONTENT,
                role="user",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=i,
            ))
            # Task transition: append <｜Assistant｜></think><｜task_token｜>.
            # This belongs to the user turn (no loss) so the next
            # assistant turn skips its own role marker.
            if nm.task is not None:
                task_str = _DS_TASK_SP_TOKENS.get(nm.task)
                if task_str is None:
                    raise ValueError(
                        f"Unknown DSv4 task: {nm.task!r}. "
                        f"Supported: {list(_DS_TASK_SP_TOKENS)}"
                    )
                segs.append(Segment(
                    text=f"{_ASSISTANT}{_THINK_E}{task_str}",
                    token_ids=None,
                    kind=SegKind.MARKER,
                    role="user",
                    is_generation=False,
                    token_class=TokenClass.PROTOCOL,
                    msg_index=i,
                ))
                prev_task = nm.task
            else:
                prev_task = None
            continue

        if role == "tool":
            # Tool result content from ToolResultBlock (or TextBlock
            # fallback for legacy bare-text tool messages).
            prev_role = msgs[i - 1].role if i > 0 else None
            opener = "" if prev_role == "tool" else _USER
            segs.append(Segment(
                text=f"{opener}<tool_result>{_extract_tool_result_text(nm.blocks)}</tool_result>",
                token_ids=None,
                kind=SegKind.TOOL_RESULT,
                role="tool",
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=i,
            ))
            prev_task = None
            continue

        if role == "assistant":
            segs.extend(_assistant_segments(nm, i, prev_task=prev_task))
            prev_task = None
            continue

        raise NotImplementedError(
            f"DeepSeekV4Encoder: unsupported role {role!r} at index {i}"
        )

    return segs


def _assistant_segments(
    nm,
    i: int,
    *,
    prev_task: str | None,
) -> list[Segment]:
    """One assistant turn -> [marker?, think?, body].

    Phase 3.7 (encoder-layer-refactor): walks ``nm.blocks`` directly.
    No more ``nm.text`` / ``nm.thinking`` / ``nm.tool_calls`` reads;
    all projections come from the structured Block tuple.
    """
    msg_weight = float(nm.weight)
    text_body = _block_text(nm.blocks)
    thinking = _extract_thinking(nm.blocks)
    tool_use_blocks = _extract_tool_use_blocks(nm.blocks)
    out: list[Segment] = []

    # Role marker + think prefix unless the prior user already emitted
    # the task transition (which includes <｜Assistant｜></think><task>).
    if prev_task is None:
        out.append(Segment(
            text=_ASSISTANT,
            token_ids=None,
            kind=SegKind.MARKER,
            role="assistant",
            is_generation=False,
            token_class=TokenClass.PROTOCOL,
            msg_index=i,
        ))
        if thinking is not None:
            think_text = f"{_THINK_S}{thinking}{_THINK_E}"
        else:
            # "skip thinking" -- bare </think> single-token marker. DSv4
            # always emits a thinking section; absence is rendered as the
            # close marker only.
            think_text = _THINK_E
        out.append(Segment(
            text=think_text,
            token_ids=None,
            kind=SegKind.THINKING,
            role="assistant",
            scale_factor=msg_weight,
            is_generation=False,
            token_class=TokenClass.DECODED,
            msg_index=i,
        ))

    # Body comes from TextBlocks; no need to strip any cross-family
    # ``<think>...</think>`` envelope (blocks-only IR has no envelope).
    # F1: the closing ``<｜end▁of▁sentence｜>`` is the model's OWN autonomous
    # stop signal — emit it as a dedicated STOP_MARKER segment (not folded
    # into CONTENT) so the stop-token representation is uniform across the
    # 7 families. _EOS is a special token (BPE never merges across it), so
    # the CONTENT|STOP_MARKER seam is stable.
    body = text_body + _render_dsml_tool_calls_text(tool_use_blocks)
    # Only emit CONTENT when there is body text — a thinking-only turn has
    # no body, and an empty segment would trip assemble()'s 0-token guard
    # (previously the folded _EOS kept it non-empty).
    if body:
        out.append(Segment(
            text=body,
            token_ids=None,
            kind=SegKind.CONTENT,
            role="assistant",
            scale_factor=msg_weight,
            is_generation=False,
            token_class=TokenClass.DECODED,
            msg_index=i,
        ))
    out.append(Segment(
        text=_EOS,
        token_ids=None,
        kind=SegKind.STOP_MARKER,
        role="assistant",
        scale_factor=msg_weight,
        is_generation=False,
        token_class=TokenClass.PROTOCOL,
        msg_index=i,
    ))
    return out


def _strip_leading_thinking_envelope(text: str, thinking: str | None) -> str:
    """[DEAD CODE — Phase 5] Removed when ``NormalizedMessage.text`` was deleted.

    Kept as a stub raising NotImplementedError so any straggler caller
    fails loud rather than silently producing the legacy behavior.
    """
    raise NotImplementedError(
        "_strip_leading_thinking_envelope was removed in Phase 5. "
        "Encoders now read nm.blocks directly; there is no <think> envelope "
        "embedded in nm.text to strip."
    )


def _segments_to_legacy_arrays(
    segments: list[Segment],
    tokenizer,
    *,
    train_on_inputs: bool,
    max_length: int | None,
) -> tuple[list[int], list[int]]:
    """Tokenize segments into the legacy ``(input_ids, labels)`` tuple.

    Legacy contract: BOS stays IGNORE even under ``train_on_inputs=True``
    (DSv4 never asks the model to predict the prelude marker). We mirror
    that by always masking the very first emitted token; the rest
    follows ``train_on_inputs`` semantics.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    for seg_idx, seg in enumerate(segments):
        if seg.token_ids is not None:
            ids = list(seg.token_ids)
        else:
            ids = list(tokenizer.encode(seg.text, add_special_tokens=False))
        input_ids.extend(ids)
        if seg_idx == 0:
            # BOS prelude: always IGNORE, regardless of train_on_inputs.
            labels.extend([IGNORE_INDEX] * len(ids))
        elif train_on_inputs:
            labels.extend(ids)
        elif seg.loss_scale > 0.0:
            labels.extend(ids)
        else:
            labels.extend([IGNORE_INDEX] * len(ids))
    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return input_ids, labels


__all__ = ["DeepSeekV4Encoder"]
