"""Llama3 / Llama3.1 family SFT encoder (segment path, blocks-only).

Renders the exact byte string the Llama3 ``tokenizer_config.json``
chat_template emits, broken into Segments with per-piece loss_scale.

Key Llama3 template quirks:

* ``content | trim`` -> ``str.strip()`` on every message body
* ``bos_token`` prepended to the FIRST message only
* Template unconditionally appends a trailing ``<|start_header_id|>
  assistant<|end_header_id|>\\n\\n`` (no add_generation_prompt gate);
  this dangling assistant header is a class-C **METADATA** token —
  vLLM/SGLang inject it as a prompt prefix, so it is never produced
  by the model AND never trained on.

Out of scope: thinking, tool_use, tool-role messages. Llama3's chat_
template has no native protocol for these. Capability dispatch
warn-and-skips offending blocks before encode_segments runs.

Phase 3.2 (encoder-layer-refactor): walks ``nm.blocks`` directly.
``_build_hf_messages`` and ``nm.text``/``nm.thinking``/``nm.tool_calls``
reads are gone. Every Segment carries an explicit ``token_class``.
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
# Current encoder targets the Llama3.0 base subset: text-only multi-turn
# with optional system. Llama3.1 added ipython tool role, <|python_tag|>
# JSON tool calling, and Environment: ipython sys-prefix; those are
# OUT-OF-SCOPE here. If a sample carries thinking / tool_use / tool_result
# / parallel_tool_calls / tools_declaration / thinking_history / thinking_intent
# dispatch will warn-and-skip. Pick qwen3 if you need tool/thinking
# training, or implement a llama3_1 encoder for ipython tool wire.
SUPPORTED_BLOCKS = frozenset({TextBlock})
SUPPORTED_FEATURES = frozenset({
    FEATURE_SYSTEM,
    FEATURE_HISTORY_POLICY_DROP,
    FEATURE_LOSS_WEIGHT_PER_MESSAGE,
    FEATURE_MULTI_TURN,
})

_BOS = "<|begin_of_text|>"
_HEADER_START = "<|start_header_id|>"
_HEADER_END = "<|end_header_id|>"
_EOT = "<|eot_id|>"


@register("llama3")
class Llama3Encoder:
    """Token-first encoder for the Llama3 / Llama3.1 family."""

    name = "llama3"

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
        # type_overrides is intentionally unsupported (audit #12, 2026-06-06):
        # llama3 emits only TextBlocks and a binary assistant/non-assistant
        # loss model (scale=msg.weight for assistant, else 0.0) — there is no
        # per-block-type (thinking/tool_use/tool_result) scaling path for the
        # dict to apply to. FEATURE_TYPE_OVERRIDES is deliberately NOT declared
        # (declaring it without an implementation would be misleading); dispatch
        # therefore warn-louds if a sample carries type_overrides for this family.
        del type_overrides

        # Build the effective message list: prepend a synthesized system
        # message when SFTSample.system is set and there's no explicit
        # system in messages[0]. Track whether a synthesized prefix was
        # added so msg_index stays aligned to sample.messages indices.
        synthesized_system: NormalizedMessage | None = None
        first = sample.messages[0] if sample.messages else None
        if sample.system and (first is None or first.role != "system"):
            synthesized_system = NormalizedMessage(
                role="system",
                blocks=(TextBlock(text=sample.system),),
                weight=0.0,
                turn_role="current",
            )

        all_messages: list[tuple[int | None, NormalizedMessage]] = []
        if synthesized_system is not None:
            all_messages.append((None, synthesized_system))
        for i, m in enumerate(sample.messages):
            self._assert_role_supported(m)
            all_messages.append((i, m))

        segs: list[Segment] = []
        for sidx, (msg_index, msg) in enumerate(all_messages):
            body = self._block_text(msg.blocks).strip()
            is_first = sidx == 0
            scale = float(msg.weight) if msg.role == "assistant" else 0.0

            # Role marker (PROTOCOL): inference framework injects identically.
            # BOS prepended on the very first message of the sample.
            marker_text = f"{_HEADER_START}{msg.role}{_HEADER_END}\n\n"
            if is_first:
                marker_text = _BOS + marker_text
            segs.append(Segment(
                text=marker_text,
                token_ids=None,
                kind=SegKind.MARKER,
                role=msg.role,
                is_generation=False,
                token_class=TokenClass.PROTOCOL,
                msg_index=msg_index,
            ))

            # Body + EOT. EOT on assistant turns is class-B (model's
            # autonomous stop signal); on user/system it's class-A
            # (framework injected). loss_scale carries this distinction.
            kind = SegKind.SYSTEM if msg.role == "system" else SegKind.CONTENT
            if msg.role == "assistant":
                # Validate non-empty body for assistant turns — silently
                # training on a bare <|eot_id|> is a data error.
                if not body:
                    raise ValueError(
                        f"Llama3Encoder: assistant turn at msg_index={msg_index} "
                        "has no learnable text content (empty body after trim). "
                        "If the sample carries tool_calls / thinking, switch "
                        "to the qwen3 family."
                    )
                # F1: the assistant's closing <|eot_id|> is the model's OWN
                # autonomous stop signal (class B). Emit it as a dedicated
                # STOP_MARKER segment, not folded into CONTENT — uniform
                # stop-token shape across families. _EOT is a special token
                # so the CONTENT|STOP_MARKER seam is BPE-stable.
                segs.append(Segment(
                    text=body,
                    token_ids=None,
                    kind=kind,
                    role=msg.role,
                    scale_factor=scale,
                    is_generation=False,
                    token_class=TokenClass.DECODED,
                    msg_index=msg_index,
                ))
                segs.append(Segment(
                    text=_EOT,
                    token_ids=None,
                    kind=SegKind.STOP_MARKER,
                    role=msg.role,
                    scale_factor=scale,
                    is_generation=False,
                    token_class=TokenClass.PROTOCOL,
                    msg_index=msg_index,
                ))
            else:
                # Non-assistant EOT is class-A (framework-injected turn end),
                # not the model's stop signal — keep it folded, loss=0.
                segs.append(Segment(
                    text=f"{body}{_EOT}",
                    token_ids=None,
                    kind=kind,
                    role=msg.role,
                    is_generation=False,
                    token_class=TokenClass.PROTOCOL,
                    msg_index=msg_index,
                ))

        # Trailing dangling assistant header (Llama3 quirk: unconditional).
        # Class-C METADATA: vLLM/SGLang inject this token as a prompt
        # prefix at inference time — the model never has to *predict*
        # it. loss_scale=0.0; token_class=METADATA so RL rollout's
        # prompt builder can structurally delete it (not just mask).
        segs.append(Segment(
            text=f"{_HEADER_START}assistant{_HEADER_END}\n\n",
            token_ids=None,
            kind=SegKind.MARKER,
            role="assistant",
            is_generation=False,
            token_class=TokenClass.METADATA,
            msg_index=None,
        ))
        return segs

    @staticmethod
    def _assert_role_supported(msg: NormalizedMessage) -> None:
        if msg.role in ("system", "user", "assistant"):
            return
        raise NotImplementedError(
            f"Llama3Encoder does not handle role={msg.role!r}. "
            "Llama3 has no native tool / thinking protocol in this template; "
            "pick the qwen3 family if you need tool use."
        )

    @staticmethod
    def _block_text(blocks) -> str:
        """Concatenate the ``text`` field of every TextBlock; ignore others.

        Capability dispatch already warned-and-skipped non-TextBlock
        types before we got here.
        """
        return _blocks.block_text(blocks)


__all__ = ["Llama3Encoder"]
