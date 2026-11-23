"""Test-only identity encoder — proves the oracle harness end-to-end.

``IdentityEncoder`` implements the ``SFTEncoder`` Protocol but
delegates every decision back to ``tokenize_with_hf_template`` (the
legacy Jinja path). When registered as the ``native`` encoder for
a family, ``mode='shadow'`` will compare jinja-vs-jinja and the
oracle must report ``ok=True``.

Plain text only by design: thinking blocks, tool_calls, and
tool-role messages all raise ``NotImplementedError``. The point is
to validate the harness wiring before M1.2's real Qwen3Encoder
lands, NOT to handle features.

Phase 5 (blocks-only): reads ``nm.blocks`` directly. Legacy
``nm.text`` / ``nm.thinking`` / ``nm.tool_calls`` fields no longer
exist.
"""

from __future__ import annotations

from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from titaninfra.data.schema import TextBlock, ThinkingBlock, ToolUseBlock
from titaninfra.data.sft.types import EncodedSample, NormalizedMessage, NormalizedSample

# Test-only registry name. Leading underscore signals "not for prod".
IDENTITY_NAME = "_identity_qwen3"


class IdentityEncoder:
    """``NormalizedSample`` → HF messages → ``tokenize_with_hf_template``.

    The reconstruction is intentionally lossy: only ``role`` and ``text``
    survive. If the sample carries any feature the encoder cannot faithfully
    reproduce (thinking / tool_calls / tool messages), we fail loudly
    rather than silently produce a wrong baseline.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(
        self,
        sample: NormalizedSample,
        *,
        max_length: int | None = None,
        loss_config: dict | None = None,  # accepted for Protocol parity; intentionally unused
        truncate_at_last_segment: bool = False,  # accepted for Protocol parity; intentionally unused
    ) -> EncodedSample:
        hf_messages = [self._to_hf_message(m) for m in sample.messages]
        tools = list(sample.tools) if sample.tools else None
        result = tokenize_with_hf_template(
            hf_messages,
            self.tokenizer,
            max_length=max_length,
            system=sample.system,
            tools=tools,
        )
        input_ids, labels = result[0], result[1]
        ids = tuple(int(t) for t in input_ids)
        lbl = tuple(int(t) for t in labels)
        return EncodedSample(
            input_ids=ids,
            labels=lbl,
            positions=tuple(range(len(ids))),
        )

    @staticmethod
    def _to_hf_message(msg: NormalizedMessage) -> dict:
        if any(isinstance(b, ThinkingBlock) for b in msg.blocks):
            raise NotImplementedError(
                "IdentityEncoder cannot reconstruct ``thinking`` blocks. "
                "Use a real native encoder for fixtures with thinking."
            )
        if any(isinstance(b, ToolUseBlock) for b in msg.blocks):
            raise NotImplementedError(
                "IdentityEncoder cannot reconstruct ``tool_calls``. "
                "Use a real native encoder for fixtures with tool use."
            )
        if msg.role == "tool":
            raise NotImplementedError(
                "IdentityEncoder cannot reconstruct ``tool`` role messages. "
                "Use a real native encoder for fixtures with tool results."
            )
        text = "".join(b.text for b in msg.blocks if isinstance(b, TextBlock))
        return {"role": msg.role, "content": text}


__all__ = ["IDENTITY_NAME", "IdentityEncoder"]
