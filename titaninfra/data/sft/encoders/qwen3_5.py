"""Qwen3.5 (Qwen3-VL) SFT encoder — qwen3 text logic + vision (ImageBlock).

qwen3_5 is the ONLY titaninfra SFT family with vision support. It reuses every text
behavior of Qwen3Encoder and only overrides user-turn emission to interleave IMAGE
segments before the user text. Image tokens (vision_start + N×image_pad + vision_end)
are expanded by the Assembler and carry label=-100 (SegKind.IMAGE base_mask=0).
"""

from __future__ import annotations

from titaninfra.data.schema import ImageBlock
from titaninfra.data.sft.capabilities import FEATURE_IMAGE
from titaninfra.data.sft.encoders import register
from titaninfra.data.sft.encoders.qwen3 import (
    _IM_END,
    _IM_START,
    Qwen3Encoder,
    _block_text,
)
from titaninfra.data.sft.encoders.qwen3 import SUPPORTED_BLOCKS as _Q3_BLOCKS
from titaninfra.data.sft.encoders.qwen3 import SUPPORTED_FEATURES as _Q3_FEATURES
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass
from titaninfra.data.sft.types import NormalizedMessage

SUPPORTED_BLOCKS = _Q3_BLOCKS | frozenset({ImageBlock})
SUPPORTED_FEATURES = _Q3_FEATURES | frozenset({FEATURE_IMAGE})


@register("qwen3_5")
class Qwen3_5Encoder(Qwen3Encoder):
    """Qwen3.5 VLM SFT encoder: qwen3 text path + vision user turns."""

    name = "qwen3_5"

    def _user_segments(self, msg: NormalizedMessage, msg_index: int | None) -> list[Segment]:
        images = [b for b in msg.blocks if isinstance(b, ImageBlock)]
        if not images:
            return super()._user_segments(msg, msg_index)
        text = _block_text(msg.blocks)
        out: list[Segment] = [Segment(
            text=f"{_IM_START}user\n", token_ids=None, kind=SegKind.MARKER,
            role="user", is_generation=False, token_class=TokenClass.PROTOCOL,
            msg_index=msg_index,
        )]
        for img in images:
            out.append(Segment(
                text=None, token_ids=None, kind=SegKind.IMAGE, role="user",
                is_generation=False, token_class=TokenClass.PROTOCOL,
                image_source=img.source, msg_index=msg_index,
            ))
        out.append(Segment(
            text=f"{text}{_IM_END}\n", token_ids=None, kind=SegKind.CONTENT,
            role="user", is_generation=False, token_class=TokenClass.PROTOCOL,
            msg_index=msg_index,
        ))
        return out


__all__ = ["Qwen3_5Encoder"]
