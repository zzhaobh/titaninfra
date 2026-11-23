"""
SFTEncoder Protocol — the per-family Python encoder contract.

An encoder is stateless and token-first:
    * stateless    — same input → same output, no hidden globals
    * token-first  — emits token ids directly, never round-trips
                     through a rendered string

Each registered family supplies one encoder. The encoder owns
protocol-level token layout (role markers, thinking wrappers, tool
shells, per-turn transition tokens) and per-token (label, position)
emission. It does NOT own:

    * BlockMask construction              — model.get_attention_masks()
    * cross-sample packing                — packer
    * sft_mask_mode / train_on_inputs     — boundary builder (upstream)
    * loss-weight cascade                 — boundary builder (upstream)

The boundary builder resolves all of those into per-message ``weight``
values on ``NormalizedMessage`` before the encoder runs, so the
encoder only expresses *protocol*, never *training policy*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from titaninfra.data.sft.types import EncodedSample, NormalizedSample

if TYPE_CHECKING:
    from titaninfra.data.sft.segment import Segment


@runtime_checkable
class SFTEncoder(Protocol):
    """Per-family encoder. One implementation per model family.

    Two methods make up the contract:

    * ``encode_segments`` — the **canonical** method. Emits the family's
      wire as a ``list[Segment]`` (role markers, thinking wrappers, tool
      shells, stop markers, content), each segment carrying its own loss
      and token-class metadata. RL rollout (``rl_prompt.py``) calls this
      directly to split prompt from response without tokenizing.
    * ``encode`` — a convenience wrapper that calls ``encode_segments``
      then ``assembler.assemble`` to produce one ``EncodedSample``. The
      SFT runtime (``dispatch.tokenize_sample``) goes through this path.

    New families MUST implement both. ``encode`` without
    ``encode_segments`` would type-check fine against an ``encode``-only
    Protocol yet crash the RL path at runtime — declaring both here makes
    that gap a static error instead.
    """

    name: str  # family identifier, e.g. "qwen3", "deepseek_v3"

    def encode_segments(
        self,
        sample: NormalizedSample,
        *,
        type_overrides: dict[str, float] | None = None,
    ) -> list[Segment]:
        """Emit the family wire as a ``list[Segment]`` (the canonical method).

        Walks ``sample.messages`` (and ``sample.system`` / ``sample.tools``)
        and produces one segment per logical wire unit. Each ``Segment``
        declares ``kind`` + ``role`` + ``token_class`` + ``scale_factor``;
        ``Segment.__post_init__`` derives the final ``loss_scale`` (see
        ``data/sft/segment.py``). ``encode`` delegates here.

        ``type_overrides`` carries ``LossConfig.type_overrides`` (e.g.
        ``{"thinking": 0.3}``) so the encoder can down-weight specific
        block species when emitting THINKING / TOOL_CALL segments.
        Families with no per-type dial accept and ignore it.
        """
        ...

    def encode(
        self,
        sample: NormalizedSample,
        *,
        max_length: int | None = None,
        loss_config: dict | None = None,
        truncate_at_last_segment: bool = False,
    ) -> EncodedSample:
        """Encode a normalized sample into (input_ids, labels, positions).

        Convenience wrapper: extracts ``type_overrides`` from
        ``loss_config``, calls ``encode_segments``, then ``assemble``.

        ``max_length`` truncates from the right and applies equally to
        all three arrays so they stay aligned. Encoders should refuse
        truncation in the middle of a tool-call shell or thinking
        block (raise ``ValueError``) rather than silently emitting a
        malformed sequence — or opt into segment-aligned truncation via
        ``truncate_at_last_segment``.

        ``loss_config`` carries per-token weight overrides
        (``type_overrides`` / ``StreamConfig.thinking_scale``) that
        surface back as ``EncodedSample.loss_weights``.
        """
        ...
