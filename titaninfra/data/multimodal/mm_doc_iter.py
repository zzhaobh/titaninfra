"""Build interleaved multimodal Pretrain documents from MDS samples.

For each image position, emit vision_start, N image_pad tokens with label -100,
and vision_end before the next text segment. decode_pool preprocesses images in
parallel, while pixel values are concatenated and grid metadata is preserved.

Invariant: N = grid_thw.prod() // merge^2 must agree with the offline estimate
and preprocess_image's internal validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from titaninfra.data.packer import MMDoc

if TYPE_CHECKING:
    from torchtitan.components.tokenizer import BaseTokenizer

    from titaninfra.data.multimodal.decode_pool import ImageDecodePool


def _resolve_vision_token_ids(tokenizer: BaseTokenizer) -> tuple[int, int, int]:
    """Resolve vision_start/image_pad/vision_end IDs or raise ValueError."""
    vocab = tokenizer.get_vocab()
    try:
        return (
            vocab["<|vision_start|>"],
            vocab["<|image_pad|>"],
            vocab["<|vision_end|>"],
        )
    except KeyError as e:
        raise ValueError(
            f"tokenizer 词表缺少 vision 特殊 token {e}；多模态 pretrain 需 qwen3_5 系 tokenizer"
        ) from e


def _grid_to_n_tokens(grid_thw: tuple[int, int, int], spatial_merge_size: int) -> int:
    """Return image_pad token count N = grid_t*h*w // merge^2."""
    grid_t, grid_h, grid_w = grid_thw
    return (grid_t * grid_h * grid_w) // (spatial_merge_size * spatial_merge_size)


def _build_interleaved_doc(
    text_segments: list[str],
    image_positions: list[int],
    decoded: list[tuple[Any, Any]],
    tokenizer: BaseTokenizer,
    vision_ids: tuple[int, int, int],
    *,
    spatial_merge_size: int,
    add_bos_eos: bool,
) -> MMDoc | None:
    """Rebuild one interleaved document as tokens and image tensors.

    Each position p inserts all images assigned to p before text_segments[p].
    Position N means after the final segment. Empty documents return None.
    """
    import torch

    vision_start_id, image_pad_id, vision_end_id = vision_ids
    n_segs = len(text_segments)

    tokens: list[tuple[int, int]] = []
    pixel_values_parts: list[Any] = []
    grid_thw_all: list[tuple[int, int, int]] = []

    if add_bos_eos and tokenizer.bos_id is not None:
        tokens.append((tokenizer.bos_id, tokenizer.bos_id))

    img_cursor = 0
    for p in range(n_segs + 1):
        # Emit all images assigned to insertion point p in their original order.
        while img_cursor < len(image_positions) and image_positions[img_cursor] == p:
            pixel_values, grid_thw = decoded[img_cursor]
            grid_tuple = tuple(int(x) for x in grid_thw.tolist())
            n_tokens = _grid_to_n_tokens(grid_tuple, spatial_merge_size)
            # Vision markers and image pads never contribute to loss.
            tokens.append((vision_start_id, -100))
            tokens.extend((image_pad_id, -100) for _ in range(n_tokens))
            tokens.append((vision_end_id, -100))
            pixel_values_parts.append(pixel_values)
            grid_thw_all.append(grid_tuple)
            img_cursor += 1

        if p < n_segs:
            seg_tokens = tokenizer.encode(text_segments[p], add_bos=False, add_eos=False)
            tokens.extend((t, t) for t in seg_tokens)

    if add_bos_eos and tokenizer.eos_id is not None:
        tokens.append((tokenizer.eos_id, tokenizer.eos_id))

    if not tokens:
        return None
    if not pixel_values_parts:
        return MMDoc(tokens=tokens, images=(None, []))

    pixel_values_cat = torch.cat(pixel_values_parts, dim=0)
    return MMDoc(tokens=tokens, images=(pixel_values_cat, grid_thw_all))


def _make_multimodal_pretrain_doc_iter(
    streaming_ds,
    tokenizer: BaseTokenizer,
    decode_pool: ImageDecodePool,
    *,
    spatial_merge_size: int = 2,
    add_bos_eos: bool = True,
) -> Iterator[MMDoc]:
    """Yield multimodal MMDoc values from one StreamingDataset.

    Every invocation creates a fresh generator and a new decode-pool lifetime,
    allowing PackedStream to restart iteration safely.
    """
    vision_ids = _resolve_vision_token_ids(tokenizer)

    def _images_of(sample: dict) -> list[bytes]:
        return list(sample.get("images", []) or [])

    with decode_pool:
        for sample, decoded in decode_pool.map_docs(iter(streaming_ds), _images_of):
            text_segments = list(sample.get("text_segments", []) or [])
            image_positions = list(sample.get("image_positions", []) or [])
            doc = _build_interleaved_doc(
                text_segments,
                image_positions,
                decoded,
                tokenizer,
                vision_ids,
                spatial_merge_size=spatial_merge_size,
                add_bos_eos=add_bos_eos,
            )
            if doc is not None:
                yield doc
