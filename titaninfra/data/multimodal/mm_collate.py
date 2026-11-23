"""Collate multimodal Pretrain batches into TokenizedTrainingMicrobatch.

Text fields are concatenated into one flat token stream. Image patches are
also flattened to (total_patches, dim) without padding or image-wise stacking,
while grid_thw retains one entry per image in the same order as placeholders.
mrope_positions is built per sample and concatenated using the same layout as
input, labels, and positions. Text-only batches follow the same path without
image tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torchtitan.components.data.types import TokenizedTrainingMicrobatch


def _resolve_special_token_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve vision special-token IDs from the tokenizer vocabulary.

    Avoid importing model modules: these IDs are tokenizer wire constants and
    should be resolved from the same vocab used by the data pipeline.
    """
    vocab = tokenizer.get_vocab()
    return {
        "image_id": vocab["<|image_pad|>"],
        "video_id": vocab["<|video_pad|>"],
        "vision_start_id": vocab["<|vision_start|>"],
        "vision_end_id": vocab["<|vision_end|>"],
    }


class _MRopeStub:
    """Provide the vision_processor.config shape required by _build_mrope_positions.

    Upstream reads spatial_merge_size through that nested path. Only the fields
    consumed by the helper are modeled; no other collator state is emulated.
    """

    @dataclass
    class _VisionProcessorConfigStub:
        spatial_merge_size: int

    @dataclass
    class _VisionProcessorStub:
        config: _MRopeStub._VisionProcessorConfigStub

    def __init__(self, spatial_merge_size: int) -> None:
        self.vision_processor = self._VisionProcessorStub(
            config=self._VisionProcessorConfigStub(spatial_merge_size=spatial_merge_size)
        )


class MMCollator:
    """Collate text and multimodal batches into one flat microbatch.

    Text-only batches use the same concatenation path without image tensors.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        num_tokens_per_microbatch: int,
        max_context_length: int,
        pad_id: int,
        spatial_merge_size: int = 2,
        build_mrope_positions: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self._size = num_tokens_per_microbatch
        self._max_context_length = max_context_length
        self._pad_id = pad_id
        self.spatial_merge_size = spatial_merge_size
        self.build_mrope_positions = build_mrope_positions
        self._special_token_ids = _resolve_special_token_ids(tokenizer)

    def __call__(
        self, batch: list[tuple[dict[str, Any], torch.Tensor]]
    ) -> TokenizedTrainingMicrobatch:
        outputs = [b[0] for b in batch]
        labels_list = [b[1] for b in batch]
        has_positions = "positions" in outputs[0]
        has_weights = "loss_weights" in outputs[0]

        num_tokens = sum(len(o["input"]) for o in outputs)
        if num_tokens > self._size:
            raise ValueError(
                f"batch 的总 token 数 {num_tokens} 超过配置的 "
                f"num_tokens_per_microbatch={self._size}（每条样本理应已被 "
                "packer pad 到 seq_len，总数应恰好等于 "
                "local_batch_size * seq_len；请检查 batch_size/seq_len 配置）"
            )

        size = self._size
        input_ids = torch.full((size,), self._pad_id, dtype=torch.int64)
        labels = torch.full((size,), -100, dtype=torch.int64)
        positions = torch.full((size,), -1, dtype=torch.int64)
        loss_weights = torch.zeros((size,), dtype=torch.float32) if has_weights else None

        torch.cat([o["input"] for o in outputs], out=input_ids[:num_tokens])
        torch.cat(labels_list, out=labels[:num_tokens])
        if has_positions:
            torch.cat([o["positions"] for o in outputs], out=positions[:num_tokens])
        else:
            # If positions are absent, generate arange per sample so each
            # document starts at zero; padded positions remain -1.
            offset = 0
            for o in outputs:
                n = len(o["input"])
                positions[offset : offset + n] = torch.arange(n)
                offset += n
        if has_weights:
            torch.cat([o["loss_weights"] for o in outputs], out=loss_weights[:num_tokens])

        pad_len = size - num_tokens
        if pad_len:
            # Fill missing whole-sample padding with safe positions and mark
            # padding_mask so those values never affect loss or routing.
            positions[num_tokens:] = torch.arange(pad_len) % self._max_context_length

        padding_mask = positions == -1
        if pad_len:
            padding_mask[num_tokens:] = True

        loss_token_counts = (labels != -100).sum()
        routing_token_counts = torch.stack([(~padding_mask).sum()])

        model_kwargs: dict[str, Any] = {
            # Model forward expects special_tokens["image_id"] even for
            # text-only batches, so always provide the small dictionary.
            "special_tokens": dict(self._special_token_ids),
        }
        if has_weights:
            model_kwargs["loss_kwargs"] = {"loss_weights": loss_weights}

        has_image = any("pixel_values" in o for o in outputs)
        if has_image:
            per_image_patches: list[torch.Tensor] = []
            grid_thw_list: list[tuple[int, int, int]] = []
            per_sample_grid_thw: list[list[tuple[int, int, int]]] = []
            for o in outputs:
                pv = o.get("pixel_values")
                gthw = o.get("grid_thw")
                per_sample_grid_thw.append(list(gthw) if gthw else [])
                if pv is None or not gthw:
                    continue
                cursor = 0
                for (t, h, w) in gthw:
                    n_patch = t * h * w
                    per_image_patches.append(pv[cursor : cursor + n_patch])
                    grid_thw_list.append((t, h, w))
                    cursor += n_patch

            if per_image_patches:
                # Flatten patches without padding or image-wise stacking, as
                # required by upstream collate_images and vision_encoder.
                model_kwargs["pixel_values"] = torch.cat(per_image_patches, dim=0)
                model_kwargs["grid_thw"] = torch.tensor(grid_thw_list, dtype=torch.long)

                if self.build_mrope_positions:
                    # Upstream moved MultiModalCollator under components.data.
                    from torchtitan.components.data.multimodal.mm_collator import (
                        MultiModalCollator,
                    )

                    stub = _MRopeStub(self.spatial_merge_size)
                    mrope_chunks = []
                    for o, gthw in zip(outputs, per_sample_grid_thw, strict=True):
                        grid_thw_i = (
                            torch.tensor(gthw, dtype=torch.long)
                            if gthw
                            else torch.zeros((0, 3), dtype=torch.long)
                        )
                        mrope_chunks.append(
                            MultiModalCollator._build_mrope_positions(
                                stub,
                                o["input"],
                                grid_thw_i,
                                None,  # Video grids are not supported by this pipeline.
                                o.get("positions"),
                                image_token_id=self._special_token_ids["image_id"],
                                video_token_id=self._special_token_ids["video_id"],
                            )
                        )
                    mrope_flat = torch.cat(mrope_chunks, dim=0)
                    mrope_positions = torch.zeros((size, 3), dtype=mrope_flat.dtype)
                    mrope_positions[:num_tokens] = mrope_flat
                    if pad_len:
                        # Fill padding with safe positions on all three MRoPE
                        # axes; padding_mask excludes them from loss/routing.
                        pad_positions = torch.arange(pad_len) % self._max_context_length
                        mrope_positions[num_tokens:] = pad_positions.unsqueeze(-1).expand(
                            -1, 3
                        )
                    model_kwargs["mrope_positions"] = mrope_positions

        return TokenizedTrainingMicrobatch(
            input=input_ids,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
            loss_token_counts=loss_token_counts,
            routing_token_counts=routing_token_counts,
            model_kwargs=model_kwargs,
        )
