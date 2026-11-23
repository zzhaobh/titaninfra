"""Kimi-VL projector, vision-position scatter, and top-level VLM.

The projector and scatter are pure torch and CPU-testable. KimiVLModel imports
TorchTitan only inside a guarded section so optional dependencies do not break
these utilities on CPU-only hosts.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KimiVLMultiModalProjector(nn.Module):
    """Project patch-merged MoonViT features into the language hidden size."""

    def __init__(self, vision_hidden: int = 1152, merge_kernel=(2, 2), text_hidden: int = 2048, ln_eps: float = 1e-5):
        super().__init__()
        self.vision_hidden = vision_hidden
        self.hidden_size = vision_hidden * merge_kernel[0] * merge_kernel[1]  # 4608
        self.pre_norm = nn.LayerNorm(vision_hidden, eps=ln_eps)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_2 = nn.Linear(self.hidden_size, text_hidden, bias=True)

    def forward(self, image_features: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(image_features, dim=0)  # (Σn_i, hidden_size=4608)
        # HF: pre_norm on vision_hidden view, then reshape to merged hidden_size.
        # GPU-VERIFY: HF pre_norm normalized_shape (1152 vs 4608) + view order needs
        # confirming against modeling_kimi_vl.py before GPU parity. Implemented per brief
        # (pre_norm on vision_hidden=1152) — HF LayerNorm(vision_config.hidden_size).
        x = self.pre_norm(x.view(-1, self.vision_hidden)).view(-1, self.hidden_size)
        x = self.linear_2(F.gelu(self.linear_1(x)))
        return x  # (Σn_i, text_hidden)


def get_vision_positions(
    tokens: torch.Tensor,
    num_vision_tokens_per_item: torch.Tensor,
    placeholder_id: int,
) -> list[tuple[int, int, int]]:
    """Locate each image's placeholder run in a flat ``(T,)`` token sequence.

    Pure-torch, no torchtitan dependency (mirrors the validation discipline of
    ``torchtitan.models.common.multimodal.get_vision_positions`` — kept local
    rather than imported so this module's projector/scatter surface stays
    Mac-testable without torchtitan/triton, matching this file's existing
    design).

    Args:
        tokens: ``(T,)`` token IDs (flat, no batch dim).
        num_vision_tokens_per_item: ``(num_images,)`` post-merge token count
            per image, in the order the images appear in ``tokens``.
        placeholder_id: token id whose contiguous runs mark image spans.

    Returns:
        ``(item_idx, start, n_tokens)`` per image.

    Raises:
        ValueError: the number of placeholder runs doesn't match the number
            of images, or a run's length doesn't match that image's token
            count — text/vision misalignment that would otherwise silently
            corrupt the scatter.
    """
    vision_mask = tokens == placeholder_id
    prev_mask = torch.zeros_like(vision_mask)
    prev_mask[1:] = vision_mask[:-1]
    next_mask = torch.zeros_like(vision_mask)
    next_mask[:-1] = vision_mask[1:]
    region_starts = torch.where(vision_mask & ~prev_mask)[0]
    region_ends = torch.where(vision_mask & ~next_mask)[0]

    num_items = int(num_vision_tokens_per_item.shape[0])
    num_runs = int(region_starts.shape[0])
    if num_runs != num_items:
        raise ValueError(
            f"scatter misalignment: found {num_runs} contiguous run(s) of "
            f"placeholder id {placeholder_id} but received {num_items} "
            f"image(s). Each image must correspond to exactly one placeholder run."
        )

    region_starts_list = region_starts.tolist()
    run_lengths = (region_ends - region_starts + 1).tolist()
    num_vision_tokens_per_item_list = num_vision_tokens_per_item.tolist()
    positions: list[tuple[int, int, int]] = []
    for i in range(num_items):
        start = int(region_starts_list[i])
        n_tokens = int(num_vision_tokens_per_item_list[i])
        if run_lengths[i] != n_tokens:
            raise ValueError(
                f"scatter misalignment: placeholder run {i} spans "
                f"{run_lengths[i]} token(s) but image {i} produced "
                f"{n_tokens} embedding(s)."
            )
        positions.append((i, start, n_tokens))
    return positions


def scatter_vision_embeds(
    inputs_embeds: torch.Tensor,
    *,
    vision_embeds: torch.Tensor,
    vision_positions: list[tuple[int, int, int]],
) -> torch.Tensor:
    """Copy packed vision features into a flat ``(T, D)`` text sequence.

    Returns a copy of ``inputs_embeds``; the input is left unchanged.

    Args:
        inputs_embeds: ``(T, D)`` text embeddings (flat, no batch dim).
        vision_embeds: packed vision features ``(total_tokens, D)``.
        vision_positions: from ``get_vision_positions``.
    """
    inputs_embeds = inputs_embeds.clone()
    vision_offset = 0
    for _, start, n_tokens in vision_positions:
        inputs_embeds[start : start + n_tokens] = vision_embeds[
            vision_offset : vision_offset + n_tokens
        ].to(inputs_embeds.dtype)
        vision_offset += n_tokens

    if vision_offset != vision_embeds.shape[0]:
        raise ValueError(
            f"scatter mismatch: placeholder runs consume {vision_offset} "
            f"embeddings but vision output contains {vision_embeds.shape[0]}."
        )
    return inputs_embeds


# ─────────────────────────────────────────────────────────────────────────────
# KimiVLModel top-level VLM composition. Guard TorchTitan imports so CPU-only
# projector/scatter tests still work without TorchTitan or Triton. When the
# dependencies are missing, KimiVLModel is intentionally undefined and structural
# integration tests skip rather than failing during import.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from dataclasses import dataclass

    from torchtitan.models.common.attention import AttentionMetadataMap

    from titaninfra.models.deepseek_v3.model import DeepSeekV3Model
    from titaninfra.models.kimi_vl.state_dict_adapter import KimiVLStateDictAdapter
    from titaninfra.models.kimi_vl.vision_encoder import MoonViT

    class KimiVLModel(DeepSeekV3Model):
        """Kimi-VL model composed of MoonViT, a projector, and a DeepSeek-V3 tower.

The language tower subclasses DeepSeekV3Model so its weights remain at the
top-level FQN expected by the language adapter. Vision features are projected
and scattered into placeholder spans before delegating to forward_embeddings.
MoonViT and the projector are bare nn.Module rather than torchtitan Module
configs, which is why protocol validation explicitly excludes them.
"""

        # Place vision_tower and projector with tok_embeddings on the first PP
        # stage through BaseModel's declarative first-stage module list.
        pipeline_first_stage_module_fqns = ("vision_tower", "projector")

        # Kimi-VL checkpoints contain vision_tower, language_model, and
        # multi_modal_projector sections, so use the three-part adapter.
        state_dict_adapter_cls = KimiVLStateDictAdapter

        @dataclass(kw_only=True, slots=True)
        class Config(DeepSeekV3Model.Config):
            # Media placeholder token ID used to locate vision scatter spans.
            media_placeholder_token_id: int

            # MoonViT parameters are constructed directly as bare nn.Module,
            # using MOONVIT_CONFIG as the source of truth.
            vision_hidden: int = 1152
            vision_num_layers: int = 27
            vision_num_heads: int = 16
            vision_mlp_dim: int = 4304
            vision_patch_size: int = 14
            vision_merge_kernel: tuple = (2, 2)
            vision_init_pos_emb_hw: int = 14
            vision_attn_impl: str = "flex"

            # Projector input is vision_hidden * merge[0] * merge[1]; output
            # defaults to the decoder hidden dimension.
            projector_text_hidden: int | None = None
            projector_ln_eps: float = 1e-5

        def __init__(self, config: KimiVLModel.Config):
            # The base DeepSeekV3Model constructor builds decoder components.
            super().__init__(config)

            self.media_placeholder_token_id = config.media_placeholder_token_id

            self.vision_tower = MoonViT(
                hidden_dim=config.vision_hidden,
                num_layers=config.vision_num_layers,
                num_heads=config.vision_num_heads,
                mlp_dim=config.vision_mlp_dim,
                patch_size=config.vision_patch_size,
                merge_kernel_size=config.vision_merge_kernel,
                init_pos_emb_hw=config.vision_init_pos_emb_hw,
                attn_impl=config.vision_attn_impl,
            )

            text_hidden = config.projector_text_hidden or config.dim
            self.projector = KimiVLMultiModalProjector(
                vision_hidden=config.vision_hidden,
                merge_kernel=config.vision_merge_kernel,
                text_hidden=text_hidden,
                ln_eps=config.projector_ln_eps,
            )

        def verify_module_protocol(self) -> None:
            """Validate the language tower only, skipping bare vision modules.

            MoonViT and the projector are plain nn.Module rather than
            torchtitan Module implementations, so BaseModel's protocol check
            intentionally excludes those two subtrees.
            """
            from torchtitan.models.base.module import Module

            failures: list[tuple[str, str]] = []
            for fqn, mod in self.named_modules():
                if fqn.startswith("vision_tower") or fqn.startswith("projector"):
                    continue
                if not isinstance(mod, Module):
                    failures.append((fqn, type(mod).__name__))
            if failures:
                details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
                raise RuntimeError(
                    f"The following modules do not satisfy the Module protocol: {details}"
                )

        def _apply_fsdp(self, *, parallelism_context, precision, parallelism) -> None:
            """vision_tower FSDP before decoder FSDP — see parallelize.py's
            ``apply_fsdp_to_kimi_vl`` docstring for the full rationale (mirrors
            upstream's ``MultimodalModel._apply_fsdp``/``MTPDecoder._apply_fsdp``
            override pattern: this is the sanctioned hook, not a free-standing
            ``parallelize_fn`` — that concept no longer exists post-ModelSpec
            removal)."""
            from titaninfra.models.kimi_vl.parallelize import apply_fsdp_to_kimi_vl

            apply_fsdp_to_kimi_vl(
                self,
                parallelism_context=parallelism_context,
                precision=precision,
                parallelism=parallelism,
            )
            super()._apply_fsdp(
                parallelism_context=parallelism_context,
                precision=precision,
                parallelism=parallelism,
            )

        def forward(  # pyrefly: ignore [bad-override]
            self,
            tokens: torch.Tensor,
            *,
            pixel_values: torch.Tensor | None = None,
            image_grid_hws: torch.Tensor | None = None,
            attention_metadata: AttentionMetadataMap | None = None,
            positions: torch.Tensor | None = None,
            padding_mask: torch.Tensor | None = None,
            aux_loss_denominators: torch.Tensor | None = None,
        ):
            """Forward pass for Kimi-VL-A3B.

            Args:
                tokens: ``(T,)`` packed token IDs (flat, no batch dim).
                pixel_values: packed image patches, or None for text-only.
                image_grid_hws: ``(num_images, 2)`` patch grid ``[h, w]`` per image.
                attention_metadata: per-layer attention metadata map (from
                    ``self._get_attention_metadata(positions)``), matching the
                    ``Decoder``/``DeepSeekV3Model`` contract.
                positions: per-token position IDs for packed sequences;
                    ``torch.arange(T)`` if omitted (single-document convenience).
                padding_mask: threaded into MoE token dispatch.
                aux_loss_denominators: per-depth routing-token counts (MTP; kimi_vl
                    has no MTP layers, so this is always None in practice).

            Returns:
                ``(T, vocab_size)`` logits.
            """
            # 1) Compute text embeddings.
            embeds = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

            # 2) For image inputs, run MoonViT and the projector, then scatter
            # features into placeholder positions. Text-only batches skip this.
            if pixel_values is not None:
                assert image_grid_hws is not None, "pixel_values 需配套 image_grid_hws"
                vis_list = self.vision_tower(pixel_values, image_grid_hws)
                num_vision_tokens_per_item = torch.tensor(
                    [v.shape[0] for v in vis_list], device=tokens.device
                )
                feats = self.projector(vis_list)
                vision_positions = get_vision_positions(
                    tokens, num_vision_tokens_per_item, self.media_placeholder_token_id
                )
                embeds = scatter_vision_embeds(
                    embeds, vision_embeds=feats, vision_positions=vision_positions
                )

            # 3) Use sequential 1D positions when the caller does not supply them.
            if positions is None:
                positions = torch.arange(tokens.shape[0], device=tokens.device)

            # 4) Delegate decoder layers, norm, and lm_head to the base
            # forward_embeddings implementation with vision-enriched embeds.
            return self.forward_embeddings(
                embeds,
                positions,
                attention_metadata,
                padding_mask=padding_mask,
                aux_loss_denominators=aux_loss_denominators,
            )

except ImportError as _exc:  # Optional TorchTitan/Triton dependencies are absent.
    logging.getLogger(__name__).info(
        "KimiVLModel 未定义：缺依赖 %s（Mac 无 torchtitan/triton 属正常；GPU 节点触发则为真实 import 错误）",
        getattr(_exc, "name", "") or _exc,
    )
