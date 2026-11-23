"""Qwen2.5 dense Decoder with QKV biases and no QK normalization.

Position padding uses -1 and is clamped to zero before RoPE. Document masks
are inherited from Decoder's positions-first metadata handling, and the model
uses standard FSDP2 sharding declarations.
"""

from dataclasses import dataclass, field

import torch
import torch_remat as remat
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention import AttentionMetadataMap
from torchtitan.models.common.decoder import Decoder, TransformerBlock

from .state_dict_adapter import Qwen2_5StateDictAdapter


class Qwen2_5TransformerBlock(TransformerBlock):
    """Qwen2.5 Transformer Block — Dense only."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: AttentionMetadataMap | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        # Dense architecture, no MoE — accept and drop (see llama3/model.py).
        del padding_mask, aux_loss_denominator
        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # torch_remat: the residual add reads attn_out with bare ops, so it
        # must survive to backward even if attention's internals are
        # selectively dropped by an enclosing AC region (same pattern as
        # llama3/qwen3/gpt_oss's identical residual-add site).
        remat.recompute_needs_tensor(attn_out)
        h = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(h))
        # Trailing add, always saved: it saves nothing for backward, so
        # recompute replay can skip it and its inputs don't need pinning --
        # matches upstream's own llama3/qwen3/etc. convention for this exact
        # residual-add site.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(h, ffn_out)


class Qwen2_5Model(Decoder):
    """Qwen2.5 dense model with QKV-biased GQA."""

    state_dict_adapter_cls = Qwen2_5StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 5120
        vocab_size: int = 152064
        # Compile boundaries for this dense family's actual call graph: loss,
        # fused_binary_activation (SwiGLU), and rope (CosSinRoPE). MoE and MLA
        # regions do not apply.
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=["loss", "fused_binary_activation", "rope"]
            )
        )

        # Decoder.Config handles RoPE length and parallel validation.
        # Parameter counts and FLOPs use the inherited estimator; no
        # family-specific get_nparams_and_flops override is needed.

    def __init__(self, config: Config):
        super().__init__(config)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        # Streaming uses -1 for padding; clamp it before RoPE to avoid
        # negative-index cache access.
        if positions is not None:
            positions = positions.clamp(min=0)
        return super().forward(
            tokens,
            positions=positions,
            attention_metadata=attention_metadata,
            padding_mask=padding_mask,
            aux_loss_denominators=aux_loss_denominators,
        )
