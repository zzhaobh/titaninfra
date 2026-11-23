"""Qwen2.5 Hugging Face to TorchTitan state-dict adapter.

This dense adapter uses StateDictAdapter rather than the MoE variant and maps
Qwen2.5's QKV projection biases without QK normalization entries.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from torchtitan.models.base.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from .model import Qwen2_5Model

logger = logging.getLogger(__name__)


class Qwen2_5StateDictAdapter(StateDictAdapter):
    """Map dense Qwen2.5 weights, including QKV biases, between formats."""

    def __init__(self, model_config: Qwen2_5Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        self.from_hf_map = {
            # Embedding
            "model.embed_tokens.weight": "tok_embeddings.weight",

            # Attention — weight
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",

            # Attention biases, a Qwen2.5-specific architectural detail.
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.qkv_linear.wq.bias",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.qkv_linear.wk.bias",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.qkv_linear.wv.bias",

            # Rotary embedding (ignored)
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,

            # FFN
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",

            # Layer Norm
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",

            # Output
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan format to HuggingFace format."""
        state_dict = self._native_fused_linears_to_hf(state_dict)
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)

                if abstract_key not in to_hf_map:
                    continue

                new_key = to_hf_map[abstract_key].format(layer_num)
                hf_state_dict[new_key] = value
            else:
                if key not in to_hf_map:
                    continue
                hf_state_dict[to_hf_map[key]] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace format to TorchTitan format."""
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                match = re.search(r"\d+", key)
                if match is None:
                    logger.warning(f"Skipping HF key with no layer index: {key}")
                    continue
                layer_num = match.group(0)

                if abstract_key not in self.from_hf_map:
                    logger.warning(f"Skipping unknown HF key: {key} (abstract: {abstract_key})")
                    continue
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                state_dict[new_key.format(layer_num)] = value
            else:
                if key not in self.from_hf_map:
                    logger.warning(f"Skipping unknown HF key: {key}")
                    continue
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
                state_dict[new_key] = value

        return self._native_fused_linears_from_hf(state_dict)
