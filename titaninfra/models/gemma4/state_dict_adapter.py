"""Gemma4 Hugging Face to TorchTitan state-dict adapter.

Handles model.language_model and text-only prefixes, fused 3D expert tensors,
gate_up splitting into w1/w3, down projection into w2, full-attention K/V
sharing, normalization/layer-scalar buffers, and tied output embeddings.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.tensor import DTensor
from torchtitan.models.base.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from .model import Gemma4Model

logger = logging.getLogger(__name__)


_HF_PREFIX_MULTIMODAL = "model.language_model"
_HF_PREFIX_TEXT = "model"


def _detect_hf_prefix(hf_state_dict: dict[str, Any]) -> str:
    """Auto-detect whether the HF checkpoint uses multimodal or text-only prefix.

    Multimodal (PaliGemma2-style): keys start with "model.language_model."
    Text-only (gemma-4-26B-A4B): keys start with "model." without "language_model"
    """
    for key in hf_state_dict:
        if key.startswith(f"{_HF_PREFIX_MULTIMODAL}."):
            return _HF_PREFIX_MULTIMODAL
        if key.startswith(f"{_HF_PREFIX_TEXT}.") and "language_model" not in key:
            return _HF_PREFIX_TEXT
    return _HF_PREFIX_MULTIMODAL


def _build_from_hf_map(prefix: str) -> dict[str, str | None]:
    """Build the HF→TorchTitan key mapping for the given prefix."""
    L = f"{prefix}.layers.{{}}"
    return {
        f"{prefix}.embed_tokens.weight": "tok_embeddings.weight",
        f"{prefix}.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",

        f"{L}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
        f"{L}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
        f"{L}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
        f"{L}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        f"{L}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
        f"{L}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",

        f"{L}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        f"{L}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        f"{L}.pre_feedforward_layernorm.weight": "layers.{}.pre_feedforward_layernorm.weight",
        f"{L}.post_feedforward_layernorm.weight": "layers.{}.post_feedforward_layernorm.weight",
        f"{L}.post_feedforward_layernorm_1.weight": "layers.{}.post_feedforward_layernorm_1.weight",
        f"{L}.pre_feedforward_layernorm_2.weight": "layers.{}.pre_feedforward_layernorm_2.weight",
        f"{L}.post_feedforward_layernorm_2.weight": "layers.{}.post_feedforward_layernorm_2.weight",

        f"{L}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        f"{L}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        f"{L}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",

        f"{L}.router.proj.weight": "layers.{}.moe.router.proj.weight",
        f"{L}.router.scale": "layers.{}.moe.router.scale",
        f"{L}.router.per_expert_scale": "layers.{}.moe.router.per_expert_scale",

        f"{L}.experts.gate_up_proj": None,  # split into w1 + w3
        f"{L}.experts.down_proj": "layers.{}.moe.experts.w2",

        f"{L}.layer_scalar": "layers.{}.layer_scalar",
    }


class Gemma4StateDictAdapter(StateDictAdapter):
    """Convert Gemma4 weights between HF and TorchTitan layouts."""

    def __init__(self, model_config: Gemma4Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self._load_prefix = self._detect_prefix_from_index()

    def _detect_prefix_from_index(self) -> str:
        """Detect the HF prefix from the safetensors index built by the base class.

        Multimodal checkpoints have both model.language_model.* AND model.vision_tower.*
        keys. Must check for multimodal prefix across ALL keys first — a vision_tower
        key would falsely match the text-only pattern if checked per-key.
        """
        if self.fqn_to_index_mapping:
            if any(k.startswith(f"{_HF_PREFIX_MULTIMODAL}.") for k in self.fqn_to_index_mapping):
                logger.info(f"Gemma4 safetensors prefix: '{_HF_PREFIX_MULTIMODAL}'")
                return _HF_PREFIX_MULTIMODAL
            if any(k.startswith(f"{_HF_PREFIX_TEXT}.") for k in self.fqn_to_index_mapping):
                logger.info(f"Gemma4 safetensors prefix: '{_HF_PREFIX_TEXT}'")
                return _HF_PREFIX_TEXT
        logger.info("Gemma4 safetensors prefix: defaulting to multimodal (no index)")
        return _HF_PREFIX_MULTIMODAL

    @staticmethod
    def _abstract_layer_key(key: str) -> str:
        return re.sub(r"\.(\d+)\.", ".{}.", key, count=1)

    @staticmethod
    def _layer_index(key: str) -> str | None:
        # (?:^|\.) rather than a bare leading "\." — native FQNs are
        # top-level ("layers.0.attention...", no leading dot), so a
        # leading-dot-only pattern never matches and _layer_index always
        # returns None on every native key passed to to_hf(). Confirmed a
        # genuine pre-existing bug (predates this migration — untouched by
        # it until this round's roundtrip verification actually exercised
        # to_hf() on a real built model's full state_dict for the first
        # time; no prior test ever did). The HF-side abstract keys DO have
        # a prefix before "layers." (e.g. "model.language_model.layers.{}
        # ...") and were never affected — only native-side lookups broke.
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
        return m.group(1) if m else None

    @staticmethod
    def _to_local(value: Any) -> torch.Tensor:
        if isinstance(value, DTensor):
            return value.to_local()
        return value

    # ------------------------------------------------------------------
    # HF → TorchTitan
    # ------------------------------------------------------------------
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}

        prefix = _detect_hf_prefix(hf_state_dict)
        from_hf_map = _build_from_hf_map(prefix)
        logger.info(f"Gemma4 HF checkpoint prefix detected: '{prefix}'")

        # Weight tying: lm_head not always saved
        embed_key = f"{prefix}.embed_tokens.weight"
        if "lm_head.weight" not in hf_state_dict and embed_key in hf_state_dict:
            hf_state_dict = dict(hf_state_dict)
            hf_state_dict["lm_head.weight"] = hf_state_dict[embed_key]

        gate_up_abstract = f"{prefix}.layers.{{}}.experts.gate_up_proj"

        for key, value in hf_state_dict.items():
            if key.startswith("model.vision_tower") or key.startswith("model.embed_vision"):
                continue

            if ".layers." in key:
                abstract = self._abstract_layer_key(key)
                layer_num = self._layer_index(key)
                if layer_num is None or abstract not in from_hf_map:
                    logger.warning(f"Skipping unknown HF key: {key}")
                    continue

                # gate_up_proj: [E, 2*H, D] → w1 [E, H, D] + w3 [E, H, D]
                if abstract == gate_up_abstract:
                    if isinstance(value, DTensor):
                        local_t = value._local_tensor
                        half = local_t.shape[1] // 2
                        w1_local = local_t[:, :half, :].contiguous()
                        w3_local = local_t[:, half:, :].contiguous()
                        state_dict[f"layers.{layer_num}.moe.experts.w1"] = DTensor.from_local(
                            w1_local, device_mesh=value.device_mesh,
                            placements=value.placements, run_check=False,
                        )
                        state_dict[f"layers.{layer_num}.moe.experts.w3"] = DTensor.from_local(
                            w3_local, device_mesh=value.device_mesh,
                            placements=value.placements, run_check=False,
                        )
                    else:
                        half = value.shape[1] // 2
                        state_dict[f"layers.{layer_num}.moe.experts.w1"] = value[:, :half, :].contiguous()
                        state_dict[f"layers.{layer_num}.moe.experts.w3"] = value[:, half:, :].contiguous()
                    continue

                target = from_hf_map[abstract]
                if target is None:
                    continue
                state_dict[target.format(layer_num)] = value

            elif key in from_hf_map:
                target = from_hf_map[key]
                if target is None:
                    continue
                state_dict[target] = value
            else:
                logger.warning(f"Skipping unknown HF key: {key}")

        return state_dict

    @staticmethod
    def _fuse_gate_up(w1: Any, w3: Any) -> Any:
        """Fuse w1 and w3 into gate_up_proj [E, 2*H, D].

        Preserves DTensor structure when running under EP so that DCP can
        shard-load from the full safetensors tensor.
        """
        if isinstance(w1, DTensor):
            local_fused = torch.cat([w1._local_tensor, w3._local_tensor], dim=1).contiguous()
            return DTensor.from_local(
                local_fused,
                device_mesh=w1.device_mesh,
                placements=w1.placements,
                run_check=False,
            )
        return torch.cat([w1, w3], dim=1).contiguous()

    # ------------------------------------------------------------------
    # TorchTitan → HF
    # ------------------------------------------------------------------
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        prefix = self._load_prefix
        export_map = _build_from_hf_map(prefix)
        to_hf_map: dict[str, str] = {
            v: k for k, v in export_map.items() if v is not None
        }
        hf_state_dict: dict[str, Any] = {}
        pending_gate_up: dict[str, dict[str, Any]] = {}

        for key, value in state_dict.items():
            # Expert w1/w3 → fuse into gate_up_proj
            if "moe.experts.w1" in key or "moe.experts.w3" in key:
                layer_num = self._layer_index(key)
                if layer_num is None:
                    continue
                slot = pending_gate_up.setdefault(layer_num, {})
                slot["w1" if key.endswith(".w1") else "w3"] = value
                if "w1" in slot and "w3" in slot:
                    fused = self._fuse_gate_up(slot["w1"], slot["w3"])
                    hf_state_dict[f"{prefix}.layers.{layer_num}.experts.gate_up_proj"] = fused
                    del pending_gate_up[layer_num]
                continue

            # Expert w2 → down_proj (keep DTensor for DCP shard-load)
            if "moe.experts.w2" in key:
                layer_num = self._layer_index(key)
                if layer_num is None:
                    continue
                hf_state_dict[f"{prefix}.layers.{layer_num}.experts.down_proj"] = value
                continue

            # Layer-scoped keys
            if "layers." in key:
                abstract = self._abstract_layer_key(key)
                layer_num = self._layer_index(key)
                if abstract in to_hf_map and layer_num is not None:
                    hf_state_dict[to_hf_map[abstract].format(layer_num)] = value
                continue

            # Weight tying: skip lm_head.weight (HF uses embed_tokens only)
            if key == "lm_head.weight":
                continue
            if key in to_hf_map:
                hf_state_dict[to_hf_map[key]] = value

        if pending_gate_up:
            raise ValueError(
                f"Incomplete expert w1/w3 pairs for layers: {list(pending_gate_up.keys())}"
            )

        return hf_state_dict
