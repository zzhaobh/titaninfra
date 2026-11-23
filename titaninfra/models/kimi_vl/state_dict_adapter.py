"""Three-segment state_dict_adapter for the full Kimi-VL VLM checkpoint.

A Kimi-VL-A3B HF checkpoint carries three disjoint key namespaces:
  * ``vision_tower.*``          — MoonViT vision tower (formalized from the
                                   retired subproject-1 weight-transfer script).
  * ``language_model.*``        — DeepSeek-V3 language tower; DELEGATED to
                                   ``KimiVLLanguageStateDictAdapter`` (subproject 2),
                                   which strips the ``language_model.`` prefix and
                                   maps to the titaninfra TOP-LEVEL FQNs.
  * ``multi_modal_projector.*`` — patch-merge → text-hidden projector.

``from_hf`` splits incoming HF keys by prefix and routes each segment;
``to_hf`` reverses all three (re-adding the HF prefixes).

# GPU-VERIFY: The three HF TOP-LEVEL attribute names (``vision_tower`` /
# ``language_model`` / ``multi_modal_projector``) and the projector sub-attr
# names (``pre_norm`` / ``linear_1`` / ``linear_2``) are taken from
# modeling_kimi_vl.py per subproject-1/2 review, but the HF file is not
# importable on Mac. Confirm against modeling_kimi_vl.py on a GPU node before
# trusting a real full-VLM checkpoint load.

This module imports torchtitan (via the language delegate + the base
``StateDictAdapter``) and is therefore GPU-node-only; it is not importable on a
Mac without torchtitan/triton. The mapping dicts themselves are pure Python.
"""

from __future__ import annotations

import re
from typing import Any

from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.models.base.state_dict_adapter import StateDictAdapter

from titaninfra.models.deepseek_v3.state_dict_adapter import (
    KimiVLLanguageStateDictAdapter,
)

# ── HF prefixes that select each segment (from_hf) ──────────────────────────
_VISION_HF_PREFIX = "vision_tower."
_LANGUAGE_HF_PREFIX = "language_model."
_PROJECTOR_HF_PREFIX = "multi_modal_projector."

# ── titaninfra TOP-LEVEL sub-module attr names (KimiVLModel, Task 2) ───────────
_VISION_JD_PREFIX = "vision_tower."
_PROJECTOR_JD_PREFIX = "projector."

# ── Vision segment: HF (vision_tower.*) → titaninfra (vision_tower.*) ──────────
# Formalizes scripts/alignment/kimi_moonvit_weight_transfer.py (retired).
# HF nests blocks/final_layernorm under an extra ``encoder.`` level; titaninfra's
# MoonViT hangs ``blocks`` / ``final_layernorm`` directly under vision_tower.
# ``{}`` = block index. ``.weight``/``.bias`` enumerated explicitly (Conv2d /
# LayerNorm / Linear all carry both).
_VISION_FROM_HF_MAP: dict[str, str] = {
    # PatchEmbed — same nesting on both sides.
    "vision_tower.patch_embed.proj.weight": "vision_tower.patch_embed.proj.weight",
    "vision_tower.patch_embed.proj.bias": "vision_tower.patch_embed.proj.bias",
    "vision_tower.patch_embed.pos_emb.weight": "vision_tower.patch_embed.pos_emb.weight",
    # Encoder blocks — drop the ``encoder.`` level.
    "vision_tower.encoder.blocks.{}.norm0.weight": "vision_tower.blocks.{}.norm0.weight",
    "vision_tower.encoder.blocks.{}.norm0.bias": "vision_tower.blocks.{}.norm0.bias",
    "vision_tower.encoder.blocks.{}.norm1.weight": "vision_tower.blocks.{}.norm1.weight",
    "vision_tower.encoder.blocks.{}.norm1.bias": "vision_tower.blocks.{}.norm1.bias",
    "vision_tower.encoder.blocks.{}.wqkv.weight": "vision_tower.blocks.{}.wqkv.weight",
    "vision_tower.encoder.blocks.{}.wqkv.bias": "vision_tower.blocks.{}.wqkv.bias",
    "vision_tower.encoder.blocks.{}.wo.weight": "vision_tower.blocks.{}.wo.weight",
    "vision_tower.encoder.blocks.{}.wo.bias": "vision_tower.blocks.{}.wo.bias",
    "vision_tower.encoder.blocks.{}.mlp.fc0.weight": "vision_tower.blocks.{}.mlp.fc0.weight",
    "vision_tower.encoder.blocks.{}.mlp.fc0.bias": "vision_tower.blocks.{}.mlp.fc0.bias",
    "vision_tower.encoder.blocks.{}.mlp.fc1.weight": "vision_tower.blocks.{}.mlp.fc1.weight",
    "vision_tower.encoder.blocks.{}.mlp.fc1.bias": "vision_tower.blocks.{}.mlp.fc1.bias",
    # Final layernorm — drop the ``encoder.`` level.
    "vision_tower.encoder.final_layernorm.weight": "vision_tower.final_layernorm.weight",
    "vision_tower.encoder.final_layernorm.bias": "vision_tower.final_layernorm.bias",
}

# ── Projector segment: HF (multi_modal_projector.*) → titaninfra (projector.*) ─
# KimiVLMultiModalProjector: pre_norm(LayerNorm) / linear_1 / linear_2 (Linear,
# bias=True) — all carry weight + bias. No layer index.
_PROJECTOR_FROM_HF_MAP: dict[str, str] = {
    "multi_modal_projector.pre_norm.weight": "projector.pre_norm.weight",
    "multi_modal_projector.pre_norm.bias": "projector.pre_norm.bias",
    "multi_modal_projector.linear_1.weight": "projector.linear_1.weight",
    "multi_modal_projector.linear_1.bias": "projector.linear_1.bias",
    "multi_modal_projector.linear_2.weight": "projector.linear_2.weight",
    "multi_modal_projector.linear_2.bias": "projector.linear_2.bias",
}


def _abstract(key: str) -> str:
    """Collapse the first run of digits to ``{}`` (block-index abstraction)."""
    return re.sub(r"(\d+)", "{}", key, count=1)


class KimiVLStateDictAdapter(StateDictAdapter):
    """Three-segment adapter: vision + language(delegate) + projector.

    Mirrors the qwen3_5 composite-adapter shape (from_hf/to_hf via
    ``BaseStateDictAdapter``) but composes the language segment out of the
    already-landed ``KimiVLLanguageStateDictAdapter`` instead of reimplementing
    the DeepSeek-V3 MLA+MoE mapping. Vision / projector segments are pure
    prefix-map lookups (fault-tolerant: unmapped keys are skipped, matching the
    deepseek_v3 from_hf "skip unmapped keys" guard).
    """

    def __init__(
        self,
        model_config: Any,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        # Language segment delegate — reuses subproject-2's prefix-stripping +
        # DeepSeek-V3 MLA/MoE mapping. KimiVLModel.Config subclasses
        # DeepSeekV3Model.Config, so it satisfies the delegate's ctor contract.
        self._language = KimiVLLanguageStateDictAdapter(model_config, hf_assets_path)
        self._vision_from_hf = dict(_VISION_FROM_HF_MAP)
        self._projector_from_hf = dict(_PROJECTOR_FROM_HF_MAP)
        self._vision_to_hf = {v: k for k, v in self._vision_from_hf.items()}
        self._projector_to_hf = {v: k for k, v in self._projector_from_hf.items()}

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        # Defer to the language delegate (handles the DeepSeek-V3 quantized path).
        return self._language.get_hf_storage_reader(path, from_quantized)

    # ── from_hf: split HF keys by prefix, route each segment ────────────────
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        vision_hf: dict[str, Any] = {}
        language_hf: dict[str, Any] = {}
        projector_hf: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            if key.startswith(_VISION_HF_PREFIX):
                vision_hf[key] = value
            elif key.startswith(_LANGUAGE_HF_PREFIX):
                language_hf[key] = value
            elif key.startswith(_PROJECTOR_HF_PREFIX):
                projector_hf[key] = value
            # else: unmapped top-level key → skipped (fault-tolerant).

        state_dict: dict[str, Any] = {}
        # Language: delegate strips language_model. + maps to top-level FQNs.
        state_dict.update(self._language.from_hf(language_hf))
        # Vision + projector: flat prefix-map lookups.
        state_dict.update(self._map_flat(vision_hf, self._vision_from_hf))
        state_dict.update(self._map_flat(projector_hf, self._projector_from_hf))
        return state_dict

    # ── to_hf: reverse each segment, re-add HF prefixes ─────────────────────
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        vision_tt: dict[str, Any] = {}
        projector_tt: dict[str, Any] = {}
        language_tt: dict[str, Any] = {}

        for key, value in state_dict.items():
            if key.startswith(_VISION_JD_PREFIX):
                vision_tt[key] = value
            elif key.startswith(_PROJECTOR_JD_PREFIX):
                projector_tt[key] = value
            else:
                # Everything else is a language-tower top-level FQN
                # (tok_embeddings / layers / norm / lm_head).
                language_tt[key] = value

        hf_state_dict: dict[str, Any] = {}
        # Language: delegate re-adds language_model. prefix.
        hf_state_dict.update(self._language.to_hf(language_tt))
        # Vision + projector: flat reverse-map lookups.
        hf_state_dict.update(self._map_flat(vision_tt, self._vision_to_hf))
        hf_state_dict.update(self._map_flat(projector_tt, self._projector_to_hf))
        return hf_state_dict

    @staticmethod
    def _map_flat(
        src: dict[str, Any], mapping: dict[str, str]
    ) -> dict[str, Any]:
        """Map keys through ``mapping`` with ``{}`` block-index abstraction.

        Unmapped keys are skipped (fault-tolerant), mirroring the deepseek_v3
        from_hf guard.
        """
        out: dict[str, Any] = {}
        for key, value in src.items():
            if key in mapping:  # non-indexed key (patch_embed / projector / norm)
                out[mapping[key]] = value
                continue
            abstract = _abstract(key)
            if abstract not in mapping:
                continue  # unmapped → skip
            match = re.search(r"\d+", key)
            idx = match.group(0) if match else ""
            out[mapping[abstract].format(idx)] = value
        return out
