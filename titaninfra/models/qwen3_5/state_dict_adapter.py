# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Qwen3.5.

Converts between HuggingFace Qwen3.5 checkpoint format and torchtitan format.

GatedDeltaNet's native fused ``in_proj`` is packed/unpacked via the base
class's generic ``_native_fused_linears_to_hf``/``_native_fused_linears_from_hf``
(``torchtitan/models/base/state_dict_adapter.py``), which finds every
``GDN.Config`` instance in the model config tree and uses its own
``in_proj_split_sizes`` to do the per-key-head-interleaved pack/unpack — this
file only handles the HF-facing rename of the resulting logical
``in_proj_q/k/v`` keys (fusing them into HF's single ``in_proj_qkv``;
``in_proj_z/a/b`` map 1:1, no fusion needed).

MoE expert weights: HF ``gate_up_proj`` (fused ``[E, 2F, D]``) <-> TT ``w13``
(``[E, 2, F, D]``) via flatten/unflatten on dim 1 (not a real split/cat,
just a reshape — same tensor, no transpose). HF ``down_proj`` and TT ``w2``
share the same ``[E, D, F]`` layout.

Other notable conversions:
- Conv3d patch embedding (HF) <-> Linear (TT) via weight reshape
- Vision block naming: HF ``blocks`` <-> TT ``layers``
- Vision QKV: HF fused qkv <-> TT separate wq/wk/wv
- GatedDeltaNet Conv1d: HF fused conv1d <-> TT separate conv_q/k/v
"""

import re
from typing import TYPE_CHECKING, Any

import torch
from torchtitan.models.base.state_dict_adapter import StateDictAdapter
from torchtitan.models.common.multimodal import MultimodalModel

if TYPE_CHECKING:
    from .model import Qwen35Model


class Qwen35StateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: "Qwen35Model.Config", hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        has_vision_encoder = isinstance(model_config, MultimodalModel.Config)
        # Multimodal HF checkpoints nest the decoder under ``model.language_model``.
        # A text-only flavor (no vision encoder) can still load one, so read the
        # layout from the checkpoint index when there is one.
        if self.fqn_to_index_mapping:
            multimodal_checkpoint = any(
                key.startswith("model.language_model.")
                for key in self.fqn_to_index_mapping
            )
        else:
            multimodal_checkpoint = has_vision_encoder
        self.hf_language_model_prefix = (
            "model.language_model" if multimodal_checkpoint else "model"
        )

        self.from_hf_map = {
            # ===== Language Model =====
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            # Full attention layers (self_attn.*)
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.language_model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # GatedDeltaNet layers (linear_attn.*)
            # QKV and Conv1d: HF fused -> TT separate logical keys (handled in to_hf/from_hf)
            "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight": None,
            "model.language_model.layers.{}.linear_attn.conv1d.weight": None,
            "model.language_model.layers.{}.linear_attn.in_proj_z.weight": "layers.{}.attention.in_proj_z.weight",
            "model.language_model.layers.{}.linear_attn.in_proj_a.weight": "layers.{}.attention.in_proj_a.weight",
            "model.language_model.layers.{}.linear_attn.in_proj_b.weight": "layers.{}.attention.in_proj_b.weight",
            "model.language_model.layers.{}.linear_attn.A_log": "layers.{}.attention.A_log",
            "model.language_model.layers.{}.linear_attn.dt_bias": "layers.{}.attention.dt_bias",
            "model.language_model.layers.{}.linear_attn.norm.weight": "layers.{}.attention.norm.weight",
            "model.language_model.layers.{}.linear_attn.out_proj.weight": "layers.{}.attention.out_proj.weight",
            # Non-MoE MLP
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Layer norms
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE (grouped 3D format, handled specially in to_hf/from_hf)
            "model.language_model.layers.{}.mlp.experts.down_proj": "layers.{}.feed_forward.routed_experts.w2.weight",
            "model.language_model.layers.{}.mlp.gate.weight": "layers.{}.feed_forward.router.gate.weight",
            # MoE shared expert
            "model.language_model.layers.{}.mlp.shared_expert.gate_proj.weight": "layers.{}.feed_forward.shared_experts.w1.weight",  # noqa: E501
            "model.language_model.layers.{}.mlp.shared_expert.up_proj.weight": "layers.{}.feed_forward.shared_experts.w3.weight",  # noqa: E501
            "model.language_model.layers.{}.mlp.shared_expert.down_proj.weight": "layers.{}.feed_forward.shared_experts.w2.weight",  # noqa: E501
            "model.language_model.layers.{}.mlp.shared_expert_gate.weight": "layers.{}.feed_forward.shared_experts.gate.weight",  # noqa: E501
            # Final norm and output
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            # ===== Vision Encoder =====
            # Patch embedding (Conv3d in HF, Linear in TT — weight reshape needed)
            "model.visual.patch_embed.proj.weight": "vision_encoder.patch_embed.weight",
            "model.visual.patch_embed.proj.bias": "vision_encoder.patch_embed.bias",
            # Position embeddings
            "model.visual.pos_embed.weight": "vision_encoder.pos_embed",
            # Vision transformer blocks (HF: blocks, TT: layers)
            "model.visual.blocks.{}.norm1.weight": "vision_encoder.layers.{}.norm1.weight",
            "model.visual.blocks.{}.norm1.bias": "vision_encoder.layers.{}.norm1.bias",
            "model.visual.blocks.{}.norm2.weight": "vision_encoder.layers.{}.norm2.weight",
            "model.visual.blocks.{}.norm2.bias": "vision_encoder.layers.{}.norm2.bias",
            # Vision QKV: HF fused -> TT separate (handled in to_hf/from_hf)
            "model.visual.blocks.{}.attn.qkv.weight": None,
            "model.visual.blocks.{}.attn.qkv.bias": None,
            "model.visual.blocks.{}.attn.proj.weight": "vision_encoder.layers.{}.attn.proj.weight",
            "model.visual.blocks.{}.attn.proj.bias": "vision_encoder.layers.{}.attn.proj.bias",
            "model.visual.blocks.{}.mlp.linear_fc1.weight": "vision_encoder.layers.{}.mlp.linear_fc1.weight",
            "model.visual.blocks.{}.mlp.linear_fc1.bias": "vision_encoder.layers.{}.mlp.linear_fc1.bias",
            "model.visual.blocks.{}.mlp.linear_fc2.weight": "vision_encoder.layers.{}.mlp.linear_fc2.weight",
            "model.visual.blocks.{}.mlp.linear_fc2.bias": "vision_encoder.layers.{}.mlp.linear_fc2.bias",
            # Merger
            "model.visual.merger.norm.weight": "vision_encoder.merger.norm.weight",
            "model.visual.merger.norm.bias": "vision_encoder.merger.norm.bias",
            "model.visual.merger.linear_fc1.weight": "vision_encoder.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.bias": "vision_encoder.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2.weight": "vision_encoder.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.bias": "vision_encoder.merger.linear_fc2.bias",
        }
        if not has_vision_encoder:
            self.from_hf_map = {
                hf_key: tt_key
                for hf_key, tt_key in self.from_hf_map.items()
                if not hf_key.startswith("model.visual.")
            }
        if not multimodal_checkpoint:
            # Text-only checkpoints use model.* instead of model.language_model.*.
            self.from_hf_map = {
                hf_key.replace("model.language_model.", "model.", 1): tt_key
                for hf_key, tt_key in self.from_hf_map.items()
            }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HuggingFace Qwen3.5 format."""
        # Unpacks GDN's native fused in_proj into logical in_proj_q/k/v/z/a/b
        # (base-class generic helper, keyed on GDN.Config.in_proj_split_sizes).
        state_dict = self._native_fused_linears_to_hf(state_dict)
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        vision_qkv_by_layer: dict[str, dict[str, Any]] = {}
        deltanet_qkv_by_layer: dict[str, dict[str, Any]] = {}

        for tt_key, value in state_dict.items():
            if ".feed_forward.routed_experts." in tt_key:
                tt_abstract_key = re.sub(r"(\d+)", "{}", tt_key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", tt_key).group(0)

                if tt_abstract_key == "layers.{}.feed_forward.routed_experts.w13.weight":
                    # w13 [E,2,F,D] -> HF gate_up_proj [E,2F,D]: reshape, no transpose.
                    hf_state_dict[
                        f"{self.hf_language_model_prefix}.layers."
                        f"{layer_num}.mlp.experts.gate_up_proj"
                    ] = value.flatten(1, 2)
                    continue
                elif tt_abstract_key == "layers.{}.feed_forward.routed_experts.w2.weight":
                    hf_key = (
                        f"{self.hf_language_model_prefix}.layers."
                        f"{layer_num}.mlp.experts.down_proj"
                    )
                    hf_state_dict[hf_key] = value
                    continue

                if tt_abstract_key not in to_hf_map:
                    continue
                hf_state_dict[to_hf_map[tt_abstract_key].format(layer_num)] = value

            elif re.search(r"\.\d+\.", tt_key):
                tt_abstract_key = re.sub(r"(\d+)", "{}", tt_key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", tt_key).group(0)

                # Collect deltanet q/k/v projections and conv weights for fusing
                if tt_abstract_key in (
                    "layers.{}.attention.in_proj_q.weight",
                    "layers.{}.attention.in_proj_k.weight",
                    "layers.{}.attention.in_proj_v.weight",
                    "layers.{}.attention.conv_q.weight",
                    "layers.{}.attention.conv_k.weight",
                    "layers.{}.attention.conv_v.weight",
                ):
                    if layer_num not in deltanet_qkv_by_layer:
                        deltanet_qkv_by_layer[layer_num] = {}
                    short_key = tt_abstract_key.split("attention.")[-1].replace("{}", "")
                    deltanet_qkv_by_layer[layer_num][short_key] = value
                    continue

                # Collect vision wq/wk/wv for fusing into qkv
                if tt_abstract_key in (
                    "vision_encoder.layers.{}.attn.wq.weight",
                    "vision_encoder.layers.{}.attn.wq.bias",
                    "vision_encoder.layers.{}.attn.wk.weight",
                    "vision_encoder.layers.{}.attn.wk.bias",
                    "vision_encoder.layers.{}.attn.wv.weight",
                    "vision_encoder.layers.{}.attn.wv.bias",
                ):
                    if layer_num not in vision_qkv_by_layer:
                        vision_qkv_by_layer[layer_num] = {}
                    short_key = tt_abstract_key.split("attn.")[-1].replace("{}", "")
                    vision_qkv_by_layer[layer_num][short_key] = value
                    continue

                if tt_abstract_key not in to_hf_map:
                    continue
                hf_state_dict[to_hf_map[tt_abstract_key].format(layer_num)] = value

            else:
                if tt_key not in to_hf_map:
                    continue
                if tt_key == "lm_head.weight" and getattr(
                    self.model_config, "enable_weight_tying", False
                ):
                    if self.fqn_to_index_mapping:
                        self.fqn_to_index_mapping.pop("lm_head.weight", None)
                    continue
                hf_value = value
                # Linear weight (out, C*T*H*W) -> Conv3d weight (out, C, T, H, W)
                if tt_key == "vision_encoder.patch_embed.weight":
                    # pyrefly: ignore [missing-attribute]
                    encoder = self.model_config.vision_encoder
                    hf_value = value.reshape(
                        value.shape[0],
                        encoder.in_channels,
                        encoder.temporal_patch_size,
                        encoder.patch_size,
                        encoder.patch_size,
                    )
                hf_state_dict[to_hf_map[tt_key]] = hf_value

        # Fuse vision wq/wk/wv -> qkv
        for layer_num, parts in vision_qkv_by_layer.items():
            for suffix in ("weight", "bias"):
                q = parts.get(f"wq.{suffix}")
                k = parts.get(f"wk.{suffix}")
                v = parts.get(f"wv.{suffix}")
                if q is not None and k is not None and v is not None:
                    hf_state_dict[
                        f"model.visual.blocks.{layer_num}.attn.qkv.{suffix}"
                    ] = torch.cat([q, k, v], dim=0)

        # Fuse deltanet in_proj_q/k/v -> in_proj_qkv, conv_q/k/v -> conv1d
        for layer_num, parts in deltanet_qkv_by_layer.items():
            q = parts.get("in_proj_q.weight")
            k = parts.get("in_proj_k.weight")
            v = parts.get("in_proj_v.weight")
            if q is not None and k is not None and v is not None:
                hf_state_dict[
                    f"{self.hf_language_model_prefix}.layers.{layer_num}.linear_attn.in_proj_qkv.weight"
                ] = torch.cat([q, k, v], dim=0)
            cq = parts.get("conv_q.weight")
            ck = parts.get("conv_k.weight")
            cv = parts.get("conv_v.weight")
            if cq is not None and ck is not None and cv is not None:
                hf_state_dict[
                    f"{self.hf_language_model_prefix}.layers.{layer_num}.linear_attn.conv1d.weight"
                ] = torch.cat([cq, ck, cv], dim=0)

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace Qwen3.5 state dict to torchtitan format."""
        tt_state_dict = {}

        # HF ties lm_head with embed_tokens — copy if missing
        if "lm_head.weight" not in hf_state_dict:
            embed_key = f"{self.hf_language_model_prefix}.embed_tokens.weight"
            if embed_key not in hf_state_dict:
                raise ValueError(
                    f"HF checkpoint missing both 'lm_head.weight' and '{embed_key}'"
                )
            hf_state_dict["lm_head.weight"] = hf_state_dict[embed_key]

        for hf_key, value in hf_state_dict.items():
            if re.search(r"\.\d+\.", hf_key):
                hf_abstract_key = re.sub(r"(\d+)", "{}", hf_key, count=1)
                # pyrefly: ignore [missing-attribute]
                idx = re.search(r"\d+", hf_key).group(0)

                # MoE gate_up_proj [E,2F,D] -> TT w13 [E,2,F,D]: reshape, no transpose.
                if (
                    hf_abstract_key
                    == f"{self.hf_language_model_prefix}.layers.{{}}.mlp.experts.gate_up_proj"
                ):
                    tt_state_dict[
                        f"layers.{idx}.feed_forward.routed_experts.w13.weight"
                    ] = value.unflatten(1, (2, -1))
                    continue

                # MoE down_proj [E,D,F] == TT w2 [E,D,F]: same layout, no transpose.
                if (
                    hf_abstract_key
                    == f"{self.hf_language_model_prefix}.layers.{{}}.mlp.experts.down_proj"
                ):
                    tt_state_dict[
                        f"layers.{idx}.feed_forward.routed_experts.w2.weight"
                    ] = value
                    continue

                # GatedDeltaNet fused in_proj_qkv -> split into logical q/k/v
                if (
                    hf_abstract_key
                    == f"{self.hf_language_model_prefix}.layers.{{}}.linear_attn.in_proj_qkv.weight"
                ):
                    # pyrefly: ignore [missing-attribute]
                    dn = self.model_config.layers[int(idx)].attention
                    kd = dn.n_key_heads * dn.key_head_dim
                    vd = dn.n_heads * dn.value_head_dim
                    q, k, v = value.split([kd, kd, vd], dim=0)
                    tt_state_dict[f"layers.{idx}.attention.in_proj_q.weight"] = q
                    tt_state_dict[f"layers.{idx}.attention.in_proj_k.weight"] = k
                    tt_state_dict[f"layers.{idx}.attention.in_proj_v.weight"] = v
                    continue

                # GatedDeltaNet fused conv1d -> split into conv_q/k/v
                if (
                    hf_abstract_key
                    == f"{self.hf_language_model_prefix}.layers.{{}}.linear_attn.conv1d.weight"
                ):
                    # pyrefly: ignore [missing-attribute]
                    dn = self.model_config.layers[int(idx)].attention
                    kd = dn.n_key_heads * dn.key_head_dim
                    vd = dn.n_heads * dn.value_head_dim
                    cq, ck, cv = value.split([kd, kd, vd], dim=0)
                    tt_state_dict[f"layers.{idx}.attention.conv_q.weight"] = cq
                    tt_state_dict[f"layers.{idx}.attention.conv_k.weight"] = ck
                    tt_state_dict[f"layers.{idx}.attention.conv_v.weight"] = cv
                    continue

                # Vision fused QKV -> split into wq/wk/wv
                if hf_abstract_key in (
                    "model.visual.blocks.{}.attn.qkv.weight",
                    "model.visual.blocks.{}.attn.qkv.bias",
                ):
                    suffix = "weight" if "weight" in hf_abstract_key else "bias"
                    q, k, v = value.chunk(3, dim=0)
                    tt_state_dict[f"vision_encoder.layers.{idx}.attn.wq.{suffix}"] = q
                    tt_state_dict[f"vision_encoder.layers.{idx}.attn.wk.{suffix}"] = k
                    tt_state_dict[f"vision_encoder.layers.{idx}.attn.wv.{suffix}"] = v
                    continue

                if hf_abstract_key not in self.from_hf_map:
                    continue
                tt_key = self.from_hf_map[hf_abstract_key]
                if tt_key is None:
                    continue
                tt_state_dict[tt_key.format(idx)] = value

            else:
                if hf_key not in self.from_hf_map:
                    continue
                tt_key = self.from_hf_map[hf_key]
                if tt_key is None:
                    continue
                tt_value = value
                # Conv3d weight (out, C, T, H, W) -> Linear weight (out, C*T*H*W)
                if hf_key == "model.visual.patch_embed.proj.weight":
                    tt_value = value.reshape(value.shape[0], -1)
                tt_state_dict[tt_key] = tt_value

        # Re-fuses logical in_proj_q/k/v/z/a/b into GDN's native fused in_proj
        # (base-class generic helper, keyed on GDN.Config.in_proj_split_sizes).
        return self._native_fused_linears_from_hf(tt_state_dict)
