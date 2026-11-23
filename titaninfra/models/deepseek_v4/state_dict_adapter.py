# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""HF ↔ titaninfra state_dict adapter for DeepSeek-V4.

Bidirectional weight-name (and MoE-layout) mapping between HuggingFace
``transformers.models.deepseek_v4`` and titaninfra's native V4 model.

**NOT the upstream torchtitan adapter** (deliberate, see
docs/torchtitan_upgrade_plan_2026-11.md §35): upstream's
``torchtitan/models/deepseek_v4/state_dict_adapter.py`` targets a DIFFERENT,
non-HF checkpoint convention (bare ``layers.{}.attn.*``/``embed.weight``, no
``model.`` prefix) that is the RAW published Hub safetensors layout for
``deepseek-ai/DeepSeek-V4-Flash`` -- NOT what loading through the real
``transformers.models.deepseek_v4.DeepseekV4ForCausalLM`` Python class
produces. titaninfra's compressor.py/indexer.py test suites instantiate that
real class directly for their HF-oracle parity checks, so THIS adapter must
keep targeting that class's own ``state_dict()`` key convention
(``model.layers.{}.self_attn.*``, confirmed by direct instantiation of
``transformers`` 5.20.2's real class -- not assumed) to stay consistent with
them. Verified this round by instantiating the real class with a small debug
config and diffing its full ``state_dict()`` key list against both
conventions.

Structural correspondence (re-verified this round against the real
transformers class, see above; unchanged across the model-space migration
for the pieces NOT discussed below -- compressor.py/indexer.py's own
submodule structure is untouched, §34.5):

* MLA:        HF ``self_attn.{q_a_proj,q_a_norm,q_b_proj,kv_proj,kv_norm,
              o_a_proj,o_b_proj,sinks}`` ↔ titaninfra ``attention.{wq_a,q_norm,
              wq_b,wkv,kv_norm,wo_a,wo_b,attn_sink}`` (byte-identical shapes
              once ``o_groups`` matches).
* Compressor: HF ``self_attn.compressor.*`` ↔ titaninfra ``attention.compressor.*``
* Indexer:    HF ``self_attn.compressor.indexer.*`` ↔ titaninfra
              ``attention.indexer.*``; the HF ``...indexer.scorer.weights_proj``
              maps to titaninfra ``attention.indexer.weights_proj`` (one extra
              ``scorer.`` level on the HF side).
* HC:         HF ``attn_hc.{fn,base,scale}`` / ``ffn_hc.*`` ↔ titaninfra
              ``hc_attn_{fn,base,scale}`` / ``hc_ffn_*``; top-level
              ``model.hc_head.hc_{fn,base,scale}`` ↔ ``hc_head_{fn,base,scale}``.
* Norms:      ``input_layernorm``↔``attention_norm``,
              ``post_attention_layernorm``↔``ffn_norm``.
* Dense FFN:  ``mlp.{gate_proj,up_proj,down_proj}`` ↔
              ``feed_forward.{w13,w13,w2}`` (``w13`` stores gate+up stacked,
              shape ``[2,F,D]`` — upstream fused the two Linears; handled by
              explicit stack/unbind code below, not the straight-rename table).
              **Field-path change from the pre-migration adapter**: the
              Decoder's unified ``feed_forward`` field now holds the
              dense-FFN module DIRECTLY (no separate ``.moe`` attribute to
              disambiguate dense vs MoE layers any more — ``isinstance(layer.
              feed_forward, MoE)`` is the new discriminator, matching every
              other already-migrated family).
* MoE:        HF packs experts as batched Parameters ``mlp.experts.gate_up_proj
              (E, 2F, D)`` and ``mlp.experts.down_proj (E, D, F)``. titaninfra's
              NEW native layout (post-migration) is ALSO a single fused
              Parameter ``routed_experts.w13 (E, 2, F, D)`` / ``w2 (E, D, F)``
              (``GroupedLinear``'s ``num_linears=2`` fusion) — so the
              conversion is now a PURE RESHAPE (``flatten(1,2)`` to_hf /
              ``unflatten(1, (2,-1))`` from_hf, no transpose), not the old
              per-expert SPLIT into separate ``w1_EFD``/``w3_EFD`` tensors
              (the old native layout, pre-migration, kept gate/up as two
              independent Parameters). ``w2``/``down_proj`` share the same
              ``[E, D, F]`` layout on both sides — a straight rename, no
              reshape needed (unchanged from before this migration).
* Router:     ``mlp.gate.weight``↔``feed_forward.router.gate.weight``;
              ``mlp.gate.tid2eid``↔``feed_forward.router.tid2eid`` (hash-layer
              buffer); ``mlp.gate.e_score_correction_bias``↔
              ``feed_forward.expert_bias_E``.
* Shared:     ``mlp.shared_experts.{gate,up,down}_proj``↔
              ``feed_forward.shared_experts.{w13,w13,w2}`` (same gate/up
              STACK fusion as dense FFN — shared experts are a
              ``DenseFeedForward``, not routed, so they use the stack/unbind
              convention, not the routed-experts reshape convention above).
* MTP:        titaninfra has ``mtp.N.*`` blocks (structural only); HF
              inference checkpoints carry no MTP weights. from_hf leaves them to
              param_init (skipped); to_hf drops them (not exported).
"""

import re
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

if TYPE_CHECKING:
    from .model import DeepSeekV4Model

# Per-layer HF→titaninfra name templates. ``{}`` is the layer index. Expert
# weights are handled separately (batched reshape), not via this table.
_FROM_HF_LAYER_MAP = {
    # MLA
    "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
    "model.layers.{}.self_attn.q_a_norm.weight": "layers.{}.attention.q_norm.weight",
    "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
    "model.layers.{}.self_attn.kv_proj.weight": "layers.{}.attention.wkv.weight",
    "model.layers.{}.self_attn.kv_norm.weight": "layers.{}.attention.kv_norm.weight",
    "model.layers.{}.self_attn.o_a_proj.weight": "layers.{}.attention.wo_a.weight",
    "model.layers.{}.self_attn.o_b_proj.weight": "layers.{}.attention.wo_b.weight",
    "model.layers.{}.self_attn.sinks": "layers.{}.attention.attn_sink",
    # Compressor
    "model.layers.{}.self_attn.compressor.kv_proj.weight": "layers.{}.attention.compressor.kv_proj.weight",
    "model.layers.{}.self_attn.compressor.kv_norm.weight": "layers.{}.attention.compressor.kv_norm.weight",
    "model.layers.{}.self_attn.compressor.gate_proj.weight": "layers.{}.attention.compressor.gate_proj.weight",
    "model.layers.{}.self_attn.compressor.position_bias": "layers.{}.attention.compressor.position_bias",
    # Indexer (CSA layers only)
    "model.layers.{}.self_attn.compressor.indexer.kv_proj.weight": "layers.{}.attention.indexer.kv_proj.weight",
    "model.layers.{}.self_attn.compressor.indexer.kv_norm.weight": "layers.{}.attention.indexer.kv_norm.weight",
    "model.layers.{}.self_attn.compressor.indexer.gate_proj.weight": "layers.{}.attention.indexer.gate_proj.weight",
    "model.layers.{}.self_attn.compressor.indexer.position_bias": "layers.{}.attention.indexer.position_bias",
    "model.layers.{}.self_attn.compressor.indexer.q_b_proj.weight": "layers.{}.attention.indexer.q_b_proj.weight",
    "model.layers.{}.self_attn.compressor.indexer.scorer.weights_proj.weight": "layers.{}.attention.indexer.weights_proj.weight",  # noqa: E501
    # HC
    "model.layers.{}.attn_hc.fn": "layers.{}.hc_attn_fn",
    "model.layers.{}.attn_hc.base": "layers.{}.hc_attn_base",
    "model.layers.{}.attn_hc.scale": "layers.{}.hc_attn_scale",
    "model.layers.{}.ffn_hc.fn": "layers.{}.hc_ffn_fn",
    "model.layers.{}.ffn_hc.base": "layers.{}.hc_ffn_base",
    "model.layers.{}.ffn_hc.scale": "layers.{}.hc_ffn_scale",
    # Norms
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    # Dense FFN down_proj (gate/up are handled separately — stacked into w13).
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    # MoE router + shared experts (shared gate/up handled separately too).
    "model.layers.{}.mlp.gate.weight": "layers.{}.feed_forward.router.gate.weight",
    "model.layers.{}.mlp.gate.tid2eid": "layers.{}.feed_forward.router.tid2eid",
    "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.feed_forward.expert_bias_E",
    "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.feed_forward.shared_experts.w2.weight",
}

# Global (non-layer) name map.
_FROM_HF_GLOBAL_MAP = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
    "model.hc_head.hc_fn": "hc_head_fn",
    "model.hc_head.hc_base": "hc_head_base",
    "model.hc_head.hc_scale": "hc_head_scale",
}

# MoE routed-expert weights: HF batches them as single fused Parameters, same
# as titaninfra's post-migration native layout — PURE RESHAPE, no transpose,
# no per-expert split (see module docstring's MoE bullet).
_HF_GATE_UP = "model.layers.{}.mlp.experts.gate_up_proj"
_HF_DOWN = "model.layers.{}.mlp.experts.down_proj"
_J_ROUTED_W13 = "layers.{}.feed_forward.routed_experts.w13.weight"
_J_ROUTED_W2 = "layers.{}.feed_forward.routed_experts.w2.weight"

# Dense FFN / shared-experts gate+up: HF keeps them as two separate Linears
# (gate_proj/up_proj); titaninfra's common FeedForward fuses them into one
# physical ``w13`` Parameter of shape [2, F, D] (STACKED, not concatenated —
# see DenseFeedForward.forward's ``gate_up.unbind(-2)``-style convention).
# One (abstract HF key template, titaninfra w13 key template) pair per site.
_GATE_UP_SITES = (
    ("model.layers.{}.mlp.gate_proj.weight", "model.layers.{}.mlp.up_proj.weight",
     "layers.{}.feed_forward.w13.weight"),
    ("model.layers.{}.mlp.shared_experts.gate_proj.weight", "model.layers.{}.mlp.shared_experts.up_proj.weight",
     "layers.{}.feed_forward.shared_experts.w13.weight"),
)
_GATE_UP_BY_HF_KEY = {gate: (gate, up, titan) for gate, up, titan in _GATE_UP_SITES}
_GATE_UP_BY_HF_KEY.update({up: (gate, up, titan) for gate, up, titan in _GATE_UP_SITES})
_GATE_UP_BY_TITAN_KEY = {titan: (gate, up, titan) for gate, up, titan in _GATE_UP_SITES}


class DeepSeekV4StateDictAdapter:
    """Bidirectional HF ↔ titaninfra state_dict adapter for DeepSeek-V4.

    Unlike DeepSeek-V3, HF V4 stores routed experts as batched Parameters
    (``gate_up_proj``/``down_proj`` with a leading expert dim), AND
    (post-migration) so does titaninfra's native ``GroupedLinear``-based
    ``routed_experts.w13``/``w2`` — so there is no per-expert split/concat,
    only a pure reshape (gate/up fuse axis) for ``w13``/``gate_up_proj``.
    """

    def __init__(
        self,
        model_config: "DeepSeekV4Model.Config",
        hf_assets_path: str | None = None,
    ):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self._to_hf_layer_map = {v: k for k, v in _FROM_HF_LAYER_MAP.items()}
        self._to_hf_global_map = {v: k for k, v in _FROM_HF_GLOBAL_MAP.items()}
        # Hash-routed MoE layers: HF registers e_score_correction_bias ONLY on
        # non-hash ("moe") layers, whereas titaninfra registers expert_bias_E on
        # every MoE layer (unless load_balance_coeff=None dropped it, see
        # MoE.Config). Compute the hash-layer index set from the router
        # configs so to_hf() drops the hash layers' expert_bias (HF has none)
        # and from_hf() tolerates their absence (titaninfra fills via param_init).
        self._hash_layer_ids: set[str] = set()
        for idx, layer in enumerate(getattr(model_config, "layers", []) or []):
            feed_forward = getattr(layer, "feed_forward", None)
            router = getattr(feed_forward, "router", None)
            if router is None:
                continue
            layer_id = getattr(router, "layer_id", None)
            num_hash = getattr(router, "n_hash_layers", 0)
            if layer_id is not None and layer_id < num_hash:
                self._hash_layer_ids.add(str(idx))

    # ---- from_hf: HF checkpoint → titaninfra state_dict --------------------
    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        """Storage reader for DCP-loading a HF safetensors checkpoint.

        deepseek_v4 HF weights ship unquantized bf16 here (the make_hf_16b init
        script), so a plain HuggingFaceStorageReader suffices. (DeepSeek-V3's
        671B fp8 path uses QuantizedHuggingFaceStorageReader; not needed for the
        16B-csa init-weight validation.)
        """
        return HuggingFaceStorageReader(path)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        skipped: list[str] = []
        # Buffer one half of each gate/up pair until its sibling shows up,
        # then stack into the fused w13 Parameter.
        gate_up_halves: dict[str, dict[str, Any]] = {}

        for key, value in hf_state_dict.items():
            # MoE routed experts (batched, fused): pure reshape, no split.
            if key.endswith("mlp.experts.gate_up_proj"):
                layer = re.search(r"layers\.(\d+)\.", key).group(1)
                # HF (E, 2F, D); native w13 (E, 2, F, D) -- same physical
                # layout, just the fused axis unflattened back out.
                state_dict[_J_ROUTED_W13.format(layer)] = value.unflatten(1, (2, -1)).contiguous()
                continue
            if key.endswith("mlp.experts.down_proj"):
                layer = re.search(r"layers\.(\d+)\.", key).group(1)
                state_dict[_J_ROUTED_W2.format(layer)] = value
                continue

            # Dense/shared-experts gate+up: stack into fused w13 once both halves seen.
            if key.startswith("model.layers."):
                abstract = re.sub(r"layers\.\d+", "layers.{}", key, count=1)
                if abstract in _GATE_UP_BY_HF_KEY:
                    gate_tmpl, up_tmpl, titan_tmpl = _GATE_UP_BY_HF_KEY[abstract]
                    layer = re.search(r"layers\.(\d+)", key).group(1)
                    slot = "gate" if abstract == gate_tmpl else "up"
                    halves = gate_up_halves.setdefault((titan_tmpl, layer), {})
                    halves[slot] = value
                    if "gate" in halves and "up" in halves:
                        state_dict[titan_tmpl.format(layer)] = torch.stack(
                            [halves["gate"], halves["up"]], dim=0
                        )
                    continue

            # Layer-templated keys.
            if key.startswith("model.layers."):
                abstract = re.sub(r"layers\.\d+", "layers.{}", key, count=1)
                layer = re.search(r"layers\.(\d+)", key).group(1)
                titan = _FROM_HF_LAYER_MAP.get(abstract)
                if titan is None:
                    skipped.append(key)
                    continue
                state_dict[titan.format(layer)] = value
                continue

            # Global keys.
            titan = _FROM_HF_GLOBAL_MAP.get(key)
            if titan is None:
                skipped.append(key)
                continue
            state_dict[titan] = value

        if skipped:
            # FAIL-LOUD: an unmapped HF key would load as an uninitialized
            # titaninfra weight under strict=False — silent numerical corruption.
            raise KeyError(
                f"from_hf: {len(skipped)} HF key(s) not in mapping — would load "
                f"as uninitialized weights. Fix the mapping or the checkpoint. "
                f"Unmapped: {skipped[:16]}" + (" ..." if len(skipped) > 16 else "")
            )

        return state_dict

    # ---- to_hf: titaninfra state_dict → HF checkpoint ----------------------
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            # MTP blocks: structural-only, not exported to HF.
            if key.startswith("mtp."):
                continue

            # MoE routed experts → recombine (pure reshape, no split/concat).
            m = re.match(
                r"layers\.(\d+)\.feed_forward\.routed_experts\.(w13|w2)\.weight",
                key,
            )
            if m:
                layer, wname = m.group(1), m.group(2)
                if wname == "w2":
                    hf_state_dict[_HF_DOWN.format(layer)] = value
                else:  # w13 (E, 2, F, D) -> gate_up_proj (E, 2F, D)
                    hf_state_dict[_HF_GATE_UP.format(layer)] = value.flatten(1, 2).contiguous()
                continue

            # Layer-templated keys.
            if key.startswith("layers."):
                abstract = re.sub(r"layers\.\d+", "layers.{}", key, count=1)
                layer = re.search(r"layers\.(\d+)", key).group(1)
                # Dense/shared-experts fused w13 → unbind back into HF's
                # separate gate_proj/up_proj Linears.
                if abstract in _GATE_UP_BY_TITAN_KEY:
                    gate_tmpl, up_tmpl, _ = _GATE_UP_BY_TITAN_KEY[abstract]
                    gate_w, up_w = value.unbind(0)
                    hf_state_dict[gate_tmpl.format(layer)] = gate_w
                    hf_state_dict[up_tmpl.format(layer)] = up_w
                    continue
                # Hash layers carry no e_score_correction_bias in HF; drop the
                # titaninfra expert_bias_E for those layers (HF would reject it).
                if abstract.endswith("feed_forward.expert_bias_E") and layer in self._hash_layer_ids:
                    continue
                hf = self._to_hf_layer_map.get(abstract)
                if hf is None:
                    raise KeyError(f"to_hf: titaninfra key not in mapping: {key}")
                hf_state_dict[hf.format(layer)] = value
                continue

            # Global keys.
            hf = self._to_hf_global_map.get(key)
            if hf is None:
                raise KeyError(f"to_hf: titaninfra global key not in mapping: {key}")
            hf_state_dict[hf] = value

        return hf_state_dict
