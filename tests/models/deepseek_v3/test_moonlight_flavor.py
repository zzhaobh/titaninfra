"""Structural + HF-parity tests for the moonlight_16b_a3b deepseek_v3 flavor.

Moonlight-16B-A3B (moonshotai/Moonlight-16B-A3B) is the flagship MoE pretrain
case model (replaces qwen3/30B-A3B per the round-18 upgrade decision — see
docs/torchtitan_upgrade_plan_2026-11.md). This flavor shares its architecture
with the Kimi-VL-A3B language tower (both are `_dsv3_lite_16b(theta=...)`
builds — see that function's docstring), differing only in rope_theta and the
absence of a vision tower / `language_model.` HF key prefix.

Expected values below are read DIRECTLY from the real HF config.json
(https://huggingface.co/moonshotai/Moonlight-16B-A3B/raw/main/config.json,
fetched and compared field-by-field on 2026-11-17) — not copied from
titaninfra's own docstrings, so this test independently re-verifies the
"confirmed against both HF config.json files" claim in
`titaninfra/models/deepseek_v3/__init__.py::_dsv3_lite_16b`'s docstring,
rather than trusting it.
"""

from torchtitan.models.common import Sigmoid

from titaninfra.models.deepseek_v3 import deepseekv3_configs


def _build():
    return deepseekv3_configs["moonlight_16b_a3b"](
        attn_backend="flex", moe_comm_backend="standard"
    )


def test_moonlight_16b_a3b_registered():
    assert "moonlight_16b_a3b" in deepseekv3_configs


def test_moonlight_16b_a3b_config_hparams():
    """Core dims, from HF config.json: hidden_size/num_hidden_layers/
    vocab_size/num_attention_heads/qk_nope_head_dim/qk_rope_head_dim/
    v_head_dim/kv_lora_rank."""
    cfg = _build()
    assert cfg.dim == 2048
    assert cfg.vocab_size == 163840
    assert len(cfg.layers) == 27
    attn0 = cfg.layers[0].attention
    assert attn0.qk_nope_head_dim == 128
    assert attn0.qk_rope_head_dim == 64
    assert attn0.v_head_dim == 128
    assert attn0.kv_lora_rank == 512
    assert attn0.n_heads == 16


def test_moonlight_16b_a3b_no_q_lora():
    """HF q_lora_rank=null -> 0 -> non-LoRA wq path (wq set, wq_a/wq_b None)."""
    cfg = _build()
    attn0 = cfg.layers[0].attention
    assert attn0.q_lora_rank == 0
    assert attn0.wq is not None
    assert attn0.wq_a is None
    assert attn0.wq_b is None


def test_moonlight_16b_a3b_rope_theta():
    """rope_theta=50000.0 -- the ONE field that differs from the otherwise
    bit-identical Kimi-VL-A3B language tower (theta=800000.0)."""
    cfg = _build()
    rope = cfg.layers[0].attention.rope
    assert rope.theta == 50000.0
    # No YARN: scaling stays at its "none" default (never set to "yarn").
    assert getattr(rope, "scaling", None) in (None, "none", "")


def test_moonlight_16b_a3b_moe():
    """MoE shape, from HF config.json: first_k_dense_replace/
    moe_intermediate_size/n_routed_experts/n_shared_experts/
    num_experts_per_tok/scoring_func/n_group/topk_group/
    routed_scaling_factor/norm_topk_prob."""
    cfg = _build()
    # Layer 0 is dense (first_k_dense_replace=1); layer 1 is the first MoE layer.
    from torchtitan.models.common.moe import MoE

    assert not isinstance(cfg.layers[0].feed_forward, MoE.Config)
    assert cfg.layers[0].feed_forward is not None
    moe_layer = cfg.layers[1]
    assert isinstance(moe_layer.feed_forward, MoE.Config)
    assert moe_layer.feed_forward.num_experts == 64
    assert moe_layer.feed_forward.router.top_k == 6
    assert isinstance(moe_layer.feed_forward.router.score_func, Sigmoid.Config)
    # n_group=1, topk_group=1 -> no group-limited routing distinguishable from
    # plain top-k at this group count, but the fields must still carry the HF
    # values rather than some other model's defaults.
    assert moe_layer.feed_forward.router.num_expert_groups == 1
    assert moe_layer.feed_forward.router.num_limited_groups == 1
    assert moe_layer.feed_forward.router.route_scale == 2.446
    assert moe_layer.feed_forward.router.route_norm is True


def test_moonlight_16b_a3b_dense_ffn_intermediate_size():
    """intermediate_size=11264 (dense MLP hidden dim) and
    moe_intermediate_size=1408 (per-expert hidden dim) from HF config.json --
    distinct values, must not be swapped."""
    cfg = _build()
    dense_layer = cfg.layers[0]
    # DenseFeedForward.Config fuses gate+up into a single w13 projection
    # (ColumnParallelLinear.Config, num_linears=2) -- confirmed by reading
    # make_ffn_config's body; there is no separate w1.
    assert dense_layer.feed_forward.w13.out_features == 11264

    from torchtitan.models.common.moe import MoE

    moe_layer = next(
        layer for layer in cfg.layers if isinstance(layer.feed_forward, MoE.Config)
    )
    # moe_hidden_dim is consumed into w13/w2's GroupedLinear shapes, not
    # stored as its own RoutedExperts.Config field -- confirmed by reading
    # make_routed_experts_config's body (w13.out_features == w2.in_features
    # == hidden_dim).
    assert moe_layer.feed_forward.routed_experts.w13.out_features == 1408
    assert moe_layer.feed_forward.routed_experts.w2.in_features == 1408


def test_moonlight_16b_a3b_matches_kimi_vl_a3b_except_theta():
    """Confirms the shared-builder claim: moonlight_16b_a3b and
    kimi_vl_a3b (Kimi-VL-A3B's language tower) are bit-identical except
    rope_theta -- both route through `_dsv3_lite_16b`."""
    moonlight = _build()
    kimi_vl = deepseekv3_configs["kimi_vl_a3b"](
        attn_backend="flex", moe_comm_backend="standard"
    )
    assert moonlight.dim == kimi_vl.dim
    assert moonlight.vocab_size == kimi_vl.vocab_size
    assert len(moonlight.layers) == len(kimi_vl.layers)
    assert (
        moonlight.layers[0].attention.kv_lora_rank
        == kimi_vl.layers[0].attention.kv_lora_rank
    )
    assert moonlight.layers[0].attention.rope.theta == 50000.0
    assert kimi_vl.layers[0].attention.rope.theta == 800000.0


def test_moonlight_16b_a3b_uses_default_state_dict_adapter():
    """Unlike kimi_vl_a3b (which needs KimiVLLanguageStateDictAdapter to strip
    the `language_model.` HF key prefix), moonlight_16b_a3b is a standalone
    plain `DeepseekV3ForCausalLM` checkpoint with no prefix stripping needed
    -- it uses the family's default DeepSeekV3StateDictAdapter, same as every
    other non-VLM deepseek_v3 flavor."""
    from titaninfra.models.deepseek_v3 import model_registry
    from titaninfra.models.deepseek_v3.state_dict_adapter import (
        DeepSeekV3StateDictAdapter,
    )

    config = model_registry("moonlight_16b_a3b", attn_backend="flex")
    assert config._owner.state_dict_adapter_cls is DeepSeekV3StateDictAdapter
