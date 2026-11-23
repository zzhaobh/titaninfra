"""Structural tests for the kimi_vl_a3b DeepSeek-V3 language-tower flavor.

Kimi-VL-A3B's language tower IS DeepSeek-V3 architecture (MLA + MoE), so it is
registered as a flavor in the deepseek_v3 family (text_config hyperparameters).
"""

from torchtitan.models.common import Sigmoid
from torchtitan.models.common.moe import MoE

from titaninfra.models.deepseek_v3 import deepseekv3_configs


def _build():
    return deepseekv3_configs["kimi_vl_a3b"](
        attn_backend="flex", moe_comm_backend="standard"
    )


def test_kimi_vl_a3b_registered():
    assert "kimi_vl_a3b" in deepseekv3_configs


def test_kimi_vl_a3b_config_hparams():
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


def test_kimi_vl_a3b_no_q_lora():
    # HF q_lora_rank=null -> 0 -> non-LoRA wq path (wq set, wq_a/wq_b None).
    cfg = _build()
    attn0 = cfg.layers[0].attention
    assert attn0.q_lora_rank == 0
    assert attn0.wq is not None
    assert attn0.wq_a is None
    assert attn0.wq_b is None


def test_kimi_vl_a3b_rope_no_yarn():
    cfg = _build()
    rope = cfg.layers[0].attention.rope
    assert rope.theta == 800000.0
    # No YARN: scaling stays at its "none" default (never set to "yarn").
    assert getattr(rope, "scaling", None) in (None, "none", "")


def test_kimi_vl_a3b_moe():
    # Post-migration: dense and MoE layers share one `feed_forward` field
    # (DenseFeedForward.Config vs MoE.Config), discriminated by isinstance
    # -- the old separate `.moe` field no longer exists (same pattern as
    # every other migrated family, see docs/torchtitan_upgrade_plan_2026-11.md).
    cfg = _build()
    # Layer 0 is dense (n_dense_layers=1); layer 1 is the first MoE layer.
    assert not isinstance(cfg.layers[0].feed_forward, MoE.Config)
    assert cfg.layers[0].feed_forward is not None
    moe_layer = cfg.layers[1]
    assert isinstance(moe_layer.feed_forward, MoE.Config)
    assert moe_layer.feed_forward.num_experts == 64
    assert moe_layer.feed_forward.router.top_k == 6
    assert isinstance(moe_layer.feed_forward.router.score_func, Sigmoid.Config)


# ---------------------------------------------------------------------------
# hf_prefix adapter tests (Task 2). Use simple non-expert keys (embed_tokens)
# to verify prefix strip/restore without expert-split complexity.
# ---------------------------------------------------------------------------


def test_adapter_default_prefix_unchanged():
    import torch

    from titaninfra.models.deepseek_v3.state_dict_adapter import (
        DeepSeekV3StateDictAdapter,
    )

    cfg = _build()
    a = DeepSeekV3StateDictAdapter(cfg, None)  # default hf_prefix=""
    out = a.from_hf({"model.embed_tokens.weight": torch.zeros(1)})
    assert "tok_embeddings.weight" in out


def test_kimi_adapter_strips_language_model_prefix():
    import torch

    from titaninfra.models.deepseek_v3.state_dict_adapter import (
        KimiVLLanguageStateDictAdapter,
    )

    cfg = _build()
    a = KimiVLLanguageStateDictAdapter(cfg, None)
    out = a.from_hf({"language_model.model.embed_tokens.weight": torch.zeros(1)})
    assert "tok_embeddings.weight" in out  # prefix stripped, then mapped


def test_kimi_adapter_roundtrip_prefix():
    import torch

    from titaninfra.models.deepseek_v3.state_dict_adapter import (
        KimiVLLanguageStateDictAdapter,
    )

    cfg = _build()
    a = KimiVLLanguageStateDictAdapter(cfg, None)
    titan = a.from_hf({"language_model.model.embed_tokens.weight": torch.zeros(1)})
    hf_out = a.to_hf(titan)
    assert "language_model.model.embed_tokens.weight" in hf_out  # prefix restored


def test_deepseek_v3_kimi_vl_a3b_flavor_uses_default_adapter():
    """Post-migration architecture note (renamed from
    test_kimi_flavor_uses_language_adapter): state_dict_adapter_cls is a
    per-CLASS ClassVar now (set via Configurable.Config._owner
    auto-wiring), not something model_registry() can swap per-flavor the
    way the old ModelSpec(state_dict_adapter=...) field let it.

    deepseek_v3.model_registry("kimi_vl_a3b", ...) returns a plain
    DeepSeekV3Model.Config, so its adapter is DeepSeekV3StateDictAdapter
    like every other deepseek_v3 flavor -- there is no standalone
    kimi_vl_a3b-flavored adapter override anymore within THIS family's
    own registry.

    The real Kimi-VL-specific language_model.-prefix handling capability
    is NOT lost: it is reached via a different path. The kimi_vl
    family's own model_registry("A3B", ...) builds a KimiVLModel.Config
    DIRECTLY (bypassing deepseek_v3.model_registry() entirely -- it
    reads deepseekv3_configs["kimi_vl_a3b"] as a plain dict lookup for
    shape hyperparameters only, see titaninfra/models/kimi_vl/__init__.py
    _a3b()), whose _owner is KimiVLModel, whose state_dict_adapter_cls is
    KimiVLStateDictAdapter (the real production adapter, a 3-segment
    composite that delegates its language segment to
    KimiVLLanguageStateDictAdapter -- verified with a 335-key real-model
    roundtrip test, see docs/torchtitan_upgrade_plan_2026-11.md §22).
    """
    from titaninfra.models.deepseek_v3 import model_registry
    from titaninfra.models.deepseek_v3.state_dict_adapter import (
        DeepSeekV3StateDictAdapter,
    )

    config = model_registry("kimi_vl_a3b", attn_backend="flex")
    assert config._owner.state_dict_adapter_cls is DeepSeekV3StateDictAdapter
