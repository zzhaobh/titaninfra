import pytest
import torch


def test_projector_dim_chain():
    from titaninfra.models.kimi_vl.model import KimiVLMultiModalProjector
    proj = KimiVLMultiModalProjector(vision_hidden=8, merge_kernel=(2, 2), text_hidden=16, ln_eps=1e-5)
    # merged hidden = 8*2*2 = 32; two images, per-image (n_i, 32)
    feats = [torch.randn(3, 32), torch.randn(5, 32)]
    out = proj(feats)
    assert out.shape == (8, 16)  # (3+5, text_hidden)


def test_scatter_replaces_placeholder_positions():
    """Verify scatter replaces placeholder positions."""
    from titaninfra.models.kimi_vl.model import get_vision_positions, scatter_vision_embeds
    dim = 4
    embeds = torch.zeros(6, dim)
    tokens = torch.tensor([10, 99, 99, 11, 99, 12])  # placeholder=99 at 1,2 (one run) and 4 (another run)
    feats = torch.arange(3 * dim, dtype=torch.float32).view(3, dim)
    # Two separate images: first run (positions 1-2, 2 tokens), second run (position 4, 1 token).
    vision_positions = get_vision_positions(
        tokens, num_vision_tokens_per_item=torch.tensor([2, 1]), placeholder_id=99
    )
    out = scatter_vision_embeds(embeds, vision_embeds=feats, vision_positions=vision_positions)
    assert torch.equal(out[1], feats[0])
    assert torch.equal(out[2], feats[1])
    assert torch.equal(out[4], feats[2])
    assert torch.equal(out[0], torch.zeros(dim))  # non-placeholder untouched


def test_scatter_run_count_mismatch_fails_loud():
    """Verify scatter run count mismatch fails loud."""
    from titaninfra.models.kimi_vl.model import get_vision_positions
    tokens = torch.tensor([99, 99, 10, 11])  # one contiguous run of 2 placeholders
    with pytest.raises(ValueError):
        # Claim 2 images (2 runs expected) but tokens only has 1 run.
        get_vision_positions(tokens, num_vision_tokens_per_item=torch.tensor([1, 1]), placeholder_id=99)


def test_scatter_run_length_mismatch_fails_loud():
    """Verify scatter run length mismatch fails loud."""
    from titaninfra.models.kimi_vl.model import get_vision_positions
    tokens = torch.tensor([99, 99, 10, 11])  # one run of length 2
    with pytest.raises(ValueError):
        # Claim the one image has 3 tokens, but its run is only 2 tokens long.
        get_vision_positions(tokens, num_vision_tokens_per_item=torch.tensor([3]), placeholder_id=99)


def test_scatter_total_count_mismatch_fails_loud():
    """Verify scatter total count mismatch fails loud."""
    from titaninfra.models.kimi_vl.model import get_vision_positions, scatter_vision_embeds
    embeds = torch.zeros(4, 4)
    tokens = torch.tensor([99, 99, 10, 11])  # 2 placeholders, one run
    vision_positions = get_vision_positions(
        tokens, num_vision_tokens_per_item=torch.tensor([2]), placeholder_id=99
    )
    feats = torch.randn(3, 4)  # 3 rows supplied, but the run only consumes 2
    with pytest.raises(ValueError):
        scatter_vision_embeds(embeds, vision_embeds=feats, vision_positions=vision_positions)


def test_kimi_vl_model_config_composes_three_parts():
    """Verify kimi vl model config composes three parts."""
    try:
        from titaninfra.models.kimi_vl.model import KimiVLModel
    except Exception as e:  # triton/torchtitan absent on Mac
        pytest.skip(f"KimiVLModel import needs torchtitan/triton (GPU node): {e}")
    cfg = KimiVLModel.Config
    assert hasattr(cfg, "__dataclass_fields__") or hasattr(cfg, "model_fields")
    fields = cfg.__dataclass_fields__
    assert "media_placeholder_token_id" in fields
    assert "vision_hidden" in fields
    assert "dim" in fields


def _build_kimi_vl_config():
    """Build kimi vl config."""
    from titaninfra.models.deepseek_v3 import deepseekv3_configs

    return deepseekv3_configs["kimi_vl_a3b"](
        attn_backend="flex", moe_comm_backend="standard"
    )


def test_adapter_routes_three_prefixes():
    """Verify adapter routes three prefixes."""
    try:
        from titaninfra.models.kimi_vl.state_dict_adapter import KimiVLStateDictAdapter
    except Exception as e:  # torchtitan/triton absent on Mac
        pytest.skip(f"adapter import needs torchtitan (GPU node): {e}")

    cfg = _build_kimi_vl_config()
    adapter = KimiVLStateDictAdapter(cfg, None)

    hf = {
        # vision (one Conv2d weight + one block norm)
        "vision_tower.patch_embed.proj.weight": torch.zeros(1),
        "vision_tower.encoder.blocks.0.norm0.weight": torch.zeros(1),
        "vision_tower.encoder.final_layernorm.weight": torch.zeros(1),
        # language (delegate strips language_model. → tok_embeddings.weight)
        "language_model.model.embed_tokens.weight": torch.zeros(1),
        # projector
        "multi_modal_projector.linear_1.weight": torch.zeros(1),
        # unmapped top-level key → fault-tolerant skip
        "some.unmapped.key": torch.zeros(1),
    }
    tt = adapter.from_hf(hf)

    # vision → vision_tower.* (encoder. level dropped)
    assert "vision_tower.patch_embed.proj.weight" in tt
    assert "vision_tower.blocks.0.norm0.weight" in tt
    assert "vision_tower.final_layernorm.weight" in tt
    # language → top-level FQN via delegate
    assert "tok_embeddings.weight" in tt
    # projector → projector.*
    assert "projector.linear_1.weight" in tt
    # unmapped key dropped
    assert "some.unmapped.key" not in tt


def test_kimi_vl_registry_contract():
    """Verify kimi vl registry contract."""
    try:
        from titaninfra.models.kimi_vl import model_registry
    except Exception as e:  # torchtitan/triton absent on Mac
        pytest.skip(f"model_registry import needs torchtitan (GPU node): {e}")

    config = model_registry("A3B")
    assert hasattr(config, "media_placeholder_token_id")
    assert hasattr(config, "dim")
    assert config._owner.state_dict_adapter_cls is not None


def test_kimi_vl_registered_in_family_modules():
    """Verify kimi vl registered in family modules."""
    from titaninfra.models import _FAMILY_MODULES

    assert "kimi_vl" in _FAMILY_MODULES


def test_adapter_to_hf_reverses_three_prefixes():
    """Verify adapter to hf reverses three prefixes."""
    try:
        from titaninfra.models.kimi_vl.state_dict_adapter import KimiVLStateDictAdapter
    except Exception as e:
        pytest.skip(f"adapter import needs torchtitan (GPU node): {e}")

    cfg = _build_kimi_vl_config()
    adapter = KimiVLStateDictAdapter(cfg, None)

    tt = {
        "vision_tower.blocks.0.norm0.weight": torch.zeros(1),
        "projector.linear_2.bias": torch.zeros(1),
        "tok_embeddings.weight": torch.zeros(1),
    }
    hf = adapter.to_hf(tt)

    assert "vision_tower.encoder.blocks.0.norm0.weight" in hf
    assert "multi_modal_projector.linear_2.bias" in hf
    assert "language_model.model.embed_tokens.weight" in hf
