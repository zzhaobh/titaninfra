"""CPU regression tests for the qwen3_5 family after its torchtitan
upstream-tip migration (see docs/torchtitan_upgrade_plan_2026-11.md
§28/§30/§32-33). Covers the hybrid attention dispatch (Qwen35Attention vs
GDN), the generic base-class in_proj pack/unpack for GatedDeltaNet (the
highest-risk piece identified during scoping -- an incorrect interleaving
would produce correct-shaped-but-wrong-numerics weights, not a construction-
time error), and both the dense and MoE variants.
"""

import torch
from torchtitan.models.common.attention.gdn import GDN

from titaninfra.engine.pretrain import _apply_attn_backend, _apply_moe_config
from titaninfra.models.qwen3_5 import model_registry, qwen3_5_configs
from titaninfra.models.qwen3_5.model import Qwen35Attention, Qwen35MultimodalModel


def test_all_flavors_registered():
    assert set(qwen3_5_configs.keys()) == {
        "debugmodel",
        "debugmodel_moe",
        "0.8B",
        "2B",
        "4B",
        "9B",
        "27B",
        "35B-A3B",
        "122B-A10B",
        "397B-A17B",
    }


def test_all_flavors_construct():
    """Cheap construction-only check (no .build()) for every flavor,
    including the full-size ones -- catches config-shape regressions
    without paying the memory cost of materializing real-size models."""
    for flavor in qwen3_5_configs:
        spec = model_registry(flavor, attn_backend="flex")
        assert isinstance(spec, Qwen35MultimodalModel.Config)
        assert len(spec.layers) > 0


def test_debugmodel_hybrid_attention_dispatch():
    """75% GatedDeltaNet (GDN) / 25% full attention (Qwen35Attention),
    every 4th layer full -- the core hybrid-attention architecture."""
    spec = model_registry("debugmodel", attn_backend="flex")
    kinds = [
        "full" if isinstance(layer.attention, Qwen35Attention.Config) else "gdn"
        for layer in spec.layers
    ]
    assert kinds == ["gdn", "gdn", "gdn", "full", "gdn", "gdn", "gdn", "full"]
    for layer in spec.layers:
        assert isinstance(layer.attention, (Qwen35Attention.Config, GDN.Config))


def _full_chain(flavor: str, *, moe_comm_backend: str | None = None):
    kwargs = {"attn_backend": "flex"}
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    spec = model_registry(flavor, **kwargs)
    spec = _apply_attn_backend(spec, "flex")
    spec = _apply_moe_config(spec, None, None)
    model = spec.build()
    model.init_states(buffer_device=torch.device("cpu"))
    model.eval()
    adapter_cls = spec._owner.state_dict_adapter_cls
    adapter = adapter_cls(spec, None)
    return spec, model, adapter


def _assert_exact_roundtrip(model, adapter):
    """to_hf() -> from_hf() must reproduce every native key bit-for-bit.

    Real (non-zero, non-meta) random weights: this is the only kind of
    check that can catch a wrong in_proj interleaving -- shapes and key
    sets would still be "correct" even with scrambled numerics.
    """
    sd = model.state_dict()
    hf_sd = adapter.to_hf(sd)
    recovered = adapter.from_hf(hf_sd)
    assert set(recovered) == set(sd), (
        f"missing={set(sd) - set(recovered)} extra={set(recovered) - set(sd)}"
    )
    for key, value in sd.items():
        assert torch.equal(value, recovered[key]), f"mismatch at {key}"


def test_debugmodel_forward_and_roundtrip():
    spec, model, adapter = _full_chain("debugmodel")
    torch.manual_seed(0)
    T = 16
    tokens = torch.randint(0, spec.vocab_size, (T,))
    positions = torch.arange(T)
    attention_metadata = model._get_attention_metadata(positions=positions)
    with torch.no_grad():
        logits = model(tokens, positions=positions, attention_metadata=attention_metadata)
    assert logits.shape == (T, spec.vocab_size)
    assert torch.isfinite(logits).all()

    _assert_exact_roundtrip(model, adapter)


def test_debugmodel_moe_forward_and_roundtrip():
    spec, model, adapter = _full_chain("debugmodel_moe", moe_comm_backend="standard")
    torch.manual_seed(1)
    T = 12
    tokens = torch.randint(0, spec.vocab_size, (T,))
    positions = torch.arange(T)
    attention_metadata = model._get_attention_metadata(positions=positions)
    with torch.no_grad():
        logits = model(tokens, positions=positions, attention_metadata=attention_metadata)
    assert logits.shape == (T, spec.vocab_size)
    assert torch.isfinite(logits).all()

    _assert_exact_roundtrip(model, adapter)


def test_gdn_in_proj_split_sizes_match_fused_width():
    """GDN.Config's own __post_init__ already asserts this, but assert it
    again explicitly here: documents the exact per-key-head-interleaved
    contract (see model.py's GatedDeltaKernel docstring) that the generic
    base-class state_dict helper relies on."""
    spec = model_registry("debugmodel", attn_backend="flex")
    gdn_layers = [layer for layer in spec.layers if isinstance(layer.attention, GDN.Config)]
    assert len(gdn_layers) > 0
    for layer in gdn_layers:
        cfg = layer.attention
        expected = cfg.n_key_heads * sum(cfg.in_proj_split_sizes)
        assert cfg.in_proj.out_features == expected
