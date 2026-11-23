"""FQN regression guard for the gemma4 26B-A4B model (DCP checkpoint safety).

The gemma4 rewrite (gemma4-owned numeric components in ``_components.py``) MUST
preserve every distributed-checkpoint parameter FQN byte-for-byte. Changing a
parameter's fully-qualified name silently breaks DCP load from the HF safetensors
(the documented prior production incidents). This test instantiates the real
``26B-A4B_titaninfra`` flavor on the meta device and asserts its ``state_dict()``
keys exactly equal the frozen expected set below.

Requires torchtitan (GPU node). Auto-skips on machines without it (e.g. macOS dev
boxes), where the component-level parity suite ``test_gemma4_hf_parity.py`` runs
instead.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torchtitan")

import torch  # noqa: E402

# Layer types for 26B-A4B: 5 sliding + 1 full, repeated 5× → full at 5,11,17,23,29.
_FULL_LAYERS = {5, 11, 17, 23, 29}
_NUM_LAYERS = 30


def _expected_param_fqns() -> set[str]:
    """The authoritative DCP parameter FQN set for 26B-A4B.

    Mirrors the state_dict_adapter key map. Full-attention layers (K==V sharing)
    have NO ``attention.qkv_linear.wv.weight``. ``moe.expert_bias`` is absent
    because 26B sets ``load_balance_coeff=None`` (the bias buffer is only created
    when the coeff is set, and it's a buffer not a parameter regardless).
    """
    fqns: set[str] = {
        "tok_embeddings.weight",
        # output.weight is tied to tok_embeddings.weight (weight sharing),
        # so named_parameters() reports it only once as tok_embeddings.weight.
        "norm.weight",
    }
    for i in range(_NUM_LAYERS):
        p = f"layers.{i}"
        fqns |= {
            f"{p}.attention_norm.weight",
            f"{p}.ffn_norm.weight",
            f"{p}.attention.qkv_linear.wq.weight",
            f"{p}.attention.qkv_linear.wk.weight",
            f"{p}.attention.wo.weight",
            f"{p}.attention.q_norm.weight",
            f"{p}.attention.k_norm.weight",
            f"{p}.pre_feedforward_layernorm.weight",
            f"{p}.post_feedforward_layernorm.weight",
            f"{p}.post_feedforward_layernorm_1.weight",
            f"{p}.pre_feedforward_layernorm_2.weight",
            f"{p}.post_feedforward_layernorm_2.weight",
            f"{p}.feed_forward.w1.weight",
            f"{p}.feed_forward.w2.weight",
            f"{p}.feed_forward.w3.weight",
            f"{p}.moe.router.scale",
            f"{p}.moe.router.per_expert_scale",
            f"{p}.moe.router.proj.weight",
            f"{p}.moe.experts.w1",
            f"{p}.moe.experts.w2",
            f"{p}.moe.experts.w3",
            f"{p}.layer_scalar",
        }
        if i not in _FULL_LAYERS:
            fqns.add(f"{p}.attention.qkv_linear.wv.weight")
    return fqns


def test_gemma4_26b_param_fqns_unchanged():
    from titaninfra.models.gemma4 import gemma4_configs

    cfg = gemma4_configs["26B-A4B"]
    with torch.device("meta"):
        model = cfg.build()

    actual = set(dict(model.named_parameters()).keys())
    expected = _expected_param_fqns()

    missing = expected - actual
    extra = actual - expected
    assert not missing, f"DCP-critical FQNs MISSING after rewrite: {sorted(missing)[:20]}"
    assert not extra, f"Unexpected new FQNs after rewrite (DCP risk): {sorted(extra)[:20]}"


def test_full_layers_have_no_v_proj():
    """K==V sharing: full-attention layers must NOT carry a wv parameter."""
    from titaninfra.models.gemma4 import gemma4_configs

    cfg = gemma4_configs["26B-A4B"]
    with torch.device("meta"):
        model = cfg.build()
    names = set(dict(model.named_parameters()).keys())
    for i in _FULL_LAYERS:
        assert f"layers.{i}.attention.qkv_linear.wv.weight" not in names
    for i in range(_NUM_LAYERS):
        if i not in _FULL_LAYERS:
            assert f"layers.{i}.attention.qkv_linear.wv.weight" in names
