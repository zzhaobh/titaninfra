"""Roundtrip parity test for the DeepSeek-V4 HF ↔ titaninfra state_dict adapter.

Deferred to a GPU node: importing titaninfra's deepseek_v4 model pulls torchtitan
(triton) and `transformers.models.deepseek_v4`. importorskip makes this a no-op
on machines lacking either.

Both sides are built at a deliberately-aligned shape so their layer schedules
match exactly: HF DeepseekV4 has NO dense layers (every layer is hash_moe/moe),
so titaninfra is built with ``n_dense_layers=0`` and ``num_hash_layers=3`` to mirror
HF's default (first 3 layers hash_moe, rest moe). The only remaining structural
asymmetry is expert-bias: HF registers ``e_score_correction_bias`` only on
non-hash MoE layers, whereas titaninfra registers ``expert_bias_E`` on every MoE
layer — the adapter drops the hash layers' bias on to_hf and tolerates its
absence on from_hf (titaninfra fills it via param_init/zeros).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.common.attention")
pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")

from transformers.models.deepseek_v4.configuration_deepseek_v4 import (  # noqa: E402
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: E402
    DeepseekV4ForCausalLM,
)

from titaninfra.models.deepseek_v4.config_registry import (  # noqa: E402
    _layer_types,
    _make_model_config,
)
from titaninfra.models.deepseek_v4.state_dict_adapter import (  # noqa: E402
    DeepSeekV4StateDictAdapter,
)

_N_LAYERS = 6
_N_HASH = 3


def _titaninfra_config():
    return _make_model_config(
        dim=256,
        n_layers=_N_LAYERS,
        n_dense_layers=0,  # HF DeepseekV4 has no dense layers
        vocab_size=151936,
        n_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        moe_hidden_dim=256,
        dense_hidden_dim=1024,
        num_experts=8,
        num_shared_experts=1,
        top_k=3,
        num_hash_layers=_N_HASH,
        routed_scaling_factor=1.5,
        max_seq_len=4096,
        sliding_window=8,
        attn_backend="flex",
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
        layer_types=_layer_types(_N_LAYERS),
    )


def _hf_model():
    cfg = DeepseekV4Config(
        hidden_size=256,
        # HF aliases intermediate_size -> moe_intermediate_size (V4 ships only
        # one FFN width; shared + routed experts share it). Pass a single value
        # matching titaninfra's moe_hidden_dim=256; do NOT also pass intermediate_size.
        moe_intermediate_size=256,
        num_hidden_layers=_N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        q_lora_rank=128,
        kv_lora_rank=64,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=3,
        o_groups=2,
        o_lora_rank=64,
        vocab_size=151936,
        num_nextn_predict_layers=1,
        num_hash_layers=_N_HASH,
    )
    return DeepseekV4ForCausalLM(cfg)


def _titaninfra_model():
    m = _titaninfra_config().build()
    m.init_states(buffer_device=torch.device("cpu"))
    return m


def _adapter():
    return DeepSeekV4StateDictAdapter(_titaninfra_config(), None)


def _is_hash_expert_bias(key: str) -> bool:
    """titaninfra expert_bias_E on a hash layer (0..N_HASH-1) — HF has none."""
    import re

    m = re.match(r"layers\.(\d+)\.feed_forward\.expert_bias_E$", key)
    return bool(m) and int(m.group(1)) < _N_HASH


def test_from_hf_covers_all_titaninfra_weights():
    """from_hf(HF sd) must produce every titaninfra weight (except MTP and the
    hash-layer expert_bias HF lacks) with matching shapes."""
    hf = _hf_model()
    jm = _titaninfra_model()
    converted = _adapter().from_hf(hf.state_dict())
    jsd = jm.state_dict()

    missing, mismatched = [], []
    for k, v in jsd.items():
        if k.startswith("mtp."):
            continue  # HF has no MTP weights
        if _is_hash_expert_bias(k):
            continue  # HF hash layers carry no e_score_correction_bias
        if k not in converted:
            missing.append(k)
        elif tuple(converted[k].shape) != tuple(v.shape):
            mismatched.append((k, tuple(converted[k].shape), tuple(v.shape)))

    assert not missing, f"from_hf missing {len(missing)} titaninfra keys: {missing[:12]}"
    assert not mismatched, f"shape mismatch: {mismatched[:12]}"


def test_to_hf_roundtrips_back_to_hf_keyset():
    """to_hf(titaninfra sd) must reproduce the HF key set with matching shapes."""
    hf = _hf_model()
    jm = _titaninfra_model()
    hf_sd = hf.state_dict()
    back = _adapter().to_hf(jm.state_dict())

    missing, mismatched, extra = [], [], []
    for k, v in hf_sd.items():
        if k not in back:
            missing.append(k)
        elif tuple(back[k].shape) != tuple(v.shape):
            mismatched.append((k, tuple(back[k].shape), tuple(v.shape)))
    # to_hf must not emit keys HF doesn't have (would fail HF load_state_dict).
    for k in back:
        if k not in hf_sd:
            extra.append(k)

    assert not missing, f"to_hf missing {len(missing)} HF keys: {missing[:12]}"
    assert not mismatched, f"shape mismatch: {mismatched[:12]}"
    assert not extra, f"to_hf emitted {len(extra)} keys HF lacks: {extra[:12]}"


def test_from_hf_to_hf_value_identity():
    """A from_hf → to_hf cycle must be value-identical (no lossy reshape)."""
    hf = _hf_model()
    hf_sd = hf.state_dict()
    cycled = _adapter().to_hf(_adapter().from_hf(hf_sd))

    for k, v in hf_sd.items():
        assert k in cycled, f"key lost in cycle: {k}"
        torch.testing.assert_close(cycled[k], v, rtol=0, atol=0)
