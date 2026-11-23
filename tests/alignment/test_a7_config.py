"""Unit tests for a7 config."""

import pytest
import transformers

_HF_MAJOR = int(transformers.__version__.split(".", 1)[0])
_REQUIRES_HF_V5 = pytest.mark.skipif(
    _HF_MAJOR < 5,
    reason=f"HF v5 rope_parameters dict only; installed transformers={transformers.__version__}",
)


TITAN_30B_A3B = {
    "vocab_size": 151936,
    "dim": 2048,
    "n_layers": 48,
    "n_heads": 32,
    "n_kv_heads": 4,
    "head_dim": 128,
    "ffn_hidden_dim": 6144,
    "moe_enabled": True,
    "moe_hidden_dim": 768,          # moe_intermediate_size
    "num_experts": 128,
    "top_k": 8,
    "num_shared_experts": 0,
    "score_func": "softmax",
    "route_norm": True,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_dim": 128,
    "rope_max_seq_len": 262144,
    "qk_norm": True,
}

TITAN_8B = {
    "vocab_size": 151936,
    "dim": 4096,
    "n_layers": 36,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "ffn_hidden_dim": 12288,
    "moe_enabled": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_dim": 128,
    "qk_norm": True,
}


class TestA7Config30BA3B:
    """Tests for test a7 config30 ba3 b."""

    @pytest.fixture(scope="class")
    def hf(self):
        from transformers import Qwen3MoeConfig
        return Qwen3MoeConfig(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=48,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            moe_intermediate_size=768,
            num_experts=128,
            num_experts_per_tok=8,
            decoder_sparse_step=1,
            num_shared_expert=0,
            norm_topk_prob=False,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            hidden_act="silu",
            attention_bias=False,
            qk_norm=True,
        )


    def test_vocab_size(self, hf):
        assert TITAN_30B_A3B["vocab_size"] == hf.vocab_size

    def test_hidden_size(self, hf):
        assert TITAN_30B_A3B["dim"] == hf.hidden_size

    def test_num_layers(self, hf):
        assert TITAN_30B_A3B["n_layers"] == hf.num_hidden_layers

    def test_num_attention_heads(self, hf):
        assert TITAN_30B_A3B["n_heads"] == hf.num_attention_heads

    def test_num_kv_heads(self, hf):
        assert TITAN_30B_A3B["n_kv_heads"] == hf.num_key_value_heads

    def test_head_dim(self, hf):
        assert TITAN_30B_A3B["head_dim"] == hf.head_dim


    def test_moe_enabled(self):
        assert TITAN_30B_A3B["moe_enabled"] is True

    def test_num_experts(self, hf):
        assert TITAN_30B_A3B["num_experts"] == hf.num_experts

    def test_top_k(self, hf):
        assert TITAN_30B_A3B["top_k"] == hf.num_experts_per_tok

    def test_moe_intermediate_size(self, hf):
        assert TITAN_30B_A3B["moe_hidden_dim"] == hf.moe_intermediate_size

    def test_num_shared_experts(self, hf):
        assert TITAN_30B_A3B["num_shared_experts"] == hf.num_shared_expert


    def test_intermediate_size(self, hf):
        assert TITAN_30B_A3B["ffn_hidden_dim"] == hf.intermediate_size

    # ── Normalization ─────────────────────────────────────────

    def test_rms_norm_eps(self, hf):
        assert TITAN_30B_A3B["rms_norm_eps"] == hf.rms_norm_eps

    def test_qk_norm_enabled(self, hf):
        assert TITAN_30B_A3B["qk_norm"] == hf.qk_norm

    # ── RoPE ──────────────────────────────────────────────────

    @_REQUIRES_HF_V5
    def test_rope_theta(self, hf):
        assert TITAN_30B_A3B["rope_theta"] == hf.rope_parameters["rope_theta"]

    def test_rope_dim_equals_head_dim(self, hf):
        assert TITAN_30B_A3B["rope_dim"] == hf.head_dim


    def test_q_proj_out_dim(self, hf):
        """q_proj output = n_heads * head_dim。"""
        titan_q_dim = TITAN_30B_A3B["n_heads"] * TITAN_30B_A3B["head_dim"]
        hf_q_dim = hf.num_attention_heads * hf.head_dim
        assert titan_q_dim == hf_q_dim

    def test_kv_proj_out_dim(self, hf):
        """kv_proj output = n_kv_heads * head_dim。"""
        titan_kv_dim = TITAN_30B_A3B["n_kv_heads"] * TITAN_30B_A3B["head_dim"]
        hf_kv_dim = hf.num_key_value_heads * hf.head_dim
        assert titan_kv_dim == hf_kv_dim


class TestA7Config8B:
    """Tests for test a7 config8 b."""

    @pytest.fixture(scope="class")
    def hf(self):
        from transformers import Qwen3Config
        return Qwen3Config(
            vocab_size=151936,
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            hidden_act="silu",
            attention_bias=False,
            qk_norm=True,
        )

    def test_vocab_size(self, hf):
        assert TITAN_8B["vocab_size"] == hf.vocab_size

    def test_hidden_size(self, hf):
        assert TITAN_8B["dim"] == hf.hidden_size

    def test_num_layers(self, hf):
        assert TITAN_8B["n_layers"] == hf.num_hidden_layers

    def test_num_heads(self, hf):
        assert TITAN_8B["n_heads"] == hf.num_attention_heads

    def test_num_kv_heads(self, hf):
        assert TITAN_8B["n_kv_heads"] == hf.num_key_value_heads

    def test_intermediate_size(self, hf):
        assert TITAN_8B["ffn_hidden_dim"] == hf.intermediate_size

    def test_rms_norm_eps(self, hf):
        assert TITAN_8B["rms_norm_eps"] == hf.rms_norm_eps

    @_REQUIRES_HF_V5
    def test_rope_theta(self, hf):
        assert TITAN_8B["rope_theta"] == hf.rope_parameters["rope_theta"]

    def test_no_moe(self):
        assert TITAN_8B["moe_enabled"] is False


class TestA7AllFlavors:
    """Tests for test a7 all flavors."""

    @pytest.fixture(autouse=True)
    def _skip_without_triton(self):
        pytest.importorskip("triton", reason="triton 仅在 GPU 环境可用")

    def test_all_configs_loadable(self):
        from titaninfra.models.qwen3 import qwen3_configs

        for flavor, config_factory in qwen3_configs.items():
            config = config_factory(attn_backend="flex")
            assert config.vocab_size > 0, f"{flavor} vocab_size 异常"
            assert config.dim > 0, f"{flavor} dim 异常"
            assert len(config.layers) > 0, f"{flavor} layers count 异常"

    def test_model_registry_returns_spec(self):
        from titaninfra.models import load_model_spec
        from titaninfra.models.qwen3 import Qwen3Model, model_registry

        config = model_registry("30B-A3B")
        assert config._owner is Qwen3Model
        assert hasattr(Qwen3Model, "parallelize")
        assert Qwen3Model.state_dict_adapter_cls is not None

        config_compat = load_model_spec("qwen3", "30B-A3B_titaninfra")
        assert config_compat._owner is Qwen3Model
