"""Unit tests for a1 param mapping."""

import pytest
import torch
import transformers

_HF_MAJOR = int(transformers.__version__.split(".", 1)[0])
_REQUIRES_HF_V5 = pytest.mark.skipif(
    _HF_MAJOR < 5,
    reason=f"HF v5 fused experts format only; installed transformers={transformers.__version__}",
)


def _get_hf_param_shapes() -> dict[str, tuple]:
    """Get hf param shapes."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
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
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    return {name: tuple(p.shape) for name, p in model.named_parameters()}


def _get_titan_param_names() -> list[str]:
    """Get titan param names."""
    params = []
    n_layers = 48

    # Embedding
    params.append("tok_embeddings.weight")

    for i in range(n_layers):
        prefix = f"layers.{i}"
        # Attention
        params.append(f"{prefix}.attention.qkv_linear.wq.weight")
        params.append(f"{prefix}.attention.qkv_linear.wk.weight")
        params.append(f"{prefix}.attention.qkv_linear.wv.weight")
        params.append(f"{prefix}.attention.wo.weight")
        params.append(f"{prefix}.attention.q_norm.weight")
        params.append(f"{prefix}.attention.k_norm.weight")
        # Layer Norm
        params.append(f"{prefix}.attention_norm.weight")
        params.append(f"{prefix}.ffn_norm.weight")
        # MoE (all layers are MoE for Qwen3-30B-A3B)
        params.append(f"{prefix}.moe.router.gate.weight")
        params.append(f"{prefix}.moe.experts.w1")  # gate_proj, shape (128, 768, 2048)
        params.append(f"{prefix}.moe.experts.w2")  # down_proj, shape (128, 2048, 768)
        params.append(f"{prefix}.moe.experts.w3")  # up_proj, shape (128, 768, 2048)

    # Output
    params.append("norm.weight")
    params.append("output.weight")

    return params


def _get_from_hf_map() -> dict[str, str | None]:
    """Get from hf map."""
    return {
        # Embedding
        "model.embed_tokens.weight": "tok_embeddings.weight",
        # Attention
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
        "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        # Layer Norm
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
        "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
        "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
        "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
        # Output
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }


def _expand_template_keys(
    template_map: dict[str, str | None],
    n_layers: int = 48,
    n_experts: int = 128,
) -> dict[str, str | None]:
    """Expand template keys."""
    expanded = {}
    for hf_key, titan_key in template_map.items():
        if "{}" not in hf_key:
            expanded[hf_key] = titan_key
            continue

        n_slots = hf_key.count("{}")

        if n_slots == 1:
            for layer_idx in range(n_layers):
                concrete_hf = hf_key.replace("{}", str(layer_idx), 1)
                concrete_titan = titan_key.replace("{}", str(layer_idx), 1) if titan_key else None
                expanded[concrete_hf] = concrete_titan
        elif n_slots == 2:
            for layer_idx in range(n_layers):
                for expert_idx in range(n_experts):
                    concrete_hf = hf_key.replace("{}", str(layer_idx), 1).replace(
                        "{}", str(expert_idx), 1
                    )
                    concrete_titan = (
                        titan_key.replace("{}", str(layer_idx), 1).replace(
                            "{}", str(expert_idx), 1
                        )
                        if titan_key
                        else None
                    )
                    expanded[concrete_hf] = concrete_titan

    return expanded


class TestA1ParamMapping:
    """Tests for test a1 param mapping."""

    @pytest.fixture(scope="class")
    def hf_params(self):
        return _get_hf_param_shapes()

    @pytest.fixture(scope="class")
    def titan_params(self):
        return _get_titan_param_names()

    @pytest.fixture(scope="class")
    def safetensors_keys(self):
        """Safetensors keys."""
        keys = []
        n_layers = 48
        n_experts = 128
        keys.append("model.embed_tokens.weight")
        for i in range(n_layers):
            prefix = f"model.layers.{i}"
            keys.extend([
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate.weight",
            ])
            for e in range(n_experts):
                keys.extend([
                    f"{prefix}.mlp.experts.{e}.gate_proj.weight",
                    f"{prefix}.mlp.experts.{e}.up_proj.weight",
                    f"{prefix}.mlp.experts.{e}.down_proj.weight",
                ])
        keys.extend(["model.norm.weight", "lm_head.weight"])
        return keys

    def test_hf_model_loads_on_meta(self, hf_params):
        """Verify hf model loads on meta."""
        assert len(hf_params) > 0
        print(f"  HF v5 参数数: {len(hf_params)}")

    def test_titan_model_params_known(self, titan_params):
        """Verify titan model params known."""
        assert len(titan_params) > 0
        print(f"  Titan 参数数: {len(titan_params)}")

    def test_from_hf_map_covers_all_safetensors_keys(self, safetensors_keys):
        """Verify from hf map covers all safetensors keys."""
        from_hf_map = _get_from_hf_map()
        expanded = _expand_template_keys(from_hf_map)

        uncovered = []
        for key in safetensors_keys:
            if "inv_freq" in key:
                continue
            if key not in expanded:
                uncovered.append(key)

        if uncovered:
            print(f"\n  未覆盖的 safetensors key ({len(uncovered)}):")
            for k in uncovered[:20]:
                print(f"    {k}")
        assert len(uncovered) == 0, f"{len(uncovered)} 个 safetensors key 未被 from_hf_map 覆盖"

    def test_from_hf_map_targets_exist_in_titan(self, titan_params):
        """Verify from hf map targets exist in titan."""
        from_hf_map = _get_from_hf_map()
        expanded = _expand_template_keys(from_hf_map)
        titan_set = set(titan_params)

        missing = []
        for hf_key, titan_key in expanded.items():
            if titan_key is None:
                continue
            if "feed_forward" in titan_key:
                continue
            if titan_key not in titan_set:
                missing.append((hf_key, titan_key))

        if missing:
            print(f"\n  映射目标不存在于 titan ({len(missing)}):")
            for hf_k, tt_k in missing[:10]:
                print(f"    {hf_k} → {tt_k}")

        assert len(missing) == 0, f"{len(missing)} 个映射目标不存在于 titan"

    @_REQUIRES_HF_V5
    def test_hf_v5_uses_fused_experts(self, hf_params):
        """Verify hf v5 uses fused experts."""
        fused_keys = [k for k in hf_params if "gate_up_proj" in k]
        per_expert_keys = [k for k in hf_params if "experts." in k and "gate_proj" in k]

        print(f"  Fused expert keys (gate_up_proj): {len(fused_keys)}")
        print(f"  Per-expert keys (experts.N.gate_proj): {len(per_expert_keys)}")

        assert len(fused_keys) > 0, "HF v5 应使用 fused gate_up_proj 格式"
        assert len(per_expert_keys) == 0, "HF v5 不应有 per-expert gate_proj keys"

    @_REQUIRES_HF_V5
    def test_expert_3d_shape_consistent(self, hf_params):
        """Verify expert 3d shape consistent."""
        n_experts = 128
        intermediate = 768
        hidden = 2048

        # HF v5 fused format
        gate_up_key = "model.layers.0.mlp.experts.gate_up_proj"
        down_key = "model.layers.0.mlp.experts.down_proj"

        assert gate_up_key in hf_params, f"缺少 {gate_up_key}"
        assert down_key in hf_params, f"缺少 {down_key}"

        assert hf_params[gate_up_key] == (n_experts, 2 * intermediate, hidden), \
            f"gate_up_proj shape 不匹配: {hf_params[gate_up_key]} != {(n_experts, 2*intermediate, hidden)}"
        assert hf_params[down_key] == (n_experts, hidden, intermediate), \
            f"down_proj shape 不匹配: {hf_params[down_key]} != {(n_experts, hidden, intermediate)}"

        print(f"  HF v5 gate_up_proj: {hf_params[gate_up_key]}")
        print(f"  HF v5 down_proj:    {hf_params[down_key]}")
        print(f"  Titan w1 (gate): ({n_experts}, {intermediate}, {hidden})")
        print(f"  Titan w3 (up):   ({n_experts}, {intermediate}, {hidden})")
        print(f"  Titan w2 (down): ({n_experts}, {hidden}, {intermediate})")

    def test_non_expert_shapes_consistent(self, hf_params):
        """Verify non expert shapes consistent."""
        # Qwen3-30B-A3B: hidden=2048, n_heads=32, n_kv_heads=4, head_dim=128
        expected_shapes = {
            "model.embed_tokens.weight": (151936, 2048),
            "model.layers.0.self_attn.q_proj.weight": (4096, 2048),   # 32 heads × 128 dim
            "model.layers.0.self_attn.k_proj.weight": (512, 2048),    # 4 kv_heads × 128 head_dim
            "model.layers.0.self_attn.v_proj.weight": (512, 2048),
            "model.layers.0.self_attn.o_proj.weight": (2048, 4096),   # hidden × (n_heads × head_dim)
            "model.layers.0.self_attn.q_norm.weight": (128,),
            "model.layers.0.self_attn.k_norm.weight": (128,),
            "model.layers.0.input_layernorm.weight": (2048,),
            "model.layers.0.post_attention_layernorm.weight": (2048,),
            "model.layers.0.mlp.gate.weight": (128, 2048),            # router gate
            "model.norm.weight": (2048,),
            "lm_head.weight": (151936, 2048),
        }

        mismatches = []
        for key, expected_shape in expected_shapes.items():
            if key in hf_params:
                actual_shape = hf_params[key]
                if actual_shape != expected_shape:
                    mismatches.append((key, expected_shape, actual_shape))

        if mismatches:
            print("\n  Shape 不匹配:")
            for k, exp, act in mismatches:
                print(f"    {k}: expected {exp}, got {act}")

        assert len(mismatches) == 0, f"{len(mismatches)} 个参数 shape 不匹配"
