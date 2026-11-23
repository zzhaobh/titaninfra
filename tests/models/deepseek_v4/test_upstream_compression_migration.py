"""Upstream compression-stack migration with a frozen legacy checkpoint tree."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.common.attention")

from torchtitan.models.base.module import Module  # noqa: E402
from torchtitan.models.deepseek_v4.attention import (  # noqa: E402
    CompressedSparseAttention,
    HeavilyCompressedAttention,
)
from torchtitan.models.deepseek_v4.compressor import InnerIndexer  # noqa: E402

from titaninfra.models.deepseek_v4.config_registry import model_registry  # noqa: E402


def _expected_compressed_parameters(config):
    """Independent legacy name/shape contract for every compressed submodule."""
    expected = {}
    layer_configs = [(f"layers.{index}", layer) for index, layer in enumerate(config.layers)]
    layer_configs.append(("mtp.0", config.layers[-1]))
    for prefix, layer in layer_configs:
        attention = layer.attention
        if attention.layer_type == "sliding_attention":
            continue
        dim = attention.dim
        head_dim = attention.head_dim
        ratio = attention.compress_rate
        compressor_width = head_dim * (2 if attention.layer_type == "compressed_sparse_attention" else 1)
        base = f"{prefix}.attention.compressor"
        expected.update(
            {
                f"{base}.kv_proj.weight": ((compressor_width, dim), "torch.float32"),
                f"{base}.gate_proj.weight": ((compressor_width, dim), "torch.float32"),
                f"{base}.kv_norm.weight": ((head_dim,), "torch.float32"),
                f"{base}.position_bias": ((ratio, compressor_width), "torch.float32"),
            }
        )
        if attention.layer_type != "compressed_sparse_attention":
            continue
        index_dim = attention.index_head_dim
        index_base = f"{prefix}.attention.indexer"
        expected.update(
            {
                f"{index_base}.kv_proj.weight": ((2 * index_dim, dim), "torch.float32"),
                f"{index_base}.gate_proj.weight": ((2 * index_dim, dim), "torch.float32"),
                f"{index_base}.kv_norm.weight": ((index_dim,), "torch.float32"),
                f"{index_base}.position_bias": ((ratio, 2 * index_dim), "torch.float32"),
                f"{index_base}.q_b_proj.weight": (
                    (attention.index_n_heads * index_dim, attention.q_lora_rank),
                    "torch.float32",
                ),
                f"{index_base}.weights_proj.weight": (
                    (attention.index_n_heads, dim),
                    "torch.float32",
                ),
            }
        )
    return expected


@pytest.mark.parametrize("flavor", ["debugmodel-p3", "16B-csa"])
def test_legacy_compressed_parameter_tree_is_unchanged(flavor):
    config = model_registry(flavor)
    with torch.device("meta"):
        model = config.build()
    actual = {
        name: (tuple(value.shape), str(value.dtype))
        for name, value in model.named_parameters()
        if ".attention.compressor" in name or ".attention.indexer" in name
    }
    expected = _expected_compressed_parameters(config)
    assert actual == expected


def test_compressed_modules_satisfy_upstream_module_and_inner_attention_protocols():
    config = model_registry("debugmodel-p3")
    with torch.device("meta"):
        model = config.build()

    csa = model.layers["3"].attention
    assert isinstance(csa.compressor, Module)
    assert isinstance(csa.indexer, Module)
    assert isinstance(csa.indexer.inner_indexer, InnerIndexer)
    assert isinstance(csa.inner_attention, CompressedSparseAttention)

    hca = model.layers["0"].attention
    assert isinstance(hca.compressor, Module)
    assert isinstance(hca.inner_attention, HeavilyCompressedAttention)

    # Read-only compatibility views must not register upstream-name aliases.
    parameter_names = dict(model.named_parameters())
    upstream_aliases = (
        ".compressor.wkv.",
        ".compressor.wgate.",
        ".compressor.norm.",
        ".indexer.wq_b.",
    )
    assert not any(alias in name for alias in upstream_aliases for name in parameter_names)
    assert not any(name.endswith(".ape") for name in parameter_names)
    assert not any(".indexer.compressor." in name for name in parameter_names)
    assert not any(".compressor_128." in name for name in parameter_names)


def test_csa_indexer_aux_loss_opt_in_backpropagates_to_indexer():
    import dataclasses

    from titaninfra.models.deepseek_v4.config_registry import (
        _debugmodel_csa_attn_config,
    )

    config = dataclasses.replace(
        _debugmodel_csa_attn_config(), indexer_loss_coeff=0.01
    )
    attention = config.build().train()
    attention.init_states(buffer_device=torch.device("cpu"))
    seqlen = 32
    x = torch.randn(seqlen, config.dim)
    positions = torch.arange(seqlen)
    metadata = config.inner_attention.build_attention_metadata(positions)
    output = attention(
        x,
        metadata,
        positions,
        aux_loss_denominator=torch.tensor(seqlen),
        padding_mask=torch.zeros(seqlen, dtype=torch.bool),
    )
    output.square().mean().backward()
    assert attention.indexer.q_b_proj.weight.grad is not None
    assert attention.indexer.weights_proj.weight.grad is not None
    assert torch.isfinite(attention.indexer.q_b_proj.weight.grad).all()
    assert torch.isfinite(attention.indexer.weights_proj.weight.grad).all()
