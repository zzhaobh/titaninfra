"""DeepSeek V4 flavor scalars and topology-aware sharding contracts.

torchtitan needs triton (absent on this Mac), so tests exercising real configs
skip locally and run on the GPU node. Pure layer-schedule helpers run anywhere.
"""

import pytest


@pytest.mark.parametrize(
    "flavor,dim,layers,experts",
    [
        ("debugmodel", 256, 6, 8),
        ("16B", 2048, 27, 64),
        ("Flash", 4096, 43, 256),
        ("Pro", 7168, 61, 384),
    ],
)
def test_flavor_scalars(flavor, dim, layers, experts):
    pytest.importorskip("triton")
    pytest.importorskip("torchtitan.models.common.attention")
    pytest.importorskip("torchtitan.models.common.moe")
    from torchtitan.models.common.moe import MoE

    from titaninfra.models.deepseek_v4.config_registry import model_registry

    cfg = model_registry(flavor)
    assert cfg.dim == dim
    assert len(cfg.layers) == layers
    moe_layer = next(layer for layer in cfg.layers if isinstance(layer.feed_forward, MoE.Config))
    assert moe_layer.feed_forward.num_experts == experts


def test_layer_types_default_schedule():
    from titaninfra.models.deepseek_v4.config_registry import _layer_types

    types = _layer_types(8)
    assert types[0] == "heavily_compressed_attention"
    assert types[1] == "heavily_compressed_attention"
    assert types[2] == "heavily_compressed_attention"  # i=0 in interior -> HCA
    assert types[3] == "compressed_sparse_attention"  # i=1 in interior -> CSA
    assert len(types) == 8


def test_csa_attn_config_carries_compress_rate():
    from titaninfra.models.deepseek_v4.config_registry import _debugmodel_csa_attn_config

    cfg = _debugmodel_csa_attn_config()
    assert cfg.layer_type == "compressed_sparse_attention"
    assert cfg.compress_rate == 4
    assert cfg.index_topk == 512


def test_hca_attn_config_no_index_semantics():
    from titaninfra.models.deepseek_v4.config_registry import _debugmodel_hca_attn_config

    cfg = _debugmodel_hca_attn_config()
    assert cfg.layer_type == "heavily_compressed_attention"
    assert cfg.compress_rate == 128


def _parallelism(*, enable_sp=True, ep=2):
    pytest.importorskip("triton")
    from torchtitan.config.parallelism import ParallelismConfig

    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=ep,
        enable_sequence_parallel=enable_sp,
    )


def test_decoder_sharding_matches_upstream_sequence_parallel_contract():
    """Regression: the five decoder boundaries must not drop their SP contract."""
    from torchtitan.models.common.decoder_sharding import (
        lm_head_sharding_config,
        norm_sharding_config,
        token_embedding_sharding_config,
    )

    from titaninfra.models.deepseek_v4.config_registry import model_registry

    cfg = model_registry("debugmodel", parallelism=_parallelism(enable_sp=True))
    expected_norm = norm_sharding_config(enable_sp=True)
    assert cfg.tok_embeddings.sharding_config == token_embedding_sharding_config(enable_sp=True)
    assert cfg.norm.sharding_config == expected_norm
    assert cfg.lm_head.sharding_config == lm_head_sharding_config()
    assert cfg.layers[0].attention_norm.sharding_config == expected_norm
    assert cfg.layers[0].ffn_norm.sharding_config == expected_norm


@pytest.mark.parametrize(
    "flavor", ["debugmodel", "debugmodel-p3", "16B", "16B-csa"]
)
def test_every_direct_state_has_a_declared_tp_ep_placement(flavor):
    """Regression: Module._parallelize rejects undeclared direct state under TP/EP."""
    import torch

    from titaninfra.models.deepseek_v4.config_registry import model_registry

    cfg = model_registry(flavor, parallelism=_parallelism())
    with torch.device("meta"):
        model = cfg.build()

    missing = {}
    for fqn, module in model.named_modules():
        direct_params = [name for name, _ in module.named_parameters(recurse=False)]
        direct_buffers = [name for name, _ in module.named_buffers(recurse=False)]
        sharding = getattr(module, "_sharding_config", None)
        declared = set() if sharding is None else set(sharding.state_shardings)
        required = set(direct_params)
        if sharding is not None:
            required.update(direct_buffers)
        undeclared = sorted(required - declared)
        if undeclared:
            missing[fqn] = undeclared
    assert missing == {}


def test_routed_expert_layout_follows_runtime_ep_degree():
    """Regression: EP=1 HSDP needs dense expert layouts, not hardcoded sparse ones."""
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.config.validation import _validate_declared_parallel_axes
    from torchtitan.models.common.moe.sharding import routed_expert_weight_placement

    from titaninfra.models.deepseek_v4.config_registry import model_registry

    for ep in (1, 2):
        parallelism = ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=2,
            expert_parallel_degree=ep,
            enable_sequence_parallel=False,
        )
        cfg = model_registry("debugmodel", parallelism=parallelism)
        routed = cfg.layers[1].feed_forward.routed_experts
        assert routed.w13.sharding_config.state_shardings["weight"] == routed_expert_weight_placement(enable_ep=ep > 1)
        _validate_declared_parallel_axes(cfg, parallelism)


def test_shared_experts_use_moe_partial_output_contract_without_sp():
    """Regression: dense FFN w2 reduces too early when shared experts run without SP."""
    from torchtitan.models.common import Linear, RowParallelLinear
    from torchtitan.models.common.moe.sharding import shared_experts_output_placement

    from titaninfra.models.deepseek_v4.config_registry import model_registry

    cfg = model_registry("debugmodel", parallelism=_parallelism(enable_sp=False))
    shared = cfg.layers[1].feed_forward.shared_experts
    assert isinstance(shared.w2, Linear.Config)
    assert not isinstance(shared.w2, RowParallelLinear.Config)
    assert shared.sharding_config is not None
    assert shared.sharding_config.output_shardings == shared_experts_output_placement(enable_sp=False)
