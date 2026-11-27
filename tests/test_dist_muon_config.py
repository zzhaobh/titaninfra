from inspect import signature

import pytest
import torch

from tests.conftest import make_test_model_config, make_test_train_config


def _k3_config_and_spec():
    from titaninfra.engine.pretrain import _load_model_spec

    model_config = make_test_model_config(
        name="kimi_k3/debugmodel",
        attn_backend="flex",
    )
    model_spec = _load_model_spec(
        "kimi_k3",
        "debugmodel",
        parallelism=model_config.to_torchtitan_parallelism_config(),
        seq_len=128,
        attn_backend="varlen",
    )
    return model_config, model_spec


def test_k3_dist_muon_has_layout_for_every_selected_parameter():
    from torchtitan.components.optim import DistMuon

    from titaninfra.engine._pretrain_helpers import _build_optimizer_config

    model_config, model_spec = _k3_config_and_spec()
    assert "model_spec" in signature(_build_optimizer_config).parameters
    train_config = make_test_train_config(optimizer_name="DistMuon")
    optimizer = _build_optimizer_config(
        train_config,
        [],
        model_config,
        model_spec=model_spec,
    )
    dist_muon = next(
        config
        for config in optimizer.optimizers
        if isinstance(config, DistMuon.Config)
    )

    assert dist_muon.compute_sharding_by_fqn
    assert dist_muon.bucket_configs
    from titaninfra.engine.pretrain import _apply_attn_backend

    model_spec = _apply_attn_backend(model_spec, "flex")
    with torch.device("meta"):
        model = model_spec.build()
    parameter_names = dict(model.named_parameters()).keys()
    assert dist_muon.compute_sharding_by_fqn.keys() <= parameter_names
    assert "layers.0.attention.q_proj.weight" in dist_muon.compute_sharding_by_fqn
    assert any(
        name.endswith("feed_forward.routed_experts.w13.weight")
        for name in dist_muon.compute_sharding_by_fqn
    )


def test_dist_muon_rejects_tensor_parallel_k3():
    from titaninfra.engine._pretrain_helpers import _build_optimizer_config

    model_config, model_spec = _k3_config_and_spec()
    model_config.tp = 2
    train_config = make_test_train_config(optimizer_name="DistMuon")
    with pytest.raises(ValueError, match="tensor_parallel_degree=1"):
        _build_optimizer_config(
            train_config,
            [],
            model_config,
            model_spec=model_spec,
        )


def test_dist_muon_rejects_non_k3_models():
    from titaninfra.engine._pretrain_helpers import _build_optimizer_config

    model_config = make_test_model_config(name="qwen3/0.6B")
    train_config = make_test_train_config(optimizer_name="DistMuon")
    assert "model_spec" in signature(_build_optimizer_config).parameters
    with pytest.raises(ValueError, match="kimi_k3"):
        _build_optimizer_config(
            train_config,
            [],
            model_config,
            model_spec=object(),
        )
