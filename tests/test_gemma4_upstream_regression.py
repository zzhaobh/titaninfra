"""Regression tests for gemma4's upstream validation and device-buffer contracts."""

from __future__ import annotations

import dataclasses
import math

import pytest

pytest.importorskip("torchtitan")

import torch  # noqa: E402
from torchtitan.config.parallelism import ParallelismConfig  # noqa: E402
from torchtitan.config.validation import _validate_declared_parallel_axes  # noqa: E402

from titaninfra.models.gemma4 import gemma4_configs, model_registry  # noqa: E402


@pytest.mark.parametrize(
    ("ep", "dp_replicate", "dp_shard"),
    [
        (1, 2, 2),
        (2, 2, 1),
        (2, 1, 2),
    ],
)
def test_declared_parallel_axes_match_runtime_ep(ep, dp_replicate, dp_shard):
    """Expert layouts must pass the same upstream validator as common MoE layouts."""
    parallelism = ParallelismConfig(
        expert_parallel_degree=ep,
        data_parallel_replicate_degree=dp_replicate,
        data_parallel_shard_degree=dp_shard,
    )
    config = model_registry("debugmodel", parallelism=parallelism)

    _validate_declared_parallel_axes(config, parallelism)


def test_ep_config_alignment_does_not_rename_runtime_experts():
    """The upstream config contract must not alter checkpoint-producing module names."""
    parallelism = ParallelismConfig(
        expert_parallel_degree=2,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=2,
    )
    config = model_registry("debugmodel", parallelism=parallelism)
    with torch.device("meta"):
        model = config.build()

    moe = model.layers["0"].moe
    assert hasattr(moe, "experts")
    assert not hasattr(moe, "routed_experts")
    assert moe.experts.w1 is not None


@pytest.mark.parametrize("embed_scale", [16.0, math.sqrt(2816)])
def test_embed_scale_materializes_as_nonpersistent_device_buffer(embed_scale):
    """The HF-rounded normalizer must not require a host-to-device copy in forward."""
    config = dataclasses.replace(gemma4_configs["debugmodel"], embed_scale=embed_scale)
    with torch.device("meta"):
        model = config.build()

    assert isinstance(model.embed_scale, torch.Tensor)
    model._init_self_buffers(buffer_device=torch.device("cpu"))

    assert model.embed_scale.device.type == "cpu"
    assert "embed_scale" in dict(model.named_buffers())
    assert "embed_scale" not in model.state_dict()

    h = torch.randn(8, dtype=torch.bfloat16)
    hf_scaled = h * torch.tensor(config.embed_scale, dtype=h.dtype)
    buffer_scaled = h * model.embed_scale.to(dtype=h.dtype)
    assert torch.equal(buffer_scaled, hf_scaled)
