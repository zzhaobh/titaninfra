"""DeepSeek V4 structural-MTP activation-remat regression tests."""

import pytest

pytest.importorskip("triton")
pytest.importorskip("torchtitan.models.common.attention")
pytest.importorskip("torchtitan.distributed.activation_remat")


def test_mtp_blocks_are_included_in_activation_remat_without_new_fqns():
    """Upstream #6246 must reach structural-only MTP blocks on this family."""
    import torch
    from torchtitan.distributed.activation_remat import FullActivationRematConfig

    from titaninfra.models.deepseek_v4.config_registry import model_registry

    cfg = model_registry("debugmodel")
    with torch.device("meta"):
        model = cfg.build()

    assert model._remat_block_container_fqns() == ["layers", "mtp_layers"]
    assert model.mtp_layers is model.mtp
    fqns_before = list(dict(model.named_parameters()).keys())

    FullActivationRematConfig().apply(
        model, block_container_fqns=model._remat_block_container_fqns()
    )

    assert len(model.mtp) > 0
    assert all("forward" in vars(block) for block in model.mtp)
    assert list(dict(model.named_parameters()).keys()) == fqns_before
