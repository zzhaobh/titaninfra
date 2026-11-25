"""DeepSeek V4 position-aware FLOPs estimator regressions."""

import pytest

pytest.importorskip("triton")
pytest.importorskip("torchtitan.models.common.attention")


@pytest.mark.parametrize("flavor", ["debugmodel", "debugmodel-p3", "16B", "16B-csa"])
def test_attention_flops_accepts_positions_for_every_layer(flavor):
    """Decoder.build_flops_estimator calls this hook on the first train step."""
    import torch

    from titaninfra.models.deepseek_v4.config_registry import model_registry

    cfg = model_registry(flavor)
    # Exceed CSA top-k after compression so the clamp branch is exercised too.
    positions = torch.arange(4096)
    for layer in cfg.layers:
        flops = layer.attention.flops(positions)
        assert isinstance(flops, int)
        assert flops > 0
