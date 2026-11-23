"""DeepSeek V4 FSDP lifecycle regression tests."""

from types import SimpleNamespace

import pytest

pytest.importorskip("triton")
pytest.importorskip("torchtitan.models.common.attention")
pytest.importorskip("torchtitan.models.common.moe")


def test_mtp_blocks_receive_decoder_fsdp_and_restore_module_tree(monkeypatch):
    """MTP blocks must not fall through to root dense-only FSDP wrapping."""
    import torch
    import torchtitan.distributed.fsdp as decoder_fsdp
    import torchtitan.models.deepseek_v3.mtp as mtp_fsdp
    from torchtitan.config import PrecisionConfig
    from torchtitan.config.parallelism import ParallelismConfig

    from titaninfra.models.deepseek_v4.config_registry import model_registry

    cfg = model_registry("debugmodel")
    with torch.device("meta"):
        model = cfg.build()

    fqns_before = list(dict(model.named_parameters()).keys())
    layer_keys_before = tuple(model.layers.keys())
    observed_layer_keys = []

    def capture_fsdp_call(model, *args, **kwargs):
        del args, kwargs
        observed_layer_keys.append(tuple(model.layers.keys()))

    monkeypatch.setattr(
        decoder_fsdp,
        "resolve_fsdp_mesh",
        lambda context: (object(), object()),
    )
    monkeypatch.setattr(
        decoder_fsdp,
        "resolve_sparse_fsdp_mesh",
        lambda context: (object(), object()),
    )
    # The decoder helper imports its own function reference, so patch both
    # references while keeping the real MTP helper and model lifecycle intact.
    monkeypatch.setattr(decoder_fsdp, "apply_fsdp_to_decoder", capture_fsdp_call)
    monkeypatch.setattr(mtp_fsdp, "apply_fsdp_to_decoder", capture_fsdp_call)

    model._apply_fsdp(
        parallelism_context=SimpleNamespace(pp_enabled=False, ep=4),
        precision=PrecisionConfig(param_dtype="float32", reduce_dtype="float32"),
        parallelism=ParallelismConfig(),
    )

    expected_during_fsdp = (*layer_keys_before, str(len(layer_keys_before)))
    assert observed_layer_keys == [expected_during_fsdp]
    assert tuple(model.layers.keys()) == layer_keys_before
    assert list(dict(model.named_parameters()).keys()) == fqns_before
