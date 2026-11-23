"""Qwen2.5 forward and SP-declaration regressions."""

import pytest

pytest.importorskip("torchtitan")


def test_forward_accepts_engine_padding_and_aux_loss_kwargs():
    """The training engine forwards both upstream keyword arguments."""
    import torch

    from titaninfra.models.qwen2_5 import model_registry

    cfg = model_registry("debugmodel")
    model = cfg.build()
    model.init_states(buffer_device=torch.device("cpu"))
    seq_len = 16
    tokens = torch.randint(0, cfg.vocab_size, (seq_len,))
    positions = torch.arange(seq_len)
    inner_attention = cfg.layers[0].attention.inner_attention.build()
    metadata = cfg.layers[0].attention.inner_attention.build_attention_metadata(positions)
    masks = {type(inner_attention): metadata}
    padding_mask = torch.zeros(seq_len, dtype=torch.bool)
    aux_loss_denominators = torch.ones(1)
    with torch.no_grad():
        logits = model(
            tokens,
            positions,
            masks,
            padding_mask=padding_mask,
            aux_loss_denominators=aux_loss_denominators,
        )
    assert logits.shape == (seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_enable_sp_is_derived_from_real_parallelism():
    """SP declarations must differ when the real topology enables SP."""
    from torchtitan.config.parallelism import ParallelismConfig

    from titaninfra.models.qwen2_5 import model_registry

    def norm_tp(enable_sp):
        parallelism = ParallelismConfig(
            tensor_parallel_degree=2,
            data_parallel_shard_degree=2,
            enable_sequence_parallel=enable_sp,
        )
        cfg = model_registry("debugmodel", parallelism=parallelism)
        root = cfg.norm.sharding_config.state_shardings["weight"].local_type
        layer = cfg.layers[0].attention_norm.sharding_config.state_shardings["weight"].local_type
        attention = cfg.layers[0].attention.sharding_config.input_shardings["x_TD"].local_type
        return root, layer, attention

    sp_on = norm_tp(True)
    sp_off = norm_tp(False)
    assert sp_on != sp_off
    for root, layer, _ in (sp_on, sp_off):
        assert root == layer
