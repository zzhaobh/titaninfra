# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""HF-oracle parity for the CSA/HCA compressed attention branch (Task 5).

Deferred to a GPU node: the Mac venv cannot import torchtitan (no triton) nor
``transformers.models.deepseek_v4``. The ``importorskip`` guards make this a
no-op skip locally and a real assertion on the GPU node.

The titaninfra compressed branch reproduces ``DeepseekV4Attention.forward``
(modeling_deepseek_v4.py L789-861) with ``_attn_implementation="eager"`` and
``past_key_values=None`` for one CSA and one HCA layer: q + base-KV rope on the
``compress`` theta, attention over ``cat([base_kv, compressed_kv])`` under one
fp32 softmax with an active per-head sink, then compress-theta inverse rope.
"""

import pytest

pytestmark = pytest.mark.xdist_group("dsv4_cuda_kernels")

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.common.attention")
pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")

from transformers.models.deepseek_v4.configuration_deepseek_v4 import (  # noqa: E402
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: E402
    DeepseekV4Attention,
    DeepseekV4Model,
)

from titaninfra.models.deepseek_v4.config_registry import (  # noqa: E402
    _debugmodel_csa_attn_config,
    _debugmodel_hca_attn_config,
)


def _oracle_config():
    """Small DeepseekV4Config matching the debugmodel CSA/HCA attn configs.

    head_dim=64, hidden=256, 4 heads, single KV head (V4 always MQA), q_lora=128,
    o_groups=2, o_lora=64; index dims match ``_debugmodel_csa_attn_config``
    (index_head_dim=32, index_n_heads=4, index_topk=8). compress_rates 4/8 so a
    32-token sequence yields several compressed slots for both layer types.
    partial_rotary_factor = 32/64 → qk_rope_head_dim=32 (matches the titaninfra
    debug attn config). sliding_window=8 (debug value).
    """
    return DeepseekV4Config(
        hidden_size=256,
        head_dim=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=128,
        o_groups=2,
        o_lora_rank=64,
        index_head_dim=32,
        index_n_heads=4,
        index_topk=8,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        compress_rope_theta=160000.0,
        partial_rotary_factor=32 / 64,
        sliding_window=8,
        attention_dropout=0.0,
    )


def _layer_idx_for(cfg, layer_type):
    for i, lt in enumerate(cfg.layer_types):
        if lt == layer_type:
            return i
    raise AssertionError(f"no {layer_type} layer in {cfg.layer_types}")


def _build_pair(layer_type):
    """Build titaninfra attn + HF oracle attn (eager, no-cache) with copied weights.

    Returns ``(mine, oracle, cfg, layer_idx)``. The titaninfra attn's compress_rate
    is overridden onto the debug config helper so it matches the oracle's
    per-type ``compress_rates`` (CSA=4, HCA=8) rather than the registry default
    (CSA=4, HCA=128) — otherwise HCA would emit 0 slots at S=32.
    """
    torch.manual_seed(0)
    cfg = _oracle_config()
    cfg._attn_implementation = "eager"
    layer_idx = _layer_idx_for(cfg, layer_type)
    oracle = DeepseekV4Attention(cfg, layer_idx).eval()
    with torch.no_grad():
        # Match DeepseekV4PreTrainedModel._init_weights for standalone modules.
        oracle.sinks.zero_()
        oracle.compressor.position_bias.zero_()
        if layer_type == "compressed_sparse_attention":
            oracle.compressor.indexer.position_bias.zero_()

    if layer_type == "compressed_sparse_attention":
        attn_cfg = _debugmodel_csa_attn_config(attn_backend="flex")
        attn_cfg.compress_rate = 4
    else:
        attn_cfg = _debugmodel_hca_attn_config(attn_backend="flex", compress_rate=8)
    mine = attn_cfg.build().eval()

    # Shared attention weights oracle -> titaninfra.
    mine.wq_a.load_state_dict(oracle.q_a_proj.state_dict())
    mine.q_norm.load_state_dict(oracle.q_a_norm.state_dict())
    mine.wq_b.load_state_dict(oracle.q_b_proj.state_dict())
    mine.wkv.load_state_dict(oracle.kv_proj.state_dict())
    mine.kv_norm.load_state_dict(oracle.kv_norm.state_dict())
    with torch.no_grad():
        # o_a_proj is a plain nn.Linear weight in both; titaninfra stores it as a
        # Linear whose .weight is reshaped group-wise at forward time.
        mine.wo_a.weight.copy_(oracle.o_a_proj.weight)
        mine.wo_b.weight.copy_(oracle.o_b_proj.weight)
        mine.attn_sink.copy_(oracle.sinks)

    # Compressor submodules (kv_proj / gate_proj / kv_norm / position_bias).
    oc = oracle.compressor
    mc = mine.compressor
    mc.kv_proj.load_state_dict(oc.kv_proj.state_dict())
    mc.gate_proj.load_state_dict(oc.gate_proj.state_dict())
    mc.kv_norm.load_state_dict(oc.kv_norm.state_dict())
    with torch.no_grad():
        mc.position_bias.copy_(oc.position_bias)

    # Indexer submodules (CSA only). Oracle indexer lives at compressor.indexer.
    if layer_type == "compressed_sparse_attention":
        oi = oc.indexer
        mi = mine.indexer
        mi.kv_proj.load_state_dict(oi.kv_proj.state_dict())
        mi.gate_proj.load_state_dict(oi.gate_proj.state_dict())
        mi.kv_norm.load_state_dict(oi.kv_norm.state_dict())
        mi.q_b_proj.load_state_dict(oi.q_b_proj.state_dict())
        oracle_weights = oi.weights_proj if hasattr(oi, "weights_proj") else oi.scorer.weights_proj
        mi.weights_proj.load_state_dict(oracle_weights.state_dict())
        with torch.no_grad():
            mi.position_bias.copy_(oi.position_bias)

    return mine, oracle, cfg, layer_idx, attn_cfg


def _oracle_forward(oracle, cfg, x, pos):
    """Call the oracle attention with the {main,compress} rope dict + causal mask
    exactly as ``DeepseekV4Model.forward`` builds them (modeling L1285-1307).

    Returns ``[B, S, hidden]``. Uses a fresh throwaway model instance only to
    reuse its rotary_emb + mask builder, then invokes the passed ``oracle``
    attention module directly so its copied weights are the ones under test.
    """
    from transformers.masking_utils import create_sliding_window_causal_mask

    B, S, _ = x.shape
    # inputs_embeds stands in for the hidden state that feeds rope/mask shape/dtype.
    inputs_embeds = x
    causal_mask = create_sliding_window_causal_mask(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=pos,
    )
    # rotary_emb is a stateless module (buffers only) — build one to produce the
    # {main, compress} cos/sin dict.
    rotary = DeepseekV4Model(cfg).rotary_emb
    position_embeddings = {
        "main": rotary(inputs_embeds, position_ids=pos, layer_type="main"),
        "compress": rotary(inputs_embeds, position_ids=pos, layer_type="compress"),
    }
    out, _ = oracle(
        hidden_states=x,
        position_embeddings=position_embeddings,
        position_ids=pos,
        attention_mask=causal_mask,
        past_key_values=None,
    )
    return out


@pytest.mark.parametrize(
    "layer_type",
    ["heavily_compressed_attention", "compressed_sparse_attention"],
)
def test_compressed_attention_matches_oracle(layer_type):
    mine, oracle, cfg, _, attn_cfg = _build_pair(layer_type)
    B, S = 1, 32
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        o = _oracle_forward(oracle, cfg, x, pos)
        # mine is flat (T, ...), no batch dim (B=1 here, so squeeze(0)/
        # unsqueeze(0) is lossless) -- see model.py's module docstring.
        metadata = attn_cfg.inner_attention.build_attention_metadata(pos.squeeze(0))
        m = mine(x.squeeze(0), metadata, positions=pos.squeeze(0)).unsqueeze(0)
    torch.testing.assert_close(m, o, rtol=2e-3, atol=2e-3)


def test_short_seq_hca_matches_oracle():
    # S=4 < rate=8 -> 0 compressed slots. The oracle still applies the
    # unconditional compress-theta inverse-RoPE (L856) on the base-only output,
    # so parity here specifically guards that the titaninfra n_comp==0 path shares
    # that inverse-rope tail (not just that the output is finite).
    mine, oracle, cfg, _, attn_cfg = _build_pair("heavily_compressed_attention")
    B, S = 1, 4
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        o = _oracle_forward(oracle, cfg, x, pos)
        metadata = attn_cfg.inner_attention.build_attention_metadata(pos.squeeze(0))
        m = mine(x.squeeze(0), metadata, positions=pos.squeeze(0)).unsqueeze(0)
    assert torch.isfinite(m).all()
    torch.testing.assert_close(m, o, rtol=2e-3, atol=2e-3)


def test_csa_gather_kernel_matches_dense():
    """Stage-B opt-in gather (``use_gather_kernel=True``) == dense CSA path.

    GPU + triton ONLY: the ``sparse_attn_gather`` Triton kernel needs CUDA. Two
    CSA ``DeepSeekV4Attention`` modules with IDENTICAL weights, one with
    ``use_gather_kernel=True`` and one dense (False), must produce
    ``assert_close`` output — the combined base-window + compressed-topk gather
    index (``_csa_combined_gather_index``) makes the single-tensor gather attend
    over exactly the non-(-inf) columns the dense combined softmax keeps.
    """
    if not torch.cuda.is_available():
        pytest.skip("gather kernel needs CUDA")
    pytest.importorskip("triton")

    torch.manual_seed(0)
    dense_cfg = _debugmodel_csa_attn_config(attn_backend="flex", use_gather_kernel=False)
    dense = dense_cfg.build().cuda().eval()

    gather_cfg = _debugmodel_csa_attn_config(attn_backend="flex", use_gather_kernel=True)
    gather = gather_cfg.build().cuda().eval()
    # Same weights so any output delta is purely the gather-vs-dense compute path.
    gather.load_state_dict(dense.state_dict())

    S = 64
    x = torch.randn(S, dense_cfg.dim, device="cuda")
    pos = torch.arange(S, device="cuda")
    metadata = dense_cfg.inner_attention.build_attention_metadata(pos)
    with torch.no_grad():
        o_dense = dense(x, metadata, positions=pos)
        o_gather = gather(x, metadata, positions=pos)
    assert torch.isfinite(o_gather).all()
    torch.testing.assert_close(o_gather, o_dense, rtol=2e-3, atol=2e-3)
