# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""HF-oracle and differentiable-score tests for upstream InnerIndexer glue."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.common.attention")
pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer  # noqa: E402

from titaninfra.models.deepseek_v4.config_registry import (  # noqa: E402
    _debugmodel_csa_attn_config,
)


def _build_pair():
    torch.manual_seed(0)
    cfg = DeepseekV4Config(
        hidden_size=256, head_dim=64, num_hidden_layers=4,
        num_attention_heads=4,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        compress_rope_theta=160000.0, q_lora_rank=128,
        partial_rotary_factor=32 / 64,
        index_head_dim=32, index_n_heads=4, index_topk=3,
    )
    oracle = DeepseekV4Indexer(cfg).eval()
    with torch.no_grad():
        oracle.position_bias.zero_()  # DeepseekV4PreTrainedModel._init_weights
    attn_cfg = _debugmodel_csa_attn_config(index_topk=3)
    mine = attn_cfg.build().indexer.eval()
    mine.init_states(buffer_device=torch.device("cpu"))
    for name in ("kv_proj", "gate_proj", "kv_norm", "q_b_proj"):
        getattr(mine, name).load_state_dict(getattr(oracle, name).state_dict())
    oracle_weights = (
        oracle.weights_proj if hasattr(oracle, "weights_proj") else oracle.scorer.weights_proj
    )
    mine.weights_proj.load_state_dict(oracle_weights.state_dict())
    with torch.no_grad():
        mine.position_bias.copy_(oracle.position_bias)
    return oracle, mine, attn_cfg, cfg


def test_indexer_topk_matches_oracle():
    oracle, mine, attn_cfg, cfg = _build_pair()
    B, S = 1, 32
    x = torch.randn(B, S, cfg.hidden_size)
    q_res = torch.randn(B, S, cfg.q_lora_rank)
    pos = torch.arange(S).unsqueeze(0)
    metadata = attn_cfg.inner_attention.build_attention_metadata(pos.squeeze(0))
    with torch.no_grad():
        oracle_indices = oracle(x, q_res, pos, None, 0).squeeze(0)
        indices, scores = mine(
            x.squeeze(0),
            q_res.squeeze(0),
            positions=pos.squeeze(0),
            attention_metadata=metadata,
            return_scores=False,
        )
    assert indices.shape == oracle_indices.shape
    assert indices.dtype == torch.int32
    assert scores.shape == (S, 0)
    # Order within a top-k row is tie-dependent; compare sorted index sets.
    torch.testing.assert_close(
        indices.sort(-1).values.to(oracle_indices.dtype),
        oracle_indices.sort(-1).values,
    )


def test_inner_indexer_scores_are_live_and_differentiable():
    _, mine, attn_cfg, cfg = _build_pair()
    mine.train()
    S = 32
    x = torch.randn(S, cfg.hidden_size)
    q_res = torch.randn(S, cfg.q_lora_rank)
    positions = torch.arange(S)
    metadata = attn_cfg.inner_attention.build_attention_metadata(positions)
    indices, scores = mine(
        x,
        q_res,
        positions=positions,
        attention_metadata=metadata,
        return_scores=True,
    )
    assert scores.shape == indices.shape
    valid = torch.isfinite(scores)
    assert valid.any()
    assert torch.isneginf(scores[~valid]).all()
    scores[valid].sum().backward()
    for parameter in (mine.q_b_proj.weight, mine.weights_proj.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
