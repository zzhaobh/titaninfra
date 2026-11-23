# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""HF-oracle parity test for the DeepSeek-V4 HCA compressor.

Deferred to a GPU node: the Mac venv cannot import torchtitan (no triton) nor
`transformers.models.deepseek_v4`. The `importorskip` guards make this a no-op
skip locally and a real assertion on the GPU node.
"""

import pytest

pytestmark = pytest.mark.xdist_group("dsv4_cuda_kernels")

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.common.attention")
pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: E402
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
)

from titaninfra.models.deepseek_v4.config_registry import (  # noqa: E402
    _debugmodel_csa_attn_config,
    _debugmodel_hca_attn_config,
)


def _oracle_config(head_dim=64, hidden=256, rate_hca=8):
    return DeepseekV4Config(
        hidden_size=hidden, head_dim=head_dim, num_hidden_layers=4,
        num_attention_heads=4,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": rate_hca},
        compress_rope_theta=160000.0, q_lora_rank=128,
        partial_rotary_factor=32 / 64,
        index_head_dim=32, index_n_heads=4, index_topk=8,
    )


def test_hca_compressor_matches_oracle():
    torch.manual_seed(0)
    cfg = _oracle_config(rate_hca=8)
    oracle = DeepseekV4HCACompressor(cfg).eval()
    with torch.no_grad():
        oracle.position_bias.zero_()  # DeepseekV4PreTrainedModel._init_weights
    attn_cfg = _debugmodel_hca_attn_config(compress_rate=8)
    mine = attn_cfg.build().compressor.eval()
    mine.init_states(buffer_device=torch.device("cpu"))
    mine.kv_proj.load_state_dict(oracle.kv_proj.state_dict())
    mine.gate_proj.load_state_dict(oracle.gate_proj.state_dict())
    mine.kv_norm.load_state_dict(oracle.kv_norm.state_dict())
    with torch.no_grad():
        mine.position_bias.copy_(oracle.position_bias)
    B, S = 1, 32
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        oc, _bias = oracle(x, None, pos, None, 0)
        metadata = attn_cfg.inner_attention.build_attention_metadata(pos.squeeze(0))
        mc = mine(x.squeeze(0), pos.squeeze(0), metadata)[None, None]
    assert mc.shape == oc.shape
    torch.testing.assert_close(mc, oc, rtol=1e-4, atol=1e-4)


def test_csa_compressor_matches_oracle_compressed_kv():
    torch.manual_seed(0)
    cfg = _oracle_config(rate_hca=8)
    oracle = DeepseekV4CSACompressor(cfg).eval()
    with torch.no_grad():
        oracle.position_bias.zero_()  # DeepseekV4PreTrainedModel._init_weights
    attn_cfg = _debugmodel_csa_attn_config()
    mine = attn_cfg.build().compressor.eval()
    mine.init_states(buffer_device=torch.device("cpu"))
    mine.kv_proj.load_state_dict(oracle.kv_proj.state_dict())
    mine.gate_proj.load_state_dict(oracle.gate_proj.state_dict())
    mine.kv_norm.load_state_dict(oracle.kv_norm.state_dict())
    with torch.no_grad():
        mine.position_bias.copy_(oracle.position_bias)
    B, S = 1, 32
    x = torch.randn(B, S, cfg.hidden_size)
    q_res = torch.randn(B, S, cfg.q_lora_rank)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        oc, _bias = oracle(x, q_res, pos, None, 0)   # (compressed_kv, block_bias)
        metadata = attn_cfg.inner_attention.build_attention_metadata(pos.squeeze(0))
        mc = mine(x.squeeze(0), pos.squeeze(0), metadata)[None, None]
    assert mc.shape == oc.shape
    torch.testing.assert_close(mc, oc, rtol=1e-4, atol=1e-4)
