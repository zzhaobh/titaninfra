"""DeepSeek V4 TransformerBlock (Hyper-Connections) shape tests.

torchtitan needs triton (absent on this Mac), so this test SKIPS locally via
importorskip and runs on the GPU node.

Post-migration (see docs/torchtitan_upgrade_plan_2026-11.md §39): flat
``(T, ...)`` token-stream tensors except the HC axis, which stays
``[T, hc_mult, dim]`` (unchanged -- HC is per-token, batch was never on this
axis). The old xfail(strict=True) is RESOLVED the same way as
test_attention.py. ``DeepSeekV4MoE`` (the custom wrapper that stashed
``input_ids`` for hash routing) was deleted: upstream's real ``MoE.forward``
already forwards ``**router_kwargs`` (including ``input_ids_T``) to the
router directly, so a bare ``MoE.Config`` built via ``make_moe_config``
works unchanged (see model.py's module docstring, §37.1).
"""

from functools import partial

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.common.attention")
pytest.importorskip("torchtitan.models.common.moe")

import torch.nn as nn  # noqa: E402
from torchtitan.models.common import HiMidLoLinear, SqrtSoftplus  # noqa: E402
from torchtitan.models.common.attention import (  # noqa: E402
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.config_utils import (  # noqa: E402
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
)
from torchtitan.models.common.linear import Linear  # noqa: E402
from torchtitan.models.common.nn_modules import RMSNorm  # noqa: E402
from torchtitan.models.common.rope import ComplexRoPE  # noqa: E402

from titaninfra.models.deepseek_v4.model import (  # noqa: E402
    DeepSeekV4Attention,
    DeepSeekV4TransformerBlock,
)
from titaninfra.models.deepseek_v4.router import DeepSeekV4Router  # noqa: E402

# Debug-scale hyperparameters (tiny shapes for the GPU node).
_DIM = 256
_N_HEADS = 16
_HEAD_DIM = 64
_ROPE = 32
_Q_LORA = 128
_O_LORA = 128
_O_GROUPS = 2
_WINDOW = 8
_N_EXPERTS = 8
_TOP_K = 3
_MOE_INTER = 128
_VOCAB = 64
_LIN_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
_NORM_INIT = {"weight": nn.init.ones_}
_EXPERTS_INIT = {name: partial(nn.init.trunc_normal_, std=0.02) for name in ("w1_EFD", "w2_EDF", "w3_EFD")}


def _mk_attn_config(layer_type="sliding_attention", compress_rate=0, index_head_dim=32):
    inner = (
        SlidingWindowFlexInnerAttention.Config(sliding_window_size=_WINDOW)
        if layer_type == "sliding_attention"
        else FlexInnerAttention.Config()  # replaced by Config.__post_init__
    )
    return DeepSeekV4Attention.Config(
        dim=_DIM, n_heads=_N_HEADS, head_dim=_HEAD_DIM, qk_rope_head_dim=_ROPE,
        q_lora_rank=_Q_LORA, o_lora_rank=_O_LORA, o_groups=_O_GROUPS,
        sliding_window=_WINDOW, layer_type=layer_type, compress_rate=compress_rate,
        index_topk=8, index_n_heads=4, index_head_dim=index_head_dim, max_seq_len=64,
        wq_a=Linear.Config(in_features=_DIM, out_features=_Q_LORA, param_init=_LIN_INIT),
        q_norm=RMSNorm.Config(normalized_shape=_Q_LORA, param_init=_NORM_INIT),
        wq_b=Linear.Config(in_features=_Q_LORA, out_features=_N_HEADS * _HEAD_DIM, param_init=_LIN_INIT),
        wkv=Linear.Config(in_features=_DIM, out_features=_HEAD_DIM, param_init=_LIN_INIT),
        kv_norm=RMSNorm.Config(normalized_shape=_HEAD_DIM, param_init=_NORM_INIT),
        wo_a=Linear.Config(
            in_features=(_N_HEADS * _HEAD_DIM) // _O_GROUPS, out_features=_O_GROUPS * _O_LORA,
            param_init=_LIN_INIT,
        ),
        wo_b=Linear.Config(in_features=_O_GROUPS * _O_LORA, out_features=_DIM, param_init=_LIN_INIT),
        rope=ComplexRoPE.Config(dim=_ROPE, max_context_length=64),
        inner_attention=inner,
    )


def _mk_dense_ffn_config():
    return make_ffn_config(dim=_DIM, hidden_dim=512, w13_param_init=_LIN_INIT, w2_param_init=_LIN_INIT, enable_sp=False)


def _mk_moe_config(layer_id: int, num_hash_layers: int):
    router = DeepSeekV4Router.Config(
        num_experts=_N_EXPERTS,
        gate=HiMidLoLinear.Config(
            in_features=_DIM, out_features=_N_EXPERTS, backward_mode="hi_mid_lo",
            bias=False, param_init=_LIN_INIT,
        ),
        top_k=_TOP_K,
        score_func=SqrtSoftplus.Config(),
        route_norm=True,
        n_hash_layers=num_hash_layers,
        layer_id=layer_id,
        vocab_size=_VOCAB,
    )
    routed_experts = make_routed_experts_config(
        dim=_DIM, hidden_dim=_MOE_INTER, num_experts=_N_EXPERTS, top_k=_TOP_K,
        param_init=_EXPERTS_INIT, enable_ep=True,
    )
    return make_moe_config(
        num_experts=_N_EXPERTS, router=router, routed_experts=routed_experts,
        load_balance_coeff=1e-3, enable_sp=False,
    )


def _mk_block_config(attention, feed_forward):
    return DeepSeekV4TransformerBlock.Config(
        dim=_DIM, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        attention=attention, feed_forward=feed_forward,
        attention_norm=RMSNorm.Config(normalized_shape=_DIM, param_init=_NORM_INIT),
        ffn_norm=RMSNorm.Config(normalized_shape=_DIM, param_init=_NORM_INIT),
    )


_HC = 4
_T = 16


def test_block_dense_sliding_preserves_hc_shape():
    cfg = _mk_block_config(_mk_attn_config("sliding_attention"), _mk_dense_ffn_config())
    block = cfg.build()
    block.init_states(buffer_device=torch.device("cpu"))
    x = torch.randn(_T, _HC, _DIM)
    positions = torch.arange(_T)
    attention_metadata = cfg.attention.inner_attention.build_attention_metadata(positions)
    with torch.no_grad():
        out = block(x, attention_metadata, positions, None)
    assert out.shape == (_T, _HC, _DIM)
    assert torch.isfinite(out).all()


def test_block_moe_non_hash_csa_preserves_hc_shape():
    cfg = _mk_block_config(
        _mk_attn_config("compressed_sparse_attention", compress_rate=4),
        _mk_moe_config(layer_id=5, num_hash_layers=0),
    )
    block = cfg.build().eval()
    block.init_states(buffer_device=torch.device("cpu"))
    x = torch.randn(_T, _HC, _DIM)
    input_ids = torch.randint(0, _VOCAB, (_T,))
    positions = torch.arange(_T)
    attention_metadata = cfg.attention.inner_attention.build_attention_metadata(positions)
    with torch.no_grad():
        out = block(x, attention_metadata, positions, input_ids)
    assert out.shape == (_T, _HC, _DIM)
    assert torch.isfinite(out).all()


def test_block_moe_hash_hca_preserves_hc_shape_and_threads_input_ids():
    """Hash-routed layer: the router must receive input_ids_T (threaded via
    TransformerBlock.forward's input_ids_T positional arg) to route at all --
    confirms the deleted DeepSeekV4MoE wrapper's job is now done unconditionally
    by upstream's own MoE.forward(**router_kwargs) passthrough."""
    cfg = _mk_block_config(
        _mk_attn_config("heavily_compressed_attention", compress_rate=8),
        _mk_moe_config(layer_id=0, num_hash_layers=2),
    )
    block = cfg.build()
    block.init_states(buffer_device=torch.device("cpu"))
    assert block.feed_forward.router.hash
    x = torch.randn(_T, _HC, _DIM)
    input_ids = torch.randint(0, _VOCAB, (_T,))
    positions = torch.arange(_T)
    attention_metadata = cfg.attention.inner_attention.build_attention_metadata(positions)
    with torch.no_grad():
        out = block(x, attention_metadata, positions, input_ids)
    assert out.shape == (_T, _HC, _DIM)
    assert torch.isfinite(out).all()
