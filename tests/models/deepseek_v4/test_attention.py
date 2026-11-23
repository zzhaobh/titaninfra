"""Tests for DeepSeek V4 attention (dense/sliding-window + CSA/HCA compressed).

Skips cleanly on machines without triton (e.g. macOS dev): torchtitan's
top-level import succeeds, but ``torchtitan.models.common.attention`` (which
model.py imports) pulls in triton via FlexInnerAttention/EP kernels. importorskip
the actual dependency so this module skips instead of erroring.

Post-migration (see docs/torchtitan_upgrade_plan_2026-11.md §39): flat
``(T, ...)`` token-stream tensors, no leading batch dim -- matching upstream's
rope.py convention and every other migrated family. The old xfail(strict=True)
this file carried ("upstream rope.py now requires flat query/positions...")
is RESOLVED: model.py now builds that exact flat convention directly.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.common.attention")

from torchtitan.models.common.attention import (  # noqa: E402
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.linear import Linear  # noqa: E402
from torchtitan.models.common.nn_modules import RMSNorm  # noqa: E402
from torchtitan.models.common.rope import ComplexRoPE  # noqa: E402

from titaninfra.models.deepseek_v4.model import DeepSeekV4Attention  # noqa: E402

_DIM = 256
_N_HEADS = 16
_HEAD_DIM = 64
_ROPE = 32
_Q_LORA = 128
_O_LORA = 128
_O_GROUPS = 2
_WINDOW = 8


def _mk_attn_config(layer_type="sliding_attention", compress_rate=0, index_head_dim=32):
    """Debug-scale config; compressed placeholders are replaced by upstream configs."""
    inner = (
        SlidingWindowFlexInnerAttention.Config(sliding_window_size=_WINDOW)
        if layer_type == "sliding_attention"
        else FlexInnerAttention.Config()
    )
    return DeepSeekV4Attention.Config(
        dim=_DIM,
        n_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        qk_rope_head_dim=_ROPE,
        q_lora_rank=_Q_LORA,
        o_lora_rank=_O_LORA,
        o_groups=_O_GROUPS,
        sliding_window=_WINDOW,
        layer_type=layer_type,
        compress_rate=compress_rate,
        index_topk=8,
        index_n_heads=4,
        index_head_dim=index_head_dim,
        max_seq_len=64,
        wq_a=Linear.Config(in_features=_DIM, out_features=_Q_LORA),
        q_norm=RMSNorm.Config(normalized_shape=_Q_LORA),
        wq_b=Linear.Config(in_features=_Q_LORA, out_features=_N_HEADS * _HEAD_DIM),
        wkv=Linear.Config(in_features=_DIM, out_features=_HEAD_DIM),
        kv_norm=RMSNorm.Config(normalized_shape=_HEAD_DIM),
        # wo_a is applied group-wise: input is (n_heads * head_dim) // o_groups
        # per group, output is o_groups * o_lora_rank.
        wo_a=Linear.Config(
            in_features=(_N_HEADS * _HEAD_DIM) // _O_GROUPS,
            out_features=_O_GROUPS * _O_LORA,
        ),
        wo_b=Linear.Config(in_features=_O_GROUPS * _O_LORA, out_features=_DIM),
        rope=ComplexRoPE.Config(dim=_ROPE, max_context_length=64),
        inner_attention=inner,
    )


def test_sliding_attention_forward_shape():
    cfg = _mk_attn_config("sliding_attention")
    attn = cfg.build()
    attn.init_states(buffer_device=torch.device("cpu"))
    T = 16
    x = torch.randn(T, _DIM)
    positions = torch.arange(T)
    attention_metadata = cfg.inner_attention.build_attention_metadata(positions)
    # forward-only shape check; no_grad so FlexInnerAttention runs on CPU
    # (its autograd path is CUDA-only).
    with torch.no_grad():
        out = attn(x, attention_metadata, positions)
    assert out.shape == (T, _DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "layer_type,compress_rate,index_head_dim",
    [
        ("heavily_compressed_attention", 8, 32),
        ("compressed_sparse_attention", 4, 32),
    ],
)
def test_compressed_attention_forward_shape(layer_type, compress_rate, index_head_dim):
    cfg = _mk_attn_config(layer_type, compress_rate, index_head_dim)
    attn = cfg.build().eval()
    attn.init_states(buffer_device=torch.device("cpu"))
    T = 32  # long enough for compress_rate=8 to have >=1 compressed block
    x = torch.randn(T, _DIM)
    positions = torch.arange(T)
    attention_metadata = cfg.inner_attention.build_attention_metadata(positions)
    with torch.no_grad():
        out = attn(x, attention_metadata, positions)
    assert out.shape == (T, _DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "layer_type,compress_rate,index_head_dim",
    [
        ("heavily_compressed_attention", 8, 32),
        ("compressed_sparse_attention", 4, 32),
    ],
)
def test_compressed_attention_short_seq_degeneracy(layer_type, compress_rate, index_head_dim):
    """Fewer tokens than compress_rate -> zero compressed slots; must still
    attend over base KV only and stay finite (the active sink keeps the
    softmax finite even with zero real compressed columns)."""
    cfg = _mk_attn_config(layer_type, compress_rate, index_head_dim)
    attn = cfg.build().eval()
    attn.init_states(buffer_device=torch.device("cpu"))
    T = 3
    x = torch.randn(T, _DIM)
    positions = torch.arange(T)
    attention_metadata = cfg.inner_attention.build_attention_metadata(positions)
    with torch.no_grad():
        out = attn(x, attention_metadata, positions)
    assert out.shape == (T, _DIM)
    assert torch.isfinite(out).all()
