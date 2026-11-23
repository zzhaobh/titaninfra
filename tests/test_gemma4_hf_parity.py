"""HF-parity tests for gemma4 self-contained numeric components.

Each test builds a gemma4-owned primitive from
``titaninfra.models.gemma4._components`` AND the matching HuggingFace transformers
component, feeds identical input, and asserts numerical equivalence. This locks
gemma4's numerics to HF (transformers >= 5.10) and guards against regression of
the loss=441 RoPE bug.

CPU-only, no GPU, no real checkpoint, no torchtitan. ``_components`` is imported
directly by file path so the package ``__init__`` (which needs torchtitan) is not
triggered.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

pytest.importorskip("transformers")
_gemma4_hf = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
_gemma4_cfg = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")


def _load_components():
    """Import _components.py by path, bypassing the torchtitan-heavy package init."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "titaninfra", "models", "gemma4", "_components.py")
    spec = importlib.util.spec_from_file_location("gemma4_components", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_components()

ATOL = 1e-4
RTOL = 1e-4


def _seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. RMSNorm
# ---------------------------------------------------------------------------


def test_rmsnorm_with_scale_matches_hf():
    _seed()
    dim = 64
    x = torch.randn(2, 7, dim)

    hf = _gemma4_hf.Gemma4RMSNorm(dim, eps=1e-6)
    hf.weight.data = torch.randn(dim)

    jd = C.Gemma4RMSNorm(dim, eps=1e-6, with_scale=True)
    jd.weight.data = hf.weight.data.clone()

    assert torch.allclose(jd(x), hf(x), atol=ATOL, rtol=RTOL)


def test_rmsnorm_without_scale_matches_hf():
    _seed()
    dim = 64
    x = torch.randn(2, 7, dim)

    hf = _gemma4_hf.Gemma4RMSNorm(dim, eps=1e-6, with_scale=False)
    jd = C.Gemma4RMSNorm(dim, eps=1e-6, with_scale=False)

    assert not hasattr(jd, "weight")  # v_norm / router-norm carry no learnable weight
    assert torch.allclose(jd(x), hf(x), atol=ATOL, rtol=RTOL)


def test_rmsnorm_bf16_roundtrip_matches_hf():
    _seed()
    dim = 64
    x = torch.randn(2, 7, dim, dtype=torch.bfloat16)
    hf = _gemma4_hf.Gemma4RMSNorm(dim, eps=1e-6)
    hf.weight.data = torch.randn(dim)
    jd = C.Gemma4RMSNorm(dim, eps=1e-6, with_scale=True)
    jd.weight.data = hf.weight.data.clone()
    out_jd, out_hf = jd(x), hf(x)
    assert out_jd.dtype == torch.bfloat16 and out_hf.dtype == torch.bfloat16
    assert torch.allclose(out_jd.float(), out_hf.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 2. RoPE — sliding (default) and full (proportional)
# ---------------------------------------------------------------------------


def _hf_rotary(head_dim, partial, theta, positions, denom=None):
    """Reproduce HF Gemma4 cos/sin for one layer type. Returns cos, sin [S, head_dim]."""
    denom = denom if denom is not None else head_dim
    if partial is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    else:
        rope_angles = int(partial * denom // 2)
        ifr = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2).float() / denom))
        nope = denom // 2 - rope_angles
        inv_freq = torch.cat([ifr, torch.zeros(nope)], dim=0)
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def test_sliding_rope_cache_matches_hf():
    _seed()
    head_dim, theta, L = 256, 10000.0, 32
    cache = C.build_default_cos_sin_cache(head_dim=head_dim, theta=theta, max_seq_len=L)
    cos_jd, sin_jd = cache[..., :head_dim], cache[..., head_dim:]
    cos_hf, sin_hf = _hf_rotary(head_dim, None, theta, torch.arange(L))
    assert torch.allclose(cos_jd, cos_hf, atol=ATOL)
    assert torch.allclose(sin_jd, sin_hf, atol=ATOL)


def test_full_proportional_rope_cache_matches_hf():
    """The loss=441 fix: proportional cache must use denom=512, pair k<->k+256."""
    _seed()
    full_head_dim, rope_dim, theta, L = 512, 128, 1_000_000.0, 32
    cache = C.build_proportional_cos_sin_cache(
        head_dim=full_head_dim, rope_dim=rope_dim, theta=theta, max_seq_len=L
    )
    assert list(cache.shape) == [L, full_head_dim * 2]
    cos_jd, sin_jd = cache[..., :full_head_dim], cache[..., full_head_dim:]
    cos_hf, sin_hf = _hf_rotary(full_head_dim, 0.25, theta, torch.arange(L), denom=full_head_dim)
    assert torch.allclose(cos_jd, cos_hf, atol=ATOL)
    assert torch.allclose(sin_jd, sin_hf, atol=ATOL)
    # exactly 64 nonzero frequencies (×2 halves = 128 nonzero channels), rest zero
    nz = (sin_jd[1] != 0).sum().item()
    assert nz == 128


def test_apply_rotary_full_head_matches_hf():
    """Applying RoPE to the full head with the proportional cache == HF."""
    _seed()
    full_head_dim, rope_dim, theta, L = 512, 128, 1_000_000.0, 16
    xq = torch.randn(1, L, 4, full_head_dim)
    xk = torch.randn(1, L, 2, full_head_dim)
    cache = C.build_proportional_cos_sin_cache(
        head_dim=full_head_dim, rope_dim=rope_dim, theta=theta, max_seq_len=L
    )
    positions = torch.arange(L).unsqueeze(0)
    q_jd, k_jd = C.apply_rotary_emb_from_cache(xq, xk, cache, positions)

    cos_hf, sin_hf = _hf_rotary(full_head_dim, 0.25, theta, torch.arange(L), denom=full_head_dim)
    # HF apply_rotary_pos_emb broadcasts cos/sin [B, L, head_dim] over heads
    # (unsqueeze_dim=2). Add the batch dim to match xq's [1, L, H, head_dim].
    cos_hf = cos_hf.unsqueeze(0)
    sin_hf = sin_hf.unsqueeze(0)
    q_hf = _gemma4_hf.apply_rotary_pos_emb(xq, cos_hf, sin_hf, unsqueeze_dim=2)
    k_hf = _gemma4_hf.apply_rotary_pos_emb(xk, cos_hf, sin_hf, unsqueeze_dim=2)
    assert torch.allclose(q_jd, q_hf, atol=ATOL)
    assert torch.allclose(k_jd, k_hf, atol=ATOL)


# ---------------------------------------------------------------------------
# 3. Attention math (SDPA) vs HF eager_attention_forward
# ---------------------------------------------------------------------------


class _AttnStub:
    def __init__(self, n_rep, head_dim, training=False):
        self.num_key_value_groups = n_rep
        self.head_dim = head_dim
        self.training = training


@pytest.mark.parametrize("enable_gqa", [False, True])
def test_sdpa_matches_hf_eager(enable_gqa):
    _seed()
    B, S, Hq, D = 1, 6, 4, 16
    Hkv = 2 if enable_gqa else 4
    xq = torch.randn(B, S, Hq, D)
    xk = torch.randn(B, S, Hkv, D)
    xv = torch.randn(B, S, Hkv, D)

    mask = torch.full((S, S), float("-inf")).triu(1).view(1, 1, S, S)

    out_jd = C.gemma4_sdpa(xq, xk, xv, attn_mask=mask, scaling=1.0, enable_gqa=enable_gqa)

    q = xq.transpose(1, 2)
    k = xk.transpose(1, 2)
    v = xv.transpose(1, 2)
    stub = _AttnStub(n_rep=Hq // Hkv, head_dim=D)
    out_hf, _ = _gemma4_hf.eager_attention_forward(
        stub, q, k, v, attention_mask=mask, scaling=1.0, dropout=0.0
    )  # returns [B, S, H, D]
    assert torch.allclose(out_jd, out_hf, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# 4. Dense MLP vs HF Gemma4TextMLP
# ---------------------------------------------------------------------------


def test_dense_mlp_matches_hf():
    _seed()
    dim, hidden = 64, 128
    x = torch.randn(2, 5, dim)
    cfg = _gemma4_cfg.Gemma4TextConfig(
        hidden_size=dim, intermediate_size=hidden, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    hf = _gemma4_hf.Gemma4TextMLP(cfg, layer_idx=0)

    import torch.nn as nn
    w1 = nn.Linear(dim, hidden, bias=False)
    w3 = nn.Linear(dim, hidden, bias=False)
    w2 = nn.Linear(hidden, dim, bias=False)
    w1.weight.data = hf.gate_proj.weight.data.clone()
    w3.weight.data = hf.up_proj.weight.data.clone()
    w2.weight.data = hf.down_proj.weight.data.clone()

    out_jd = C.gemma4_mlp_forward(x, w1, w2, w3)
    out_hf = hf(x)
    assert torch.allclose(out_jd, out_hf, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# 5. MoE experts (for-loop path) — gate=w1/up=w3 split + gelu-tanh + down
# ---------------------------------------------------------------------------


def test_experts_for_loop_split_and_activation():
    _seed()
    E, dim, inter = 4, 32, 48
    cfg = _gemma4_cfg.Gemma4TextConfig(
        hidden_size=dim, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        enable_moe_block=True, num_experts=E, top_k_experts=2, moe_intermediate_size=inter,
    )
    hf = _gemma4_hf.Gemma4TextExperts(cfg)
    # HF allocates expert weights with torch.empty (uninitialized → NaN); fill them.
    with torch.no_grad():
        hf.gate_up_proj.normal_(0, 0.02)
        hf.down_proj.normal_(0, 0.02)
    gate_up = hf.gate_up_proj.data  # [E, 2*inter, dim]
    down = hf.down_proj.data        # [E, dim, inter]
    w1 = gate_up[:, :inter, :].contiguous()   # gate
    w3 = gate_up[:, inter:, :].contiguous()   # up
    w2 = down.contiguous()

    counts = torch.tensor([3, 1, 2, 4])
    total = int(counts.sum())
    x = torch.randn(total, dim)

    out_jd = C.gemma4_experts_for_loop(w1, w2, w3, x, counts)

    import torch.nn.functional as F
    outs = []
    start = 0
    for e, n in enumerate(counts.tolist()):
        xe = x[start:start + n]
        start += n
        gate = xe @ gate_up[e, :inter, :].T
        up = xe @ gate_up[e, inter:, :].T
        h = F.gelu(gate, approximate="tanh") * up
        outs.append(h @ down[e].T)
    out_ref = torch.cat(outs, dim=0)
    assert torch.allclose(out_jd, out_ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# 6. Router math (selected-experts set) vs HF Gemma4TextRouter
# ---------------------------------------------------------------------------


def test_router_selection_matches_hf():
    _seed()
    dim, E, topk = 32, 8, 2
    cfg = _gemma4_cfg.Gemma4TextConfig(
        hidden_size=dim, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        enable_moe_block=True, num_experts=E, top_k_experts=topk, moe_intermediate_size=48,
    )
    hf = _gemma4_hf.Gemma4TextRouter(cfg)
    with torch.no_grad():
        hf.proj.weight.normal_(0, 0.02)
        hf.scale.normal_(1.0, 0.05)
        hf.per_expert_scale.normal_(1.0, 0.05)
    x = torch.randn(5, dim)

    # Replicate titaninfra router math with HF weights (norm has no scale).
    # HF order (modeling_gemma4.py:1348-1365): norm -> *scale*hidden^-0.5 -> proj
    # -> softmax(dim=-1) -> topk -> /sum -> *per_expert_scale[idx].
    norm = C.Gemma4RMSNorm(dim, with_scale=False)
    h = norm(x) * hf.scale * (dim ** -0.5)
    scores = h @ hf.proj.weight.T
    probs = torch.softmax(scores, dim=-1)
    top_w, sel = torch.topk(probs, k=topk, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w * hf.per_expert_scale[sel]

    _, hf_w, hf_sel = hf(x)  # (router_probabilities, top_k_weights, top_k_index)
    assert torch.equal(sel, hf_sel)
    assert torch.allclose(top_w, hf_w, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
