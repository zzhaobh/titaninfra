"""Tests for titaninfra.profiling.op_efficiency (CPU-safe, pure functions)."""

import math

from titaninfra.profiling.op_efficiency import (
    REFMOE_20B_DIMS,
    ROOFS,
    achieved_flops,
    bandwidth_util,
    headroom_ms,
    mfu,
    per_call_flops,
)

D = REFMOE_20B_DIMS
TOKENS = 8 * 4096  # batch 8 × seq_len 4096


def test_gemm_regions_have_positive_flops():
    for region in ("attn", "moe.experts", "moe.shared", "moe.router"):
        f = per_call_flops(region, D, tokens=TOKENS, batch=8, seq_len=4096)
        assert f is not None and f > 0


def test_memory_bound_region_returns_none():
    assert per_call_flops("attn_norm", D, tokens=TOKENS, batch=8, seq_len=4096) is None
    assert per_call_flops("ffn_norm", D, tokens=TOKENS, batch=8, seq_len=4096) is None


def test_flops_scale_linearly_with_tokens():
    f1 = per_call_flops("moe.experts", D, tokens=TOKENS, batch=8, seq_len=4096)
    f2 = per_call_flops("moe.experts", D, tokens=2 * TOKENS, batch=16, seq_len=4096)
    assert math.isclose(f2, 2 * f1, rel_tol=1e-9)


def test_experts_dominate_router_flops():
    # experts (top_k=4 × SwiGLU 768) ≫ router gate (dim×128)
    exp = per_call_flops("moe.experts", D, tokens=TOKENS, batch=8, seq_len=4096)
    rtr = per_call_flops("moe.router", D, tokens=TOKENS, batch=8, seq_len=4096)
    assert exp > 50 * rtr


def test_mfu_in_unit_range_and_monotonic_in_time():
    roof = ROOFS["H200"]
    pc = per_call_flops("moe.experts", D, tokens=TOKENS, batch=8, seq_len=4096)
    flop_bound_ms = (pc * 31) / roof.peak_flops * 1000.0
    assert math.isclose(mfu(pc, 31, flop_bound_ms, roof), 1.0, rel_tol=1e-6)
    fast = mfu(pc, 31, 2 * flop_bound_ms, roof)
    slow = mfu(pc, 31, 4 * flop_bound_ms, roof)
    assert 0.0 < slow < fast < 1.0


def test_achieved_flops_zero_on_zero_time():
    assert achieved_flops(1e12, 10, 0.0) == 0.0


def test_bandwidth_util_ratio():
    roof = ROOFS["H200"]  # nvlink 900e9 B/s
    u = bandwidth_util(bytes_moved=900e9, time_ms=1000.0, roof=roof)
    assert math.isclose(u, 1.0, rel_tol=1e-6)


def test_headroom_decreases_with_utilization():
    assert headroom_ms(10.0, 0.0) == 10.0
    assert headroom_ms(10.0, 1.0) == 0.0
    assert math.isclose(headroom_ms(10.0, 0.25), 7.5)
    assert headroom_ms(10.0, 1.5) == 0.0


def test_h20_low_compute_roof_inflates_mfu():
    pc = per_call_flops("attn", D, tokens=TOKENS, batch=8, seq_len=4096)
    m_h200 = mfu(pc, 32, 20.0, ROOFS["H200"])
    m_h20 = mfu(pc, 32, 20.0, ROOFS["H20"])
    assert m_h20 > m_h200
