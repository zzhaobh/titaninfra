import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

# Load the pure-torch HC helper DIRECTLY from its file, bypassing the package
# __init__ (which imports config_registry → torchtitan → triton, unavailable on
# this Mac). This keeps the Task-1 helper fully testable locally.
_HC_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "titaninfra"
    / "models"
    / "deepseek_v4"
    / "hyper_connection.py"
)
_spec = importlib.util.spec_from_file_location("_dsv4_hyper_connection", _HC_PATH)
_hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc)
HC_MIX_WIDTH = _hc.HC_MIX_WIDTH
hc_split_sinkhorn = _hc.hc_split_sinkhorn
hc_pre_fn = _hc.hc_pre_fn
hc_head_fn = _hc.hc_head_fn


def test_mix_width():
    assert HC_MIX_WIDTH(4) == (2 + 4) * 4  # 24


def test_split_shapes_and_stochastic():
    hc_mult, iters, eps = 4, 20, 1e-6
    B, S = 2, 3
    mixes = torch.randn(B, S, HC_MIX_WIDTH(hc_mult), dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(HC_MIX_WIDTH(hc_mult), dtype=torch.float32)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc_mult, iters, eps)
    assert pre.shape == (B, S, hc_mult)
    assert post.shape == (B, S, hc_mult)
    assert comb.shape == (B, S, hc_mult, hc_mult)
    row_sums = comb.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert (pre > 0).all() and (post > 0).all()


def _col_dev(comb):
    # deviation of column-sums from 1 (0 == perfectly doubly-stochastic columns)
    return (comb.sum(dim=-2) - 1).abs().max()


def test_sinkhorn_iters_reduce_col_deviation():
    # The final row-normalization makes row-sums ~1 regardless of the loop, so
    # to prove the Sinkhorn loop does real work we check COLUMN-sum deviation:
    # more iterations must push comb closer to doubly-stochastic.
    torch.manual_seed(0)
    hc_mult, eps = 4, 1e-6
    B, S = 2, 3
    mixes = torch.randn(B, S, HC_MIX_WIDTH(hc_mult), dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(HC_MIX_WIDTH(hc_mult), dtype=torch.float32)
    _, _, comb_0 = hc_split_sinkhorn(mixes, scale, base, hc_mult, 0, eps)
    _, _, comb_20 = hc_split_sinkhorn(mixes, scale, base, hc_mult, 20, eps)
    assert _col_dev(comb_20) < _col_dev(comb_0)


def test_wrong_width_mixes_raises():
    hc_mult, iters, eps = 4, 20, 1e-6
    bad = torch.randn(2, 3, HC_MIX_WIDTH(hc_mult) + 1, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(HC_MIX_WIDTH(hc_mult), dtype=torch.float32)
    with pytest.raises(ValueError):
        hc_split_sinkhorn(bad, scale, base, hc_mult, iters, eps)


def test_hc_pre_fn_bf16_weight_fp32_math():
    # Regression: FSDP2 mixed precision casts hc_fn to the bf16 compute dtype
    # while hc_pre_fn upcasts the activation to fp32. F.linear(fp32, bf16) used
    # to raise "expected mat1 and mat2 to have the same dtype" at GPU runtime.
    # The weight must be cast to fp32 internally so the all-fp32 HC math holds.
    hc_mult, iters, eps = 4, 20, 1e-6
    B, S, d = 2, 3, 16
    x = torch.randn(B, S, hc_mult, d, dtype=torch.bfloat16)
    hc_fn = torch.randn(HC_MIX_WIDTH(hc_mult), hc_mult * d, dtype=torch.bfloat16)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(HC_MIX_WIDTH(hc_mult), dtype=torch.float32)
    y, post, comb = hc_pre_fn(x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps)
    assert y.dtype == torch.bfloat16 and y.shape == (B, S, d)
    assert post.shape == (B, S, hc_mult) and comb.shape == (B, S, hc_mult, hc_mult)


def test_hc_head_fn_bf16_weight_fp32_math():
    # Same regression as hc_pre_fn, for the model-level head collapse.
    eps = 1e-6
    B, S, d, hc_mult = 2, 3, 16, 4
    x = torch.randn(B, S, hc_mult, d, dtype=torch.bfloat16)
    head_fn = torch.randn(hc_mult, hc_mult * d, dtype=torch.bfloat16)
    head_scale = torch.randn(hc_mult, dtype=torch.float32)
    head_base = torch.randn(hc_mult, dtype=torch.float32)
    y = hc_head_fn(x, head_fn, head_scale, head_base, eps)
    assert y.dtype == torch.bfloat16 and y.shape == (B, S, d)
