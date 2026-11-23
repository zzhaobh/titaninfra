"""Regression test locking in hc_split_sinkhorn's `post` gate formula against
the REAL DeepSeek-V4-Flash reference numerics (not just structural properties
like shape/sign/convergence, which the existing test_hyper_connection.py
already covers and which would have passed even with the bug).

Context (see docs/torchtitan_upgrade_plan_2026-11.md §34.2/§35): titaninfra's
hc_split_sinkhorn previously computed `post = sigmoid(x) + eps`. Independently
verified via (a) upstream torchtitan's new HcSplitSinkhorn.forward()
(torchtitan/models/deepseek_v4/mhc.py) and (b) the real DeepSeek-V4-Flash
reference kernel.py (hc_split_sinkhorn_kernel_, fetched from the exact HF
commit URL upstream's own docstring cites: deepseek-ai/DeepSeek-V4-Flash/
blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L394) that
the correct formula is `post = 2 * sigmoid(x)` (no +eps) -- exactly 2x off
from what titaninfra had, not a rounding difference. No prior test caught
this because none compared against real reference values numerically.

This test hand-computes the reference formula directly (not importing
hc_split_sinkhorn's own `post` line, which would trivially match itself) and
asserts the module's output matches it bit-for-bit.
"""

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

_HC_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "titaninfra"
    / "models"
    / "deepseek_v4"
    / "hyper_connection.py"
)
_spec = importlib.util.spec_from_file_location("_dsv4_hyper_connection_regress", _HC_PATH)
_hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc)
HC_MIX_WIDTH = _hc.HC_MIX_WIDTH
hc_split_sinkhorn = _hc.hc_split_sinkhorn


def test_post_gate_matches_real_dsv4_flash_reference_formula():
    """post = 2 * sigmoid(post_raw * scale[1] + base_post) -- no +eps.

    Hand-derives the expected `post` from the raw inputs using the formula
    confirmed against the real reference kernel, independent of
    hc_split_sinkhorn's own implementation.
    """
    torch.manual_seed(0)
    hc_mult, iters, eps = 4, 20, 1e-6
    B, S = 2, 3
    mixes = torch.randn(B, S, HC_MIX_WIDTH(hc_mult), dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(HC_MIX_WIDTH(hc_mult), dtype=torch.float32)

    _, post_raw, _ = torch.split(mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    _, base_post, _ = torch.split(hc_base, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    expected_post = 2 * torch.sigmoid(post_raw * hc_scale[1] + base_post)

    _, post, _ = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    assert torch.equal(post, expected_post)

    # The old (buggy) formula would have been sigmoid(x) + eps -- exactly
    # half of the correct 2*sigmoid(x) value (eps is negligible at this
    # scale). Explicitly assert the fix did NOT regress to that formula.
    buggy_post = torch.sigmoid(post_raw * hc_scale[1] + base_post) + eps
    assert not torch.allclose(post, buggy_post, atol=1e-3)


def test_post_gate_range_is_0_to_2_not_0_to_1():
    """2*sigmoid(x) ranges over (0, 2); the old sigmoid(x)+eps ranged over
    roughly (eps, 1+eps) -- a coarse but direct range-based sanity check
    that would catch a regression to the old formula even without hand-
    deriving the exact expected tensor."""
    torch.manual_seed(1)
    hc_mult, iters, eps = 4, 20, 1e-6
    B, S = 4, 8
    mixes = torch.randn(B, S, HC_MIX_WIDTH(hc_mult), dtype=torch.float32) * 5
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(HC_MIX_WIDTH(hc_mult), dtype=torch.float32)

    _, post, _ = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    assert post.max() > 1.0, (
        f"post.max()={post.max().item()} <= 1.0 -- looks like the old "
        "sigmoid(x)+eps formula (range ~(eps, 1+eps)), not 2*sigmoid(x) "
        "(range (0, 2))"
    )
    assert post.min() > 0.0
    assert post.max() <= 2.0
