"""Parity + micro-benchmark for the DeepSeek V4 Hyper-Connection kernels.

Parity: the compiled ``custom_impl`` is the SAME pure fn as ``pytorch_ref``
wrapped in ``torch.compile``, so fwd+bwd must match tightly. This runs on CPU
(inductor CPU backend) — no triton/GPU needed, so it's a real local gate.

Benchmark: a rough CPU wall-clock ratio (compiled vs eager) as a smoke signal.
The authoritative >=3% MFU decision is made from the GPU profiling run
(case_40 + op_attribution ``refmoe/hc_*`` regions), NOT from this CPU timing.
"""

import time

import pytest

torch = pytest.importorskip("torch")

# The kernel ops module loads the pure HC math by file path (no torchtitan), so
# importing the registry works on this Mac.
import titaninfra.kernels  # noqa: E402,F401  (triggers registration)
from titaninfra.kernels import assert_parity, get_spec  # noqa: E402

_HC_MULT = 4
_DIM = 256
_ITERS = 20
_EPS = 1e-6
_MIX = (2 + _HC_MULT) * _HC_MULT


def _mk_hc_pre_inputs():
    torch.manual_seed(0)
    B, L = 2, 32
    x = torch.randn(B, L, _HC_MULT, _DIM, dtype=torch.float32, requires_grad=True)
    hc_fn = torch.randn(_MIX, _HC_MULT * _DIM, dtype=torch.float32, requires_grad=True) * 0.02
    hc_scale = torch.ones(3, dtype=torch.float32, requires_grad=True)
    hc_base = torch.zeros(_MIX, dtype=torch.float32, requires_grad=True)
    return (x, hc_fn, hc_scale, hc_base, _HC_MULT, _ITERS, _EPS)


def _mk_hc_head_inputs():
    torch.manual_seed(0)
    B, L = 2, 32
    x = torch.randn(B, L, _HC_MULT, _DIM, dtype=torch.float32, requires_grad=True)
    head_fn = torch.randn(_HC_MULT, _HC_MULT * _DIM, dtype=torch.float32, requires_grad=True) * 0.02
    head_scale = torch.ones(1, dtype=torch.float32, requires_grad=True)
    head_base = torch.zeros(_HC_MULT, dtype=torch.float32, requires_grad=True)
    return (x, head_fn, head_scale, head_base, _EPS)


def _mk_hc_post_inputs():
    torch.manual_seed(0)
    B, L = 2, 32
    x = torch.randn(B, L, _DIM, dtype=torch.float32, requires_grad=True)
    residual = torch.randn(B, L, _HC_MULT, _DIM, dtype=torch.float32, requires_grad=True)
    post = torch.rand(B, L, _HC_MULT, dtype=torch.float32, requires_grad=True)
    comb = torch.rand(B, L, _HC_MULT, _HC_MULT, dtype=torch.float32, requires_grad=True)
    return (x, residual, post, comb)


def test_hc_kernels_registered_with_sm80_gate():
    # torch.compile-fused HC has no H200/fp8 requirement (unlike the tilelang
    # sm90 default). Gate must be (8,0) so the fused path activates on A100+,
    # not silently fall back to eager. Regression guard for that fix.
    for name in ("hc_pre", "hc_post", "hc_head"):
        spec = get_spec(name)
        assert spec.custom_impl is not None, f"{name} custom_impl missing"
        assert spec.min_cuda_capability == (8, 0), (
            f"{name} min_cuda_capability={spec.min_cuda_capability}, expected (8,0)"
        )


def test_hc_pre_parity():
    # hc_pre returns a 3-tuple (y, post, comb); the shared assert_parity assumes
    # a single-tensor output, so compare tuple elements + backward here directly.
    spec = get_spec("hc_pre")
    base = _mk_hc_pre_inputs()

    def _clone(t):
        if isinstance(t, torch.Tensor):
            c = t.detach().clone()
            c.requires_grad_(t.requires_grad)
            return c
        return t

    ref_in = tuple(_clone(t) for t in base)
    impl_in = tuple(_clone(t) for t in base)
    ref_out = spec.pytorch_ref(*ref_in)
    impl_out = spec.custom_impl(*impl_in)
    for r, i in zip(ref_out, impl_out, strict=True):
        torch.testing.assert_close(i, r, rtol=spec.rtol, atol=spec.atol)

    # backward through the primary output y (element 0).
    g = torch.randn_like(ref_out[0])
    ref_out[0].backward(g)
    impl_out[0].backward(g.clone())
    for ri, ii in zip(ref_in, impl_in, strict=True):
        if isinstance(ri, torch.Tensor) and ri.requires_grad:
            torch.testing.assert_close(ii.grad, ri.grad, rtol=spec.rtol, atol=spec.atol)


def test_hc_head_parity():
    assert_parity("hc_head", _mk_hc_head_inputs, check_backward=True)


def test_hc_post_parity():
    assert_parity("hc_post", _mk_hc_post_inputs, check_backward=True)


@pytest.mark.parametrize("name,mk", [("hc_pre", _mk_hc_pre_inputs), ("hc_head", _mk_hc_head_inputs)])
def test_hc_compiled_runs_and_report_cpu_timing(name, mk):
    # Smoke: compiled impl runs fwd+bwd; print a rough CPU ratio (NOT the ship
    # gate — GPU profiling owns that). Warm up to exclude compile time.
    spec = get_spec(name)
    ref, impl = spec.pytorch_ref, spec.custom_impl

    def _run(fn, inputs):
        out = fn(*inputs)
        out0 = out[0] if isinstance(out, tuple) else out
        out0.sum().backward()

    for _ in range(3):  # warmup (triggers compile for impl)
        _run(ref, mk())
        _run(impl, mk())

    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        _run(ref, mk())
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n):
        _run(impl, mk())
    t_impl = time.perf_counter() - t0
    print(f"[{name}] CPU eager={t_ref/n*1e3:.3f}ms compiled={t_impl/n*1e3:.3f}ms "
          f"ratio={t_ref/max(t_impl,1e-9):.2f}x (GPU profile is the real gate)")
    assert t_impl > 0  # ran without error
