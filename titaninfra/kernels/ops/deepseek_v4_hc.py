"""DeepSeek V4 Hyper-Connection kernels (torch.compile-fused, opt-in).

Registers three kernels — ``hc_pre`` / ``hc_post`` / ``hc_head`` — into the
titaninfra kernel registry. Each is launch-bound (RMS → linear → 20-iter Sinkhorn →
weighted-sum, run 2×/layer + once at the head → ~90-120 calls/step on Flash/Pro,
each spraying dozens of tiny fp32 kernels). The optimization is to let
``torch.compile`` fuse that chain — NOT a hand-written ``autograd.Function``
(which would sever compile fusion; see titaninfra/kernels/wiring.py).

Contract:
- ``pytorch_ref`` = the eager pure fn (titaninfra.models.deepseek_v4.hyper_connection).
- ``custom_impl`` = the SAME pure fn wrapped in ``torch.compile`` (fullgraph,
  dynamic shapes). Parity is exact by construction (identical fn, compiled vs
  eager) — ``assert_parity`` therefore verifies the compiled graph is
  numerically faithful fwd+bwd, and guards against a torch/inductor regression.

Because the compiled callables are module-level singletons, all layers share one
compiled artifact (avoids per-layer recompiles — the same reasoning as
FlexInnerAttention's ClassVar compiled fn).
"""

from __future__ import annotations

import importlib.util
import pathlib

import torch

from titaninfra.kernels.registry import KernelSpec, register_kernel

# Import the pure HC math functions WITHOUT triggering the deepseek_v4 package
# __init__ (which pulls torchtitan → triton, unavailable in --no-deps / Mac).
# hyper_connection.py is pure-torch, so loading it by file path is safe and lets
# these kernels register in any environment (registration must not depend on the
# torchtitan-heavy model package).
_HC_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "titaninfra"
    / "models"
    / "deepseek_v4"
    / "hyper_connection.py"
)
_spec = importlib.util.spec_from_file_location("_dsv4_hc_pure", _HC_PATH)
_hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc)
hc_pre_fn = _hc.hc_pre_fn
hc_post_fn = _hc.hc_post_fn
hc_head_fn = _hc.hc_head_fn

# Module-level compiled singletons (one artifact shared across all layers).
# dynamic=True: seq_len varies across recipes; avoid a recompile per shape.
_hc_pre_compiled = torch.compile(hc_pre_fn, fullgraph=True, dynamic=True)
_hc_post_compiled = torch.compile(hc_post_fn, fullgraph=True, dynamic=True)
_hc_head_compiled = torch.compile(hc_head_fn, fullgraph=True, dynamic=True)


register_kernel(
    KernelSpec(
        name="hc_pre",
        pytorch_ref=hc_pre_fn,
        custom_impl=_hc_pre_compiled,
        # fp32 Sinkhorn + weighted sum; tight tol (same fn, compiled vs eager).
        rtol=1e-4,
        atol=1e-4,
        # torch.compile/inductor Triton has no H200/fp8 requirement (unlike the
        # tilelang default sm90). Any sm80+ GPU (A100 and up) can run the fused
        # path; gating at (9,0) would silently fall back to eager on A100.
        min_cuda_capability=(8, 0),
    )
)

register_kernel(
    KernelSpec(
        name="hc_post",
        pytorch_ref=hc_post_fn,
        custom_impl=_hc_post_compiled,
        rtol=1e-4,
        atol=1e-4,
        min_cuda_capability=(8, 0),
    )
)

register_kernel(
    KernelSpec(
        name="hc_head",
        pytorch_ref=hc_head_fn,
        custom_impl=_hc_head_compiled,
        rtol=1e-4,
        atol=1e-4,
        min_cuda_capability=(8, 0),
    )
)
