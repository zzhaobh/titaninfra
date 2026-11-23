# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GPU + Triton parity test for the CSA sparse-gather attention forward.

The kernel is GPU/Triton only; this module skips wholesale on a CPU/Mac box and
becomes a real assertion on the GPU node. The oracle (``_dense_ref``) is the
dense "attend-over-all-T + scatter(-inf on non-topk) + active sink" path the
gather kernel replaces.
"""

import pytest

pytestmark = pytest.mark.xdist_group("dsv4_cuda_kernels")

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("gather kernel is GPU-only", allow_module_level=True)
pytest.importorskip("triton")

from titaninfra.models.deepseek_v4.sparse_attn import (  # noqa: E402
    sparse_attn_gather,
    sparse_attn_gather_fwd,
)


def _dense_ref(q, kv, attn_sink, topk_idxs, scale):
    B, H, S, D = q.shape
    T = kv.shape[2]
    k = torch.repeat_interleave(kv, H, dim=1)  # [B,H,T,D]
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B,H,S,T]
    valid = topk_idxs >= 0
    safe = torch.where(valid, topk_idxs, torch.full_like(topk_idxs, T))
    bias = q.new_full((B, 1, S, T + 1), float("-inf"))
    bias.scatter_(-1, safe.long().unsqueeze(1), 0.0)
    attn = attn + bias[..., :T]
    sinks = attn_sink.view(1, H, 1, 1).expand(B, H, S, 1)
    logits = torch.cat([attn, sinks], dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = logits.softmax(dim=-1)[..., :-1]
    return torch.matmul(probs, k)


def test_gather_fwd_matches_dense():
    torch.manual_seed(0)
    B, H, S, T, D, k = 1, 4, 64, 32, 64, 8
    q = torch.randn(B, H, S, D, device="cuda")
    kv = torch.randn(B, 1, T, D, device="cuda")
    sink = torch.randn(H, device="cuda")
    topk = torch.randint(0, T, (B, S, k), device="cuda", dtype=torch.int32)
    scale = D**-0.5
    ref = _dense_ref(q, kv, sink, topk, scale)
    out = sparse_attn_gather_fwd(q, kv, sink, topk, scale)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_gather_bwd_matches_dense_autograd():
    torch.manual_seed(0)
    B, H, S, T, D, k = 1, 4, 32, 16, 32, 6
    topk = torch.randint(0, T, (B, S, k), device="cuda", dtype=torch.int32)
    scale = D**-0.5
    q1 = torch.randn(B, H, S, D, device="cuda", dtype=torch.float64, requires_grad=True)
    kv1 = torch.randn(B, 1, T, D, device="cuda", dtype=torch.float64, requires_grad=True)
    s1 = torch.randn(H, device="cuda", dtype=torch.float64, requires_grad=True)
    out1 = sparse_attn_gather(q1, kv1, s1, topk, scale)
    out1.sum().backward()
    q2 = q1.detach().clone().requires_grad_()
    kv2 = kv1.detach().clone().requires_grad_()
    s2 = s1.detach().clone().requires_grad_()
    out2 = _dense_ref(q2, kv2, s2, topk, scale)
    out2.sum().backward()
    torch.testing.assert_close(q1.grad, q2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(kv1.grad, kv2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s1.grad, s2.grad, rtol=1e-5, atol=1e-5)


def test_gather_fwd_matches_dense_with_masked_slots():
    """Rows carrying ``-1`` (skipped) slots, incl. a fully-masked row where only
    the sink survives (output must be all-zeros there, denom == exp(0))."""
    torch.manual_seed(1)
    B, H, S, T, D, k = 2, 3, 48, 40, 64, 6
    q = torch.randn(B, H, S, D, device="cuda")
    kv = torch.randn(B, 1, T, D, device="cuda")
    sink = torch.randn(H, device="cuda")
    topk = torch.randint(0, T, (B, S, k), device="cuda", dtype=torch.int32)
    # Scatter some -1 masks; force the first query row of batch 0 fully masked.
    mask = torch.rand(B, S, k, device="cuda") < 0.3
    topk = torch.where(mask, torch.full_like(topk, -1), topk)
    topk[0, 0, :] = -1
    scale = D**-0.5
    ref = _dense_ref(q, kv, sink, topk, scale)
    out = sparse_attn_gather_fwd(q, kv, sink, topk, scale)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
