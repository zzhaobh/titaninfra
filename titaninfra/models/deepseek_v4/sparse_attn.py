# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DeepSeek-V4 CSA sparse (gathered) attention forward — Triton (STAGE B).

Computes, for each query, an online-softmax attention over ONLY its ``<= k``
gathered compressed-KV slots (plus a per-head sink logit), instead of the dense
"attend-over-all-T-slots + scatter a ``-inf`` bias on non-topk columns" path
It is numerically equivalent to upstream CSA inner-attention — the parity oracle
is the ``_dense_ref`` in ``tests/models/deepseek_v4/test_sparse_attn_gather.py``
plus the model-level upstream comparison in ``test_compressed_attention.py`` —
but never materializes the ``[B, H, S, T]`` score matrix.

GPU + Triton ONLY. Forward + backward with an explicit opt-in attention branch.

Key semantics mirrored from the dense oracle:

* ``k = repeat_interleave(kv, H)`` — the single shared V4 latent KV head is
  broadcast to all ``H`` query heads; every head reads the SAME kv rows.
* ``attn = q @ kᵀ * scale`` — the ``scale`` multiplies the q·k dot product.
* The per-head sink ``attn_sink[h]`` is concatenated as an extra logit column
  AFTER scaling, so it is a RAW logit (never multiplied by ``scale``). It
  participates in the softmax denominator, then is dropped from the output —
  so the returned attention can sum to ``< 1`` (this is intentional; the sink
  also guarantees a finite softmax when every real column is masked out).
* ``topk_idxs == -1`` entries are masked out (skipped).
* Duplicate indices within a query's topk row collapse to a SINGLE column: the
  dense oracle builds the mask with ``bias.scatter_(-1, safe, 0.0)``, so an
  index appearing twice still yields one ``0.0`` (kept) column. A gather that
  looped over raw slots would double-count. We therefore dedup each row to
  ``-1``-padded distinct indices before the kernel loop (real Lightning-Indexer
  topk is already distinct, so this is a safety net, not the common path).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["sparse_attn_gather", "sparse_attn_gather_fwd", "SparseAttnGather"]


@triton.jit
def _sparse_attn_gather_kernel(
    q_ptr,
    kv_ptr,
    sink_ptr,
    idx_ptr,
    o_ptr,
    lse_ptr,
    scale,
    H,
    S,
    K,
    D,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kvb,
    stride_kvt,
    stride_kvd,
    stride_ib,
    stride_is,
    stride_ik,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    BLOCK_D: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    """One program == one (b, h, s) query row.

    Loads the ``[D]`` query vector, seeds the running softmax state with the
    raw per-head sink logit, then streams the query's ``K`` gathered kv rows
    through an online softmax (``ACC_DTYPE`` accumulation — fp32 for bf16/fp32
    inputs, fp64 for float64 inputs so the gradient-parity test is exact).
    Invalid slots (``idx < 0``, including deduped duplicates) contribute nothing
    and leave the state intact. Also stores the log-sum-exp ``lse = m + log(l)``
    (over the gathered slots AND the sink column) so the backward can recompute
    the softmax probabilities flash-attention style without materializing a
    ``[B, H, S, k]`` prob tensor.
    """
    pid = tl.program_id(0)
    # Decode the flattened (b, h, s) program id.
    s = pid % S
    hb = pid // S
    h = hb % H
    b = hb // H

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    # Load q row [D] (ACC_DTYPE accumulation regardless of input dtype).
    q_ptrs = q_ptr + b * stride_qb + h * stride_qh + s * stride_qs + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(ACC_DTYPE)

    # Seed the online softmax with the sink column: m = sink, l = exp(0) = 1,
    # acc = 0 (the sink carries no value vector — it is dropped from the output).
    sink = tl.load(sink_ptr + h).to(ACC_DTYPE)
    m_i = sink
    l_i = tl.cast(1.0, ACC_DTYPE)  # triton>=3.9 rejects loop-carried fp32->fp64 retype
    acc = tl.zeros([BLOCK_D], dtype=ACC_DTYPE)

    idx_row = idx_ptr + b * stride_ib + s * stride_is
    # Runtime trip count K (no per-seq_len recompile).
    for i in range(0, K):
        idx = tl.load(idx_row + i * stride_ik)
        valid = idx >= 0
        safe_idx = tl.where(valid, idx, 0)  # keep pointer arithmetic in-bounds

        kv_ptrs = kv_ptr + b * stride_kvb + safe_idx * stride_kvt + offs_d * stride_kvd
        kvec = tl.load(kv_ptrs, mask=d_mask & valid, other=0.0).to(ACC_DTYPE)

        score = tl.sum(q * kvec, axis=0) * scale
        score = tl.where(valid, score, float("-inf"))

        m_new = tl.maximum(m_i, score)
        corr = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)  # invalid -> exp(-inf) == 0, no contribution
        l_i = l_i * corr + p
        acc = acc * corr + p * kvec
        m_i = m_new

    out = acc / l_i  # l_i >= exp(sink - m) > 0 always (sink guarantees finite denom)
    o_ptrs = o_ptr + b * stride_ob + h * stride_oh + s * stride_os + offs_d * stride_od
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=d_mask)

    lse = m_i + tl.log(l_i)
    tl.store(lse_ptr + b * stride_lb + h * stride_lh + s * stride_ls, lse)


@triton.jit
def _sparse_attn_gather_bwd_kernel(
    q_ptr,
    kv_ptr,
    sink_ptr,
    idx_ptr,
    o_ptr,
    lse_ptr,
    do_ptr,
    dq_ptr,
    dkv_ptr,
    dsink_ptr,
    scale,
    H,
    S,
    K,
    D,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kvb,
    stride_kvt,
    stride_kvd,
    stride_ib,
    stride_is,
    stride_ik,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dqd,
    stride_dkvb,
    stride_dkvt,
    stride_dkvd,
    BLOCK_D: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    """One program == one (b, h, s) query row — the backward of the fwd kernel.

    Recomputes each gathered slot's softmax weight ``p_t = exp(score_t - lse)``
    (flash-attention style, from the saved ``lse``) and the sink weight
    ``p_sink = exp(sink - lse)``, then applies the standard softmax-with-value
    backward. Because the dense oracle uses ``kv`` as BOTH the key (``q@kᵀ``) and
    the value (``probs@k``), every kv row accumulates two contributions:

      * value path : ``dv_t = p_t · do``
      * key   path : ``dk_t = ds_t · scale · q``

    where ``ds_t = p_t (dp_t − δ)``, ``dp_t = do · v_t`` and the softmax-jacobian
    correction ``δ = Σ_j p_j dp_j`` collapses to ``do · o`` (the sink term drops
    since ``v_sink = 0``). The sink logit's gradient is ``−p_sink · δ``. ``dq`` is
    written once per row (no atomics); ``dkv`` and ``d_sink`` are scattered with
    ``tl.atomic_add`` (many query rows / heads hit the same kv row / head). All
    accumulation is in ``ACC_DTYPE`` (fp32 for bf16/fp32 inputs, fp64 for float64
    inputs — matching the ``dkv``/``dsink`` scratch buffer dtype so the atomic
    scatter with signed cancellation is exact for the gradient-parity test).
    """
    pid = tl.program_id(0)
    s = pid % S
    hb = pid // S
    h = hb % H
    b = hb // H

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q_ptrs = q_ptr + b * stride_qb + h * stride_qh + s * stride_qs + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(ACC_DTYPE)

    o_ptrs = o_ptr + b * stride_ob + h * stride_oh + s * stride_os + offs_d * stride_od
    o = tl.load(o_ptrs, mask=d_mask, other=0.0).to(ACC_DTYPE)

    do_ptrs = do_ptr + b * stride_ob + h * stride_oh + s * stride_os + offs_d * stride_od
    do = tl.load(do_ptrs, mask=d_mask, other=0.0).to(ACC_DTYPE)

    sink = tl.load(sink_ptr + h).to(ACC_DTYPE)
    lse = tl.load(lse_ptr + b * stride_lb + h * stride_lh + s * stride_ls).to(ACC_DTYPE)

    # δ = Σ_j p_j (do·v_j) = do·o  (sink drops: v_sink = 0). Flash-attn's D term.
    delta = tl.sum(do * o, axis=0)

    # d(sink logit) = p_sink (dp_sink − δ) = −p_sink·δ  (dp_sink = do·v_sink = 0).
    p_sink = tl.exp(sink - lse)
    tl.atomic_add(dsink_ptr + h, -p_sink * delta)

    dq_acc = tl.zeros([BLOCK_D], dtype=ACC_DTYPE)

    idx_row = idx_ptr + b * stride_ib + s * stride_is
    for i in range(0, K):
        idx = tl.load(idx_row + i * stride_ik)
        valid = idx >= 0
        safe_idx = tl.where(valid, idx, 0)

        kv_ptrs = kv_ptr + b * stride_kvb + safe_idx * stride_kvt + offs_d * stride_kvd
        kvec = tl.load(kv_ptrs, mask=d_mask & valid, other=0.0).to(ACC_DTYPE)

        score = tl.sum(q * kvec, axis=0) * scale
        score = tl.where(valid, score, float("-inf"))
        p_t = tl.exp(score - lse)  # invalid -> exp(-inf) == 0

        dp_t = tl.sum(do * kvec, axis=0)  # do · v_t
        ds_t = p_t * (dp_t - delta)  # softmax-jacobian: p_t(dp_t − δ)

        # dq += Σ_t ds_t · scale · k_t   (score_t = scale · q·k_t).
        dq_acc += ds_t * scale * kvec

        # dkv row t += value path (p_t·do) + key path (ds_t·scale·q).
        dkv_vec = p_t * do + ds_t * scale * q
        dkv_ptrs = dkv_ptr + b * stride_dkvb + safe_idx * stride_dkvt + offs_d * stride_dkvd
        tl.atomic_add(dkv_ptrs, dkv_vec, mask=d_mask & valid)

    dq_ptrs = dq_ptr + b * stride_dqb + h * stride_dqh + s * stride_dqs + offs_d * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(dq_ptr.dtype.element_ty), mask=d_mask)


def _dedup_topk(topk_idxs: torch.Tensor) -> torch.Tensor:
    """Return a new ``[B, S, k]`` int tensor with per-row duplicates set to -1.

    Matches the dense oracle's ``scatter_`` set-semantics: each distinct valid
    index survives once; every other slot (repeats and the original ``-1``
    padding) becomes ``-1`` and is skipped by the kernel. Order within a row is
    irrelevant because the kernel sums over slots. ``O(k log k)`` via a sort —
    no ``[B, S, k, k]`` blow-up.
    """
    sorted_idx, _ = torch.sort(topk_idxs, dim=-1)
    same_as_prev = torch.zeros_like(sorted_idx, dtype=torch.bool)
    same_as_prev[..., 1:] = sorted_idx[..., 1:] == sorted_idx[..., :-1]
    neg_one = torch.full_like(sorted_idx, -1)
    return torch.where(same_as_prev, neg_one, sorted_idx)


def sparse_attn_gather_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """CSA sparse-gather attention forward (parity with the dense scatter path).

    Args:
        q: ``[B, H, S, D]`` query (already rope'd by the caller).
        kv: ``[B, 1, T, D]`` single shared-latent KV head, broadcast to all H.
        attn_sink: ``[H]`` per-head raw sink logit.
        topk_idxs: ``[B, S, k]`` int gather indices into ``T``; ``-1`` skipped.
        scale: softmax scale applied to the ``q·k`` dot product (NOT the sink).

    Returns:
        ``[B, H, S, D]`` attention output (element-wise close to ``_dense_ref``).
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [B, H, S, D], got shape {tuple(q.shape)}")
    if kv.ndim != 4 or kv.shape[1] != 1:
        raise ValueError(f"kv must be [B, 1, T, D], got shape {tuple(kv.shape)}")
    if topk_idxs.ndim != 3:
        raise ValueError(f"topk_idxs must be [B, S, k], got shape {tuple(topk_idxs.shape)}")

    B, H, S, D = q.shape
    if kv.shape[0] != B or kv.shape[3] != D:
        raise ValueError(f"kv {tuple(kv.shape)} incompatible with q {tuple(q.shape)}")
    if topk_idxs.shape[0] != B or topk_idxs.shape[1] != S:
        raise ValueError(f"topk_idxs {tuple(topk_idxs.shape)} incompatible with q {tuple(q.shape)}")
    if attn_sink.ndim != 1 or attn_sink.shape[0] != H:
        raise ValueError(f"attn_sink must be [H={H}], got shape {tuple(attn_sink.shape)}")

    # Set-semantics: collapse per-row duplicates to -1 (new tensor, no mutation).
    idx = _dedup_topk(topk_idxs.to(torch.int64).contiguous())
    o, _ = _run_fwd_kernel(q, kv, attn_sink, idx, float(scale))
    return o


def _acc_dtypes(q: torch.Tensor) -> tuple[torch.dtype, tl.dtype]:
    """Pick the kernel accumulator dtype from the input dtype.

    float64 inputs → fp64 accumulation (exact gradient parity for the float64
    autograd test); every other dtype (bf16/fp16/fp32) → fp32 accumulation
    (unchanged production behaviour/perf). Returns ``(torch_dtype, tl_dtype)``
    for the scratch buffers and the ``ACC_DTYPE`` constexpr respectively.
    """
    if q.dtype == torch.float64:
        return torch.float64, tl.float64
    return torch.float32, tl.float32


def _run_fwd_kernel(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    idx: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the fwd kernel over ALREADY-deduped ``idx``. Returns ``(o, lse)``.

    Shared by the public ``sparse_attn_gather_fwd`` and the autograd ``forward``
    so both run byte-identical kernels; ``lse`` (``[B, H, S]`` in the accumulator
    dtype) is consumed only by the backward.
    """
    B, H, S, D = q.shape
    k = idx.shape[2]
    acc_torch, acc_tl = _acc_dtypes(q)
    sink = attn_sink.to(acc_torch).contiguous()

    o = torch.empty_like(q)
    lse = torch.empty((B, H, S), device=q.device, dtype=acc_torch)

    grid = (B * H * S,)
    _sparse_attn_gather_kernel[grid](
        q,
        kv,
        sink,
        idx,
        o,
        lse,
        scale,
        H,
        S,
        k,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(2),
        kv.stride(3),
        idx.stride(0),
        idx.stride(1),
        idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        BLOCK_D=triton.next_power_of_2(D),
        ACC_DTYPE=acc_tl,
    )
    return o, lse


def sparse_attn_gather_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    idx: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of the sparse-gather attention. Returns ``(dq, dkv, d_attn_sink)``.

    ``idx`` MUST be the SAME deduped indices the forward used (set-semantics
    parity with ``_dense_ref``). ``dkv`` and ``d_attn_sink`` are scattered with
    ``tl.atomic_add`` into accumulator-dtype scratch (fp32 for bf16/fp32 inputs,
    fp64 for float64 inputs — signed atomic cancellation stays exact for the
    gradient-parity test), then cast back to the input dtypes. ``do``/``o`` share
    q's ``[B,H,S,D]`` layout (the bwd kernel indexes ``do`` with ``o``'s strides),
    so ``do`` is made contiguous here. Full derivation: see the bwd kernel
    docstring and ``task-8-report.md``.
    """
    B, H, S, D = q.shape
    k = idx.shape[2]
    acc_torch, acc_tl = _acc_dtypes(q)
    sink = attn_sink.to(acc_torch).contiguous()

    # do shares o's [B,H,S,D] layout; make contiguous so o's strides index it.
    do = do.contiguous()
    o = o.contiguous()
    dq = torch.empty_like(q)
    # Accumulator-dtype scratch for the atomic scatter-adds (see docstring).
    dkv_acc = torch.zeros((B, 1, kv.shape[2], D), device=q.device, dtype=acc_torch)
    dsink_acc = torch.zeros((H,), device=q.device, dtype=acc_torch)

    grid = (B * H * S,)
    _sparse_attn_gather_bwd_kernel[grid](
        q,
        kv,
        sink,
        idx,
        o,
        lse,
        do,
        dq,
        dkv_acc,
        dsink_acc,
        scale,
        H,
        S,
        k,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(2),
        kv.stride(3),
        idx.stride(0),
        idx.stride(1),
        idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        dkv_acc.stride(0),
        dkv_acc.stride(2),
        dkv_acc.stride(3),
        BLOCK_D=triton.next_power_of_2(D),
        ACC_DTYPE=acc_tl,
    )
    return dq, dkv_acc.to(kv.dtype), dsink_acc.to(attn_sink.dtype)


class SparseAttnGather(torch.autograd.Function):
    """Autograd wrapper around the Triton sparse-gather fwd/bwd kernels.

    Non-differentiable inputs (``topk_idxs``, ``scale``) return ``None`` grads.
    The deduped indices + ``lse`` are saved so the backward's recomputed
    softmax matches the forward exactly (set-semantics + flash-attn recompute).
    """

    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, scale):
        idx = _dedup_topk(topk_idxs.to(torch.int64).contiguous())
        o, lse = _run_fwd_kernel(q, kv, attn_sink, idx, float(scale))
        ctx.save_for_backward(q, kv, attn_sink, idx, o, lse)
        ctx.scale = float(scale)
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, attn_sink, idx, o, lse = ctx.saved_tensors
        dq, dkv, dsink = sparse_attn_gather_bwd(do, q, kv, attn_sink, idx, o, lse, ctx.scale)
        return dq, dkv, dsink, None, None


def sparse_attn_gather(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Autograd-enabled CSA sparse-gather attention (fwd + hand-written bwd).

    Same semantics/parity as ``sparse_attn_gather_fwd`` (see its docstring) but
    differentiable w.r.t. ``q``, ``kv``, ``attn_sink``. ``topk_idxs`` / ``scale``
    are non-differentiable. Gradients are element-wise close to autograd through
    the dense oracle ``_dense_ref``.
    """
    return SparseAttnGather.apply(q, kv, attn_sink, topk_idxs, scale)
