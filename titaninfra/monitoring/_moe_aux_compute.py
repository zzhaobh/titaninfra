"""Pure compute for MoE auxiliary load-balancing loss (CPU-testable).

L_aux = N * sum_i(f_i * P_i), where f_i is the detached routed-slot fraction
and P_i is the differentiable mean routing probability. Balanced routing
approaches 1.0; collapsed routing approaches N. This matches the Switch/GShard
convention and HF load_balancing_loss_func divided by top_k.
"""

from __future__ import annotations

import torch


def normalize_counts(counts: torch.Tensor) -> torch.Tensor:
    """Normalize expert counts to a detached fraction vector summing to one.

    All-zero input returns a uniform vector instead of NaN.
    """
    counts = counts.detach().float()
    total = counts.sum()
    n = counts.shape[0]
    if total <= 0:
        return torch.full((n,), 1.0 / n, dtype=counts.dtype, device=counts.device)
    return counts / total


def compute_load_balancing_loss(
    p_sum: torch.Tensor,
    token_count: int | float,
    f_global: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Compute N * sum_i(f_global[i].detach() * P_i).

    Args:
        p_sum: Gradient-carrying per-expert summed probabilities, shape (N,).
        token_count: Number of tokens contributing to p_sum.
        f_global: Detached routed-slot fractions, shape (N,).
        n_experts: Number of experts, N.

    Returns:
        Scalar tensor whose gradient flows only through p_sum.
    """
    if token_count <= 0:
        return p_sum.sum() * 0.0  # Preserve graph connectivity with value zero.
    p_mean = p_sum / float(token_count)
    return n_experts * (f_global.detach() * p_mean).sum()


def compute_sequence_load_balancing_loss(
    probs: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_ids: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Compute the mean per-sequence DeepSeek V3/V4 load-balancing loss.

    For each sequence, f_i is the detached top-k hit fraction and P_i is the
    mean differentiable routing probability. Padding sequence IDs below zero
    are ignored. Gradient flows only through probs.
    """
    if probs.shape[0] == 0:
        return probs.sum() * 0.0
    device = probs.device
    seq_losses = []
    for s in torch.unique(seq_ids).tolist():
        if s < 0:
            continue  # Padding tokens do not contribute to sequence statistics.
        mask = seq_ids == s
        t_seq = int(mask.sum().item())
        if t_seq == 0:
            continue
        p_seq = probs[mask]
        sel_seq = topk_indices[mask]
        # f_i: detached top-k hit fraction within this sequence.
        f = torch.zeros(n_experts, dtype=torch.float32, device=device)
        f.scatter_add_(
            0,
            sel_seq.reshape(-1),
            torch.ones(sel_seq.numel(), dtype=torch.float32, device=device),
        )
        total = f.sum()
        if total > 0:
            f = (f / total).detach()
        # P_i: differentiable mean routing probability within this sequence.
        p_mean = p_seq.mean(dim=0)
        seq_losses.append(n_experts * (f * p_mean).sum())
    if not seq_losses:
        return probs.sum() * 0.0
    return torch.stack(seq_losses).mean()
