"""
QK-Clip — Per-head Q/K weight clipping for attention logit stability.

After each optimizer step, checks whether the maximum possible attention logit
||q_h|| * ||k_h|| / sqrt(d) exceeds threshold τ for any head. If so, scales
Q and K weights down so the product is bounded by τ.

For MLA (DeepSeek-V3/V4), follows Kimi K2's per-component scaling:
  - q_nope, k_nope (head-specific): scaled by √γ
  - q_rope (head-specific): scaled by γ (stronger, since rope contributes additively)
  - k_rope (shared across heads, lives in wkv_a): untouched

Prevents softmax saturation during Muon training where orthogonalized updates
can grow Q/K norms unboundedly.

Reference: Kimi K2 Technical Report §3.2 (QK-Clip + MuonClip Algorithm 1).
"""

import logging
import math
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_STANDARD_Q = re.compile(r"layers\.(\d+)\..*?\.wq\.weight$")
_STANDARD_K = re.compile(r"layers\.(\d+)\..*?\.wk\.weight$")
_FUSED_QKV = re.compile(r"layers\.(\d+)\..*?\.wqkv\.weight$")
_MLA_Q = re.compile(r"layers\.(\d+)\..*?\.wq_b\.weight$")
_MLA_KV = re.compile(r"layers\.(\d+)\..*?\.wkv_b\.weight$")


def _find_qk_params(
    model_parts: list[nn.Module],
) -> list[tuple[str, nn.Parameter, str, nn.Parameter, str]]:
    """Find paired Q/K weight parameters across all model parts.

    Supports three attention styles:
    - "standard": separate wq/wk (Qwen3, Qwen2.5, Gemma4, GPT-OSS)
    - "fused": wqkv single param (torchtitan FusedQKV variant)
    - "mla": wq_b/wkv_b (DeepSeek-V3/V4 Multi-head Latent Attention)

    Returns list of (q_name, q_param, k_name, k_param, mode) tuples.
    """
    std_q: dict[str, tuple[str, nn.Parameter]] = {}
    std_k: dict[str, tuple[str, nn.Parameter]] = {}
    fused: dict[str, tuple[str, nn.Parameter]] = {}
    mla_q: dict[str, tuple[str, nn.Parameter]] = {}
    mla_kv: dict[str, tuple[str, nn.Parameter]] = {}

    for model in model_parts:
        for name, param in model.named_parameters():
            m = _STANDARD_Q.search(name)
            if m:
                std_q[m.group(1)] = (name, param)
                continue
            m = _STANDARD_K.search(name)
            if m:
                std_k[m.group(1)] = (name, param)
                continue
            m = _FUSED_QKV.search(name)
            if m:
                fused[m.group(1)] = (name, param)
                continue
            m = _MLA_Q.search(name)
            if m:
                mla_q[m.group(1)] = (name, param)
                continue
            m = _MLA_KV.search(name)
            if m:
                mla_kv[m.group(1)] = (name, param)
                continue

    pairs = []

    for layer_idx in sorted(std_q.keys(), key=int):
        if layer_idx in std_k:
            qn, qp = std_q[layer_idx]
            kn, kp = std_k[layer_idx]
            pairs.append((qn, qp, kn, kp, "standard"))

    for layer_idx in sorted(mla_q.keys(), key=int):
        if layer_idx in mla_kv:
            qn, qp = mla_q[layer_idx]
            kn, kp = mla_kv[layer_idx]
            pairs.append((qn, qp, kn, kp, "mla"))

    for layer_idx in sorted(fused.keys(), key=int):
        fn, fp = fused[layer_idx]
        pairs.append((fn, fp, fn, fp, "fused"))

    return pairs


def _mla_clip_heads(
    q_data: torch.Tensor,
    k_data: torch.Tensor,
    head_dim: int,
    tau: float,
) -> int:
    """Apply per-component QK-Clip for MLA following Kimi K2's algorithm.

    MLA structure:
      wq_b: [n_heads * (nope_dim + rope_dim), q_lora_rank]
      wkv_b: [n_heads * (nope_dim + v_dim), kv_lora_rank]

    Per Kimi K2 Algorithm 1:
      γ_h = min(1, τ / S_max^h)
      q_nope_h ← √γ_h · q_nope_h
      k_nope_h ← √γ_h · k_nope_h
      q_rope_h ← γ_h · q_rope_h
      k_rope (shared, in wkv_a) ← unchanged

    We approximate S_max^h with the weight-norm upper bound:
      S_max^h ≈ ‖q_nope_h‖ · ‖k_nope_h‖ / √d_nope

    This is conservative (overestimates logit), so we clip slightly more than
    runtime-triggered clipping would. Safe for stability.

    Returns number of heads clipped.
    """
    q_out = q_data.shape[0]
    k_out = k_data.shape[0]

    # Infer per-head structure by trying common rope_dim values
    for rope_dim in (64, 128, 32):
        q_per_head = head_dim + rope_dim
        if q_out % q_per_head != 0:
            continue
        n_heads = q_out // q_per_head
        if k_out % n_heads != 0:
            continue
        k_per_head = k_out // n_heads
        if k_per_head < head_dim:
            continue

        lora_rank_q = q_data.shape[1]
        lora_rank_k = k_data.shape[1]

        # Reshape to [n_heads, per_head_dim, lora_rank]
        q_reshaped = q_data.view(n_heads, q_per_head, lora_rank_q)
        k_reshaped = k_data.view(n_heads, k_per_head, lora_rank_k)

        # Slice views: nope = [:head_dim], rope = [head_dim:]
        q_nope = q_reshaped[:, :head_dim, :]  # [n_heads, nope_dim, lora_rank_q]
        q_rope = q_reshaped[:, head_dim:, :]  # [n_heads, rope_dim, lora_rank_q]
        k_nope = k_reshaped[:, :head_dim, :]  # [n_heads, nope_dim, lora_rank_k]
        # k_rope is shared (in wkv_a), not in wkv_b — don't touch

        # Compute per-head nope norm product as S_max proxy
        sqrt_d = math.sqrt(head_dim)
        q_norms = q_nope.flatten(1).norm(dim=1)  # [n_heads]
        k_norms = k_nope.flatten(1).norm(dim=1)  # [n_heads]
        s_max = q_norms * k_norms / sqrt_d       # [n_heads]

        # Per-head γ_h = min(1, τ / S_max^h)
        gamma = torch.clamp(tau / (s_max + 1e-12), max=1.0)  # [n_heads]
        exceed_mask = gamma < 1.0

        if not exceed_mask.any():
            return 0

        # Apply Kimi K2 scaling:
        #   q_nope, k_nope ← √γ · (nope)
        #   q_rope ← γ · (rope)
        sqrt_gamma = torch.sqrt(gamma).view(-1, 1, 1)
        gamma_full = gamma.view(-1, 1, 1)

        q_nope.mul_(sqrt_gamma)
        k_nope.mul_(sqrt_gamma)
        q_rope.mul_(gamma_full)

        return int(exceed_mask.sum().item())

    logger.warning(
        f"QK-Clip MLA: cannot infer head structure from wq_b={list(q_data.shape)}, "
        f"wkv_b={list(k_data.shape)}, head_dim={head_dim}. Skipping."
    )
    return 0


@torch.no_grad()
def _qk_clip_step(
    qk_pairs: list[tuple[str, nn.Parameter, str, nn.Parameter, str]],
    tau: float,
    head_dim: int,
) -> int:
    """Clip Q/K weights per head where max attention logit exceeds tau.

    Handles three modes:
    - "standard": separate Q/K, symmetric √γ scaling on both
    - "fused": single wqkv, symmetric √γ scaling on Q and K slices
    - "mla": per-component asymmetric scaling (Kimi K2 Algorithm 1)

    Returns number of heads clipped.
    """
    sqrt_d = math.sqrt(head_dim)
    clipped = 0

    for _, q_param, _, k_param, mode in qk_pairs:
        q_data = q_param.data
        k_data = k_param.data

        if mode == "mla":
            clipped += _mla_clip_heads(q_data, k_data, head_dim, tau)
            continue

        if mode == "standard":
            n_q_heads = q_data.shape[0] // head_dim
            n_k_heads = k_data.shape[0] // head_dim
            q_heads = q_data.view(n_q_heads, head_dim, -1)
            k_heads = k_data.view(n_k_heads, head_dim, -1)

        elif mode == "fused":
            total_out = q_data.shape[0]
            if total_out % (3 * head_dim) == 0:
                n_heads = total_out // (3 * head_dim)
                q_dim = n_heads * head_dim
                k_dim = n_heads * head_dim
            else:
                found = False
                for ratio in (8, 4, 2, 1):
                    denom = head_dim * (1 + 2.0 / ratio)
                    if total_out % int(denom) == 0:
                        n_q = total_out // int(denom)
                        q_dim = n_q * head_dim
                        k_dim = (n_q // ratio) * head_dim
                        found = True
                        break
                if not found:
                    continue

            q_slice = q_data[:q_dim]
            k_slice = q_data[q_dim : q_dim + k_dim]
            n_q_heads = q_dim // head_dim
            n_k_heads = k_dim // head_dim
            q_heads = q_slice.view(n_q_heads, head_dim, -1)
            k_heads = k_slice.view(n_k_heads, head_dim, -1)
        else:
            continue

        q_norms = q_heads.flatten(1).norm(dim=1)
        k_norms = k_heads.flatten(1).norm(dim=1)

        # For GQA, expand K norms to match Q heads
        if len(q_norms) != len(k_norms) and len(k_norms) > 0:
            repeat_factor = len(q_norms) // len(k_norms)
            k_norms_expanded = k_norms.repeat_interleave(repeat_factor)
        else:
            k_norms_expanded = k_norms

        max_logits = q_norms * k_norms_expanded / sqrt_d
        exceed_mask = max_logits > tau

        if not exceed_mask.any():
            continue

        scale = tau * sqrt_d / (q_norms * k_norms_expanded + 1e-12)
        scale = torch.where(exceed_mask, torch.sqrt(scale), torch.ones_like(scale))

        q_scale = scale.view(-1, 1, 1)
        q_heads.mul_(q_scale)

        if mode == "standard" and len(q_norms) == len(k_norms):
            k_heads.mul_(q_scale[:len(k_norms)])
        elif mode == "fused" and n_q_heads == n_k_heads:
            k_heads.mul_(q_scale)

        clipped += exceed_mask.sum().item()

    return clipped


def register_qk_clip_hook(
    optimizer,
    model_parts: list[nn.Module],
    tau: float,
    head_dim: int,
) -> None:
    """Register a post-step hook on the optimizer for QK-Clip.

    Args:
        optimizer: OptimizersContainer (has register_step_post_hook)
        model_parts: List of model modules to scan for Q/K params
        tau: Clipping threshold for max attention logit
        head_dim: Attention head dimension (e.g. 128 for Qwen3-30B-A3B)
    """
    qk_pairs = _find_qk_params(model_parts)
    if not qk_pairs:
        logger.warning("QK-Clip: No Q/K parameter pairs found, hook not registered")
        return

    modes = set(m for *_, m in qk_pairs)
    state = {"step": 0, "n_clipped": 0}

    def _clip_hook(optimizer, args, kwargs):
        state["step"] += 1
        state["n_clipped"] = _qk_clip_step(qk_pairs, tau, head_dim)
        if state["n_clipped"] > 0 and state["step"] % 100 == 0:
            logger.info(f"QK-Clip step {state['step']}: clipped {state['n_clipped']} heads (tau={tau})")

    optimizer.register_step_post_hook(_clip_hook)
    optimizer._qk_clip_state = state
    logger.info(
        f"QK-Clip hook registered: {len(qk_pairs)} Q/K pairs, "
        f"modes={modes}, tau={tau}, head_dim={head_dim}"
    )
