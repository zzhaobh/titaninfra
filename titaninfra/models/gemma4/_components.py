"""Gemma4 self-contained numeric components (HF-aligned).

This module owns gemma4's numeric primitives so the forward path can be aligned
component-by-component with HuggingFace transformers ``modeling_gemma4.py`` (the
ground-truth reference), without borrowing torchtitan's leaf numerics
(``torchtitan...rmsnorm.RMSNorm`` / ``...rope.apply_rotary_emb_cos_sin`` /
``...attention`` math).

Distributed infrastructure (FSDP2/TP/EP/CP, DCP checkpointing, the
``Decoder``/``TransformerBlock``/``BaseAttention``/``Module`` base classes and
``parallelize.py``) is intentionally NOT touched here — only the per-token math.

Every public symbol cites its HF reference (file: transformers 5.10.2
``models/gemma4/modeling_gemma4.py`` unless noted) so future drift is auditable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_remat as remat
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig

# ============================================================================
# RMSNorm — HF Gemma4RMSNorm @ modeling_gemma4.py:193
# ============================================================================


class Gemma4RMSNorm(Module):
    """RMSNorm matching HF ``Gemma4RMSNorm`` exactly.

    HF semantics (modeling_gemma4.py:202-211):
      - compute in float32: ``normed = x.float() * pow(mean(x^2)+eps, -0.5)``
      - eps added INSIDE the rsqrt (``mean_squared + eps``)
      - plain ``normed * weight`` — NO Gemma2-style ``(1 + weight)`` offset
      - ``pow(mean_sq, -0.5)`` (NOT ``rsqrt``) to match Torch/JAX compiler parity
      - cast back to input dtype at the very end
      - ``with_scale=False`` → no learnable weight (used by v_norm and the
        router's pre-norm)

    Inherits torchtitan ``Module`` (nn.Module + Configurable) so it satisfies
    the ``verify_module_protocol`` check the trainer runs at construction, and
    so its ``weight`` is materialized under the meta-device → ``to_empty`` →
    ``init_states`` flow (``Module.init_states`` only recurses into ``Module``
    children; a plain ``nn.Module`` child would be left uninitialized). The
    positional ``__init__`` is kept (no ``Config``) — ``Module`` requires no
    ``Config`` unless one is declared, and the four construction sites pass
    positional args. ``reset_parameters`` provides the ones-init that
    ``Module._init_self_parameters`` looks up when ``_param_init`` is unset,
    matching the eager ``torch.ones(dim)`` default (identity scale). Forward
    math and the ``weight`` FQN are unchanged — DCP-compatible, numerics
    byte-identical.

    ``sharding_config`` is an optional keyword-only param (not a ``.Config``
    field — this class deliberately has no ``.Config``, see above). Callers
    that need this instance to participate in ``Module._parallelize()``
    (FSDP2 sharding) pass a ``ShardingConfig`` here; it is stored directly on
    ``self._sharding_config``, the same private attribute
    ``Module.Config.build()`` would set for a Configured module — this is a
    protocol-sanctioned pattern, not a workaround: see
    ``torchtitan/experiments/transformers_modeling_backend/hf_sharding.py``
    (sets ``module._sharding_config = ...`` directly on plain HF modules with
    no ``.Config`` at all — the exact same situation as this class) and
    ``tests/unit_tests/cpu/models/base/test_module.py::
    TestModuleRedistribution.WeightModule`` (a Config-less ``Module`` subclass
    that sets ``self._sharding_config`` in its own ``__init__``). Without a
    sharding config, ``Module._parallelize()`` silently skips distributing
    this instance's own state (``if self._sharding_config is None: return``),
    leaving ``weight`` a plain (non-DTensor) tensor — harmless under pure DDP,
    but the eventual cause of FSDP2's ``_init_sharding_spec``: "Got plain
    tensor for parameter 'weight'" once ``fully_shard()`` wraps an enclosing
    module (2026-11-20, SFT case_28 root cause).
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        with_scale: bool = True,
        *,
        sharding_config: ShardingConfig | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        if sharding_config is not None:
            self._sharding_config = sharding_config

    def reset_parameters(self) -> None:
        """Ones-init the scale weight (identity at init).

        ``Module._init_self_parameters`` calls this when ``_param_init`` is
        unset. For ``with_scale=False`` there is no ``weight`` — no-op. Matches
        gemma4's ``_NORM_INIT = {"weight": nn.init.ones_}`` used by the native
        RMSNorm layers, and the plain ``normed * weight`` forward (ones →
        identity scale at init, no ``(1 + weight)`` offset).
        """
        if self.with_scale:
            nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self._norm(hidden_states.float())
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(hidden_states)


# ============================================================================
# RoPE — HF rotate_half@780, apply_rotary_pos_emb@787,
#        Gemma4TextRotaryEmbedding@1088 (+ modeling_rope_utils proportional@187)
# ============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF rotate_half (modeling_gemma4.py:780): pairs channel k with k+dim/2."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """HF apply_rotary_pos_emb (modeling_gemma4.py:787) on a single tensor.

    ``x`` is ``[B, S, n_heads, head_dim]`` (unsqueeze_dim=2 broadcasts cos/sin
    over the head axis). cos/sin are ``[B, S, head_dim]``. Returns same shape
    as ``x``. Computation is done in float32 then cast back (HF upcasts via the
    rotary embedding producing float32 cos/sin; we mirror by upcasting x).

    Kept batched (not the flat ``(T, H, D)`` convention used elsewhere in the
    post-migration forward path) so this module stays byte-for-byte auditable
    against HF's own calling convention — see module docstring and
    ``tests/test_gemma4_hf_parity.py``, which exercises this exact signature
    with real HF imports. Callers in the flat-token-stream forward path
    (``model.py``) are responsible for inserting/removing a size-1 batch axis
    around calls into this module, not the other way around.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_f = x.float()
    out = (x_f * cos) + (rotate_half(x_f) * sin)
    return out.type_as(x)


def build_default_cos_sin_cache(
    *, head_dim: int, theta: float, max_seq_len: int
) -> torch.Tensor:
    """Default (full-rotation) RoPE cache — HF compute_default_rope_parameters.

    inv_freq = ``1 / theta^(arange(0,head_dim,2)/head_dim)`` (head_dim/2 freqs),
    cos/sin doubled via ``cat(freqs, freqs)`` to head_dim width. Used by sliding
    layers (head_dim=256, theta=10000).

    Returns ``[max_seq_len, head_dim * 2]`` = ``cat([cos, sin], -1)``, the layout
    consumed by :func:`apply_rotary_emb_from_cache`.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq).float()
    emb = torch.cat([freqs, freqs], dim=-1)
    return torch.cat([emb.cos(), emb.sin()], dim=-1)


def build_proportional_cos_sin_cache(
    *, head_dim: int, rope_dim: int, theta: float, max_seq_len: int
) -> torch.Tensor:
    """Proportional RoPE cache — HF _compute_proportional_rope_parameters
    (modeling_rope_utils.py:187) used by full-attention layers.

    Only the first ``rope_dim`` channels of a ``head_dim``-wide head rotate, but
    the inverse frequencies use ``head_dim`` (NOT ``rope_dim``) as the
    denominator, then the remaining ``head_dim/2 - rope_dim/2`` angles are
    zero-padded. cos/sin are ``cat(freqs, freqs)`` to full ``head_dim`` width, so
    rotate_half pairs channel k with ``k + head_dim/2`` (e.g. 0↔256 for the
    512-dim full head — NOT 0↔64).

    A plain default cache with ``head_dim=rope_dim`` is wrong on BOTH counts: it
    uses ``rope_dim`` as denominator and pairs k with ``k + rope_dim/2``. This
    builder reproduces HF byte-for-byte (verified max diff 0.0).

    Returns ``[max_seq_len, head_dim * 2]`` = ``cat([cos, sin], -1)``.
    """
    rope_angles = int(rope_dim) // 2
    inv_freq_rotated = 1.0 / (
        theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
    )
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            [inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)], dim=0
        )
    else:
        inv_freq = inv_freq_rotated

    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq).float()
    emb = torch.cat([freqs, freqs], dim=-1)
    return torch.cat([emb.cos(), emb.sin()], dim=-1)


def _select_cos_sin(
    cache: torch.Tensor, positions: torch.Tensor | None, seqlen: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice cos/sin from a ``[max_seq_len, head_dim*2]`` cache.

    ``positions`` (``[B, S]`` or ``[S]``) indexes the cache; when None, uses
    ``arange(seqlen)``. Returns cos, sin each broadcastable to ``[B, S, head_dim]``.
    """
    if positions is None:
        idx = torch.arange(seqlen, device=cache.device)
    else:
        idx = positions.clamp(min=0)
    sel = cache[idx]  # [..., head_dim*2]
    cos = sel[..., :head_dim]
    sin = sel[..., head_dim:]
    return cos, sin


def apply_rotary_emb_from_cache(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full-head RoPE to q/k using a precomputed cos/sin cache.

    xq: ``[B, S, n_heads, head_dim]``, xk: ``[B, S, n_kv_heads, head_dim]``.
    RoPE is applied to the FULL head_dim for both sliding and full layers; for
    full layers the cache carries zero frequencies beyond ``rope_dim`` so those
    channels pass through unchanged (cos=1, sin=0). This matches HF, which never
    slices the head and relies on the proportional cache's zero padding.

    Batched convention preserved for HF-parity auditability — see
    ``apply_rotary_pos_emb``'s docstring. ``model.py``'s flat-token-stream
    forward path inserts/removes the size-1 batch axis around this call.
    """
    head_dim = xq.shape[-1]
    cos, sin = _select_cos_sin(cache, positions, xq.shape[1], head_dim)
    xq_out = apply_rotary_pos_emb(xq, cos, sin, unsqueeze_dim=2)
    xk_out = apply_rotary_pos_emb(xk, cos, sin, unsqueeze_dim=2)
    return xq_out, xk_out


# ============================================================================
# Attention math — HF eager_attention_forward @ modeling_gemma4.py:821
# ============================================================================


def gemma4_sdpa(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    scaling: float = 1.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Scaled-dot-product attention matching HF eager_attention_forward numerics.

    HF (modeling_gemma4.py:838-851): ``attn = (q @ k^T) * scaling`` (scaling=1.0
    for gemma4 — normalization comes from q_norm/k_norm), softmax in float32 cast
    back to q dtype, then ``attn @ v``. GQA handled by repeating kv heads.

    Inputs are ``[B, S, n_heads, head_dim]`` (titaninfra layout). Internally
    transposed to ``[B, n_heads, S, head_dim]`` for SDPA, which fuses the
    float32-softmax path and is numerically equivalent to HF's eager kernel.
    Returns ``[B, S, n_heads, head_dim]``.
    """
    # [B, S, H, D] -> [B, H, S, D]
    q = xq.transpose(1, 2)
    k = xk.transpose(1, 2)
    v = xv.transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        scale=scaling,
        enable_gqa=enable_gqa,
    )
    # [B, H, S, D] -> [B, S, H, D]
    return out.transpose(1, 2).contiguous()


# ============================================================================
# Dense MLP — HF Gemma4TextMLP @ modeling_gemma4.py:1069
# ============================================================================


def gemma4_mlp_forward(
    x: torch.Tensor, w1: nn.Module, w2: nn.Module, w3: nn.Module
) -> torch.Tensor:
    """HF Gemma4TextMLP: ``down(gelu_tanh(gate(x)) * up(x))``.

    w1=gate_proj, w3=up_proj, w2=down_proj. Activation is gelu with the tanh
    approximation (``hidden_activation="gelu_pytorch_tanh"``).
    """
    gate = w1(x)
    up = w3(x)
    # gelu and the multiply read w1/w3's remat.region outputs (w1/w3 are
    # Linear instances; Linear.forward always wraps its local projection in
    # a remat.region) with bare ops -- the same pattern as upstream
    # DenseFeedForward.forward's `remat.recompute_needs_tensor(gate_up_T2F)`
    # before its activation_fn call (torchtitan/models/common/
    # feed_forward.py), adapted to gemma4's separate w1/w3 projections
    # instead of one fused w13.
    remat.recompute_needs_tensor(gate, up)
    return w2(F.gelu(gate, approximate="tanh") * up)


# ============================================================================
# MoE experts — HF Gemma4TextExperts @ modeling_gemma4.py:1295
# ============================================================================


def gemma4_experts_for_loop(
    w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
    x: torch.Tensor, num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Per-expert GELU MLP (reference / non-grouped path).

    Weight layout matches HF's gate_up_proj split: w1=gate (first half of
    gate_up_proj on dim 1), w3=up (second half), w2=down. Activation gelu-tanh.
    ``x`` is the dispatched token stream ``[total_tokens, dim]`` partitioned by
    ``num_tokens_per_expert``.
    """
    # The split (a view op, needs no marker itself) and the subsequent
    # matmuls read the token_dispatcher.dispatch() output `x` with bare ops.
    # At ep=1 (LocalTokenDispatcher.dispatch(), e.g. this family's GPU
    # verification script) that output is already a bare tensor, so this is
    # a no-op; at ep>1 (AllToAllTokenDispatcher.dispatch()), it is the
    # output of a genuine "<fqn>.dispatch" remat.region -- gemma4's own raw-
    # parameter expert matmuls (no GroupedLinear) have no other mechanism to
    # declare this dependency the way upstream's GroupedLinear-based routed
    # experts do automatically. num_tokens_per_expert needs no marker here:
    # AllToAllTokenDispatcher.dispatch() already calls
    # remat.recompute_needs_tensor on it (and the splits) before returning.
    remat.recompute_needs_tensor(x)
    counts = num_tokens_per_expert.tolist()
    x_splits = torch.split(x, split_size_or_sections=counts, dim=0)
    outs = []
    for e, xe in enumerate(x_splits):
        h = F.gelu(torch.matmul(xe, w1[e].transpose(-2, -1)), approximate="tanh")
        h = h * torch.matmul(xe, w3[e].transpose(-2, -1))
        outs.append(torch.matmul(h, w2[e].transpose(-2, -1)))
    return torch.cat(outs, dim=0)


def gemma4_experts_grouped_mm(
    w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
    x: torch.Tensor, num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Grouped-GEMM GELU experts (perf path), numerically equal to the for-loop.

    Uses ``torch._grouped_mm`` with cumulative offsets. bf16 compute matches the
    training dtype; output cast back to ``x`` dtype.
    """
    # Same dispatch-output dependency as gemma4_experts_for_loop above: the
    # ``.bfloat16()`` cast below is the first bare read of `x`'s data.
    remat.recompute_needs_tensor(x)
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    h = F.gelu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets),
        approximate="tanh",
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    return torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
