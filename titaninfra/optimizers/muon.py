"""
Muon optimizer — MomentUm Orthogonalized by Newton-Schulz.

Applies Newton-Schulz (NS) iteration to orthogonalize momentum for 2D parameters,
producing spectral-norm-like updates. Non-2D parameters should use AdamW instead
(see MixedOptimizersContainer for the combined approach).

Key design choices:
- Per-shard NS: Each FSDP shard runs NS independently (zero extra communication).
  Validated at 1T scale by Kimi K2.
- @torch.compile on NS inner loop only (pure tensor ops, no Python control flow).
- Per-step quintic coefficients from NVIDIA emerging_optimizers for faster convergence.

References:
- Jordan et al., "Muon: An optimizer for hidden layers in neural networks"
- Kimi K2 Technical Report (per-shard blockwise NS at scale)
- NVIDIA Megatron-LM emerging_optimizers (OrthogonalizedOptimizer)
- Su Jianlin blog: kexue.fm/archives/10592, 10739
"""

import math

import torch
from torch.optim import Optimizer

# Compile static Newton-Schulz kernels for each weight shape. DeepSeek V3 has
# more than eight distinct matrix shapes, so raise Dynamo's recompile limit to
# avoid silently falling back to eager execution.
torch._dynamo.config.recompile_limit = 16


def _should_gather(update) -> bool:
    """Decide whether a Muon update tensor needs all-gather for exact NS.

    Returns True only for 2D DTensors with at least one sharded dim. Returns
    False for:
    - plain tensors (single-card / non-distributed) — already a full matrix;
    - fully-replicated DTensors — every rank already holds the full matrix;
    - 3D expert tensors — Shard(0) makes per-expert local NS exact, and
      gathering Shard(1) experts across EP is prohibitive (~1GB+) and undoes
      expert parallelism. Experts always stay on per-shard NS.

    See docs/superpowers/specs/2026-06-16-muon-gather-mode-design.md §4.
    """
    if not hasattr(update, "to_local"):
        return False
    if all(p.is_replicate() for p in update.placements):
        return False
    if update.ndim == 3:
        return False
    return True


_NS_COEFFS_QUINTIC = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def _newton_schulz_impl(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration (pure computation, no compile decorator).

    Uses per-step quintic polynomial coefficients (NVIDIA emerging_optimizers)
    for faster convergence than fixed coefficients.

    Operates on whichever Gram matrix is smaller — min(m,n)². The iteration
    X_{k+1} = aX + (bA + cA²)X with A = XXᵀ is algebraically identical to
    X_{k+1} = aX + X(bG + cG²) with G = XᵀX, because (XXᵀ)X = X(XᵀX). When the
    input is tall (m > n) — which is the production case, since step()
    transposes every rectangular weight to tall before calling NS — the Gram
    form runs the dominant matmuls on the smaller n×n matrix instead of m×m,
    cutting NS FLOPs 3-18× on typical attention/FFN weights. For square or wide
    inputs the m ≤ n branch is byte-identical to the original formulation.
    """
    X = X / (X.norm() + 1e-7)
    m, n = X.shape[-2], X.shape[-1]
    if m <= n:
        for i in range(steps):
            a, b, c = _NS_COEFFS_QUINTIC[min(i, len(_NS_COEFFS_QUINTIC) - 1)]
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
    else:
        for i in range(steps):
            a, b, c = _NS_COEFFS_QUINTIC[min(i, len(_NS_COEFFS_QUINTIC) - 1)]
            G = X.mT @ X
            B = b * G + c * G @ G
            X = a * X + X @ B
    return X


def _batched_newton_schulz_impl(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz for 3D expert tensors [E, m, n]. Per-expert independent.

    Treats dim-0 as a batch of independent matrices, running NS on each
    [m, n] slice via torch.bmm. Used for MoE expert weights where each
    expert's FFN projection is a separate linear transformation.

    Like the 2D path, picks the smaller Gram matrix (min(m,n)²) per the
    (XXᵀ)X = X(XᵀX) identity. step() transposes experts to tall before NS, so
    the m > n branch — Gram on the smaller n×n per expert — is the hot path.
    """
    X = X / (X.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-7)
    m, n = X.shape[-2], X.shape[-1]
    if m <= n:
        for i in range(steps):
            a, b, c = _NS_COEFFS_QUINTIC[min(i, len(_NS_COEFFS_QUINTIC) - 1)]
            A = torch.bmm(X, X.transpose(-2, -1))
            B = b * A + c * torch.bmm(A, A)
            X = a * X + torch.bmm(B, X)
    else:
        for i in range(steps):
            a, b, c = _NS_COEFFS_QUINTIC[min(i, len(_NS_COEFFS_QUINTIC) - 1)]
            G = torch.bmm(X.transpose(-2, -1), X)
            B = b * G + c * torch.bmm(G, G)
            X = a * X + torch.bmm(X, B)
    return X


@torch.library.custom_op("titaninfra::newton_schulz", mutates_args=())
def _newton_schulz_op(X: torch.Tensor, steps: int) -> torch.Tensor:
    return _newton_schulz_impl(X, steps)


@_newton_schulz_op.register_fake
def _newton_schulz_fake(X: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.empty_like(X)


@torch.library.custom_op("titaninfra::batched_newton_schulz", mutates_args=())
def _batched_newton_schulz_op(X: torch.Tensor, steps: int) -> torch.Tensor:
    return _batched_newton_schulz_impl(X, steps)


@_batched_newton_schulz_op.register_fake
def _batched_newton_schulz_fake(X: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.empty_like(X)


_newton_schulz_compiled = torch.compile(_newton_schulz_impl, dynamic=False)
_batched_newton_schulz_compiled = torch.compile(_batched_newton_schulz_impl, dynamic=False)

_newton_schulz_iter = _newton_schulz_impl
_batched_newton_schulz_iter = _batched_newton_schulz_impl


class Muon(Optimizer):
    """Muon optimizer for 2D weight matrices.

    Applies Newton-Schulz orthogonalization to momentum before updating parameters.
    Only handles 2D parameters — use MixedOptimizersContainer for models with
    embeddings, norms, and biases.

    Args:
        params: Iterable of 2D parameters to optimize.
        lr: Learning rate (default: 0.02).
        momentum: Momentum decay factor / EMA beta (default: 0.95).
        nesterov: Use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        ns_dtype: Dtype for NS computation (default: torch.bfloat16).
        scale_factor: Multiplier for sqrt(max(m,n)) scale (default: 1.0).
            Use 0.2 for Moonlight/Kimi K2 style shared LR.
        gather_mode: When True, 2D params needing gather (per _should_gather)
            are redistributed to the full matrix for exact Newton-Schulz, then
            re-sharded — replacing the default per-shard NS approximation.
            3D experts always use per-shard NS. (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        ns_dtype: torch.dtype = torch.bfloat16,
        scale_factor: float = 1.0,
        gather_mode: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "scale_factor": scale_factor,
        }
        super().__init__(params, defaults)
        self._ns_dtype = ns_dtype
        self._gather_mode = gather_mode



    def _compute_ns_update(self, p, buf, grad, group):
        """nesterov + Newton-Schulz (gather / per-shard 2D / batched 3D) + scale fold.

        Caller must have already updated `buf` (momentum lerp). Returns the
        orthogonalized update with the sqrt(max(m,n))·scale_factor scale folded
        in, so write-back is a plain `p.add_(update, alpha=-lr)`. Pure NS
        compute, no write-back. Does NOT mutate `buf`.
        """
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        sf = group["scale_factor"]

        update = grad.lerp(buf, group["momentum"]) if nesterov else buf

        gather = self._gather_mode and _should_gather(update)

        if gather:
            # Gather mode: redistribute the full matrix, run exact NS, then
            # restore the original sharding. 3D experts stay per-shard.
            from torch.distributed.tensor import DTensor, Replicate

            orig_placements = update.placements
            mesh = update.device_mesh
            full = update.redistribute(placements=[Replicate()] * mesh.ndim)
            full_local = full.to_local()

            fm, fn = full_local.shape[-2], full_local.shape[-1]
            transposed = fm < fn
            if transposed:
                full_local = full_local.mT
            ns_input = full_local.to(self._ns_dtype)
            full_local = _newton_schulz_compiled(ns_input, steps=ns_steps).to(
                full_local.dtype
            )
            if transposed:
                full_local = full_local.mT

            full = DTensor.from_local(full_local, mesh, [Replicate()] * mesh.ndim)
            update = full.redistribute(placements=orig_placements)
            is_dtensor = True
            # Scaling uses the global parameter shape in the DTensor branch.
            m, n = fm, fn
        else:
            is_dtensor = hasattr(update, "to_local")
            local = update.to_local() if is_dtensor else update

            orig_shape = local.shape
            if local.ndim == 1:
                local = local.unsqueeze(0)

            if local.ndim == 3:
                # 3D expert tensor [E, m, n]: run batched NS per expert.
                _, m, n = local.shape
                transposed = m < n
                if transposed:
                    local = local.transpose(-2, -1)
                ns_input = local.to(self._ns_dtype)
                local = _batched_newton_schulz_compiled(ns_input, steps=ns_steps).to(local.dtype)
                if transposed:
                    local = local.transpose(-2, -1)
            else:
                # 2D matrix: standard Newton-Schulz iteration.
                m, n = local.shape[-2], local.shape[-1]
                transposed = m < n
                if transposed:
                    local = local.mT

                ns_input = local.to(self._ns_dtype)
                local = _newton_schulz_compiled(ns_input, steps=ns_steps).to(local.dtype)

                if transposed:
                    local = local.mT

            local = local.view(orig_shape)

            if is_dtensor:
                update = torch.distributed.tensor.DTensor.from_local(
                    local, update.device_mesh, update.placements
                )
            else:
                update = local

        if is_dtensor:
            gs = p.shape
            scale = sf * math.sqrt(max(gs[-2], gs[-1]))
        else:
            scale = sf * math.sqrt(max(m, n))

        # Fold scale into a fresh update tensor for uniform foreach write-back
        # without mutating the momentum buffer.
        return update * scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]

            params, bufs, grads = [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                state["step"] += 1
                params.append(p)
                bufs.append(state["momentum_buffer"])
                grads.append(p.grad)

            if not params:
                continue

            # Batch momentum EMA across all params (foreach — 8× fewer launches).
            torch._foreach_lerp_(bufs, grads, 1.0 - beta)
            # Per-param NS (matmul — not foreach-able; gather redistributes per-param).
            updates = [
                self._compute_ns_update(p, buf, g, group)
                for p, buf, g in zip(params, bufs, grads, strict=True)
            ]

            # Apply decoupled weight decay and scaled updates with uniform alpha.
            if wd != 0.0:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            torch._foreach_add_(params, updates, alpha=-lr)

        return loss
