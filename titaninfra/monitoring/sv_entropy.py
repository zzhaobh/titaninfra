"""
SVEntropyPlugin — Singular value entropy monitoring for Muon-trained weights.

Periodically samples 2D weight matrices managed by Muon and computes their
singular value entropy on CPU. Reports to WandB as sv_entropy/* metrics.

Entropy H(σ) = -Σ p_i log(p_i) / log(min(m,n)) where p_i = σ_i / Σσ_j.
- H ≈ 1.0: singular values are uniform (near-orthogonal, Muon's goal)
- H ≈ 0.0: rank-1 collapse (training instability signal)

Design constraints:
- CPU-only: .detach().cpu() then torch.linalg.svdvals (no GPU compute)
- Low-frequency: every sv_entropy_interval steps (default 100)
- Capped: at most sv_entropy_max_layers sampled per collection

Reference: Su Jianlin blog (kexue.fm/archives/10592).
"""

import logging
import math
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_MUON_EXCLUDE_PATTERNS = ("tok_embeddings", "embed_tokens", "output", "lm_head", "gate")


def _compute_sv_entropy(weight_cpu: torch.Tensor) -> float:
    """Compute normalized singular value entropy on a CPU tensor."""
    sv = torch.linalg.svdvals(weight_cpu.float())
    sv = sv[sv > 1e-12]
    if len(sv) == 0:
        return 0.0

    p = sv / sv.sum()
    log_p = torch.log(p)
    entropy = -(p * log_p).sum().item()

    m, n = weight_cpu.shape
    max_entropy = math.log(min(m, n))
    if max_entropy < 1e-12:
        return 0.0

    return entropy / max_entropy


class SVEntropyPlugin:
    """MetricsPlugin for singular value entropy monitoring.

    Implements the collect(step, trainer) -> dict protocol used by
    inject_metrics_plugins().
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        interval: int = 100,
        max_layers: int = 4,
        min_ndim_size: int = 4096,
    ):
        candidates: list[tuple[str, nn.Parameter]] = []
        router_candidates: list[tuple[str, nn.Parameter]] = []
        for model in model_parts:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim != 2 or param.numel() < min_ndim_size:
                    continue
                # Track router/gate matrices separately because their orthogonality
                # can provide an early expert-imbalance signal.
                if "gate" in name:
                    router_candidates.append((name, param))
                    continue
                if any(pat in name for pat in _MUON_EXCLUDE_PATTERNS):
                    continue
                candidates.append((name, param))

        self._params = self._subsample(candidates, max_layers)
        self._router_params = self._subsample(router_candidates, max_layers)

        self._interval = interval

        if self._params:
            names = [n for n, _ in self._params]
            logger.info(f"SVEntropyPlugin: monitoring {len(self._params)} layers: {names}")
        if self._router_params:
            rnames = [n for n, _ in self._router_params]
            logger.info(f"SVEntropyPlugin: monitoring {len(self._router_params)} router layers: {rnames}")

    @staticmethod
    def _subsample(
        candidates: list[tuple[str, nn.Parameter]], max_layers: int
    ) -> list[tuple[str, nn.Parameter]]:
        """Evenly subsample at most ``max_layers`` entries from candidates."""
        if len(candidates) > max_layers:
            step_size = len(candidates) / max_layers
            indices = [int(i * step_size) for i in range(max_layers)]
            return [candidates[i] for i in indices]
        return candidates

    def collect(self, step: int, trainer) -> dict:
        if step % self._interval != 0:
            return {}
        if not self._params and not self._router_params:
            return {}

        entropies = {}
        entropies.update(self._collect_group(self._params, "layer"))
        entropies.update(self._collect_group(self._router_params, "router_layer"))
        return entropies

    def _collect_group(
        self,
        params: list[tuple[str, nn.Parameter]],
        label_kind: str,
    ) -> dict:
        """Compute SV-entropy for one param group + its mean/min/max aggregates.

        ``label_kind`` distinguishes attention layers (``layer``) from router
        layers (``router_layer``) so the two never collide in WandB. Router
        aggregate keys are suffixed (``router_mean`` etc.) to keep the original
        ``sv_entropy/mean`` semantics for attention untouched.
        """
        if not params:
            return {}

        out = {}
        for i, (name, param) in enumerate(params):
            weight_cpu = _materialize_full_cpu(param)
            h = _compute_sv_entropy(weight_cpu)
            # Use the real layer number from the parameter name rather than the
            # sampling index, which can point to a different transformer layer.
            m = re.search(r"\.?layers\.(\d+)\.", name)
            layer_label = m.group(1) if m else str(i)
            out[f"sv_entropy/{label_kind}_{layer_label}"] = h

        values = list(out.values())
        agg = "" if label_kind == "layer" else "router_"
        out[f"sv_entropy/{agg}mean"] = sum(values) / len(values)
        out[f"sv_entropy/{agg}min"] = min(values)
        out[f"sv_entropy/{agg}max"] = max(values)
        return out


def _materialize_full_cpu(param: torch.Tensor) -> torch.Tensor:
    """Move param to CPU as a full local tensor.

    In FSDP2 setups, ``param`` is a ``DTensor`` whose local shard contains
    only 1/N of the full weight. Naive ``.cpu()`` returns a CPU-side DTensor
    wrapper that ``torch.linalg.svdvals`` cannot dispatch on, raising
    ``"No backend type associated with device type cpu"``.

    For DTensor inputs, ``full_tensor()`` performs an all-gather across the
    sharding mesh to reconstruct the complete parameter, then ``.cpu()``
    transfers to host. Peak GPU transient is the full param size (bf16 ≈ 8MB
    for a 2048×2048 weight) — negligible vs. total model footprint, and
    released as soon as ``.cpu()`` returns.

    For plain tensors (e.g. unit tests on a single-device model), this is
    a no-op aside from ``.detach().cpu()``.
    """
    detached = param.detach()
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None  # type: ignore[assignment]

    if DTensor is not None and isinstance(detached, DTensor):
        detached = detached.full_tensor()
    return detached.cpu()
