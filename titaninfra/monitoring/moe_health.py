"""Non-intrusive MoE expert-load health monitoring.

Router hooks accumulate per-expert token counts. Metrics include coefficient of
variation, min/max utilization, imbalance ratio, sliding-window dead-expert
ratio, per-layer histograms, and optional periodic global reduction.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)

_DEAD_EXPERT_THRESHOLD_FACTOR = 0.1


def _resolve_tokens_buffer(moe) -> torch.Tensor:
    """Resolve tokens_per_expert_E with fallback to the older field name."""
    buf = getattr(moe, "tokens_per_expert_E", None)
    if buf is None:
        buf = moe.tokens_per_expert
    return buf


@dataclass(slots=True)
class LayerLoadMetrics:
    expert_load_cv: float
    expert_util_min: float
    expert_util_max: float
    imbalance_ratio: float
    dead_expert_ratio: float

    def to_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/expert_load_cv": self.expert_load_cv,
            f"{prefix}/expert_util_min": self.expert_util_min,
            f"{prefix}/expert_util_max": self.expert_util_max,
            f"{prefix}/imbalance_ratio": self.imbalance_ratio,
            f"{prefix}/dead_expert_ratio": self.dead_expert_ratio,
        }


def _compute_layer_load_metrics(
    all_counts: list[torch.Tensor],
    num_experts: int,
    dead_threshold_factor: float = _DEAD_EXPERT_THRESHOLD_FACTOR,
) -> LayerLoadMetrics | None:
    """Compute aggregate health metrics from per-layer token counts.

    Return None when no layer contains tokens.
    """
    layer_cvs: list[float] = []
    layer_util_mins: list[float] = []
    layer_util_maxs: list[float] = []
    layer_imbalance: list[float] = []

    for layer_counts in all_counts:
        layer_mean = layer_counts.mean()
        if layer_mean <= 0:
            continue
        layer_cvs.append((layer_counts.std() / layer_mean).item())
        layer_util_mins.append((layer_counts.min() / layer_mean).item())
        layer_util_maxs.append((layer_counts.max() / layer_mean).item())
        floor = (layer_mean * 0.001).item()
        effective_min = max(layer_counts.min().item(), floor)
        layer_imbalance.append(min(layer_counts.max().item() / effective_min, 1000.0))

    if not layer_cvs:
        return None

    dead_threshold = dead_threshold_factor / num_experts
    total_dead = 0
    total_count = 0
    for layer_counts in all_counts:
        layer_total = layer_counts.sum()
        if layer_total > 0:
            layer_p = layer_counts / layer_total
            total_dead += (layer_p < dead_threshold).sum().item()
            total_count += num_experts

    return LayerLoadMetrics(
        expert_load_cv=sum(layer_cvs) / len(layer_cvs),
        expert_util_min=sum(layer_util_mins) / len(layer_util_mins),
        expert_util_max=sum(layer_util_maxs) / len(layer_util_maxs),
        imbalance_ratio=sum(layer_imbalance) / len(layer_imbalance) if layer_imbalance else 0.0,
        dead_expert_ratio=total_dead / total_count if total_count > 0 else 0.0,
    )


def _find_moe_layers(model_parts: list[nn.Module]) -> list[nn.Module]:
    """Collect MoE submodules from moe_enabled transformer blocks."""
    moe_layers: list[nn.Module] = []
    for model_part in model_parts:
        layers = model_part.get_submodule("layers")
        for block in layers.values():
            if getattr(block, "moe_enabled", False):
                moe_layers.append(block.moe)
    return moe_layers


def _try_wandb_histogram(values: torch.Tensor):
    """Return a wandb.Histogram, or None when WandB is unavailable."""
    try:
        import wandb
        return wandb.Histogram(values.cpu().numpy())
    except Exception:
        return None


class MoEHealthPlugin:
    """Monitor MoE load through router forward hooks.

    Reconstruct token counts from top-k expert IDs rather than reusing the
    optimizer's tokens_per_expert_E buffer. Dead experts use a sliding window
    to avoid false alarms from a single step.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        has_ac: bool = False,
        dead_window_size: int = 100,
        ep_reduce_interval: int = 0,
        parallel_dims: Any = None,
    ) -> None:
        self._moe_layers = _find_moe_layers(model_parts)
        self._has_ac = has_ac
        self._counters: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        # Keep recent load fractions per layer for dead-expert smoothing.
        self._dead_window_size = dead_window_size
        self._load_history: list[collections.deque] = []

        # Configure optional global reduction across expert-parallel ranks.
        self._ep_reduce_interval = ep_reduce_interval
        self._loss_pg = None
        self._rank = 0
        if ep_reduce_interval > 0 and parallel_dims is not None:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                self._loss_pg = loss_mesh.get_group()
                self._rank = dist.get_rank()

        for moe in self._moe_layers:
            tokens_buffer = _resolve_tokens_buffer(moe)
            num_experts = tokens_buffer.shape[0]
            counter = torch.zeros(num_experts, dtype=torch.float32, device=tokens_buffer.device)
            self._counters.append(counter)
            self._load_history.append(collections.deque(maxlen=dead_window_size))

            router = moe.router
            hook = router.register_forward_hook(self._make_router_hook(counter))
            self._hooks.append(hook)

        if self._moe_layers:
            ep_info = f", ep_reduce_interval={ep_reduce_interval}" if self._loss_pg else ""
            logger.info(
                f"MoEHealthPlugin: 监控 {len(self._moe_layers)} 个 MoE 层, "
                f"dead_window={dead_window_size}{ep_info}"
            )

    def _make_router_hook(self, counter: torch.Tensor):
        """Router forward hook: capture per-expert token counts."""

        def hook_fn(module: nn.Module, input_tuple, output):
            # Upstream routers return top-k scores, expert IDs, and full scores.
            # Derive the count histogram with a top-k one-hot sum rather than
            # relying on the optimizer-owned buffer.
            topk_expert_ids_BLK = output[1]
            scores_BLE = output[2]
            routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
                -1, topk_expert_ids_BLK, True
            )
            reduce_dims = tuple(range(routing_map_BLE.ndim - 1))
            num_tokens_per_expert = routing_map_BLE.sum(dim=reduce_dims).detach().float()
            if hasattr(num_tokens_per_expert, "to_local"):
                num_tokens_per_expert = num_tokens_per_expert.to_local()
            counter.add_(num_tokens_per_expert)

        return hook_fn

    @property
    def has_moe(self) -> bool:
        return len(self._moe_layers) > 0

    def _maybe_reduce_global(self, step: int) -> dict[str, Any]:
        """Periodically reduce expert counts and compute rank-0 global metrics."""
        if self._loss_pg is None or self._ep_reduce_interval <= 0:
            return {}
        if step % self._ep_reduce_interval != 0:
            return {}

        stacked = torch.stack([c.clone() for c in self._counters])
        if self._has_ac:
            stacked = stacked / 2.0

        dist.reduce(stacked, dst=0, group=self._loss_pg, op=dist.ReduceOp.SUM)

        if self._rank != 0:
            return {}

        metrics: dict[str, Any] = {}
        num_experts = stacked.shape[1]

        layer_metrics = _compute_layer_load_metrics(list(stacked), num_experts)
        if layer_metrics is not None:
            metrics.update(layer_metrics.to_dict("moe_global"))

        return metrics

    def _sliding_window_dead_ratio(self, num_experts: int) -> float | None:
        """Return the sliding-window dead-expert ratio when history exists."""
        dead_threshold = _DEAD_EXPERT_THRESHOLD_FACTOR / num_experts
        total_experts = 0
        dead_count = 0
        for history in self._load_history:
            if len(history) > 0:
                window_avg = torch.stack(list(history)).mean(dim=0)
                dead_count += (window_avg < dead_threshold).sum().item()
                total_experts += window_avg.shape[0]
        return dead_count / total_experts if total_experts > 0 else None

    def _collect_per_layer_counts(self, metrics: dict[str, Any]) -> list[torch.Tensor]:
        """Emit per-layer metrics and return cloned per-layer counts."""
        all_counts: list[torch.Tensor] = []
        for i, counter in enumerate(self._counters):
            tpe = counter.clone()
            if self._has_ac:
                tpe = tpe / 2.0

            total = tpe.sum()
            p = tpe / total if total > 0 else tpe

            mean_val = tpe.mean()
            if mean_val > 0:
                cv = tpe.std() / mean_val
                metrics[f"moe/layer_{i}/expert_load_cv"] = cv.item()

            hist = _try_wandb_histogram(p)
            if hist is not None:
                metrics[f"moe/layer_{i}/expert_load_hist"] = hist

            self._load_history[i].append(p.cpu())
            all_counts.append(tpe)
        return all_counts

    def collect(self, step: int, trainer: Any) -> dict[str, Any]:  # noqa: ARG002
        if not self._moe_layers:
            return {}

        metrics: dict[str, Any] = {}
        num_experts = self._counters[0].shape[0] if self._counters else 1

        all_counts = self._collect_per_layer_counts(metrics)

        # Aggregate metrics across layers.
        layer_metrics = _compute_layer_load_metrics(all_counts, num_experts)
        if layer_metrics is not None:
            # Use per-step CV/utilization/imbalance; dead ratio is computed
            # separately from the sliding window.
            step_dict = layer_metrics.to_dict("moe")
            del step_dict["moe/dead_expert_ratio"]
            metrics.update(step_dict)

        # Dead-expert ratio uses the sliding window, not a single step.
        dead_ratio = self._sliding_window_dead_ratio(num_experts)
        if dead_ratio is not None:
            metrics["moe/dead_expert_ratio"] = dead_ratio

        # Add optional periodic global metrics.
        global_metrics = self._maybe_reduce_global(step)
        metrics.update(global_metrics)

        for counter in self._counters:
            counter.zero_()

        return metrics

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
