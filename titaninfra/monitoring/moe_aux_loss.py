"""MoE auxiliary-loss load-balancing plugin (spec 2026-06-09).

Grad-carrying router forward hooks capture live softmax(gate(x)) per microbatch.
The aux term is added inside loss_fn (same per-microbatch graph torchtitan
backprops). The f_i weighting is "stale-global": previous step's globally-reduced
expert counts (detached integer all-reduce at the step boundary).

Structural opposite of MoEHealthPlugin: the hook must NOT use torch.no_grad().

WandB: moe/aux_loss (unscaled L_aux ∈ [1, N_experts]).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torchtitan.models.common import Sigmoid

from titaninfra.monitoring._moe_aux_compute import (
    compute_load_balancing_loss,
    compute_sequence_load_balancing_loss,
    normalize_counts,
)
from titaninfra.monitoring.moe_health import _find_moe_layers, _resolve_tokens_buffer

logger = logging.getLogger(__name__)


class MoEAuxLossPlugin:
    def __init__(
        self,
        model_parts: list[nn.Module],
        coeff: float,
        has_ac: bool = False,
        parallel_dims: Any = None,
        aux_scope: str = "global",
    ) -> None:
        self._moe_layers = _find_moe_layers(model_parts)
        self._coeff = coeff
        self._has_ac = has_ac
        self._aux_scope = aux_scope
        self._cur_positions: torch.Tensor | None = None

        self._hooks: list[Any] = []
        self._f_global: list[torch.Tensor] = []
        self._mb_p_sum: list[torch.Tensor | None] = []
        self._mb_token_count: list[int] = []
        self._step_count_accum: list[torch.Tensor] = []
        # Sequence scope additionally stores full probabilities and top-k IDs.
        self._mb_probs_full: list[torch.Tensor | None] = []
        self._mb_topk: list[torch.Tensor | None] = []

        self._step_scale: float = 0.0
        self._last_aux: float = 0.0

        self._loss_pg = None
        if parallel_dims is not None:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                self._loss_pg = loss_mesh.get_group()

        for moe in self._moe_layers:
            tokens_buffer = _resolve_tokens_buffer(moe)
            n = tokens_buffer.shape[0]
            device = tokens_buffer.device
            self._f_global.append(torch.full((n,), 1.0 / n, dtype=torch.float32, device=device))
            self._mb_p_sum.append(None)
            self._mb_token_count.append(0)
            self._mb_probs_full.append(None)
            self._mb_topk.append(None)
            self._step_count_accum.append(torch.zeros(n, dtype=torch.float32, device=device))
            hook = moe.router.register_forward_hook(self._make_hook(len(self._hooks)))
            self._hooks.append(hook)

        if self._moe_layers:
            logger.info(
                f"MoEAuxLossPlugin: {len(self._moe_layers)} 个 MoE 层, coeff={coeff}, "
                f"global_reduce={'on' if self._loss_pg else 'local'}"
            )

    @property
    def has_moe(self) -> bool:
        return len(self._moe_layers) > 0

    def set_step_scale(self, gvt: float, gas: int) -> None:
        """Per-step scale so the aux term survives loss_fn's ÷global_valid_tokens.

        Each microbatch injects coeff·L_aux·gvt/gas; summed over gas microbatches
        and ÷gvt → coeff·mean_mb(L_aux).
        """
        gas = max(int(gas), 1)
        self._step_scale = self._coeff * float(gvt) / gas

    def set_positions(self, positions: torch.Tensor | None) -> None:
        """Set positions before forward for sequence-scoped auxiliary loss.

        positions==0 marks document boundaries; padding is excluded from
        per-sequence statistics.
        """
        self._cur_positions = positions

    def _build_seq_ids(self) -> torch.Tensor | None:
        """Convert positions to flattened sequence IDs, or None if absent.

        positions==0 starts a document and cumsum assigns IDs; positions==-1
        remains padding with ID -1 and is excluded from statistics.
        """
        pos = self._cur_positions
        if pos is None:
            return None
        flat = pos.reshape(-1)
        is_start = (flat == 0).to(torch.long)
        seq_ids = torch.cumsum(is_start, dim=0) - 1  # Zero-based sequence IDs
        seq_ids = seq_ids.masked_fill(flat == -1, -1)  # Exclude padding from statistics
        return seq_ids

    def _make_hook(self, idx: int):
        plugin = self

        def hook_fn(module: nn.Module, input_tuple, output):
            # Upstream routers return (topk_scores, topk_expert_ids, scores).
            # scores is the normalized full routing probability; derive expert
            # counts from the top-k one-hot selection.
            topk_expert_ids = output[1]
            scores = output[2]
            probs = scores.reshape(-1, scores.shape[-1])  # (T, N), grad-carrying
            # score_func is normally a UnaryActivationFn instance; DeepSeekV4Router
            # may instead expose a string for its sqrtsoftplus branch.
            score_func = getattr(module, "score_func", None)
            is_sigmoid = isinstance(score_func, Sigmoid) or score_func == "sigmoid"
            if is_sigmoid:
                # Bare sigmoid does not renormalize, so normalize probabilities
                # across experts before computing the balanced-loss baseline.
                probs = probs / probs.sum(dim=-1, keepdim=True)
            topk = topk_expert_ids.reshape(-1, topk_expert_ids.shape[-1])  # (T, K)
            plugin._mb_p_sum[idx] = probs.sum(dim=0)
            plugin._mb_token_count[idx] = probs.shape[0]
            if plugin._aux_scope == "sequence":
                plugin._mb_probs_full[idx] = probs
                plugin._mb_topk[idx] = topk.detach()
            # Routed-slot count per expert is the one-hot top-k sum over tokens.
            n = plugin._step_count_accum[idx].shape[0]
            onehot = torch.zeros(probs.shape[0], n, dtype=torch.float32, device=probs.device)
            onehot.scatter_(1, topk, 1.0)
            counts = onehot.sum(dim=0)
            if plugin._has_ac:
                counts = counts / 2.0
            plugin._step_count_accum[idx] += counts

        return hook_fn

    def compute_microbatch_aux_loss(self):
        """Called by loss_fn (same microbatch graph). Returns scaled, grad-carrying scalar."""
        if not self._moe_layers:
            return 0.0
        seq_ids = self._build_seq_ids() if self._aux_scope == "sequence" else None
        layer_losses = []
        for idx, moe in enumerate(self._moe_layers):
            p_sum = self._mb_p_sum[idx]
            if p_sum is None:
                continue
            n = _resolve_tokens_buffer(moe).shape[0]
            if (
                self._aux_scope == "sequence"
                and seq_ids is not None
                and self._mb_probs_full[idx] is not None
                and self._mb_topk[idx] is not None
            ):
                layer_losses.append(
                    compute_sequence_load_balancing_loss(
                        self._mb_probs_full[idx],
                        self._mb_topk[idx],
                        seq_ids,
                        n,
                    )
                )
            else:
                layer_losses.append(
                    compute_load_balancing_loss(
                        p_sum,
                        token_count=self._mb_token_count[idx],
                        f_global=self._f_global[idx],
                        n_experts=n,
                    )
                )
            self._mb_p_sum[idx] = None
            self._mb_token_count[idx] = 0
            self._mb_probs_full[idx] = None
            self._mb_topk[idx] = None
        if not layer_losses:
            return 0.0
        l_aux = torch.stack(layer_losses).mean()
        self._last_aux = float(l_aux.detach().item())
        return l_aux * self._step_scale

    def reduce_and_cache_counts(self, loss_pg=None) -> None:
        """Step boundary, OFF-GRAPH: all-reduce integer counts → refresh f_global."""
        if not self._moe_layers:
            return
        pg = loss_pg if loss_pg is not None else self._loss_pg
        for idx in range(len(self._moe_layers)):
            counts = self._step_count_accum[idx]
            if pg is not None:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=pg)
            self._f_global[idx] = normalize_counts(counts)
            counts.zero_()

    def collect(self, step: int, trainer: Any) -> dict[str, Any]:  # noqa: ARG002
        if not self._moe_layers:
            return {}
        return {"moe/aux_loss": self._last_aux}

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
