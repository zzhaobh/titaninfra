"""
MixedOptimizersContainer — Muon for 2D linear params + AdamW for the rest.

Subclasses torchtitan's OptimizersContainer to provide full interface compatibility
(step, zero_grad, state_dict, load_state_dict, Stateful) while using two different
optimizers under the hood.

Param classification (per Moonlight/Kimi K2 best practices):
- param.ndim == 2 and param.numel() >= min_ndim_size
  AND not tok_embeddings/embed_tokens/output/lm_head/gate → Muon
- Everything else (embeddings, lm_head, output projection, MoE gate, norms, biases, small 2D) → AdamW

Param-group support:
- AdamW base hyperparams come from the catch-all `pattern='.*'` entry in
  `config.optimizers` (a real `AdamW.Config` -- upstream's current shape is a
  list of per-pattern, self-contained optimizer configs, first-match-wins;
  there is no more separate "ParamGroupConfig.optimizer_kwargs" indirection).
- Specific per-pattern groups (`layer_lr_overrides` / `freeze_params`) are not yet
  supported here — see `_extract_adamw_base_kwargs`'s fail-loud guard.

This module is imported lazily from _pretrain_helpers.py only when
optimizer_name="MuonAdamW", so it has zero import cost for regular AdamW training.
"""

import functools
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torchtitan.components.optim import AdamW as AdamWConfigurable
from torchtitan.components.optim import OptimizersContainer

from titaninfra.optimizers.muon import Muon

logger = logging.getLogger(__name__)


# Embedding, lm_head (output), and MoE gate are 2D but should NOT use Muon:
# - Embedding/lm_head: lookup tables, not linear transformations
# - MoE gate: low-dim router, orthogonalization may disrupt softmax routing
#   (KellerJordan small-model finding). BUT Moonlight (arXiv 2502.16982) shows
#   router-under-Muon has higher SVD-entropy → benefits MoE at 16B+ scale.
#   Controlled by the `muon_include_router` flag — when True, "gate" is dropped
#   from the exclude set so the router goes to Muon.
_MUON_EXCLUDE_PATTERNS = ("tok_embeddings", "embed_tokens", "output", "lm_head", "gate")
_ROUTER_PATTERN = "gate"


def _muon_exclude_patterns(include_router: bool) -> tuple[str, ...]:
    """Exclude patterns for Muon, optionally letting the MoE router through."""
    if include_router:
        return tuple(p for p in _MUON_EXCLUDE_PATTERNS if p != _ROUTER_PATTERN)
    return _MUON_EXCLUDE_PATTERNS

# 3D MoE expert weight patterns eligible for batched Muon Newton-Schulz.
_EXPERT_PARAM_PATTERNS = ("experts.w1", "experts.w2", "experts.w3")


def _is_expert_param(name: str) -> bool:
    """Check if param name matches a 3D MoE expert weight pattern."""
    return any(pat in name for pat in _EXPERT_PARAM_PATTERNS)


def _classify_params(
    model: nn.Module,
    min_ndim_size: int,
    freeze_patterns: list[str] | None = None,
    include_router: bool = False,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split model params into (muon_params, adamw_params).

    A parameter goes to Muon if: ndim == 2, numel >= min_ndim_size, requires_grad,
    not matched by any freeze pattern, and not an embedding/lm_head/gate param.
    Everything else goes to AdamW. Frozen params are excluded from both lists.
    When ``include_router`` is True, the MoE router/gate is allowed into Muon.

    This is the sole classification path (the old multiplier-aware
    ``_classify_and_group_params`` was removed when upstream dropped
    ``ParamGroupConfig.lr_multiplier``).
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    exclude_patterns = _muon_exclude_patterns(include_router)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if freeze_patterns and any(pat in name for pat in freeze_patterns):
            continue

        is_excluded = any(pat in name for pat in exclude_patterns)

        is_muon_2d = param.ndim == 2 and param.numel() >= min_ndim_size and not is_excluded
        is_muon_3d = param.ndim == 3 and param.numel() >= min_ndim_size and _is_expert_param(name)

        if is_muon_2d or is_muon_3d:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


def _extract_adamw_base_kwargs(optimizers: list) -> dict[str, Any]:
    """Read base AdamW hyperparameters from the catch-all config entry.

    Upstream optimizer entries are self-contained configs with first-match-wins
    patterns. The mixed container supports exactly one catch-all AdamW entry;
    specific per-pattern groups fail loudly because classification does not yet
    assign different learning rates by pattern.
    """
    catch_all = None
    specific = []
    for opt_cfg in optimizers:
        if opt_cfg.pattern == r".*" and isinstance(opt_cfg, AdamWConfigurable.Config):
            catch_all = opt_cfg
        else:
            specific.append(opt_cfg)

    if specific:
        raise NotImplementedError(
            "MixedOptimizersContainer 暂不支持 layer_lr_overrides / freeze_params "
            "（specific optimizer pattern 条目）—— 每个条目自带 lr，但 "
            "MixedOptimizer 的参数分类目前只有 Muon/AdamW 二分，不认识按 pattern "
            "分派不同 lr 这层，需要单独设计才能支持。当前只支持单一 catch-all "
            f"AdamW 条目的标准 Muon 配方。收到的 specific patterns: "
            f"{[opt_cfg.pattern for opt_cfg in specific]}"
        )
    if catch_all is None:
        raise ValueError(
            "MixedOptimizersContainer 需要一个 pattern='.*' 的 catch-all "
            "AdamW.Config 条目承载 lr/betas/eps/weight_decay/fused/foreach/"
            "moment_dtype（由 _build_optimizer_config 组装）。"
            "config.optimizers 为空或缺 catch-all。"
        )
    return {
        "lr": catch_all.lr,
        "betas": catch_all.betas,
        "eps": catch_all.eps,
        "weight_decay": catch_all.weight_decay,
        "fused": catch_all.fused,
        "foreach": catch_all.foreach,
    }


class MixedOptimizersContainer(OptimizersContainer):
    """Mixed Muon+AdamW optimizer container compatible with torchtitan Trainer.

    For each model_part, creates:
    - Muon optimizer for 2D linear weight matrices (≥ min_ndim_size elements)
    - AdamW optimizer for everything else (embeddings, norms, biases, small matrices)

    AdamW base hyperparams (lr/betas/eps/weight_decay/fused/foreach) come from
    the catch-all ``pattern='.*'`` ``AdamW.Config`` entry in ``config.optimizers``
    (assembled by ``_build_optimizer_config``). Specific per-pattern entries
    (``layer_lr_overrides`` / ``freeze_params``) are not yet supported on this
    path — see ``_extract_adamw_base_kwargs`` (fail-loud).

    Inherits OptimizersContainer's full interface: step(), zero_grad(),
    state_dict(), load_state_dict(), and Stateful protocol.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        muon_lr: float = 0.02
        """Learning rate for Muon (2D linear params)."""

        muon_momentum: float = 0.95
        """Momentum decay factor for Muon."""

        muon_nesterov: bool = True
        """Use Nesterov momentum in Muon."""

        muon_ns_steps: int = 5
        """Number of Newton-Schulz iterations."""

        muon_weight_decay: float = 0.0
        """Decoupled weight decay for Muon params."""

        muon_min_ndim_size: int = 4096
        """Minimum numel for a 2D param to use Muon (smaller → AdamW)."""

        muon_ns_dtype: str = "bf16"
        """NS computation dtype: 'bf16' (default) or 'fp32'."""

        muon_scale_factor: float = 1.0
        """Muon update scale multiplier (1.0=original, 0.2=Moonlight)."""

        muon_include_router: bool = False
        """Route the MoE gate to Muon (default False = AdamW). See Moonlight 2502.16982."""

        muon_gather_mode: bool = False
        """Gather full matrix for exact NS (router/attn/dense); experts stay per-shard.
        See docs/superpowers/specs/2026-06-16-muon-gather-mode-design.md."""

    _muon_optimizers: list[Muon | None]
    _adamw_optimizers: list[torch.optim.AdamW | None]

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        enable_accelerator_graph: bool = False,
    ) -> None:
        # Accept the upstream enable_accelerator_graph keyword for interface
        # compatibility, but reject enabling it until Muon's Newton-Schulz and
        # dtype-conversion paths are proven graph-capturable.
        if enable_accelerator_graph:
            raise NotImplementedError(
                "MixedOptimizersContainer（Muon+AdamW）尚未验证是否支持 "
                "accelerator graph 捕获——Muon 的 Newton-Schulz 迭代路径未经过 "
                "capturable 兼容性审计，强行开启可能静默产出错误的更新量。"
                "如需要这个特性，先完成针对 Muon.step() 的 capturable 审计。"
            )
        self.model_parts = model_parts
        self._muon_optimizers = []
        self._adamw_optimizers = []
        all_optimizers: list[torch.optim.Optimizer] = []
        all_params: list[nn.Parameter] = []

        # Read self-contained AdamW hyperparameters from the catch-all entry.
        # Plain torch.optim.AdamW is used here; bf16-moment hooks are not part
        # of the mixed Muon path.
        adamw_kwargs = _extract_adamw_base_kwargs(config.optimizers)
        ns_dtype = torch.bfloat16 if config.muon_ns_dtype == "bf16" else torch.float32
        muon_kwargs: dict[str, Any] = {
            "lr": config.muon_lr,
            "momentum": config.muon_momentum,
            "nesterov": config.muon_nesterov,
            "ns_steps": config.muon_ns_steps,
            "weight_decay": config.muon_weight_decay,
            "scale_factor": config.muon_scale_factor,
            "gather_mode": config.muon_gather_mode,
        }

        total_muon_params = 0
        total_adamw_params = 0

        for model in self.model_parts:
            muon_params, adamw_params = _classify_params(
                model,
                min_ndim_size=config.muon_min_ndim_size,
                include_router=config.muon_include_router,
            )

            if muon_params:
                muon_opt = Muon(
                    [{"params": muon_params, **muon_kwargs}],
                    ns_dtype=ns_dtype,
                    **muon_kwargs,
                )
                self._muon_optimizers.append(muon_opt)
                all_optimizers.append(muon_opt)
                total_muon_params += len(muon_params)
                all_params.extend(muon_params)
            else:
                self._muon_optimizers.append(None)

            if adamw_params:
                adamw_opt = torch.optim.AdamW(
                    [{"params": adamw_params, **adamw_kwargs}]
                )
                self._adamw_optimizers.append(adamw_opt)
                all_optimizers.append(adamw_opt)
                total_adamw_params += len(adamw_params)
                all_params.extend(adamw_params)
            else:
                self._adamw_optimizers.append(None)

        self.optimizers = all_optimizers

        logger.info(
            f"MixedOptimizer: {total_muon_params} Muon params "
            f"(2D, ≥{config.muon_min_ndim_size}), "
            f"{total_adamw_params} AdamW params (rest)"
        )

        if not all_params:
            raise ValueError(
                "MixedOptimizer: no parameters left to optimize. Check your "
                "model / param_group configuration."
            )

        self._post_init(all_params)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Serialize all optimizer states via DCP.

        For each model_part, merge the state dicts from both the Muon and AdamW
        optimizers. Since they hold disjoint parameter sets, keys don't collide.
        """
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        merged = {}
        for i, model in enumerate(self.model_parts):
            muon_opt = self._muon_optimizers[i]
            adamw_opt = self._adamw_optimizers[i]

            if muon_opt is not None:
                merged.update(func(model, muon_opt))
            if adamw_opt is not None:
                merged.update(func(model, adamw_opt))

        return merged

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer states from DCP.

        Delegates to set_optimizer_state_dict for each (model_part, optimizer) pair.
        DCP resolves which keys belong to which optimizer based on parameter FQNs.
        """
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        for i, model in enumerate(self.model_parts):
            muon_opt = self._muon_optimizers[i]
            adamw_opt = self._adamw_optimizers[i]

            if muon_opt is not None:
                func(model, muon_opt)
            if adamw_opt is not None:
                func(model, adamw_opt)
