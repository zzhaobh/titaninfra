"""Optimizer helpers for per-parameter learning-rate overrides."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _build_param_lr_map(
    model: nn.Module,
    layer_lr_overrides: dict[str, float],
    base_lr: float,
) -> dict[torch.Tensor, float]:
    """Build a parameter-to-learning-rate map in one model traversal.

    The first matching pattern wins; base_lr is used for unmatched parameters.
    """
    param_lr_map: dict[torch.Tensor, float] = {}
    matched_counts: dict[str, int] = {pattern: 0 for pattern in layer_lr_overrides}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        matched_lr = base_lr
        for pattern, lr in layer_lr_overrides.items():
            if pattern in name:
                matched_lr = lr
                matched_counts[pattern] += 1
                break

        param_lr_map[param] = matched_lr

    stats = [f"{p}: {c} 个参数" for p, c in matched_counts.items() if c > 0]
    logger.info(
        f"参数学习率覆盖：找到 {len(param_lr_map)} 个可训练参数，"
        f"其中 {sum(matched_counts.values())} 个使用覆盖学习率 ({', '.join(stats)})"
    )
    return param_lr_map


def setup_layer_lr_hook(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    layer_lr_overrides: dict[str, float],
    base_lr: float,
) -> None:
    """Register a pre-step hook that applies per-parameter learning rates.

    This leaves TorchTitan and optimizer initialization unchanged while allowing
    patterns such as {"router": 1e-3, "embed": 1e-5} to override base_lr.
    """
    param_lr_map = _build_param_lr_map(model, layer_lr_overrides, base_lr)

    def lr_adjust_hook(
        optimizer: torch.optim.Optimizer,
        args: tuple,  # noqa: ARG001 — required by register_step_pre_hook
        kwargs: dict,  # noqa: ARG001 — required by register_step_pre_hook
    ) -> None:
        """Apply the mapped learning rate before each optimizer step."""
        for param_group in optimizer.param_groups:
            if not param_group["params"]:
                continue
            first_param = param_group["params"][0]
            if first_param in param_lr_map:
                param_group["lr"] = param_lr_map[first_param]

    optimizer.register_step_pre_hook(lr_adjust_hook)
    logger.info("✓ 参数学习率调整 hook 已注册，将在每个 optimizer step 前自动调整")
