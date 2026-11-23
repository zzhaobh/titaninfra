"""Wire opt-in kernels into model forward seams without changing state_dict.

No replacement occurs when ModelConfig.kernel_impls is empty. Enabled kernels
are resolved through the registry and swapped only on existing bound impl
methods, preserving module structure and DCP checkpoint compatibility.

FSDP2 pure-DP forwards may expose plain Parameters, while SP/TP paths can still
expose DTensor. Kernels that require plain tensors must use an appropriate
to_local/from_local boundary rather than silently accepting the wrong layout.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def wire_custom_kernels(model_parts, model_config) -> None:
    """Wire enabled kernels into existing model forward seams.

    The default empty kernel_impls list is a no-op. New operators add a branch
    here and resolve the callable with get_kernel, which handles fallback.
    """
    enabled = getattr(model_config, "kernel_impls", None)
    if not enabled:
        return

    hc_names = {"hc_pre", "hc_post", "hc_head"}
    if hc_names & set(enabled):
        _wire_hc_kernels(model_parts, enabled)

    mooned = _wire_moonvit(model_parts, enabled)

    # csa_gather is dispatched internally by use_gather_kernel rather than this
    # wiring layer. Exclude it from unwired-operator warnings.
    gather_names = {"csa_gather"}
    for name in gather_names & set(enabled):
        logger.info(
            "%s 由 use_gather_kernel flag 派发（B 方案），不经 kernel_impls wiring，已跳过。",
            name,
        )

    remaining = [n for n in enabled if n not in (hc_names | gather_names | mooned)]
    if remaining:
        # Add wiring branches for new operators here.
        logger.warning(
            "kernel_impls=%s 含未接线算子。新算子需在 wire_custom_kernels 加分支。",
            remaining,
        )


def _wire_hc_kernels(model_parts, enabled) -> None:
    """Swap DeepSeek V4 Hyper-Connection impls for the torch.compile-fused ones.

    Replaces the per-instance ``_hc_pre_impl`` / ``_hc_post_impl`` on each
    ``DeepSeekV4TransformerBlock`` and ``_hc_head_impl`` on ``DeepSeekV4Model``
    with the compiled callable from the registry. get_kernel already handles
    hardware fallback. Structure/state_dict untouched (the params live on the
    module; only the bound impl function is swapped) → DCP-compatible.
    """
    from titaninfra.kernels.registry import get_kernel
    from titaninfra.models.deepseek_v4.model import (
        DeepSeekV4Model,
        DeepSeekV4TransformerBlock,
    )

    hc_pre = get_kernel("hc_pre", enabled) if "hc_pre" in enabled else None
    hc_post = get_kernel("hc_post", enabled) if "hc_post" in enabled else None
    hc_head = get_kernel("hc_head", enabled) if "hc_head" in enabled else None

    swapped = {"hc_pre": 0, "hc_post": 0, "hc_head": 0}
    for part in model_parts:
        for module in part.modules():
            if isinstance(module, DeepSeekV4TransformerBlock):
                if hc_pre is not None:
                    module._hc_pre_impl = hc_pre
                    swapped["hc_pre"] += 1
                if hc_post is not None:
                    module._hc_post_impl = hc_post
                    swapped["hc_post"] += 1
            if isinstance(module, DeepSeekV4Model):
                if hc_head is not None:
                    module._hc_head_impl = hc_head
                    swapped["hc_head"] += 1
    logger.info("deepseek_v4 HC kernels wired (torch.compile-fused): %s", swapped)


def _wire_moonvit(model_parts, enabled) -> set[str]:
    """Swap MoonViT VisionEncoderLayer._rope_impl / PatchMerger._merge_impl for registry kernels.

    DP-only: the vision tower is not TP/CP sharded, so kernels receive plain tensors.
    """
    rope_on = "moonvit_rope" in enabled
    merge_on = "moonvit_patch_merge" in enabled
    if not (rope_on or merge_on):
        return set()

    from titaninfra.kernels.registry import get_kernel
    from titaninfra.models.kimi_vl.vision_encoder import PatchMerger, VisionEncoderLayer

    rope_impl = get_kernel("moonvit_rope", enabled) if rope_on else None
    merge_impl = get_kernel("moonvit_patch_merge", enabled) if merge_on else None
    swapped = {"moonvit_rope": 0, "moonvit_patch_merge": 0}
    handled: set[str] = set()
    for part in model_parts:
        for module in part.modules():
            if rope_impl is not None and isinstance(module, VisionEncoderLayer):
                module._rope_impl = rope_impl
                swapped["moonvit_rope"] += 1
            if merge_impl is not None and isinstance(module, PatchMerger):
                module._merge_impl = merge_impl
                swapped["moonvit_patch_merge"] += 1
    if rope_on:
        handled.add("moonvit_rope")
    if merge_on:
        handled.add("moonvit_patch_merge")
    logger.info("moonvit kernels wired: %s", swapped)
    return handled
