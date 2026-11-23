"""Sharding config for the Kimi-VL VLM family.

Kimi-VL's decoder IS DeepSeek-V3 (MLA + MoE), so the decoder sharding is
DeepSeek-V3's ``set_deepseek_v3_sharding_config`` verbatim. Unlike qwen3_5 — whose
vision encoder is a torchtitan ``Module`` that needs per-linear TP sharding
configs — Kimi-VL's ``vision_tower`` (MoonViT) and ``projector`` are bare
``nn.Module``s. They carry NO ``sharding_config`` and are therefore invisible to
``model.parallelize()`` (the TP/EP Module-protocol walk); they are handled purely
by whole-unit FSDP in ``parallelize.py``. So this module only fills the DeepSeek-V3
decoder sub-tree.

This mirrors how ``KimiVLModel.Config`` inherits ``DeepSeekV3Model.Config`` — the
base ``update_from_config`` already calls ``set_deepseek_v3_sharding_config`` on
the decoder sub-configs. This wrapper is provided for parity with the other
families' ``sharding.py`` surface and as the single documented seam for any
future Kimi-VL-specific TP override on the vision tower.

GPU-VERIFY: confirm on a GPU node that ``KimiVLModel.Config.update_from_config``
(inherited from ``DeepSeekV3Model.Config``) fills the decoder sharding, and that
the bare vision_tower/projector are correctly skipped by ``model.parallelize()``
(no ``sharding_config`` attribute → not TP-sharded).
"""

from typing import TYPE_CHECKING

from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_sharding_config

if TYPE_CHECKING:
    from titaninfra.models.kimi_vl.model import KimiVLModel


def set_kimi_vl_sharding_config(
    config: "KimiVLModel.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on the Kimi-VL DeepSeek-V3 decoder sub-configs.

    Delegates to ``set_deepseek_v3_sharding_config`` — ``KimiVLModel.Config``
    subclasses ``DeepSeekV3Model.Config`` so it exposes the decoder sub-configs
    (``tok_embeddings`` / ``layers`` / ``norm`` / ``lm_head``) at the top level.
    The bare ``vision_tower`` / ``projector`` need no sharding config (FSDP-only,
    see ``parallelize.py``).
    """
    set_deepseek_v3_sharding_config(
        config,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
    )
