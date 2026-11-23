"""Kimi-VL family registration and MoonViT configuration.

MOONVIT_CONFIG is a plain dictionary usable without TorchTitan. The model
registry and KimiVLModel.Config assembly are guarded so optional TorchTitan or
Triton dependencies do not break projector/scatter imports on CPU-only hosts.
Unexpected internal import errors are re-raised rather than hidden.
"""

import logging

logger = logging.getLogger(__name__)

# MoonViT hyperparameters from the HF Kimi-VL-A3B vision_config.
MOONVIT_CONFIG = {
    "patch_size": 14,
    "in_dim": 3,
    "hidden_size": 1152,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,  # head_dim=72, verified on GPU
    "intermediate_size": 4304,  # Verified on GPU
    "merge_kernel_size": (2, 2),
    "init_pos_emb_height": 14,
    "init_pos_emb_width": 14,
    "rope_max_height": 512,
    "rope_max_width": 512,
}

# Media placeholder token ID from the HF Kimi-VL config. Visual features are
# scattered into token positions matching this ID.
KIMI_VL_MEDIA_PLACEHOLDER_TOKEN_ID = 163605


# ─────────────────────────────────────────────────────────────────────────────
# TorchTitan-dependent registry surface, guarded for optional dependencies.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import dataclasses

    from titaninfra.models.deepseek_v3 import deepseekv3_configs
    from titaninfra.models.kimi_vl.model import KimiVLModel
    from titaninfra.models.kimi_vl.state_dict_adapter import KimiVLStateDictAdapter

    __all__ = [
        "MOONVIT_CONFIG",
        "KIMI_VL_MEDIA_PLACEHOLDER_TOKEN_ID",
        "KimiVLModel",
        "KimiVLStateDictAdapter",
        "kimi_vl_configs",
        "model_registry",
    ]

    def _a3b(
        attn_backend: str,
        moe_comm_backend: str = "standard",
        non_blocking_capacity_factor: float | None = None,
        parallelism=None,
    ) -> "KimiVLModel.Config":
        """Assemble the Kimi-VL-A3B decoder, vision tower, and projector.

        The decoder reuses deepseek_v3's kimi_vl_a3b config and forwards its
        complete field set to KimiVLModel.Config. MoonViT dimensions come only
        from MOONVIT_CONFIG, keeping one source of truth.
        """
        decoder_cfg = deepseekv3_configs["kimi_vl_a3b"](
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
            parallelism=parallelism,
        )
        # Forward every decoder field to the top-level VLM config.
        decoder_kwargs = {
            f.name: getattr(decoder_cfg, f.name)
            for f in dataclasses.fields(decoder_cfg)
        }
        return KimiVLModel.Config(
            **decoder_kwargs,
            media_placeholder_token_id=KIMI_VL_MEDIA_PLACEHOLDER_TOKEN_ID,
            # MoonViT dimensions use MOONVIT_CONFIG as the source of truth.
            vision_hidden=MOONVIT_CONFIG["hidden_size"],
            vision_num_layers=MOONVIT_CONFIG["num_hidden_layers"],
            vision_num_heads=MOONVIT_CONFIG["num_attention_heads"],
            vision_mlp_dim=MOONVIT_CONFIG["intermediate_size"],
            vision_patch_size=MOONVIT_CONFIG["patch_size"],
            vision_merge_kernel=MOONVIT_CONFIG["merge_kernel_size"],
            vision_init_pos_emb_hw=MOONVIT_CONFIG["init_pos_emb_height"],
            vision_attn_impl=attn_backend,
            # Projector output follows the decoder hidden dimension.
            projector_text_hidden=None,
        )

    kimi_vl_configs = {
        "A3B": _a3b,
    }

    def model_registry(
        flavor: str,
        attn_backend: str = "flex",
        moe_comm_backend: str = "standard",
        non_blocking_capacity_factor: float | None = None,
        parallelism=None,
    ) -> "KimiVLModel.Config":
        """Build a KimiVLModel.Config for the selected flavor.

        parallelism is forwarded to the DeepSeek-V3 language tower so its
        sharding declarations follow the actual topology. The configuration
        does not claim that Kimi-VL with TP>1 has been GPU-verified.
        """
        config = kimi_vl_configs[flavor](
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
            parallelism=parallelism,
        )
        return config

except ImportError as _exc:
    # Tolerate only optional TorchTitan/Triton dependency failures so CPU-only
    # imports keep working. Every other import error is re-raised.
    _missing = getattr(_exc, "name", "") or ""
    if _missing in ("torchtitan", "triton") or _missing.startswith(
        ("torchtitan", "triton")
    ):
        __all__ = ["MOONVIT_CONFIG", "KIMI_VL_MEDIA_PLACEHOLDER_TOKEN_ID"]
        logger.info(
            "kimi_vl model_registry 未定义：缺可选依赖 %s（Mac / --no-deps 属正常）。",
            _missing or "torchtitan",
        )
    else:
        raise  # Propagate genuine internal import errors to the registry.
