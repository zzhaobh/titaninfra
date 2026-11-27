"""Kimi K3 model family, maintained by titaninfra.

K3 uses KDA (Kimi Delta Attention) + Gated MLA hybrid layers with a
QuantileBalancedTopKRouter and SiTU-GLU experts. The KDA layers require
Attention Gym kernels (SM90+ GPU) and the ``varlen`` attention backend.

Only text-only flavors are registered. The ``_mm`` (multimodal) variants
require additional VLM dependencies and are deferred.
"""

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TorchTitan-dependent registry surface, guarded for optional dependencies.
# KDA depends on attention-gym / torch_remat / spmd_types which may be absent
# on CPU-only hosts.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from titaninfra.models.kimi_k3.flavors import (
        MODEL_FLAVORS as _UPSTREAM_FLAVORS,
        build_model_config as _build_model_config,
    )

    __all__ = [
        "kimi_k3_configs",
        "model_registry",
        "MODEL_FLAVORS",
    ]

    MODEL_FLAVORS = _UPSTREAM_FLAVORS

    _TEXT_ONLY_FLAVORS = frozenset({"debugmodel", "Kimi-K3"})

    kimi_k3_configs = {
        flavor: _UPSTREAM_FLAVORS[flavor]
        for flavor in _TEXT_ONLY_FLAVORS
        if flavor in _UPSTREAM_FLAVORS
    }

    def model_registry(
        flavor: str,
        attn_backend: str = "varlen",
        parallelism=None,
        seq_len: int | None = None,
    ):
        """Build a Kimi K3 model config from titaninfra-maintained flavors.

        Defaults to ``varlen`` because KDA layers do not support flex
        attention. The engine's post-hoc ``_apply_attn_backend`` override
        should be suppressed for this family.
        """
        if flavor not in kimi_k3_configs:
            raise KeyError(
                f"kimi_k3 不支持 flavor '{flavor}'。"
                f"可用（text-only）：{sorted(kimi_k3_configs)}"
            )

        if parallelism is None:
            from torchtitan.config.parallelism import ParallelismConfig

            parallelism = ParallelismConfig()

        return _build_model_config(
            flavor,
            attn_backend=attn_backend,
            parallelism=parallelism,
            seq_len=seq_len,
        )

except ImportError as _exc:
    _missing = getattr(_exc, "name", "") or ""
    if _missing.startswith("torchtitan") or _missing in (
        "spmd_types",
        "torch_remat",
    ):
        logger.info(
            "kimi_k3 未注册：缺少可选依赖 %r（--no-deps 环境下属正常）。",
            _missing,
        )
    else:
        logger.warning(
            "kimi_k3 注册失败（非依赖缺失，疑似内部 import 错误）：%s",
            _exc,
        )
