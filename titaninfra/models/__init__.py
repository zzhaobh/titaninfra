"""TitanInfra model-family registry with TorchTitan fallback support.

TitanInfra-maintained families take precedence over upstream implementations.
The optional _titaninfra flavor suffix is accepted for backward compatibility
and stripped before calling each family's model_registry.
"""

import importlib
import logging
from typing import TYPE_CHECKING

from titaninfra.models.moe_balance import MoEBalanceSpec, register_moe_balance

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


# TitanInfra-maintained families, extensible at runtime.
_SUPPORTED_FAMILIES: set[str] = set()


def register_family(family: str) -> None:
    """Add a model family to the TitanInfra-supported set."""
    _SUPPORTED_FAMILIES.add(family)


def is_supported(family: str) -> bool:
    """Return whether TitanInfra maintains this model family."""
    return family in _SUPPORTED_FAMILIES


def load_model_spec(family: str, flavor: str, *, parallelism=None) -> "ModelSpec":
    """Load a model specification from a TitanInfra family.

    parallelism is forwarded to the family model_registry so sharding
    declarations follow the real topology. Unsupported families and flavors
    raise ValueError and KeyError respectively.
    """
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"模型系列 '{family}' 不在 TitanInfra 维护列表中。\n"
            f"已注册系列：{sorted(_SUPPORTED_FAMILIES)}\n"
            f"请通过 register_family('{family}') 注册，"
            f"或使用 torchtitan 原生模型。"
        )

    try:
        module = importlib.import_module(f"titaninfra.models.{family}")
    except ImportError as e:
        raise ImportError(
            f"无法加载 titaninfra.models.{family}：{e}\n"
            f"请检查模块是否存在。"
        ) from e

    # Strip the optional suffix before model_registry; attention backend is
    # applied separately by the engine.
    base_flavor, _is_titaninfra = strip_titaninfra_suffix(flavor)
    try:
        model_spec = module.model_registry(base_flavor, parallelism=parallelism)
    except KeyError as e:
        configs_dict = getattr(module, f"{family}_configs", {})
        available = sorted(configs_dict.keys())
        raise KeyError(
            f"TitanInfra '{family}' 系列不支持 flavor '{flavor}'（base='{base_flavor}'），\n"
            f"可用版本：{available}"
        ) from e

    return model_spec


def strip_titaninfra_suffix(flavor: str) -> tuple[str, bool]:
    """Return (flavor_without_suffix, suffix_was_present)."""
    if flavor.endswith("_titaninfra"):
        return flavor.removesuffix("_titaninfra"), True
    return flavor, False


# Import each family independently to trigger registration. Optional dependency
# failures are logged at info level; genuine internal import errors emit a
# warning instead of being hidden as an unsupported-model error.
_FAMILY_MODULES: tuple[str, ...] = (
    "qwen3",        # Qwen3 dense and MoE variants
    "qwen2_5",      # Dense with QKV biases
    "gemma4",       # Hybrid attention and parallel MLP/MoE
    "gpt_oss",      # MoE architecture
    "deepseek_v3",  # MLA + MoE
    "deepseek_v4",  # Hyper-Connections and compressed attention
    "llama3",       # Dense with attention-owned RoPE
    "qwen3_5",      # Gated-delta linear attention and vision
    "kimi_vl",      # MoonViT plus DeepSeek-V3 language tower
    "kimi_k3",      # KDA + Gated MLA hybrid, maintained in-tree
)

for _family in _FAMILY_MODULES:
    try:
        importlib.import_module(f"titaninfra.models.{_family}")  # Registration side effect
        _SUPPORTED_FAMILIES.add(_family)
    except ImportError as _exc:
        _missing = getattr(_exc, "name", "") or ""
        if _missing in (f"titaninfra.models.{_family}", "torchtitan") or _missing.startswith("torchtitan"):
            logger.info(
                "模型系列 %r 未注册：缺少可选依赖 %r（--no-deps 环境下属正常）。",
                _family, _missing or "torchtitan",
            )
        else:
            logger.warning(
                "模型系列 %r 注册失败（非依赖缺失，疑似模块内部 import 错误）：%s",
                _family, _exc,
            )
del _family


# MoE balancing is an architectural declaration independent of optional model
# imports. DeepSeek V3/V4 combine expert-bias feedback with sequence-scoped
# auxiliary loss; post-train families declare no balancing by default.
register_moe_balance(
    "deepseek_v3",
    MoEBalanceSpec(bias_gamma=1e-3, aux_coeff=1e-4, aux_scope="sequence"),
)

# DeepSeek V4 shares V3's balance recipe (bias feedback + sequence-level aux-loss).
register_moe_balance(
    "deepseek_v4",
    MoEBalanceSpec(bias_gamma=1e-3, aux_coeff=1e-4, aux_scope="sequence"),
)


# Public API.
__all__ = [
    "register_family",
    "is_supported",
    "load_model_spec",
    "strip_titaninfra_suffix",
]
