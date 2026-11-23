"""Per-template ToolUseAdapter registry.

Dispatch in tokenize_with_hf_template uses get_adapter(template_name) to pick
the correct adapter. Templates not in _REGISTRY get DefaultAdapter (no-op).
"""

from __future__ import annotations

from tests.oracle._legacy.tool_use_adapters.base import DefaultAdapter, ToolUseAdapter
from tests.oracle._legacy.tool_use_adapters.deepseek_v3 import DeepSeekV3Adapter
from tests.oracle._legacy.tool_use_adapters.gemma4 import Gemma4Adapter

_REGISTRY: dict[str, type[ToolUseAdapter]] = {
    "deepseek_r1": DeepSeekV3Adapter,
    "deepseek_v3": DeepSeekV3Adapter,
    "gemma4": Gemma4Adapter,
}

_DEFAULT = DefaultAdapter()


def get_adapter(template_name: str | None) -> ToolUseAdapter:
    """Return the adapter for the given template name (DefaultAdapter if unmapped)."""
    if not template_name:
        return _DEFAULT
    cls = _REGISTRY.get(template_name)
    if cls is None:
        return _DEFAULT
    return cls()


__all__ = [
    "ToolUseAdapter",
    "DefaultAdapter",
    "DeepSeekV3Adapter",
    "Gemma4Adapter",
    "get_adapter",
]
