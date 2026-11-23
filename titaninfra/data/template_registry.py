"""Template configurations for precise token-boundary control.

Each template defines role and block start/end tags plus BOS/EOS markers.
Use get_template(name) to resolve canonical names and aliases.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlockTag:
    """Start and end tags for one block type."""
    start: str = ""
    end: str = ""


@dataclass(frozen=True, slots=True)
class RoleTag:
    """Start and end tags for one message role."""
    start: str = ""
    end: str = ""


@dataclass(frozen=True)
class TemplateConfig:
    """Immutable template configuration for one model family."""
    name: str
    roles: dict[str, RoleTag]
    blocks: dict[str, BlockTag]
    bos: str = ""
    eos: str = ""
    image_placeholder: str = ""
    image_token_id: int | None = None
    default_image_tokens: int = 576


# Built-in templates.

_EMPTY = BlockTag()

QWEN3 = TemplateConfig(
    name="qwen3",
    roles={
        "system":    RoleTag("<|im_start|>system\n",    "<|im_end|>\n"),
        "user":      RoleTag("<|im_start|>user\n",      "<|im_end|>\n"),
        "assistant": RoleTag("<|im_start|>assistant\n",  "<|im_end|>\n"),
        "tool":      RoleTag("<|im_start|>user\n<tool_response>\n", "\n</tool_response><|im_end|>\n"),
    },
    blocks={
        "text":        _EMPTY,
        "thinking":    BlockTag("<think>\n", "\n</think>\n\n"),
        "tool_use":    BlockTag("<tool_call>\n", "\n</tool_call>"),
        "tool_result": _EMPTY,
        "image":       _EMPTY,
    },
    eos="<|im_end|>",
)

DEEPSEEK_V3 = TemplateConfig(
    name="deepseek_v3",
    roles={
        "system":    RoleTag("<｜begin▁of▁sentence｜>", ""),
        "user":      RoleTag("<｜User｜>", ""),
        "assistant": RoleTag("<｜Assistant｜>", "<｜end▁of▁sentence｜>"),
        # DeepSeekV3BlockRenderer handles tool-role boundaries and embeds tool
        # output in TextBlock, so the tags are empty here.
        "tool":      RoleTag("", ""),
    },
    blocks={
        "text":        _EMPTY,
        "thinking":    BlockTag("<think>\n", "\n</think>\n\n"),
        "tool_use":    _EMPTY,
        "tool_result": _EMPTY,
        "image":       _EMPTY,
    },
    bos="<｜begin▁of▁sentence｜>",
    eos="<｜end▁of▁sentence｜>",
)

LLAMA4 = TemplateConfig(
    name="llama4",
    roles={
        "system":    RoleTag("<|header_start|>system<|header_end|>\n\n", "<|eot|>"),
        "user":      RoleTag("<|header_start|>user<|header_end|>\n\n",   "<|eot|>"),
        "assistant": RoleTag("<|header_start|>assistant<|header_end|>\n\n", "<|eot|>"),
        "tool":      RoleTag("<|header_start|>ipython<|header_end|>\n\n", "<|eot|>"),
    },
    blocks={
        "text":        _EMPTY,
        "thinking":    BlockTag("<think>\n", "\n</think>\n\n"),
        "tool_use":    _EMPTY,
        "tool_result": _EMPTY,
        "image":       _EMPTY,
    },
    bos="<|begin_of_text|>",
    eos="<|eot|>",
)


# Registry.

DEEPSEEK_V4 = TemplateConfig(
    name="deepseek_v4",
    roles={},
    blocks={},
    bos="<｜begin▁of▁sentence｜>",
    eos="<｜end▁of▁sentence｜>",
)

GPT_OSS = TemplateConfig(
    name="gpt_oss",
    roles={
        "system":    RoleTag("<|start|>system<|message|>",                        "<|end|>"),
        "user":      RoleTag("<|start|>user<|message|>",                          "<|end|>"),
        "assistant": RoleTag("<|start|>assistant<|channel|>final<|message|>",     "<|end|>"),
        "tool":      RoleTag("<|start|>user<|message|>",                          "<|end|>"),
    },
    blocks={
        "text":        _EMPTY,
        "thinking":    BlockTag("<|channel|>analysis<|message|>", "<|end|>"),
        "tool_use":    _EMPTY,
        "tool_result": _EMPTY,
        "image":       _EMPTY,
    },
    eos="<|end|>",
)

GEMMA4 = TemplateConfig(
    name="gemma4",
    roles={
        "system":    RoleTag("<|turn>system\n",    "<turn|>\n"),
        "user":      RoleTag("<|turn>user\n",      "<turn|>\n"),
        "assistant": RoleTag("<|turn>model\n",     "<turn|>\n"),
        "tool":      RoleTag("<|turn>user\n",      "<turn|>\n"),
    },
    blocks={
        "text":        _EMPTY,
        "thinking":    BlockTag("<|channel>thought\n", "\n<channel|>"),
        "tool_use":    BlockTag("<|tool_call>", "<tool_call|>"),
        "tool_result": _EMPTY,
        "image":       _EMPTY,
    },
    eos="<turn|>",
)

_REGISTRY: dict[str, TemplateConfig] = {
    "qwen3": QWEN3,
    "deepseek_v3": DEEPSEEK_V3,
    "deepseek_v4": DEEPSEEK_V4,
    "llama4": LLAMA4,
    "gemma4": GEMMA4,
    "gpt_oss": GPT_OSS,
}

_ALIASES: dict[str, str] = {
    "qwen": "qwen3",
    "qwen2": "qwen3",
    "qwen2.5": "qwen3",
    "qwen2_5": "qwen3",
    "chatml": "qwen3",
    "deepseek": "deepseek_v3",
    "deepseek_r1": "deepseek_v3",
    "llama": "llama4",
    "llama3": "llama4",
}


def get_template(name: str) -> TemplateConfig:
    """Resolve a template by case-insensitive name or alias."""
    key = name.lower()
    canonical = _ALIASES.get(key, key)
    if canonical not in _REGISTRY:
        available = sorted(set(_REGISTRY.keys()) | set(_ALIASES.keys()))
        raise ValueError(f"未知模板 '{name}'，可用: {available}")
    return _REGISTRY[canonical]


def list_templates() -> list[str]:
    """Return sorted canonical template names and aliases."""
    return sorted(set(_REGISTRY.keys()) | set(_ALIASES.keys()))


def register_template(config: TemplateConfig) -> None:
    """Register a custom template configuration."""
    _REGISTRY[config.name] = config
