"""Register and resolve chat templates and aliases.

get_template performs case-insensitive alias resolution, list_templates returns
canonical names and aliases, and register_template adds custom definitions.
"""


from titaninfra.data.templates.definition import BUILTIN_TEMPLATES, ChatTemplate

# Canonical template registry: name -> template.
_REGISTRY: dict[str, ChatTemplate] = {
    t.name: t for t in BUILTIN_TEMPLATES
}

# Aliases: multiple keys can resolve to one canonical template.
_ALIASES: dict[str, str] = {
    # Qwen families
    "qwen": "qwen3",
    "qwen2": "qwen2_5",
    "qwen2.5": "qwen2_5",
    # Qwen reasoning aliases
    "qwen3_reasoning": "qwen3",
    "qwen3_thinking": "qwen3",
    # Llama families
    "llama": "llama3",
    "llama3.1": "llama3",
    "llama3.2": "llama3",
    "llama3.3": "llama3",
    # Mistral families
    "mistral_v1": "mistral",
    "mixtral": "mistral",
    # ChatML aliases
    "im": "chatml",
    "openai": "chatml",
    # DeepSeek
    "deepseek": "deepseek_r1",
    "r1": "deepseek_r1",
    # DeepSeek reasoning aliases
    "deepseek_r1_thinking": "deepseek_r1_reasoning",
    "r1_thinking": "deepseek_r1_reasoning",
}


def get_template(name: str) -> ChatTemplate:
    """Resolve a template by case-insensitive name or alias.

    Raise ValueError with the available names when no template matches.
    """
    key = name.lower()
    # Resolve aliases before the canonical registry lookup.
    canonical = _ALIASES.get(key, key)
    if canonical not in _REGISTRY:
        available = sorted(set(_REGISTRY.keys()) | set(_ALIASES.keys()))
        raise ValueError(
            f"未知模板 '{name}'。可用模板：{available}"
        )
    return _REGISTRY[canonical]


def list_templates() -> list[str]:
    """Return sorted canonical template names and aliases."""
    return sorted(set(_REGISTRY.keys()) | set(_ALIASES.keys()))


def register_template(template: ChatTemplate) -> None:
    """Register a custom ChatTemplate under its name."""
    _REGISTRY[template.name] = template


# Internal accessors for tests and advanced consumers.

def _get_registry() -> dict[str, ChatTemplate]:
    """Return the canonical registry for internal consumers."""
    return _REGISTRY


def _get_aliases() -> dict[str, str]:
    """Return the alias map for internal consumers."""
    return _ALIASES
