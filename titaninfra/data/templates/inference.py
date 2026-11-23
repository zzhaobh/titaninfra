"""Infer an appropriate chat template from a model family name."""

from titaninfra.data.templates.definition import ChatTemplate
from titaninfra.data.templates.registry import get_template

# Family name to registered template name.
_FAMILY_TO_TEMPLATE = {
    "qwen3": "qwen3",
    "qwen2": "qwen3",
    "llama3": "llama3",
    "llama4": "llama3",
    "llama2": "llama2",
    "mistral": "mistral",
    "deepseek_v3": "deepseek_r1",
    "gpt_oss": "chatml",
}


def infer_template_from_model_name(model_name: str) -> ChatTemplate | None:
    """Return the template registered for model_name, or None if unknown.

    The family is the portion before the first slash, for example "qwen3" in
    "qwen3/8B". Callers can provide their own fallback template when this
    function returns None.
    """
    if not model_name:
        return None

    family = model_name.split("/")[0].lower()
    template_name = _FAMILY_TO_TEMPLATE.get(family)
    if template_name:
        return get_template(template_name)

    return None
