"""Chat-template definitions, registry, formatting, and validation.

Templates are immutable data; formatting and encoding remain independent of
tokenizer-specific business logic. Use get_template for lookup, apply_template
for loss-tagged text turns, and encode_with_template for token IDs and labels.
Message validation, model-name inference, and thinking removal are re-exported
here for backward compatibility.
"""

# Re-export public APIs from the template submodules.
from titaninfra.data.templates.definition import (
    IGNORE_INDEX,
    ChatTemplate,
    Role,
)

# Formatting and encoding.
from titaninfra.data.templates.formatters import (
    apply_template,
    encode_with_template,
)

# Model-name inference.
from titaninfra.data.templates.inference import (
    infer_template_from_model_name,
)

# Template registry.
from titaninfra.data.templates.registry import (
    get_template,
    list_templates,
    register_template,
)

# Thinking-content helpers.
from titaninfra.data.templates.thinking import (
    remove_thought,
)

# Message validation.
from titaninfra.data.templates.validation import (
    ValidationResult,
    validate_messages,
)

__all__ = [
    # Constants
    "IGNORE_INDEX",

    # Data structures
    "Role",
    "ChatTemplate",
    "ValidationResult",

    # Template management
    "get_template",
    "list_templates",
    "register_template",

    # Formatting
    "apply_template",
    "encode_with_template",

    # Validation
    "validate_messages",

    # Thinking helpers
    "remove_thought",

    # Model-name inference
    "infer_template_from_model_name",
]
