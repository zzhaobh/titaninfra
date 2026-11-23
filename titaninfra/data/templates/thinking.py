"""Handle reasoning content in assistant messages.

Three strategies cover a separate think field, tags embedded in content, and
messages without reasoning. Historical turns can strip reasoning while the
current turn retains it according to enable_thinking.
"""

import re
from enum import Enum

from titaninfra.data.templates.definition import ChatTemplate

# Thinking-strategy types.


class ThinkingStrategy(Enum):
    """Strategy used to process reasoning content."""

    THINK_FIELD = "think_field"  # Separate think field
    CONTENT_TAG = "content_tag"  # Thinking tags embedded in content
    NO_THOUGHT = "no_thought"  # No reasoning content


# Reasoning removal.

def remove_thought(content: str, thought_words: tuple[str, str]) -> str:
    """Remove all text between the supplied start/end thinking tags.

    Tags are regex-escaped and matched non-greedily across lines. Plain text
    without matching tags is returned unchanged apart from outer whitespace.
    """
    start_tag, end_tag = thought_words
    # Escape tags and match reasoning non-greedily across lines.
    start_escaped = re.escape(start_tag)
    end_escaped = re.escape(end_tag)
    pattern = f"{start_escaped}.*?{end_escaped}"
    return re.sub(pattern, "", content, flags=re.DOTALL).strip()


# Assistant reasoning strategies.

def handle_assistant_with_think_field(
    template: ChatTemplate,
    content: str,
    think_content: str,
    is_last_message: bool,
    enable_thinking: bool,
) -> list[tuple[str, bool]]:
    """Handle an assistant message with a separate think field.

    The current turn includes reasoning in loss when enabled; historical turns
    and disabled thinking use only the answer content.
    """
    start_tag, end_tag = template.thought_words
    thought_part = f"{start_tag}{think_content}{end_tag}"

    if is_last_message and enable_thinking:
        # Include reasoning in loss for the current turn when enabled.
        return [
            (template.assistant_role.prefix, False),
            (thought_part + content + template.assistant_role.suffix, True),
        ]
    else:
        # History and disabled thinking use only answer content.
        return [
            (template.assistant_role.prefix, False),
            (content + template.assistant_role.suffix, True),
        ]


def handle_assistant_with_thought_in_content(
    template: ChatTemplate,
    content: str,
    is_last_message: bool,
    enable_thinking: bool,
) -> list[tuple[str, bool]]:
    """Handle reasoning tags embedded in assistant content.

    Historical turns remove reasoning. For the current turn, enable_thinking
    determines whether tagged reasoning is preserved or removed.
    """
    # Historical turns strip reasoning while preserving answer text.
    if not is_last_message:
        content = remove_thought(content, template.thought_words)
        return [
            (template.assistant_role.prefix, False),
            (content + template.assistant_role.suffix, True),
        ]

    # The current turn preserves reasoning only when enabled.
    if not enable_thinking:
        content = remove_thought(content, template.thought_words)

    return [
        (template.assistant_role.prefix, False),
        (content + template.assistant_role.suffix, True),
    ]


def handle_assistant_no_thought(
    template: ChatTemplate,
    content: str,
) -> list[tuple[str, bool]]:
    """Format an assistant message without reasoning content."""
    return [
        (template.assistant_role.prefix, False),
        (content + template.assistant_role.suffix, True),
    ]


# Unified strategy selection.

def _detect_thinking_strategy(
    template: ChatTemplate,
    content: str,
    think_content: str | None,
) -> ThinkingStrategy:
    """Select THINK_FIELD, CONTENT_TAG, or NO_THOUGHT from the input."""
    # Prefer an explicit think field.
    if think_content:
        return ThinkingStrategy.THINK_FIELD

    # Otherwise inspect content for reasoning tags.
    if template.thought_words:
        start_tag, end_tag = template.thought_words
        if start_tag.strip() in content or end_tag.strip() in content:
            return ThinkingStrategy.CONTENT_TAG

    # Fall back to a plain assistant message.
    return ThinkingStrategy.NO_THOUGHT


def handle_assistant_thinking(
    template: ChatTemplate,
    content: str,
    think_content: str | None,
    is_last_message: bool,
    enable_thinking: bool,
) -> list[tuple[str, bool]]:
    """Dispatch assistant reasoning to the appropriate strategy handler."""
    # Select the strategy and invoke its handler.
    strategy = _detect_thinking_strategy(template, content, think_content)
    if strategy == ThinkingStrategy.THINK_FIELD:
        return handle_assistant_with_think_field(
            template, content, think_content, is_last_message, enable_thinking
        )
    elif strategy == ThinkingStrategy.CONTENT_TAG:
        return handle_assistant_with_thought_in_content(
            template, content, is_last_message, enable_thinking
        )
    else:  # ThinkingStrategy.NO_THOUGHT
        return handle_assistant_no_thought(template, content)
