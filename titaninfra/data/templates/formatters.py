"""Legacy template formatting and tokenization.

apply_template returns text turns with loss flags; encode_with_template also
tokenizes each turn and builds IGNORE_INDEX labels for masked positions. Inputs
are never mutated.
"""

import logging

from titaninfra.data.templates.definition import IGNORE_INDEX, ChatTemplate
from titaninfra.data.templates.thinking import handle_assistant_thinking
from titaninfra.data.templates.validation import validate_messages

logger = logging.getLogger(__name__)


# Core formatting functions.

def apply_template(  # noqa: C901 - deprecated legacy formatter
    template: ChatTemplate,
    messages: list[dict],
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
    default_system: str | None = None,
) -> list[tuple[str, bool]]:
    """Format messages into (text, is_loss) turns without mutating input.

    Validation, BOS, default-system injection, role formatting, and thinking
    handling run before optional generation-prompt output. The caller owns
    tokenization and label construction.
    """
    # Validate message structure before formatting.
    validation_result = validate_messages(messages, template.name)

    # Emit validation warnings without stopping valid processing.
    import logging
    logger = logging.getLogger(__name__)
    for warning in validation_result.warnings:
        logger.warning(f"数据完整性警告: {warning}")

    # Errors stop processing, except for plain assistant-only samples.
    if not validation_result.is_valid:
        # Allow plain samples that intentionally contain only assistant text.
        is_plain_special_case = (
            template.name == "plain" and
            len([e for e in validation_result.errors if "user" in e.lower()]) > 0 and
            len([e for e in validation_result.errors if "assistant" in e.lower()]) == 0
        )

        if not is_plain_special_case:
            raise ValueError(f"数据格式错误: {'; '.join(validation_result.errors)}")

    # Resolve the effective thinking setting.
    if enable_thinking is None:
        enable_thinking = template.enable_thinking

    turns: list[tuple[str, bool]] = []

    # Prepend BOS when configured.
    if template.bos_token:
        turns.append((template.bos_token, False))

    # Caller-supplied default_system overrides the template default when no
    # system message is present.
    has_system = any(m.get("role") == "system" for m in messages)
    effective_default_system = default_system if default_system is not None else template.default_system

    if not has_system and effective_default_system:
        system_text = (
            template.system_role.prefix
            + effective_default_system
            + template.system_role.suffix
        )
        turns.append((system_text, False))

    # Format each message by role.
    for i, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            text = template.system_role.prefix + content + template.system_role.suffix
            turns.append((text, False))

        elif role == "user":
            if template.image_token and "images" in message:
                # Insert image placeholder tokens for multimodal user messages.
                images = message["images"]
                if isinstance(images, list) and len(images) > 0:
                    # Repeat the placeholder once per image.
                    image_tokens = template.image_token * len(images)
                    content = image_tokens + content

            user_pre = template.user_role.prefix + content + template.user_role.suffix
            turns.append((user_pre, False))

        elif role == "assistant":
            # Serialize assistant tool calls when a prefix is configured.
            tool_calls_text = ""
            if "tool_calls" in message and template.tool_call_prefix:
                tool_calls = message["tool_calls"]
                if isinstance(tool_calls, list) and len(tool_calls) > 0:
                    # Serialize tool calls as JSON.
                    import json
                    tool_calls_str = json.dumps(tool_calls, ensure_ascii=False)
                    tool_calls_text = template.tool_call_prefix + tool_calls_str

            # A separate think field takes precedence over tags in content.
            think_content = message.get("think", "")

            # Tool-call text precedes assistant content.
            if tool_calls_text:
                content = tool_calls_text + content

            # Apply reasoning strategy only when the template declares tags.
            if template.thought_words:
                is_last_message = (i == len(messages) - 1)
                turns.extend(
                    handle_assistant_thinking(
                        template, content, think_content,
                        is_last_message, enable_thinking
                    )
                )
            else:
                # Without thinking tags, use ordinary assistant formatting.
                turns.append((template.assistant_role.prefix, False))
                turns.append((content + template.assistant_role.suffix, True))

        elif role == "tool":
            # Tool results use tool_role when present and user_role otherwise.
            role_format = template.tool_role if template.tool_role else template.user_role
            tool_text = role_format.prefix + content + role_format.suffix
            turns.append((tool_text, False))

        # Unknown roles are ignored.

        if template.sep:
            turns.append((template.sep, False))

    # Optionally append an assistant prefix for inference, with no loss.
    if add_generation_prompt:
        turns.append((template.assistant_role.prefix, False))

    return turns


def encode_with_template(
    tokenizer,
    template: ChatTemplate,
    messages: list[dict],
    max_length: int = 4096,
    train_on_inputs: bool = False,
    enable_thinking: bool | None = None,
    default_system: str | None = None,
    debug: bool = False,
) -> tuple[list[int], list[int]]:
    """Format and tokenize messages into input_ids and IGNORE_INDEX labels.

    train_on_inputs=True trains prompt tokens too; otherwise only turns marked
    is_loss contribute labels. Both lists are truncated to max_length.
    """
    turns = apply_template(
        template,
        messages,
        enable_thinking=enable_thinking,
        default_system=default_system,
    )

    input_ids: list[int] = []
    labels: list[int] = []

    for text, is_loss in turns:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(tokens)

        if train_on_inputs or is_loss:
            labels.extend(tokens)
        else:
            labels.extend([IGNORE_INDEX] * len(tokens))

    # Truncate ids and labels consistently.
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    # Optionally print detailed debugging information.
    if debug:
        _print_debug_info(
            tokenizer=tokenizer,
            template=template,
            messages=messages,
            turns=turns,
            input_ids=input_ids,
            labels=labels,
            max_length=max_length,
            train_on_inputs=train_on_inputs,
            enable_thinking=enable_thinking,
        )

    return input_ids, labels


def _print_debug_info(
    tokenizer,
    template: ChatTemplate,
    messages: list[dict],
    turns: list[tuple[str, bool]],
    input_ids: list[int],
    labels: list[int],
    max_length: int,
    train_on_inputs: bool,
    enable_thinking: bool | None,
):
    """Print configuration, messages, turns, and tokenization diagnostics."""

    # Separator line.
    sep = "=" * 80

    print(f"\n{sep}")
    print("[DEBUG] 样本调试信息")
    print(sep)

    # 1. Configuration.
    print("\n[配置]")
    print(f"  模板名称: {template.name}")
    print(f"  最大长度: {max_length}")
    print(f"  train_on_inputs: {train_on_inputs}")
    print(f"  enable_thinking: {enable_thinking}")

    # 2. Original messages.
    print(f"\n[原始消息] ({len(messages)} 条)")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        think = msg.get("think", None)
        print(f"  [{i}] role={role}")
        if think:
            print(f"       think={repr(think[:100])}...")
        print(f"       content={repr(content[:100])}...")

    # 3. Formatted text turns.
    print(f"\n[模板处理后] ({len(turns)} 个片段)")
    for i, (text, is_loss) in enumerate(turns):
        loss_mark = "✓ loss" if is_loss else "✗ no-loss"
        text_preview = text[:80].replace("\n", "\\n")
        if len(text) > 80:
            text_preview += "..."
        print(f"  [{i}] {loss_mark}: {repr(text_preview)}")

    # 4. Tokenization result.
    print("\n[Tokenize 结果]")
    print(f"  总 token 数: {len(input_ids)}")

    # 5. Loss-mask statistics.
    num_loss = sum(1 for lbl in labels if lbl != IGNORE_INDEX)
    num_no_loss = sum(1 for lbl in labels if lbl == IGNORE_INDEX)
    print(f"  Loss tokens: {num_loss} ({num_loss/len(labels)*100:.1f}%)")
    print(f"  No-loss tokens: {num_no_loss} ({num_no_loss/len(labels)*100:.1f}%)")

    # 6. Decoded prompt and response.
    print("\n[解码文本]")

    # Locate the first trainable token.
    loss_start = 0
    for i, lbl in enumerate(labels):
        if lbl != IGNORE_INDEX:
            loss_start = i
            break

    # Decode the prompt region excluded from loss.
    if loss_start > 0:
        prompt_ids = input_ids[:loss_start]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        print("\n--- PROMPT (不计算 loss) ---")
        print(prompt_text[:500])

    # Decode the response region included in loss.
    response_labels = [lbl for lbl in labels[loss_start:] if lbl != IGNORE_INDEX]

    if response_labels:
        response_text = tokenizer.decode(
            [tid for tid, lbl in zip(input_ids[loss_start:], labels[loss_start:], strict=False) if lbl != IGNORE_INDEX],
            skip_special_tokens=False
        )
        print("\n--- RESPONSE (计算 loss) ---")
        print(response_text[:500])

    # 7. Decode the complete sequence for verification.
    print("\n--- 完整文本 ---")
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(full_text[:800])

    print(f"\n{sep}\n")
