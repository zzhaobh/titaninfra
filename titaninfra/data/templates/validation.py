"""Validate structural integrity of SFT message lists.

Errors prevent processing; warnings are advisory and may be reported while
processing continues.
"""

from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Validation outcome with is_valid, errors, and warnings fields."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_messages(
    messages: list[dict],
    template_name: str | None = None,
) -> ValidationResult:
    """Validate required roles and message structure.

    Errors cover empty/non-list messages, malformed records, and missing user or
    assistant roles. template_name is included in applicable error messages.
    """
    errors = []
    warnings = []
    ctx = f" (template={template_name})" if template_name else ""

    if not messages:
        return ValidationResult(False, [f"messages 为空{ctx}"], [])

    if not isinstance(messages, list):
        return ValidationResult(
            False,
            [f"messages 必须是列表，实际类型: {type(messages).__name__}{ctx}"],
            []
        )

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"消息 {i} 不是字典类型")
        elif "role" not in msg:
            errors.append(f"消息 {i} 缺少必需的 'role' 字段")
        elif msg.get("role") not in ["system", "user", "assistant", "tool"]:
            errors.append(f"消息 {i} 的 role 字段值无效: {msg.get('role')}")

    if errors:
        return ValidationResult(False, errors, warnings)

    roles = {m.get("role") for m in messages if isinstance(m, dict)}
    if "user" not in roles:
        errors.append("缺少 user 消息（SFT 数据必须有用户输入）")
    if "assistant" not in roles:
        errors.append("缺少 assistant 消息（SFT 数据必须有模型回复）")

    return ValidationResult(len(errors) == 0, errors, warnings)
