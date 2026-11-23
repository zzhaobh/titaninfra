"""Vendored Jinja renderer — TEST-ONLY oracle baseline.

After Phase 2 of the jinja-purge (native llama3 + qwen2_5 encoders
landed), the production tree no longer contains a Jinja renderer.
This copy lives ONLY inside the oracle harness so
``_jinja_baseline.render_with_jinja`` can still produce a Jinja-
rendered reference for the bit-exact parity tests in
``tests/oracle/test_*_parity.py``.

DO NOT import this from production code. It exists solely as a frozen
reference implementation for regression detection.
"""

import logging

from titaninfra.data.sft.message_converter import normalize_to_hf_messages
from titaninfra.data.sft.types import IGNORE_INDEX

logger = logging.getLogger(__name__)


def _find_subseq(seq: list[int], pattern: list[int], start: int, end: int) -> int:
    plen = len(pattern)
    if plen == 0:
        return -1
    last = min(end, len(seq)) - plen + 1
    for i in range(start, last):
        if seq[i : i + plen] == pattern:
            return i
    return -1


def _find_thinking_ranges(
    input_ids: list[int],
    start: int,
    end: int,
    open_ids: list[int],
    close_ids: list[int],
) -> list[tuple[int, int]]:
    if not open_ids or not close_ids:
        return []
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        open_pos = _find_subseq(input_ids, open_ids, cursor, end)
        if open_pos < 0:
            break
        inner_start = open_pos + len(open_ids)
        close_pos = _find_subseq(input_ids, close_ids, inner_start, end)
        if close_pos < 0:
            break
        ranges.append((inner_start, close_pos))
        cursor = close_pos + len(close_ids)
    return ranges


def _mask_empty_thinking(
    input_ids: list[int],
    labels: list[int],
    weights: list[float],
    roles: list[str],
    boundaries: list[int],
    content_starts: list[int],
    tokenizer,
    thinking_markers: tuple[str, str] | None = None,
) -> None:
    open_str, close_str = thinking_markers or ("<think>", "</think>")
    try:
        open_ids = tokenizer.encode(open_str, add_special_tokens=False)
        close_ids = tokenizer.encode(close_str, add_special_tokens=False)
    except Exception:
        return

    if not open_ids or not close_ids:
        return

    newline_id_set: set[int] = set()
    for ch in ("\n", "\n\n", "\r\n", " ", "\t", "  ", "    "):
        try:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            if len(ids) == 1:
                newline_id_set.add(ids[0])
        except Exception:
            pass

    for i, weight in enumerate(weights):
        if weight <= 0 or roles[i] != "assistant":
            continue
        m_start = content_starts[i]
        m_end = boundaries[i + 1] if i + 1 < len(boundaries) else len(input_ids)
        cursor = m_start
        while cursor < m_end:
            open_pos = _find_subseq(input_ids, open_ids, cursor, m_end)
            if open_pos < 0:
                break
            inner_start = open_pos + len(open_ids)
            close_pos = _find_subseq(input_ids, close_ids, inner_start, m_end)
            if close_pos < 0:
                break
            block_end = close_pos + len(close_ids)
            is_empty = all(input_ids[j] in newline_id_set for j in range(inner_start, close_pos))
            if is_empty:
                for j in range(open_pos, block_end):
                    labels[j] = IGNORE_INDEX
            cursor = block_end


def tokenize_with_hf_template(
    messages: list[dict],
    tokenizer,
    max_length: int | None = None,
    system: str | None = None,
    tools: list[dict] | None = None,
    train_on_inputs: bool = False,
    sft_mask_mode: str = "auto",
    loss_config: dict | None = None,
    debug: bool = False,
    return_weights: bool = False,
    template_name: str | None = None,
    mask_empty_thinking: bool = False,
    thinking_markers: tuple[str, str] | None = None,
):
    """Tokenize with hf template."""
    # Encoder path: models with no Jinja chat_template (e.g. deepseek_v4)
    if template_name == "deepseek_v4":
        from titaninfra.data.sft.encoders import get_encoder
        enc = get_encoder(template_name, tokenizer)
        result = enc.encode(messages, tools=tools, max_length=max_length, train_on_inputs=train_on_inputs)
        return (*result, None) if return_weights else result

    # 0. Resolve template-declared behaviors
    # Best-effort: template_name may legitimately reference a tool_use_adapter
    # name (e.g. "deepseek_v3") that has no ChatTemplate entry. Fall back
    # silently in that case, but surface unknown names via warning so typos
    # don't go undetected.
    if template_name:
        try:
            from titaninfra.data.templates import get_template
            _tmpl = get_template(template_name)
            if thinking_markers is None and _tmpl.thought_words:
                thinking_markers = (_tmpl.thought_words[0].strip(), _tmpl.thought_words[1].strip())
            if not mask_empty_thinking:
                mask_empty_thinking = _tmpl.mask_empty_thinking
        except (KeyError, ImportError):
            pass
        except ValueError:
            logger.warning(
                "ChatTemplate registry has no entry for template_name=%r; "
                "skipping template-declared thinking_markers / mask_empty_thinking. "
                "HF apply_chat_template will still run via the tokenizer's own chat_template.",
                template_name,
            )

    hf_messages, weights = normalize_to_hf_messages(
        messages, system=system
    )

    if not hf_messages:
        return ([], [], None) if return_weights else ([], [])

    # 1.5 Per-template adapter dispatch.
    # qwen3_5 needs JSON-string args parsed to dict (Jinja |items requires Mapping);
    # deepseek_v3/gemma4 have template-specific markers that the default
    # _ROLE_PATTERNS cannot detect. Templates not in the registry get
    # DefaultAdapter (identity preprocess + _compute_message_boundaries) —
    # bit-identical to the previous path.
    from tests.oracle._legacy.tool_use_adapters import get_adapter
    adapter = get_adapter(template_name)
    hf_messages, tools = adapter.preprocess_messages(hf_messages, tools)

    roles = [m.get("role", "user") for m in hf_messages]
    weights = _apply_mask_strategy(weights, roles, sft_mask_mode)

    full_ids, boundaries, content_starts = adapter.compute_boundaries(
        hf_messages, tokenizer, tools=tools
    )

    if not train_on_inputs:
        for i, weight in enumerate(weights):
            role = hf_messages[i].get("role", "user")
            if role == "tool" and weight > 0:
                raise ValueError(
                    f"Message {i} (role='tool') 在多数 chat_template 下没有专属 "
                    f"role marker（如 Qwen 把 tool 渲染成 <|im_start|>user\\n"
                    f"<tool_response>...）：weight={weight} > 0 会让 loss 范围"
                    f"扩张到前一个 assistant 的 tool_call 内容，训练目标会漂移。\n"
                    f"建议：(1) 把 weight 设为 0（默认行为），或 "
                    f"(2) 用 train_on_inputs=True 旁路边界校验。"
                )
            if weight > 0 and boundaries[i] == content_starts[i]:
                raise ValueError(
                    f"Message {i} (role={role!r}) 在 tokenizer 的 chat_template 下没有"
                    f"专属 role marker，titaninfra 无法精确划分 token 边界。\n"
                    f"但 weight={weight} > 0：loss 范围会扩张到相邻消息内容，训练目标会漂移。\n"
                    f"已知场景：Qwen 模板把 role='tool' 渲染成 <|im_start|>user\\n"
                    f"<tool_response>...</tool_response>，没有 <|im_start|>tool marker；"
                    f"此时给 tool 消息设 weight>0 会把前一个 assistant 的 tool_call 内容"
                    f"错位计入 loss。\n"
                    f"建议：(1) 把消息 role 改成模板支持的角色（user/assistant），或 "
                    f"(2) 显式设 weight=0 跳过它，或 (3) 用 train_on_inputs=True 旁路边界校验。"
                )

    if train_on_inputs:
        labels = list(full_ids)
    else:
        labels = [IGNORE_INDEX] * len(full_ids)
        for i, weight in enumerate(weights):
            if weight <= 0:
                continue
            start = content_starts[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(full_ids)
            labels[start:end] = full_ids[start:end]

    if mask_empty_thinking and not train_on_inputs:
        _mask_empty_thinking(
            full_ids, labels, weights, roles, boundaries, content_starts, tokenizer,
            thinking_markers=thinking_markers,
        )

    loss_weights: list[float] | None = None
    if return_weights:
        type_overrides = (loss_config or {}).get("type_overrides", {})
        thinking_scale = type_overrides.get("thinking")

        if thinking_scale is not None:
            open_str, close_str = thinking_markers or ("<think>", "</think>")
            try:
                open_ids = tokenizer.encode(open_str, add_special_tokens=False)
                close_ids = tokenizer.encode(close_str, add_special_tokens=False)
            except Exception:
                open_ids, close_ids = [], []

            if open_ids and close_ids:
                loss_weights = [1.0 if lab != IGNORE_INDEX else 0.0 for lab in labels]
                for i, weight in enumerate(weights):
                    if weight <= 0 or roles[i] != "assistant":
                        continue
                    m_start = content_starts[i]
                    m_end = boundaries[i + 1] if i + 1 < len(boundaries) else len(full_ids)
                    for inner_start, inner_end in _find_thinking_ranges(
                        full_ids, m_start, m_end, open_ids, close_ids
                    ):
                        if thinking_scale == 0.0:
                            for j in range(inner_start, inner_end):
                                labels[j] = IGNORE_INDEX
                                loss_weights[j] = 0.0
                        else:
                            for j in range(inner_start, inner_end):
                                loss_weights[j] *= thinking_scale

                if all(w in (0.0, 1.0) for w in loss_weights):
                    loss_weights = None

        tool_use_scale = type_overrides.get("tool_use")
        if tool_use_scale is not None:
            tool_call_markers = None
            if template_name:
                try:
                    from titaninfra.data.templates import get_template
                    _tc_tmpl = get_template(template_name)
                    if _tc_tmpl.tool_call_words:
                        tool_call_markers = (
                            _tc_tmpl.tool_call_words[0].strip(),
                            _tc_tmpl.tool_call_words[1].strip(),
                        )
                except (KeyError, ImportError):
                    pass

            if tool_call_markers:
                tc_open_str, tc_close_str = tool_call_markers
                try:
                    tc_open_ids = tokenizer.encode(tc_open_str, add_special_tokens=False)
                    tc_close_ids = tokenizer.encode(tc_close_str, add_special_tokens=False)
                except Exception:
                    tc_open_ids, tc_close_ids = [], []

                if tc_open_ids and tc_close_ids:
                    if loss_weights is None:
                        loss_weights = [1.0 if lab != IGNORE_INDEX else 0.0 for lab in labels]
                    for i, weight in enumerate(weights):
                        if weight <= 0 or roles[i] != "assistant":
                            continue
                        m_start = content_starts[i]
                        m_end = boundaries[i + 1] if i + 1 < len(boundaries) else len(full_ids)
                        for inner_start, inner_end in _find_thinking_ranges(
                            full_ids, m_start, m_end, tc_open_ids, tc_close_ids
                        ):
                            if tool_use_scale == 0.0:
                                for j in range(inner_start, inner_end):
                                    labels[j] = IGNORE_INDEX
                                    loss_weights[j] = 0.0
                            else:
                                for j in range(inner_start, inner_end):
                                    loss_weights[j] *= tool_use_scale

                    if all(w in (0.0, 1.0) for w in loss_weights):
                        loss_weights = None

    if max_length is not None:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
        if loss_weights is not None:
            loss_weights = loss_weights[:max_length]

    if debug:
        _print_debug(
            tokenizer, hf_messages, weights, full_ids, labels, boundaries, content_starts
        )

    if return_weights:
        return full_ids, labels, loss_weights
    return full_ids, labels


def _apply_mask_strategy(
    weights: list[float],
    roles: list[str],
    sft_mask_mode: str,
) -> list[float]:
    """Apply mask strategy."""
    if sft_mask_mode == "manual":
        return weights

    new_weights = []
    for w, role in zip(weights, roles, strict=False):
        if w == 0:
            new_weights.append(0.0)
        elif role == "assistant":
            new_weights.append(1.0)
        else:
            new_weights.append(w)

    return new_weights


def _print_debug(
    tokenizer,
    hf_messages: list[dict],
    weights: list[float],
    input_ids: list[int],
    labels: list[int],
    boundaries: list[int],
    content_starts: list[int],
) -> None:
    """Print debug."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("[DEBUG] HF Template Tokenizer (pattern-based boundaries)")
    print(sep)

    print(f"\n[消息] ({len(hf_messages)} 条)")
    for i, (msg, w) in enumerate(zip(hf_messages, weights, strict=False)):
        role = msg.get("role", "?")
        content = msg.get("content", "")[:80]
        token_start = boundaries[i]
        token_end = boundaries[i + 1] if i + 1 < len(boundaries) else len(input_ids)
        loss_start = content_starts[i] if i < len(content_starts) else token_start
        marker_len = loss_start - token_start
        print(
            f"  [{i}] role={role}, weight={w}, "
            f"tokens=[{token_start}:{token_end}] ({token_end - token_start} tokens, "
            f"role-marker={marker_len}, loss=[{loss_start}:{token_end}])"
        )
        print(f"       content={content!r}...")

    num_loss = sum(1 for label in labels if label != IGNORE_INDEX)
    num_total = len(labels)
    print("\n[统计]")
    print(f"  总 token 数: {num_total}")
    print(f"  Loss tokens: {num_loss} ({num_loss / max(num_total, 1) * 100:.1f}%)")
    print(f"  No-loss tokens: {num_total - num_loss}")

    loss_ids = [tid for tid, label in zip(input_ids, labels, strict=False) if label != IGNORE_INDEX]
    if loss_ids:
        print("\n[Loss 部分解码]")
        print(tokenizer.decode(loss_ids, skip_special_tokens=False)[:500])

    print(f"\n{sep}\n")
