"""Unit tests for template decoupling."""

import json
import re
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from titaninfra.data.sft.types import IGNORE_INDEX

TOKENIZERS_DIR = Path(__file__).parent.parent / "fixtures" / "tokenizers"


def _load_tokenizer(name: str):
    """Load tokenizer."""
    base = TOKENIZERS_DIR / name
    if not base.exists():
        return None
    cfg_path = base / "tokenizer_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    chat_tpl_path = base / "chat_template.jinja"
    chat_tpl = chat_tpl_path.read_text() if chat_tpl_path.exists() else None
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        kwargs = {
            k: v
            for k, v in cfg.items()
            if k not in ("tokenizer_class", "auto_map", "chat_template", "processor_class")
        }
        tok = PreTrainedTokenizerFast(tokenizer_file=str(base / "tokenizer.json"), **kwargs)
        if chat_tpl:
            tok.chat_template = chat_tpl
        return tok
    try:
        tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
        if not getattr(tok, "chat_template", None):
            return None
        return tok
    except Exception:
        return None


def _hf_tokenize_with_mask(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Hf tokenize with mask."""
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    assert full_ids[: len(prompt_ids)] == prompt_ids, (
        "HF 模板异常：full_ids 不以 prompt_ids 开头"
    )
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def _require(name: str):
    """Require."""
    tok = _load_tokenizer(name)
    if tok is None:
        pytest.skip(f"tokenizer fixture {name!r} not available locally")
    return tok


# ── R1: Qwen tool-role embedded in user block ───────────────────────────────


@pytest.mark.parametrize("family", ["qwen2_5", "qwen3"])
class TestR1QwenToolRoleEmbedded:
    """Tests for test r1 qwen tool role embedded."""

    def test_no_tool_marker_in_rendered_text(self, family):
        tok = _require(family)
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        assert "<|im_start|>tool" not in text, (
            f"[{family}] 模板行为变化：现在渲染 <|im_start|>tool marker 了"
        )
        assert "<tool_response>" in text and "</tool_response>" in text, (
            f"[{family}] 模板行为变化：tool 不再被 <tool_response> 包裹"
        )

    def test_tool_default_weight_zero_passes(self, family):
        """Verify tool default weight zero passes."""
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "sunny"},  # default weight=0
            {"role": "assistant", "content": "It is sunny."},
        ]
        tok = _require(family)
        ids, labels = tokenize_with_hf_template(
            msgs, tok, max_length=8192, sft_mask_mode="manual",
        )
        assert len(ids) > 0
        assert sum(1 for v in labels if v != IGNORE_INDEX) > 0, (
            f"[{family}] 默认配置下没有任何 loss token"
        )

    def test_tool_explicit_weight_raises(self, family):
        """Verify tool explicit weight raises."""
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "sunny", "weight": 1.0},
            {"role": "assistant", "content": "It is sunny."},
        ]
        tok = _require(family)
        with pytest.raises(ValueError, match="role marker"):
            tokenize_with_hf_template(
                msgs, tok, max_length=8192, sft_mask_mode="manual",
            )

    def test_tool_weight_with_train_on_inputs_bypasses_check(self, family):
        """Verify tool weight with train on inputs bypasses check."""
        msgs = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "sunny", "weight": 1.0},
            {"role": "assistant", "content": "It is sunny."},
        ]
        tok = _require(family)
        ids, labels = tokenize_with_hf_template(
            msgs, tok, max_length=8192,
            sft_mask_mode="manual", train_on_inputs=True,
        )
        assert all(v != IGNORE_INDEX for v in labels), (
            f"[{family}] train_on_inputs=True 但 labels 仍有 IGNORE_INDEX"
        )


# ── R2: Llama 4 maps tool → ipython ─────────────────────────────────────────


def test_r2_llama4_tool_renders_as_ipython():
    """Verify r2 llama4 tool renders as ipython."""
    tok = _require("llama4")
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert "<|header_start|>ipython<|header_end|>" in text, (
        "Llama 4 模板行为变化：tool 不再渲染为 ipython"
    )
    ids, labels = tokenize_with_hf_template(msgs, tok, max_length=8192)
    assert len(ids) > 0
    assert sum(1 for v in labels if v != IGNORE_INDEX) > 0


# ── R3: DeepSeek V3 system in BOS section ───────────────────────────────────


def test_r3_deepseek_system_in_bos_no_loss():
    """Verify r3 deepseek system in bos no loss."""
    tok = _require("deepseek_v3")
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ]
    hf_ids, hf_labels = _hf_tokenize_with_mask(msgs, tok)
    jd_ids, jd_labels = tokenize_with_hf_template(msgs, tok, max_length=4096)
    assert hf_ids == jd_ids, "[deepseek_v3] input_ids 与 HF 不一致"
    for pos in range(len(hf_labels)):
        assert jd_labels[pos] == hf_labels[pos], (
            f"[deepseek_v3] pos {pos}: hf={hf_labels[pos]} vs jd={jd_labels[pos]}"
        )

    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    user_marker_char = text.find("<｜User｜>")
    assert user_marker_char > 0, "rendered text 应包含 <｜User｜>"
    pre_user_text = text[:user_marker_char]
    pre_user_token_count = len(tok.encode(pre_user_text, add_special_tokens=False))
    for pos in range(min(pre_user_token_count, len(jd_labels))):
        assert jd_labels[pos] == IGNORE_INDEX, (
            f"[deepseek_v3] BOS+system 区间 pos {pos} 不应有 loss"
        )


# ── R4: reasoning_content field on Qwen3 ────────────────────────────────────


def test_r4_qwen3_reasoning_content_field_preserved_and_aligned():
    """Verify r4 qwen3 reasoning content field preserved and aligned."""
    tok = _require("qwen3")
    msgs_field = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A", "reasoning_content": "thinking"},
    ]

    text_field = tok.apply_chat_template(
        msgs_field, tokenize=False, add_generation_prompt=False
    )
    assert "<think>" in text_field and "</think>" in text_field
    assert "thinking" in text_field, (
        "Qwen3 模板没有读到 reasoning_content 字段；"
        "可能 normalize_to_hf_messages 没透传该字段，或模板版本不支持"
    )

    hf_field_ids, hf_field_labels = _hf_tokenize_with_mask(msgs_field, tok)
    jd_field_ids, jd_field_labels = tokenize_with_hf_template(
        msgs_field, tok, max_length=4096
    )
    assert hf_field_ids == jd_field_ids, (
        "[qwen3 reasoning_content field] input_ids 不一致"
    )
    for pos in range(len(hf_field_labels)):
        assert jd_field_labels[pos] == hf_field_labels[pos], (
            f"[qwen3 reasoning_content field] pos {pos}: "
            f"hf={hf_field_labels[pos]} vs jd={jd_field_labels[pos]}"
        )


# ── R5: Multi-turn Qwen3 — every assistant turn gets loss ──────────────────


def test_r5_qwen3_multi_turn_all_assistants_have_loss():
    """Verify r5 qwen3 multi turn all assistants have loss."""
    tok = _require("qwen3")
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
    ]
    jd_ids, jd_labels = tokenize_with_hf_template(
        msgs, tok, max_length=4096,
    )

    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    for content in ["A1", "A2", "A3"]:
        char_pos = text.find(content)
        assert char_pos > 0
        prefix_token_count = len(
            tok.encode(text[:char_pos], add_special_tokens=False)
        )
        window = jd_labels[max(0, prefix_token_count - 1): prefix_token_count + 3]
        assert any(v != IGNORE_INDEX for v in window), (
            f"[qwen3 multi-turn] assistant content {content!r} 没有任何 loss token"
        )

    for content in ["Q1", "Q2", "Q3"]:
        char_pos = text.find(content)
        assert char_pos > 0
        prefix_token_count = len(tok.encode(text[:char_pos], add_special_tokens=False))
        target_token = tok.encode(content, add_special_tokens=False)
        for offset in range(len(target_token)):
            pos = prefix_token_count + offset
            if pos < len(jd_labels):
                assert jd_labels[pos] == IGNORE_INDEX, (
                    f"[qwen3 multi-turn] user content {content!r} pos {pos} 不应有 loss"
                )


# ── R6: Gemma 3/4 dialect ───────────────────────────────────────────────────


def test_r6_gemma_dialect_recognized():
    """Verify r6 gemma dialect recognized."""
    tok = _require("gemma4")
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    jd_ids, jd_labels = tokenize_with_hf_template(msgs, tok, max_length=4096)

    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    user_pos = text.find("Hi")
    assert user_pos > 0, "rendered text 应包含 'Hi'"
    user_token_count = len(tok.encode(text[:user_pos], add_special_tokens=False))
    assert jd_labels[user_token_count] == IGNORE_INDEX, (
        f"[gemma4] user content 'Hi' pos {user_token_count} 不应有 loss"
    )

    assistant_pos = text.find("Hello!")
    assert assistant_pos > 0, "rendered text 应包含 'Hello!'"
    assistant_token_count = len(tok.encode(text[:assistant_pos], add_special_tokens=False))
    window = jd_labels[max(0, assistant_token_count - 1): assistant_token_count + 3]
    assert any(v != IGNORE_INDEX for v in window), (
        "[gemma4] assistant content 'Hello!' 没有任何 loss token"
    )

    first_loss_pos = next(
        (i for i, v in enumerate(jd_labels) if v != IGNORE_INDEX), -1
    )
    assert first_loss_pos > 0, "[gemma4] 没有任何 loss token"
    for pos in range(first_loss_pos):
        assert jd_labels[pos] == IGNORE_INDEX, (
            f"[gemma4] OOB pos {pos} 不应有 loss"
        )


# ── R6b: Gemma4 comprehensive bit-exact loss mask ────────────────────────────
#
# Gemma4 violates the HF prompt-prefix invariant (add_generation_prompt injects
# <|channel>thought\n<channel|>), so standard bit-exact comparison is impossible.
#
# Strategy: render text with add_generation_prompt=False, scan <|turn>model\n
# blocks, find the last <|tool_response>...<tool_response|> pair per block.
# If found → loss = (tr_close.end, turn_end); else → loss = (body_start, turn_end).
# Map char ranges to token positions via offset_mapping.
#
# Reference scanner independently mirrors Gemma4Adapter.compute_boundaries.

_G4_TURN_MODEL = re.compile(r"<\|turn>model\n?")
_G4_TURN_ANY   = re.compile(r"<\|turn>")
_G4_TR_OPEN    = re.compile(r"<\|tool_response>")
_G4_TR_CLOSE   = re.compile(r"<tool_response\|>")

_WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

_SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Web search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}]


def _gemma4_ref_labels(
    msgs: list,
    tok,
    tools: list | None = None,
) -> tuple[list, list]:
    """Build (input_ids, labels) by scanning rendered text — no titaninfra code used.

    Per <|turn>model\\n block: find last <|tool_response>...<tool_response|> pair.
    If found  → loss = chars after <tool_response|> to turn_end.
    If absent → loss = full block body (body_start → turn_end).
    """
    kwargs: dict = {"tokenize": False, "add_generation_prompt": False}
    if tools is not None:
        kwargs["tools"] = tools
    text = tok.apply_chat_template(msgs, **kwargs)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    full_ids = enc["input_ids"]
    offsets  = enc["offset_mapping"]

    loss_ranges: list[tuple[int, int]] = []
    for m in _G4_TURN_MODEL.finditer(text):
        body_start = m.end()
        nxt = _G4_TURN_ANY.search(text, body_start)
        turn_end = nxt.start() if nxt else len(text)

        last_open = last_close = None
        for mo in _G4_TR_OPEN.finditer(text, body_start, turn_end):
            last_open = mo
        if last_open is not None:
            for mc in _G4_TR_CLOSE.finditer(text, last_open.end(), turn_end):
                last_close = mc

        if last_open is not None and last_close is not None:
            loss_ranges.append((last_close.end(), turn_end))
        else:
            loss_ranges.append((body_start, turn_end))

    ref_labels = [
        tid if (cs < ce and any(cs >= lo and ce <= hi for lo, hi in loss_ranges))
        else IGNORE_INDEX
        for tid, (cs, ce) in zip(full_ids, offsets, strict=False)
    ]
    return full_ids, ref_labels


def _assert_gemma4_bit_exact(
    msgs: list,
    tok,
    *,
    tools: list | None = None,
    label: str = "",
) -> None:
    """Run tokenize_with_hf_template and assert bit-exact match with reference labels."""
    tag = label or str([m["role"] for m in msgs])
    jd_ids, jd_labels = tokenize_with_hf_template(
        msgs, tok,
        max_length=4096,
        template_name="gemma4",
        tools=tools,
    )
    ref_ids, ref_labels = _gemma4_ref_labels(msgs, tok, tools=tools)

    assert ref_ids == jd_ids, (
        f"[gemma4/{tag}] input_ids mismatch: ref_len={len(ref_ids)}, jd_len={len(jd_ids)}"
    )
    assert len(ref_labels) == len(jd_labels), (
        f"[gemma4/{tag}] labels length mismatch"
    )
    for pos, (r, j) in enumerate(zip(ref_labels, jd_labels, strict=False)):
        assert r == j, (
            f"[gemma4/{tag}] pos={pos}: ref={r}, jd={j}, token_id={jd_ids[pos]!r}"
        )
    assert any(v != IGNORE_INDEX for v in jd_labels), (
        f"[gemma4/{tag}] no loss tokens — mask is all IGNORE"
    )


class TestGemma4BitExact:
    """Tests for test gemma4 bit exact."""

    @pytest.fixture(scope="class")
    def tok(self):
        return _require("gemma4")

    # ── no-tool scenarios ─────────────────────────────────────────────────────

    def test_01_simple_user_assistant(self, tok):
        """Verify 01 simple user assistant."""
        _assert_gemma4_bit_exact(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            tok,
            label="01_simple",
        )

    def test_02_multi_turn_no_tool(self, tok):
        """Verify 02 multi turn no tool."""
        _assert_gemma4_bit_exact(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
            tok,
            label="02_multi_turn",
        )

    def test_03_system_user_assistant(self, tok):
        """Verify 03 system user assistant."""
        _assert_gemma4_bit_exact(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Sure!"},
            ],
            tok,
            label="03_system",
        )

    # ── tool-use scenarios ────────────────────────────────────────────────────

    def test_04_basic_tool_call(self, tok):
        """user→assistant(tool_call)→tool→assistant(continuation)。

        continue_same_model_turn=True: continuation text appears in the SAME
        <|turn>model block as the tool_call + tool_response.
        loss = only the text after the last <tool_response|>.
        """
        msgs = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
            },
            {"role": "tool", "content": "Sunny, 22°C", "tool_call_id": "c1", "name": "get_weather"},
            {"role": "assistant", "content": "It's sunny in Paris, 22°C."},
        ]
        _assert_gemma4_bit_exact(msgs, tok, tools=_WEATHER_TOOL, label="04_basic_tool")

    def test_05_system_tools_decl_tool_call(self, tok):
        """Verify 05 system tools decl tool call."""
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}],
            },
            {"role": "tool", "content": "Cloudy, 18°C", "tool_call_id": "c1", "name": "get_weather"},
            {"role": "assistant", "content": "Tokyo is cloudy and 18°C."},
        ]
        _assert_gemma4_bit_exact(msgs, tok, tools=_WEATHER_TOOL, label="05_system_tools")

    def test_06_parallel_tool_calls(self, tok):
        """Verify 06 parallel tool calls."""
        msgs = [
            {"role": "user", "content": "Tell me about cats and dogs."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "search", "arguments": {"query": "cats"}}},
                    {"id": "c2", "function": {"name": "search", "arguments": {"query": "dogs"}}},
                ],
            },
            {"role": "tool", "content": "Cats are independent.", "tool_call_id": "c1", "name": "search"},
            {"role": "tool", "content": "Dogs are loyal.", "tool_call_id": "c2", "name": "search"},
            {"role": "assistant", "content": "Cats are independent while dogs are loyal."},
        ]
        _assert_gemma4_bit_exact(msgs, tok, tools=_SEARCH_TOOL, label="06_parallel_tools")

    def test_07_multi_turn_then_tool(self, tok):
        """Verify 07 multi turn then tool."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What's the weather in Berlin?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}],
            },
            {"role": "tool", "content": "Rainy, 12°C", "tool_call_id": "c1", "name": "get_weather"},
            {"role": "assistant", "content": "Berlin is rainy and 12°C."},
        ]
        _assert_gemma4_bit_exact(msgs, tok, tools=_WEATHER_TOOL, label="07_multi_then_tool")

    def test_08_thinking_with_tool_calls(self, tok):
        """Verify 08 thinking with tool calls."""
        msgs = [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need to check the weather API.",
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "London"}}}],
            },
            {"role": "tool", "content": "Overcast, 15°C", "tool_call_id": "c1", "name": "get_weather"},
            {"role": "assistant", "content": "London is overcast and 15°C."},
        ]
        _assert_gemma4_bit_exact(msgs, tok, tools=_WEATHER_TOOL, label="08_thinking_tool")

    def test_09_full_combination(self, tok):
        """system + tools + reasoning_content + 2 parallel tool_calls + continuation。"""
        msgs = [
            {"role": "system", "content": "You are a weather expert."},
            {"role": "user", "content": "Compare Paris and London weather."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I'll check both cities.",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                    {"id": "c2", "function": {"name": "get_weather", "arguments": {"city": "London"}}},
                ],
            },
            {"role": "tool", "content": "Sunny, 22°C", "tool_call_id": "c1", "name": "get_weather"},
            {"role": "tool", "content": "Rainy, 15°C", "tool_call_id": "c2", "name": "get_weather"},
            {"role": "assistant", "content": "Paris is sunny at 22°C, London is rainy at 15°C."},
        ]
        _assert_gemma4_bit_exact(msgs, tok, tools=_WEATHER_TOOL, label="09_full_combo")


# ── GPT_OSS bit-exact tests ──────────────────────────────────────────────────
#
# Template facts:
#   - analysis channel (<|channel|>analysis) is always IGNORE_INDEX
#   - final channel (<|channel|>final) content gets loss
#   - last turn uses <|return|> (training EOS); intermediate turns use <|end|>
#   - only tool_calls[0] rendered — no parallel tool call support in template
#   - thinking is STATIC: analysis channel emitted only when "thinking" field present

_GPTOS_FINAL_MSG_RE = re.compile(r"<\|start\|>assistant<\|channel\|>final<\|message\|>")
_GPTOS_BLOCK_END_RE = re.compile(r"<\|end\|>|<\|return\|>")


def _gpt_oss_ref_labels(msgs, tok, tools=None):
    """Reference loss mask for GPT_OSS test cases.

    These tests pass ``thinking`` as a top-level dict field, but titaninfra's
    ``normalize_to_hf_messages`` only extracts thinking from
    ``ThinkingBlock`` entries inside ``blocks`` (see
    ``titaninfra/data/sft/normalize.py::_extract_thinking``). Consequently
    the encoder emits NO analysis channel for these messages. Jinja, on
    the other hand, always renders an analysis block when ``thinking`` is
    set. To compare apples to apples, we strip jinja's analysis blocks
    and then label each ``<|start|>...<|message|>BODY<closer>`` block:

      * header contains ``final`` / ``commentary`` → BODY+closer get loss
      * header contains ``analysis`` (none here after the strip)        → IGNORE
      * everything else (user / system / developer / tool)              → IGNORE

    This is structural so the channel-tag content-type slot
    (``commentary <|constrain|>json``) doesn't trip a flat regex.
    """
    kwargs: dict = {"tokenize": False, "add_generation_prompt": False}
    if tools is not None:
        kwargs["tools"] = tools
    text = tok.apply_chat_template(msgs, **kwargs)

    # 1) Encoder doesn't read top-level ``thinking`` field → no analysis.
    #    Strip jinja's analysis emission so input_ids match jd.
    text = re.sub(
        r"<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>",
        "", text, flags=re.DOTALL,
    )

    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # 2) Per-block structural walk.
    #    Loss classification: header must mention `final` or `commentary` AND
    #    be an *assistant*-emitted block. Tool-result blocks
    #    (`<|start|>functions.X to=assistant<|channel|>commentary<|message|>`)
    #    also contain "commentary" in the header but are framework-injected
    #    (class A tokens) — the encoder masks them, so the reference must too.
    loss_ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"<\|start\|>", text):
        block_start = m.start()
        msg_m = re.search(r"<\|message\|>", text[block_start:])
        if not msg_m:
            continue
        header = text[block_start : block_start + msg_m.start()]
        body_start = block_start + msg_m.end()
        closer_m = re.search(r"<\|end\|>|<\|return\|>|<\|call\|>", text[body_start:])
        if not closer_m:
            continue
        body_end = body_start + closer_m.end()
        is_tool_result = bool(
            re.match(r"<\|start\|>functions\.\w+\s+to=assistant", header)
        )
        if not is_tool_result and ("final" in header or "commentary" in header):
            loss_ranges.append((body_start, body_end))

    ref_labels = [
        tid if any(cs >= lo and ce <= hi for lo, hi in loss_ranges) else IGNORE_INDEX
        for tid, (cs, ce) in zip(full_ids, offsets, strict=False)
    ]
    return full_ids, ref_labels


def _assert_gpt_oss_bit_exact(msgs, tok, *, tools=None, label=""):
    tag = label or str([m["role"] for m in msgs])
    jd_ids, jd_labels = tokenize_with_hf_template(
        msgs, tok, max_length=4096, template_name="gpt_oss", tools=tools,
    )
    ref_ids, ref_labels = _gpt_oss_ref_labels(msgs, tok, tools=tools)
    assert ref_ids == jd_ids, f"[gpt_oss/{tag}] input_ids mismatch"
    assert len(ref_labels) == len(jd_labels), f"[gpt_oss/{tag}] labels length mismatch"
    for pos, (r, j) in enumerate(zip(ref_labels, jd_labels, strict=False)):
        assert r == j, f"[gpt_oss/{tag}] pos={pos}: ref={r}, jd={j}"
    assert any(v != IGNORE_INDEX for v in jd_labels), f"[gpt_oss/{tag}] no loss tokens"


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "fixtures" / "tokenizers" / "gpt_oss").exists(),
    reason="gpt_oss tokenizer fixture not found; run tests/fixtures/tokenizers/download_all.py",
)
class TestGptOssBitExact:
    """GPT_OSS token-exact loss mask: 6 scenarios (no-tool, tool-call, thinking).

    Template: analysis channel=IGNORE; final channel=loss; <|return|> is training EOS.
    Thinking is STATIC: analysis channel only emitted when 'thinking' field present.
    No parallel tool call support — template only renders tool_calls[0].
    """

    @pytest.fixture(scope="class")
    def tok(self):
        return _require("gpt_oss")

    def test_01_simple_user_assistant(self, tok):
        _assert_gpt_oss_bit_exact(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            tok,
            label="01_simple",
        )

    def test_02_multi_turn_no_tool(self, tok):
        _assert_gpt_oss_bit_exact(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
            tok,
            label="02_multi_turn",
        )

    def test_03_basic_tool_call(self, tok):
        msgs = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
            },
            {"role": "tool", "content": "Sunny, 22°C", "tool_call_id": "c1", "name": "get_weather"},
            {"role": "assistant", "content": "It's sunny in Paris, 22°C."},
        ]
        _assert_gpt_oss_bit_exact(msgs, tok, tools=_WEATHER_TOOL, label="03_basic_tool")

    def test_04_system_tools_decl_tool_call(self, tok):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}],
            },
            {"role": "tool", "content": "Cloudy, 18°C", "tool_call_id": "c1", "name": "get_weather"},
            {"role": "assistant", "content": "Tokyo is cloudy and 18°C."},
        ]
        _assert_gpt_oss_bit_exact(msgs, tok, tools=_WEATHER_TOOL, label="04_system_tools")

    def test_05_thinking_final_turn(self, tok):
        """Static thinking: analysis channel rendered (IGNORE), final channel gets loss."""
        msgs = [
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "Gravity attracts masses.", "thinking": "Let me think about physics."},
        ]
        _assert_gpt_oss_bit_exact(msgs, tok, label="05_thinking_static")

    def test_06_multi_turn_then_tool(self, tok):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What's the weather in Berlin?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}],
            },
            {"role": "tool", "content": "Rainy, 12°C", "tool_call_id": "c1", "name": "get_weather"},
            {"role": "assistant", "content": "Berlin is rainy and 12°C."},
        ]
        _assert_gpt_oss_bit_exact(msgs, tok, tools=_WEATHER_TOOL, label="06_multi_turn_tool")


# ── P0.1: Per-family thinking-mode characterization ────────────────────────
#
# Locks in the observed behavior of each family's chat_template / encoder
# regarding `reasoning_content` (HF chat-template field). Three modes exist:
#
#   STATIC_ON  — template always emits the thinking marker, regardless of
#                whether `reasoning_content` is present (qwen3).
#   STATIC_OFF — template never emits the thinking marker via the standard
#                `reasoning_content` field; either drops it silently
#                (qwen2_5/deepseek_v3/gemma4/gpt_oss-jinja) or requires a
#                different invocation path (gpt_oss-analysis-channel).
#   DYNAMIC    — encoder emits <think>...</think> when a ThinkingBlock is
#                provided, else falls back to single </think> marker
#                (deepseek_v4 custom encoder; no jinja template exists).
#
# Why this matters:
#   - SFT loss-mask correctness depends on which tokens are emitted; if a
#     future template change starts emitting <think></think> for a previously
#     STATIC_OFF family, the existing loss-mask boundaries can silently miss
#     a region.
#   - DSV4 is the only family with a custom (non-jinja) encoder; its
#     dynamic behavior must be locked in to catch accidental drift.

_THINKING_MODE_EXPECTED = {
    # family: (with_reasoning_renders_thinking_section, without_reasoning_still_emits_marker)
    "qwen3":       (True,  True),   # STATIC_ON: always emits <think></think>
    "qwen2_5":     (False, False),  # STATIC_OFF: reasoning_content silently dropped
    "deepseek_v3": (False, False),  # STATIC_OFF: reasoning_content silently dropped
    "gemma4":      (False, False),  # STATIC_OFF
    "gpt_oss":     (False, False),  # STATIC_OFF via apply_chat_template
}

_THINKING_MARKERS = {
    "qwen3":       "<think>",
    "qwen2_5":     "<think>",
    "deepseek_v3": "<think>",
    "gemma4":      "<|channel>thought",
    "gpt_oss":     "<|channel|>analysis",
}


@pytest.mark.parametrize("family", list(_THINKING_MODE_EXPECTED.keys()))
def test_thinking_mode_per_family_static_behavior(family):
    """Lock per-family thinking-mode behavior for jinja-rendered families.

    Compares two scenarios:
      A. Assistant message has `reasoning_content` field.
      B. Same conversation but without `reasoning_content`.

    Asserts:
      - (with_renders): scenario A and B differ iff reasoning_content was rendered.
      - (without_marker): the thinking marker appears in B iff the template
        unconditionally emits one.

    Failing this test means upstream chat_template semantics changed; revisit
    the corresponding SFT loss-mask boundary logic before updating expectations.
    """
    tok = _require(family)
    msgs_with = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "thinking step", "content": "Hello!"},
    ]
    msgs_without = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    text_with = tok.apply_chat_template(msgs_with, tokenize=False, add_generation_prompt=False)
    text_without = tok.apply_chat_template(msgs_without, tokenize=False, add_generation_prompt=False)

    with_renders_expected, without_marker_expected = _THINKING_MODE_EXPECTED[family]
    marker = _THINKING_MARKERS[family]

    differs = text_with != text_without
    assert differs == with_renders_expected, (
        f"[{family}] reasoning_content rendering behavior changed: "
        f"differs={differs}, expected={with_renders_expected}.\n"
        f"with[-200:]={text_with[-200:]!r}\nwithout[-200:]={text_without[-200:]!r}"
    )

    has_marker = marker in text_without
    assert has_marker == without_marker_expected, (
        f"[{family}] without-reasoning marker presence changed: "
        f"has_marker={has_marker}, expected={without_marker_expected}.\n"
        f"without[-200:]={text_without[-200:]!r}"
    )


def _load_tokenizer_no_template(name: str):
    """Load tokenizer fixture without requiring chat_template (for DSV4)."""
    base = TOKENIZERS_DIR / name
    if not base.exists():
        return None
    try:
        return AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    except Exception:
        return None


def test_thinking_mode_deepseek_v4_dynamic_encoder():
    """Lock DSV4 encoder DYNAMIC thinking behavior.

    DSV4 has no jinja template; the custom DeepSeekV4Encoder emits:
      - ThinkingBlock present → [<think>, content_ids..., </think>]
      - ThinkingBlock absent  → [</think>] only (skip-thinking marker)

    This contract is consumed by SFT loss-mask logic; any change breaks
    label alignment.
    """
    from titaninfra.data.schema import Message, TextBlock, ThinkingBlock
    from titaninfra.data.sft.encoders.deepseek_v4 import DeepSeekV4Encoder

    tok = _load_tokenizer_no_template("deepseek_v4")
    if tok is None:
        pytest.skip("deepseek_v4 tokenizer fixture not available locally")
    enc = DeepSeekV4Encoder(tok)

    msgs_with = [
        Message(role="user", blocks=[TextBlock(text="Hi")]),
        Message(role="assistant", blocks=[ThinkingBlock(text="thinking"), TextBlock(text="Hi!")]),
    ]
    msgs_without = [
        Message(role="user", blocks=[TextBlock(text="Hi")]),
        Message(role="assistant", blocks=[TextBlock(text="Hi!")]),
    ]

    ids_with, _ = enc.encode(msgs_with)
    ids_without, _ = enc.encode(msgs_without)

    ai_with = ids_with.index(enc._asst)
    ai_without = ids_without.index(enc._asst)

    # With ThinkingBlock: asst, <think>, ..., </think>, content...
    assert ids_with[ai_with + 1] == enc._think_s, (
        f"DSV4: expected <think> right after <｜Assistant｜> when ThinkingBlock present, "
        f"got token id {ids_with[ai_with + 1]}"
    )
    assert enc._think_e in ids_with[ai_with:], (
        "DSV4: expected </think> closing marker when ThinkingBlock present"
    )

    # Without ThinkingBlock: asst, </think>, content...  (no <think> opening)
    assert ids_without[ai_without + 1] == enc._think_e, (
        f"DSV4: expected </think> right after <｜Assistant｜> when ThinkingBlock absent, "
        f"got token id {ids_without[ai_without + 1]}"
    )
    assert enc._think_s not in ids_without, (
        "DSV4: <think> opening marker must NOT appear when ThinkingBlock absent "
        "(skip-thinking single-marker contract)"
    )


#
#
#


_FIXTURE_FAMILIES = ["qwen2_5", "qwen3", "llama4", "deepseek_v3", "gemma4", "gpt_oss"]


def _read_jinja_template(family: str) -> str:
    """Read jinja template."""
    base = TOKENIZERS_DIR / family
    if not base.exists():
        return ""
    jinja_path = base / "chat_template.jinja"
    if jinja_path.exists():
        return jinja_path.read_text()
    cfg_path = base / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        return cfg.get("chat_template", "") or ""
    return ""


def test_canary_no_template_has_generation_tag_yet():
    """Verify canary no template has generation tag yet."""
    families_with_tag = []
    for family in _FIXTURE_FAMILIES:
        tpl = _read_jinja_template(family)
        if tpl and "{% generation %}" in tpl:
            families_with_tag.append(family)
        elif tpl and "{%- generation" in tpl:
            families_with_tag.append(family)
    assert not families_with_tag, (
        f"以下 fixture 模板已引入 {{% generation %}} 块：{families_with_tag}\n"
        f"该信号意味着上游已采用 HF transformers 4.45+ 的 canonical assistant-mask "
        f"机制。请按测试文档的 4 步迁移检查清单操作。"
    )


@pytest.mark.parametrize("family", _FIXTURE_FAMILIES)
def test_canary_when_generation_tag_present_assistant_mask_aligns(family):
    """Verify canary when generation tag present assistant mask aligns."""
    tpl = _read_jinja_template(family)
    if not tpl or ("{% generation %}" not in tpl and "{%- generation" not in tpl):
        pytest.skip(f"{family!r} 模板没有 {{% generation %}} 标签")

    tok = _require(family)
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = tok.apply_chat_template(
        msgs,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        add_generation_prompt=False,
    )
    official_mask = result["assistant_masks"]
    official_loss_positions = {i for i, v in enumerate(official_mask) if v == 1}

    _, jd_labels = tokenize_with_hf_template(msgs, tok, max_length=4096)
    jd_loss_positions = {i for i, v in enumerate(jd_labels) if v != IGNORE_INDEX}

    diff = official_loss_positions ^ jd_loss_positions
    assert not diff, (
        f"[{family}] titaninfra 边界与官方 assistant_masks 不一致，"
        f"differing positions: {sorted(diff)[:20]}. "
        f"迁移到 return_assistant_tokens_mask 时需要修复 _compute_message_boundaries"
    )


# ── OOB: BOS/EOT/system-prefix tokens have no loss ─────────────────────────


@pytest.mark.parametrize("family", ["qwen2_5", "qwen3", "llama4", "deepseek_v3"])
def test_oob_pre_first_marker_tokens_no_loss(family):
    """Verify oob pre first marker tokens no loss."""
    tok = _require(family)
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    hf_ids, hf_labels = _hf_tokenize_with_mask(msgs, tok)
    jd_ids, jd_labels = tokenize_with_hf_template(msgs, tok, max_length=4096)
    assert hf_ids == jd_ids
    first_loss_pos = next(
        (i for i, v in enumerate(jd_labels) if v != IGNORE_INDEX), -1
    )
    assert first_loss_pos > 0, f"[{family}] 没有任何 loss token"
    for pos in range(first_loss_pos):
        assert jd_labels[pos] == IGNORE_INDEX, (
            f"[{family}] OOB pos {pos} 有 loss，预期前缀全部 -100"
        )
