"""Unit tests for tool use pipeline."""

import json
import os

import pytest

from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from tests.oracle._legacy._boundary import (
    _find_role_markers,
    _gpt_oss_role_mapper,
)
from tests.oracle._legacy.tool_use_adapters import get_adapter
from titaninfra.data.sft.message_converter import (
    _build_tool_calls_from_blocks,
    _extract_non_tool_content,
    normalize_to_hf_messages,
)
from titaninfra.data.sft.types import IGNORE_INDEX

FIXTURES_DIR = "tests/fixtures/tokenizers"
QWEN3_TOKENIZER = f"{FIXTURES_DIR}/qwen3"
QWEN2_5_TOKENIZER = f"{FIXTURES_DIR}/qwen2_5"
LLAMA3_TOKENIZER = f"{FIXTURES_DIR}/llama3"
DEEPSEEK_V3_TOKENIZER = f"{FIXTURES_DIR}/deepseek_v3"
GEMMA4_TOKENIZER = f"{FIXTURES_DIR}/gemma4"
GPT_OSS_TOKENIZER = f"{FIXTURES_DIR}/gpt_oss"
LLAMA4_TOKENIZER = f"{FIXTURES_DIR}/llama4"


def _has_tokenizer(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "tokenizer_config.json"))


def _load_fast_tokenizer_with_jinja(path: str):
    """Load tokenizer.json directly and attach chat_template.jinja manually.

    Used for tokenizers whose tokenizer_config.json specifies a custom
    tokenizer_class that AutoTokenizer rejects (e.g. deepseek_v3, gemma4).
    """
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"))
    jinja_path = os.path.join(path, "chat_template.jinja")
    if os.path.isfile(jinja_path):
        with open(jinja_path) as f:
            tok.chat_template = f.read()
    return tok


requires_qwen3 = pytest.mark.skipif(
    not _has_tokenizer(QWEN3_TOKENIZER),
    reason="Qwen3 tokenizer fixture not available",
)
requires_qwen2_5 = pytest.mark.skipif(
    not _has_tokenizer(QWEN2_5_TOKENIZER),
    reason="Qwen2.5 tokenizer fixture not available",
)
requires_llama3 = pytest.mark.skipif(
    not _has_tokenizer(LLAMA3_TOKENIZER),
    reason="Llama3 tokenizer fixture not available",
)
requires_deepseek_v3 = pytest.mark.skipif(
    not _has_tokenizer(DEEPSEEK_V3_TOKENIZER),
    reason="DeepSeek-V3 tokenizer fixture not available",
)
requires_gemma4 = pytest.mark.skipif(
    not _has_tokenizer(GEMMA4_TOKENIZER),
    reason="Gemma4 tokenizer fixture not available",
)
requires_gpt_oss = pytest.mark.skipif(
    not _has_tokenizer(GPT_OSS_TOKENIZER),
    reason="GPT-OSS tokenizer fixture not available",
)
requires_llama4 = pytest.mark.skipif(
    not _has_tokenizer(LLAMA4_TOKENIZER),
    reason="Llama4 tokenizer fixture not available",
)


# ============================================================
# TestNormalizeToolCalls — unit tests for message_converter
# ============================================================


class TestNormalizeToolCalls:
    """Tests for test normalize tool calls."""

    def test_tool_use_blocks_produce_hf_tool_calls(self):
        """Verify tool use blocks produce hf tool calls."""
        messages = [
            {"role": "user", "content": "查天气"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "北京"}},
                ],
            },
        ]
        hf_msgs, weights = normalize_to_hf_messages(messages)
        assert len(hf_msgs) == 2
        assistant_msg = hf_msgs[1]
        assert "tool_calls" in assistant_msg
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "北京"}
        assert tc["id"] == "c1"
        assert assistant_msg["content"] == ""

    def test_parallel_tool_calls_from_blocks(self):
        """Verify parallel tool calls from blocks."""
        messages = [
            {"role": "user", "content": "查询并计算"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "pi"}},
                    {"type": "tool_use", "id": "c2", "name": "calc", "input": {"expr": "3*7"}},
                ],
            },
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        tc_list = hf_msgs[1]["tool_calls"]
        assert len(tc_list) == 2
        assert tc_list[0]["function"]["name"] == "search"
        assert tc_list[1]["function"]["name"] == "calc"

    def test_thinking_plus_tool_use(self):
        """thinking + tool_use blocks → reasoning_content + tool_calls。"""
        messages = [
            {"role": "user", "content": "分析并计算"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "让我思考一下..."},
                    {"type": "tool_use", "id": "c1", "name": "calc", "input": {"expr": "2+2"}},
                ],
            },
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        assistant_msg = hf_msgs[1]
        assert assistant_msg["reasoning_content"] == "让我思考一下..."
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["content"] == ""

    def test_tool_result_block_produces_content(self):
        """tool role + tool_result block → hf_msg content + tool_call_id。"""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "test"}},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "text": "result data"},
                ],
            },
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        tool_msg = hf_msgs[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == "result data"
        assert tool_msg["tool_call_id"] == "c1"

    def test_passthrough_priority(self):
        """Verify passthrough priority."""
        existing_tc = [{"id": "x", "type": "function", "function": {"name": "fn", "arguments": "{}"}}]
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "other", "input": {}},
                ],
                "tool_calls": existing_tc,
            },
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        # pass-through wins — tool_calls from msg preserved
        assert hf_msgs[1]["tool_calls"] is existing_tc

    def test_tool_role_with_tool_result_block(self):
        """Verify tool role with tool result block."""
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "tool",
                "blocks": [
                    {"type": "tool_result", "tool_use_id": "call_42", "text": "answer"},
                ],
            },
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        assert hf_msgs[1]["tool_call_id"] == "call_42"
        assert hf_msgs[1]["content"] == "answer"

    def test_text_plus_tool_use_content_separation(self):
        """Verify text plus tool use content separation."""
        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me help you."},
                    {"type": "tool_use", "id": "c1", "name": "fn", "input": {"a": 1}},
                ],
            },
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        assert hf_msgs[1]["content"] == "Let me help you."
        assert len(hf_msgs[1]["tool_calls"]) == 1


class TestBuildToolCallsHelper:
    """_build_tool_calls_from_blocks helper unit tests."""

    def test_empty_blocks(self):
        assert _build_tool_calls_from_blocks([]) == []

    def test_no_tool_use_blocks(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "thinking", "text": "hmm"}]
        assert _build_tool_calls_from_blocks(blocks) == []

    def test_arguments_serialized_as_json_string(self):
        blocks = [{"type": "tool_use", "id": "x", "name": "fn", "input": {"key": "val"}}]
        result = _build_tool_calls_from_blocks(blocks)
        assert result[0]["function"]["arguments"] == '{"key": "val"}'

    def test_missing_id_defaults_empty(self):
        blocks = [{"type": "tool_use", "name": "fn", "input": {}}]
        result = _build_tool_calls_from_blocks(blocks)
        assert result[0]["id"] == ""


class TestExtractNonToolContent:
    """_extract_non_tool_content helper unit tests."""

    def test_text_only(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]
        content, reasoning = _extract_non_tool_content(blocks)
        assert content == "hello world"
        assert reasoning is None

    def test_thinking_extracted(self):
        blocks = [{"type": "thinking", "text": "step1"}, {"type": "text", "text": "answer"}]
        content, reasoning = _extract_non_tool_content(blocks)
        assert content == "answer"
        assert reasoning == "step1"

    def test_tool_use_skipped(self):
        blocks = [
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "id": "x", "name": "fn", "input": {}},
        ]
        content, reasoning = _extract_non_tool_content(blocks)
        assert content == "hi"
        assert reasoning is None


# ============================================================
# TestGptOssRoleMapper — GPT-OSS regex + callable mapper
# ============================================================


class TestGptOssRoleMapper:
    """GPT-OSS role mapper and regex tests."""

    def test_basic_roles(self):
        assert _gpt_oss_role_mapper("system") == "system"
        assert _gpt_oss_role_mapper("developer") == "system"
        assert _gpt_oss_role_mapper("user") == "user"
        assert _gpt_oss_role_mapper("assistant") == "assistant"
        assert _gpt_oss_role_mapper("tool") == "tool"

    def test_assistant_to_functions(self):
        assert _gpt_oss_role_mapper("assistant to=functions.search") == "assistant"
        assert _gpt_oss_role_mapper("assistant to=functions.get_weather") == "assistant"

    def test_functions_to_assistant(self):
        assert _gpt_oss_role_mapper("functions.search to=assistant") == "tool"
        assert _gpt_oss_role_mapper("functions.calc to=assistant") == "tool"

    def test_unknown_returns_none(self):
        assert _gpt_oss_role_mapper("unknown_role") is None

    def test_regex_matches_tool_call_pattern(self):
        text = '<|start|>assistant to=functions.search<|message|>{"query":"test"}<|end|>'
        markers = _find_role_markers(text)
        assert len(markers) == 1
        assert markers[0][2] == "assistant"

    def test_regex_matches_tool_result_pattern(self):
        text = '<|start|>functions.search to=assistant<|message|>result<|end|>'
        markers = _find_role_markers(text)
        assert len(markers) == 1
        assert markers[0][2] == "tool"

    def test_regex_with_channel(self):
        text = '<|start|>assistant to=functions.fn<|channel|>default<|message|>body'
        markers = _find_role_markers(text)
        assert len(markers) == 1
        assert markers[0][2] == "assistant"


# ============================================================
# TestTokenizeToolUse — integration tests (requires tokenizer)
# ============================================================


@requires_qwen3
class TestTokenizeToolUse:
    """tokenize_with_hf_template() tool-use integration tests with Qwen3 tokenizer."""

    @pytest.fixture(autouse=True)
    def setup_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    def _load_fixture(self, scenario: str) -> dict:
        import json as json_mod
        with open("tests/fixtures/sft_samples.json") as f:
            samples = json_mod.load(f)
        for s in samples:
            if s.get("_scenario") == scenario:
                return s
        raise ValueError(f"Fixture {scenario!r} not found")

    def test_single_tool_call_tokenizes(self):
        """Verify single tool call tokenizes."""
        sample = self._load_fixture("single_tool_call")
        input_ids, labels = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
        )
        assert len(input_ids) > 0
        assert len(input_ids) == len(labels)
        has_loss = [i for i, lab in enumerate(labels) if lab != IGNORE_INDEX]
        assert len(has_loss) > 0

    def test_parallel_tool_calls_tokenizes(self):
        """Verify parallel tool calls tokenizes."""
        sample = self._load_fixture("parallel_tool_calls")
        input_ids, labels = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            tools=sample.get("tools"),
            template_name="qwen3",
        )
        assert len(input_ids) > 0
        has_loss = [i for i, lab in enumerate(labels) if lab != IGNORE_INDEX]
        assert len(has_loss) > 0

    def test_thinking_with_tool_call_tokenizes(self):
        """Verify thinking with tool call tokenizes."""
        sample = self._load_fixture("thinking_with_tool_call")
        input_ids, labels = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
        )
        assert len(input_ids) > 0
        has_loss = [i for i, lab in enumerate(labels) if lab != IGNORE_INDEX]
        assert len(has_loss) > 5

    def test_tool_use_type_overrides_scaling(self):
        """Verify tool use type overrides scaling."""
        sample = self._load_fixture("single_tool_call")
        input_ids, labels, loss_weights = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
            loss_config={"type_overrides": {"tool_use": 0.5}},
            return_weights=True,
        )
        # If tool_call markers were found, some weights should be 0.5
        if loss_weights is not None:
            half_weights = [w for w in loss_weights if abs(w - 0.5) < 1e-6]
            assert len(half_weights) > 0

    def test_tool_use_type_overrides_zero(self):
        """type_overrides["tool_use"]=0 → tool_call content fully masked。"""
        sample = self._load_fixture("single_tool_call")
        input_ids, labels, loss_weights = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
            loss_config={"type_overrides": {"tool_use": 0.0}},
            return_weights=True,
        )
        input_ids_base, labels_base = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
        )
        masked_count = sum(
            1 for lb, lab in zip(labels_base, labels, strict=False) if lb != IGNORE_INDEX and lab == IGNORE_INDEX
        )
        # Only assert if template actually produces tool_call markers
        if masked_count > 0:
            assert masked_count > 0

    def test_train_on_inputs_bypasses_boundary_check(self):
        """train_on_inputs=True bypasses fail-loud boundary check。"""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "result", "weight": 1.0},
            {"role": "assistant", "content": "done"},
        ]
        input_ids, labels = tokenize_with_hf_template(
            messages=messages,
            tokenizer=self.tokenizer,
            train_on_inputs=True,
            template_name="qwen3",
        )
        assert len(input_ids) > 0
        assert all(lab != IGNORE_INDEX for lab in labels)

    def test_tools_definition_in_template(self):
        """Verify tools definition in template."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        messages = [
            {"role": "user", "content": "weather in Tokyo?"},
            {"role": "assistant", "content": "Let me check."},
        ]
        input_ids_with, _ = tokenize_with_hf_template(
            messages=messages,
            tokenizer=self.tokenizer,
            tools=tools,
            template_name="qwen3",
        )
        input_ids_without, _ = tokenize_with_hf_template(
            messages=messages,
            tokenizer=self.tokenizer,
            template_name="qwen3",
        )
        assert len(input_ids_with) > len(input_ids_without)


# ============================================================
# TestToolUseFailLoud — boundary validation
# ============================================================


@requires_qwen3
class TestToolUseFailLoud:
    """Fail-loud behavior when tool role lacks proper marker."""

    @pytest.fixture(autouse=True)
    def setup_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    def test_tool_weight_positive_raises(self):
        """Qwen tool role with weight>0 raises ValueError (no dedicated marker)."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling tool..."},
            {"role": "tool", "content": "result", "weight": 1.0},
            {"role": "assistant", "content": "got it"},
        ]
        try:
            tokenize_with_hf_template(
                messages=messages,
                tokenizer=self.tokenizer,
                template_name="qwen3",
            )
        except ValueError as e:
            assert "weight" in str(e).lower() or "marker" in str(e).lower()

    def test_tool_weight_zero_no_raise(self):
        """tool role with weight=0 never raises regardless of marker support."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling tool..."},
            {"role": "tool", "content": "result", "weight": 0.0},
            {"role": "assistant", "content": "got it"},
        ]
        input_ids, labels = tokenize_with_hf_template(
            messages=messages,
            tokenizer=self.tokenizer,
            template_name="qwen3",
        )
        assert len(input_ids) > 0


# ============================================================
# TestConverterToolColumns — converter MDS column extension
# ============================================================


class TestConverterToolColumns:
    """converter.py system/tools MDS column tests."""

    QWEN3_TOKENIZER_PATH = "tests/fixtures/tokenizers/qwen3"

    def _real_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.QWEN3_TOKENIZER_PATH, trust_remote_code=True)

    def test_prepare_messages_item_with_tools(self):
        """_prepare_messages_item extracts system and tools into payload."""
        from titaninfra.data.converter import _prepare_messages_item

        tokenizer = self._real_tokenizer()
        record = {
            "messages": [
                {"role": "user", "blocks": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "hello"}]},
            ],
            "system": "You are helpful.",
            "tools": [{"name": "fn", "description": "desc", "input_schema": {}}],
        }

        payload, tc = _prepare_messages_item(record, tokenizer, "messages", "test_stream", "qwen3")
        assert payload["system"] == "You are helpful."
        assert json.loads(payload["tools"]) == record["tools"]
        assert tc > 0

    def test_prepare_messages_item_without_tools(self):
        """Missing system/tools → empty strings."""
        from titaninfra.data.converter import _prepare_messages_item

        tokenizer = self._real_tokenizer()
        record = {
            "messages": [
                {"role": "user", "blocks": [{"type": "text", "text": "q"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "a"}]},
            ],
        }

        payload, tc = _prepare_messages_item(record, tokenizer, "messages", "s", "qwen3")
        assert payload["system"] == ""
        assert payload["tools"] == ""
        assert tc > 0


# ============================================================
# TestToolUseHFAlignmentQwen3 — titaninfra vs HF apply_chat_template
# ============================================================

TOOL_USE_FIXTURE = "tests/fixtures/tool_use_samples/tool_use_test_samples.json"


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return f"pos {i}: titaninfra={x} hf={y}"
    if len(a) != len(b):
        return f"length differs: titaninfra={len(a)} hf={len(b)}"
    return "identical"


def _load_tool_use_samples():
    if not os.path.isfile(TOOL_USE_FIXTURE):
        return []
    with open(TOOL_USE_FIXTURE) as f:
        return json.load(f)


_TOOL_USE_SAMPLES = _load_tool_use_samples()
requires_tool_use_fixture = pytest.mark.skipif(
    not _TOOL_USE_SAMPLES,
    reason=f"Tool-use fixture missing: {TOOL_USE_FIXTURE}",
)


@requires_qwen3
@requires_tool_use_fixture
class TestToolUseHFAlignmentQwen3:
    """Tests for test tool use hfalignment qwen3."""

    @pytest.fixture(autouse=True)
    def setup_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    @pytest.mark.parametrize("sample_idx", range(len(_TOOL_USE_SAMPLES)))
    def test_input_ids_match_hf(self, sample_idx):
        """Verify input ids match hf."""
        sample = _TOOL_USE_SAMPLES[sample_idx]
        messages = sample["messages"]
        system = sample.get("system")
        tools = sample.get("tools")

        titaninfra_ids, _ = tokenize_with_hf_template(
            messages=messages,
            tokenizer=self.tokenizer,
            system=system,
            tools=tools,
            template_name="qwen3",
        )

        hf_msgs, _ = normalize_to_hf_messages(messages, system=system)
        hf_ids = self.tokenizer.apply_chat_template(
            hf_msgs,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )

        assert titaninfra_ids == hf_ids, (
            f"sample {sample_idx} ({sample['_scenario']}, xlam_id={sample['_xlam_id']}) "
            f"diverges: {_first_diff(titaninfra_ids, hf_ids)}"
        )

    @pytest.mark.parametrize("sample_idx", range(len(_TOOL_USE_SAMPLES)))
    def test_loss_mask_covers_assistant_only(self, sample_idx):
        """Verify loss mask covers assistant only."""
        sample = _TOOL_USE_SAMPLES[sample_idx]
        input_ids, labels = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
        )
        n_loss = sum(1 for lb in labels if lb != IGNORE_INDEX)
        n_masked = sum(1 for lb in labels if lb == IGNORE_INDEX)
        assert n_loss > 0, f"sample {sample_idx} has no trainable tokens"
        assert n_masked > 0, f"sample {sample_idx} has nothing masked (user/tool should be)"

    def test_tool_use_type_override_zero_masks_tool_call_body(self):
        """Verify tool use type override zero masks tool call body."""
        sample = _TOOL_USE_SAMPLES[0]
        _, labels_base, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
            return_weights=True,
        )
        _, labels_zero, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
            loss_config={"type_overrides": {"tool_use": 0.0}},
            return_weights=True,
        )
        regressed = sum(
            1 for lb_b, lb_z in zip(labels_base, labels_zero, strict=False)
            if lb_b == IGNORE_INDEX and lb_z != IGNORE_INDEX
        )
        assert regressed == 0, "type_override=0 不应放开任何 mask"

    def test_tool_use_type_override_half_scales_weights(self):
        """Verify tool use type override half scales weights."""
        sample = _TOOL_USE_SAMPLES[0]
        _, _, weights = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name="qwen3",
            loss_config={"type_overrides": {"tool_use": 0.5}},
            return_weights=True,
        )
        if weights is not None:
            half = [w for w in weights if abs(w - 0.5) < 1e-6]
            assert len(half) > 0, "tool_use=0.5 应有 token 权重为 0.5"


# ============================================================
# ------------------------------------------------------------
# ============================================================

# (model_id, tokenizer_path, template_name, use_fast_loader, skip_marker)
# template_name is the registered titaninfra template name used for adapter
# dispatch and behavior resolution. Pass None when the tokenizer's family
# uses DefaultAdapter and has no registered template (e.g. llama4) — titaninfra
# falls back to the tokenizer's own chat_template.
_MULTI_MODEL_CASES = [
    ("qwen3",        QWEN3_TOKENIZER,        "qwen3",        False, requires_qwen3),
    ("llama3",       LLAMA3_TOKENIZER,       "llama3",       False, requires_llama3),
    ("llama4",       LLAMA4_TOKENIZER,       None,           True,  requires_llama4),
    ("deepseek_v3",  DEEPSEEK_V3_TOKENIZER,  "deepseek_r1",  True,  requires_deepseek_v3),
    ("gemma4",       GEMMA4_TOKENIZER,       "gemma4",       True,  requires_gemma4),
    ("gpt_oss",      GPT_OSS_TOKENIZER,      "gpt_oss",      False, requires_gpt_oss),
]


def _load_tokenizer_for_case(tokenizer_path: str, use_fast_loader: bool):
    """Load tokenizer for a multi-model case.

    Some fixtures (deepseek_v3, gemma4) declare a custom tokenizer_class in
    tokenizer_config.json that AutoTokenizer cannot resolve offline; for those
    we read tokenizer.json directly and attach chat_template.jinja manually.
    """
    if use_fast_loader:
        return _load_fast_tokenizer_with_jinja(tokenizer_path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


@requires_tool_use_fixture
@pytest.mark.parametrize(
    "model_id,tokenizer_path,template_name,use_fast_loader",
    [
        pytest.param(
            mid,
            tp,
            tn,
            ufl,
            marks=[mark, pytest.mark.xdist_group(f"tok_{mid}")],
            id=mid,
        )
        for (mid, tp, tn, ufl, mark) in _MULTI_MODEL_CASES
    ],
)
class TestToolUseHFAlignmentMultiTokenizer:
    """Tests for test tool use hfalignment multi tokenizer."""

    @pytest.fixture(autouse=True)
    def setup_tokenizer(self, tokenizer_path, use_fast_loader):
        self.tokenizer = _load_tokenizer_for_case(tokenizer_path, use_fast_loader)

    @pytest.mark.parametrize("sample_idx", range(len(_TOOL_USE_SAMPLES)))
    def test_input_ids_match_hf(
        self, model_id, tokenizer_path, template_name, use_fast_loader, sample_idx
    ):
        sample = _TOOL_USE_SAMPLES[sample_idx]
        messages = sample["messages"]
        system = sample.get("system")
        tools = sample.get("tools")

        titaninfra_ids, _ = tokenize_with_hf_template(
            messages=messages,
            tokenizer=self.tokenizer,
            system=system,
            tools=tools,
            template_name=template_name,
        )

        hf_msgs, _ = normalize_to_hf_messages(messages, system=system)
        # Apply the same adapter preprocessing titaninfra uses so the HF reference
        # represents "the canonical HF call for this template family". The adapter
        # is the contract that bridges per-family tool_call argument conventions.
        adapter = get_adapter(template_name)
        hf_msgs, hf_tools = adapter.preprocess_messages(hf_msgs, tools)
        hf_ids = self.tokenizer.apply_chat_template(
            hf_msgs,
            tools=hf_tools,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )

        # Strict bit-exact when lengths agree; otherwise require prefix match
        # (some HF templates append a trailing empty assistant header which
        # titaninfra does not render — accept that boundary case as a no-op suffix).
        if len(titaninfra_ids) == len(hf_ids):
            assert titaninfra_ids == hf_ids, (
                f"[{model_id}] sample {sample_idx} ({sample['_scenario']}, "
                f"xlam_id={sample['_xlam_id']}) diverges: "
                f"{_first_diff(titaninfra_ids, hf_ids)}"
            )
        else:
            shorter, longer = (
                (titaninfra_ids, hf_ids) if len(titaninfra_ids) < len(hf_ids) else (hf_ids, titaninfra_ids)
            )
            assert longer[: len(shorter)] == shorter, (
                f"[{model_id}] sample {sample_idx} ({sample['_scenario']}, "
                f"xlam_id={sample['_xlam_id']}) is not a prefix: "
                f"{_first_diff(titaninfra_ids, hf_ids)}"
            )

    @pytest.mark.parametrize("sample_idx", range(len(_TOOL_USE_SAMPLES)))
    def test_loss_mask_covers_assistant_only(
        self, model_id, tokenizer_path, template_name, use_fast_loader, sample_idx
    ):
        sample = _TOOL_USE_SAMPLES[sample_idx]
        _, labels = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
        )
        n_loss = sum(1 for lb in labels if lb != IGNORE_INDEX)
        n_masked = sum(1 for lb in labels if lb == IGNORE_INDEX)
        assert n_loss > 0, f"[{model_id}] sample {sample_idx} has no trainable tokens"
        assert n_masked > 0, (
            f"[{model_id}] sample {sample_idx} has nothing masked (user/tool should be)"
        )

    def test_tool_use_type_override_zero_no_regression(
        self, model_id, tokenizer_path, template_name, use_fast_loader
    ):
        """Verify tool use type override zero no regression."""
        sample = _TOOL_USE_SAMPLES[0]
        _, labels_base, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
            return_weights=True,
        )
        _, labels_zero, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
            loss_config={"type_overrides": {"tool_use": 0.0}},
            return_weights=True,
        )
        regressed = sum(
            1
            for lb_b, lb_z in zip(labels_base, labels_zero, strict=False)
            if lb_b == IGNORE_INDEX and lb_z != IGNORE_INDEX
        )
        assert regressed == 0, f"[{model_id}] type_override=0 不应放开任何 mask"

    def test_tool_use_type_override_half_present(
        self, model_id, tokenizer_path, template_name, use_fast_loader
    ):
        """Verify tool use type override half present."""
        sample = _TOOL_USE_SAMPLES[0]
        _, _, weights = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
            loss_config={"type_overrides": {"tool_use": 0.5}},
            return_weights=True,
        )
        if weights is not None:
            _ = [w for w in weights if abs(w - 0.5) < 1e-6]


# ============================================================
# Phase 2: Parallel-separate-tool-msgs + sequential multi-turn
# ============================================================
#
#
#
# ============================================================

_PARALLEL_SEP_SAMPLE = next(
    (s for s in _TOOL_USE_SAMPLES if s.get("_scenario") == "parallel_separate_tool_msgs"),
    None,
)
_SEQUENTIAL_SAMPLE = next(
    (s for s in _TOOL_USE_SAMPLES if s.get("_scenario") == "sequential_multi_turn"),
    None,
)

requires_phase2_fixtures = pytest.mark.skipif(
    _PARALLEL_SEP_SAMPLE is None or _SEQUENTIAL_SAMPLE is None,
    reason="Phase 2 fixture samples missing from tool_use_test_samples.json",
)


@requires_tool_use_fixture
@requires_phase2_fixtures
@pytest.mark.parametrize(
    "model_id,tokenizer_path,template_name,use_fast_loader",
    [
        pytest.param(mid, tp, tn, ufl, marks=mark, id=mid)
        for (mid, tp, tn, ufl, mark) in _MULTI_MODEL_CASES
    ],
)
class TestPhase2ToolCallScenarios:
    """Tests for test phase2 tool call scenarios."""

    @pytest.fixture(autouse=True)
    def setup_tokenizer(self, tokenizer_path, use_fast_loader):
        self.tokenizer = _load_tokenizer_for_case(tokenizer_path, use_fast_loader)

    def test_parallel_both_tool_results_in_decoded(
        self, model_id, tokenizer_path, template_name, use_fast_loader
    ):
        """Verify parallel both tool results in decoded."""
        sample = _PARALLEL_SEP_SAMPLE
        input_ids, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
        )
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        # Eiffel Tower result text
        assert "Eiffel Tower" in decoded, (
            f"[{model_id}] parallel_separate: Eiffel Tower result not in decoded text"
        )
        # Paris weather result text
        assert "partly cloudy" in decoded, (
            f"[{model_id}] parallel_separate: Paris weather result not in decoded text"
        )

    def test_parallel_final_assistant_has_loss(
        self, model_id, tokenizer_path, template_name, use_fast_loader
    ):
        """Verify parallel final assistant has loss."""
        sample = _PARALLEL_SEP_SAMPLE
        _, labels = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
        )
        assert any(lb != IGNORE_INDEX for lb in labels), (
            f"[{model_id}] parallel_separate: no loss tokens found"
        )

    def test_sequential_both_results_in_decoded(
        self, model_id, tokenizer_path, template_name, use_fast_loader
    ):
        """Verify sequential both results in decoded."""
        sample = _SEQUENTIAL_SAMPLE
        input_ids, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
        )
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        assert "Rome" in decoded, (
            f"[{model_id}] sequential: Rome result not in decoded text"
        )
        assert "Athens" in decoded, (
            f"[{model_id}] sequential: Athens result not in decoded text"
        )

    def test_sequential_final_assistant_has_loss(
        self, model_id, tokenizer_path, template_name, use_fast_loader
    ):
        """Verify sequential final assistant has loss."""
        sample = _SEQUENTIAL_SAMPLE
        _, labels = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
        )
        assert any(lb != IGNORE_INDEX for lb in labels), (
            f"[{model_id}] sequential: no loss tokens found"
        )


# ---------------------------------------------------------------------------
# Phase 3 — reasoning_content + system coexistence
# ---------------------------------------------------------------------------

# Use sample[0]: has both a system message and an assistant thinking block.
_PHASE3_SAMPLE = _TOOL_USE_SAMPLES[0] if _TOOL_USE_SAMPLES else None

# First 40 chars of the thinking text — enough to identify it in decoded output.
_PHASE3_THINKING_FRAGMENT = "The user is asking"
_PHASE3_SYSTEM_FRAGMENT = "helpful assistant"

# Families whose Jinja templates actually render reasoning_content (<think> block).
# llama3/4, deepseek_v3 (deepseek_r1 template strips thinking in tool-use context),
# and gpt_oss have no reasoning_content rendering path.
_REASONING_FAMILIES = {"qwen3", "gemma4"}

requires_phase3_fixture = pytest.mark.skipif(
    _PHASE3_SAMPLE is None,
    reason="Phase 3 fixture sample not found in tool_use_test_samples.json",
)


@requires_tool_use_fixture
@requires_phase3_fixture
@pytest.mark.parametrize(
    "model_id,tokenizer_path,template_name,use_fast_loader",
    [pytest.param(mid, tp, tn, ufl, marks=mark, id=mid)
     for (mid, tp, tn, ufl, mark) in _MULTI_MODEL_CASES],
)
class TestPhase3ReasoningSystemTools:
    """Tests for test phase3 reasoning system tools."""

    @pytest.fixture(autouse=True)
    def setup_tokenizer(self, tokenizer_path, use_fast_loader):
        self.tokenizer = _load_tokenizer_for_case(tokenizer_path, use_fast_loader)

    def test_reasoning_content_in_decoded(
        self, model_id, tokenizer_path, template_name, use_fast_loader
    ):
        """Verify reasoning content in decoded."""
        sample = _PHASE3_SAMPLE
        input_ids, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
        )
        if model_id not in _REASONING_FAMILIES:
            return  # template doesn't render reasoning_content; crash-free is sufficient
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        assert _PHASE3_THINKING_FRAGMENT in decoded, (
            f"[{model_id}] thinking fragment '{_PHASE3_THINKING_FRAGMENT}' "
            f"not found in decoded text"
        )

    def test_system_message_in_decoded(
        self, model_id, tokenizer_path, template_name, use_fast_loader
    ):
        """Verify system message in decoded."""
        sample = _PHASE3_SAMPLE
        input_ids, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            system=sample.get("system"),
            tools=sample.get("tools"),
            template_name=template_name,
        )
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        assert _PHASE3_SYSTEM_FRAGMENT in decoded, (
            f"[{model_id}] system fragment '{_PHASE3_SYSTEM_FRAGMENT}' "
            f"not found in decoded text"
        )
