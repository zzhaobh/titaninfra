"""
Tests for titaninfra.data.sft.message_converter

Covers:
- normalize_content(): string passthrough, content blocks (text, thinking, tool_use, image)
- normalize_to_hf_messages(): system injection, weight extraction, role defaults,
  tool message passthrough, edge cases (empty, mixed)
"""


from titaninfra.data.sft.message_converter import normalize_content, normalize_to_hf_messages

# ---------------------------------------------------------------------------
# normalize_content tests
# ---------------------------------------------------------------------------


class TestNormalizeContent:
    """Tests for normalize_content()."""

    def test_string_passthrough(self):
        """String content returned unchanged."""
        assert normalize_content("hello world") == "hello world"

    def test_empty_string(self):
        """Empty string returned unchanged."""
        assert normalize_content("") == ""

    def test_single_text_block(self):
        """Single text block extracted correctly."""
        blocks = [{"type": "text", "text": "Hello"}]
        assert normalize_content(blocks) == "Hello"

    def test_multiple_text_blocks(self):
        """Multiple text blocks concatenated."""
        blocks = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "World"},
        ]
        assert normalize_content(blocks) == "Hello World"

    def test_thinking_block(self):
        """Thinking block wrapped in <think> tags."""
        blocks = [{"type": "thinking", "text": "Let me think..."}]
        result = normalize_content(blocks)
        assert "<think>" in result
        assert "Let me think..." in result
        assert "</think>" in result

    def test_thinking_block_empty(self):
        """Empty thinking block skipped."""
        blocks = [{"type": "thinking", "text": ""}]
        result = normalize_content(blocks)
        assert "<think>" not in result

    def test_tool_use_block(self):
        """Tool use block formatted as JSON."""
        blocks = [
            {
                "type": "tool_use",
                "name": "get_weather",
                "input": {"city": "Beijing"},
            }
        ]
        result = normalize_content(blocks)
        assert "get_weather" in result
        assert "Beijing" in result

    def test_tool_use_block_with_id(self):
        """Tool use block includes id when present."""
        blocks = [
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "search",
                "input": {"q": "test"},
            }
        ]
        result = normalize_content(blocks)
        assert "call_123" in result

    def test_image_block_skipped(self):
        """Image block silently skipped in text mode."""
        blocks = [
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
            {"type": "text", "text": "description"},
        ]
        result = normalize_content(blocks)
        assert result == "description"
        assert "base64" not in result

    def test_unknown_block_type_skipped(self):
        """Unknown block type skipped with warning."""
        blocks = [
            {"type": "audio", "data": "..."},
            {"type": "text", "text": "hello"},
        ]
        result = normalize_content(blocks)
        assert result == "hello"

    def test_mixed_blocks(self):
        """Mixed text + thinking + tool_use blocks concatenated correctly."""
        blocks = [
            {"type": "thinking", "text": "analyzing"},
            {"type": "text", "text": "The answer is 42."},
            {"type": "tool_use", "name": "calc", "input": {"expr": "6*7"}},
        ]
        result = normalize_content(blocks)
        assert "<think>" in result
        assert "The answer is 42." in result
        assert "calc" in result

    def test_non_dict_block_converted_to_str(self):
        """Non-dict items in list converted to str."""
        blocks = ["plain text", {"type": "text", "text": " more"}]
        result = normalize_content(blocks)
        assert "plain text" in result
        assert " more" in result

    def test_non_list_non_string_content(self):
        """Non-list, non-string content converted via str()."""
        result = normalize_content(42)
        assert result == "42"

    def test_text_block_missing_text_key(self):
        """Text block without 'text' key returns empty string."""
        blocks = [{"type": "text"}]
        assert normalize_content(blocks) == ""


# ---------------------------------------------------------------------------
# normalize_to_hf_messages tests
# ---------------------------------------------------------------------------


class TestNormalizeToHfMessages:
    """Tests for normalize_to_hf_messages()."""

    def test_simple_conversation(self):
        """Basic user + assistant conversation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        hf_msgs, weights = normalize_to_hf_messages(messages)

        assert len(hf_msgs) == 2
        assert hf_msgs[0] == {"role": "user", "content": "Hello"}
        assert hf_msgs[1] == {"role": "assistant", "content": "Hi there!"}

    def test_default_weights(self):
        """Default weights: user=0, assistant=1."""
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        _, weights = normalize_to_hf_messages(messages)
        assert weights == [0.0, 1.0]

    def test_explicit_weights(self):
        """Explicit weight overrides defaults."""
        messages = [
            {"role": "user", "content": "Q", "weight": 1},
            {"role": "assistant", "content": "A", "weight": 0},
        ]
        _, weights = normalize_to_hf_messages(messages)
        assert weights == [1.0, 0.0]

    def test_system_injection(self):
        """System prompt injected as first message with weight=0."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        hf_msgs, weights = normalize_to_hf_messages(messages, system="Be helpful")

        assert len(hf_msgs) == 3
        assert hf_msgs[0] == {"role": "system", "content": "Be helpful"}
        assert weights[0] == 0.0

    def test_system_none_not_injected(self):
        """system=None does not inject a system message."""
        messages = [{"role": "user", "content": "Hi"}]
        hf_msgs, _ = normalize_to_hf_messages(messages, system=None)
        assert len(hf_msgs) == 1

    def test_system_empty_string_not_injected(self):
        """system='' does not inject a system message."""
        messages = [{"role": "user", "content": "Hi"}]
        hf_msgs, _ = normalize_to_hf_messages(messages, system="")
        assert len(hf_msgs) == 1

    def test_tool_message_passthrough(self):
        """Tool message preserves tool_call_id and name."""
        messages = [
            {
                "role": "tool",
                "tool_use_id": "toolu_123",
                "name": "search",
                "content": "result data",
                "weight": 0,
            },
        ]
        hf_msgs, weights = normalize_to_hf_messages(messages)

        assert hf_msgs[0]["role"] == "tool"
        assert hf_msgs[0]["content"] == "result data"
        assert hf_msgs[0]["tool_call_id"] == "toolu_123"
        assert hf_msgs[0]["name"] == "search"
        assert weights[0] == 0.0

    def test_rich_content_blocks(self):
        """Content blocks normalized to string."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "hmm"},
                    {"type": "text", "text": "answer"},
                ],
            },
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        assert "<think>" in hf_msgs[0]["content"]
        assert "answer" in hf_msgs[0]["content"]

    def test_empty_messages(self):
        """Empty messages list returns empty."""
        hf_msgs, weights = normalize_to_hf_messages([])
        assert hf_msgs == []
        assert weights == []

    def test_weights_aligned_with_messages(self):
        """weights list length always matches hf_messages length."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        hf_msgs, weights = normalize_to_hf_messages(messages, system="injected")
        # injected system + 5 original messages = 6
        assert len(hf_msgs) == 6
        assert len(weights) == 6

    def test_missing_role_defaults_to_user(self):
        """Missing role defaults to 'user'."""
        messages = [{"content": "no role"}]
        hf_msgs, weights = normalize_to_hf_messages(messages)
        assert hf_msgs[0]["role"] == "user"
        assert weights[0] == 0.0  # user default weight

    def test_missing_content_defaults_to_empty(self):
        """Missing content defaults to empty string."""
        messages = [{"role": "user"}]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        assert hf_msgs[0]["content"] == ""


# ---------------------------------------------------------------------------
# Assistant tool_calls: blocks → HF tool_calls synthesis (template-agnostic)
# ---------------------------------------------------------------------------
#
# normalize_to_hf_messages is template-agnostic — it does not know whether the
# downstream Jinja is DSV3 (`content is none` gate), gpt_oss (string iteration
# on content), llama4, etc. Its job is to lift blocks-format tool_use into the
# HF-standard `tool_calls` list and leave content as a string (empty when no
# text/thinking blocks).
#
# Template-specific rewrites (e.g. DSV3/R1 requiring `content=None` to trigger
# the <｜tool▁call▁begin｜> branch) belong in that template's
# ToolUseAdapter.preprocess_messages — see
# `titaninfra/data/sft/tool_use_adapters/deepseek_v3.py` and
# `tests/test_dsv3_adapter_preprocess.py`.
#
# Earlier this file enforced `content=None` here, which broke gpt_oss/llama4
# Jinja (they iterate the content string and crash on None). The per-family
# adapter layer is the correct fix point.


class TestAssistantToolCallsBlocksSynthesis:
    """blocks-format tool_use → HF tool_calls list (template-agnostic layer)."""

    def test_blocks_pure_tool_use_synthesises_tool_calls(self):
        """Assistant with only tool_use blocks → tool_calls list + content=''."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "x"}},
                ],
            }
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        # Content stays as empty string at this layer; DSV3-specific None
        # rewrite happens in DeepSeekV3Adapter.preprocess_messages.
        assert hf_msgs[0]["content"] == ""
        assert hf_msgs[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_blocks_text_plus_tool_use_keeps_text_content(self):
        """Same-turn text + tool_use → keep text string, attach tool_calls list."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Sure, searching..."},
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "x"}},
                ],
            }
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        assert hf_msgs[0]["content"] == "Sure, searching..."
        assert "tool_calls" in hf_msgs[0]

    def test_blocks_thinking_only_plus_tool_use_extracts_reasoning(self):
        """thinking-only (no text) + tool_use → content='', reasoning_content set, tool_calls list."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Need to search"},
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "x"}},
                ],
            }
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        # reasoning splits to its own field; main content is empty (no text blocks).
        assert hf_msgs[0]["content"] == ""
        assert hf_msgs[0]["reasoning_content"] == "Need to search"
        assert "tool_calls" in hf_msgs[0]

    def test_passthrough_tool_calls_with_empty_content_unchanged(self):
        """msg.tool_calls passthrough preserves content='' (no None rewrite at this layer)."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q":"x"}'},
                    }
                ],
            }
        ]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        # Empty string preserved; DSV3 adapter handles the None rewrite per-family.
        assert hf_msgs[0]["content"] == ""
        assert hf_msgs[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_assistant_text_only_unchanged(self):
        """No tool_calls → content stays as string (regression guard)."""
        messages = [{"role": "assistant", "content": "Hello"}]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        assert hf_msgs[0]["content"] == "Hello"
        assert "tool_calls" not in hf_msgs[0]

    def test_assistant_empty_text_no_tool_calls_unchanged(self):
        """Empty content + no tool_calls → content stays as '' (regression guard)."""
        messages = [{"role": "assistant", "content": ""}]
        hf_msgs, _ = normalize_to_hf_messages(messages)
        assert hf_msgs[0]["content"] == ""
        assert "tool_calls" not in hf_msgs[0]
