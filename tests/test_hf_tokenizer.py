"""
Tests for titaninfra.data.sft._jinja_renderer and helpers from encoders._boundary

Covers:
- _find_role_markers(): pattern detection for ChatML, Llama3, DeepSeek, etc.
- _compute_message_boundaries(): boundary calculation with offset_mapping
- tokenize_with_hf_template(): end-to-end tokenize + loss mask
- Edge cases: empty messages, max_length truncation, train_on_inputs

Uses a mock tokenizer that mimics ChatML format with offset_mapping support.
"""

import pytest

from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from tests.oracle._legacy._boundary import (
    _compute_message_boundaries,
    _find_role_markers,
)
from titaninfra.data.sft.types import IGNORE_INDEX

# ---------------------------------------------------------------------------
# Mock tokenizer for ChatML format
# ---------------------------------------------------------------------------


class MockChatMLTokenizer:
    """
    Deterministic mock tokenizer that simulates ChatML format.

    Produces char-level tokenization (1 char = 1 token) for predictable
    boundary testing. Supports apply_chat_template and offset_mapping.
    """

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
        )

    def apply_chat_template(self, messages, **kwargs):
        tokenize = kwargs.get("tokenize", True)
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        text = "".join(parts)
        if tokenize:
            return [ord(c) % 256 for c in text]
        return text

    def __call__(self, text, **kwargs):
        """Simulate tokenizer call with offset_mapping support."""
        ids = [ord(c) % 256 for c in text]
        result = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]
        return result

    def encode(self, text, **kwargs):
        return [ord(c) % 256 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# _find_role_markers tests
# ---------------------------------------------------------------------------


class TestFindRoleMarkers:
    """Tests for _find_role_markers()."""

    def test_chatml_markers(self):
        """Detects ChatML role markers; content_start lands after `<|im_start|>role\\n`."""
        text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n"
        markers = _find_role_markers(text)
        assert len(markers) == 2
        h0, c0, role0 = markers[0]
        h1, c1, role1 = markers[1]
        assert role0 == "user"
        assert role1 == "assistant"
        assert text[c0:].startswith("Hello")
        assert text[c1:].startswith("Hi")

    def test_chatml_system(self):
        """Detects system role in ChatML."""
        text = "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
        markers = _find_role_markers(text)
        assert len(markers) == 2
        assert markers[0][2] == "system"
        assert markers[1][2] == "user"

    def test_chatml_tool(self):
        """Detects tool role in ChatML."""
        text = "<|im_start|>tool\nresult<|im_end|>\n"
        markers = _find_role_markers(text)
        assert len(markers) == 1
        assert markers[0][2] == "tool"

    def test_llama3_markers(self):
        """Detects Llama 3 style markers (no trailing \\n\\n in this fixture)."""
        text = "<|start_header_id|>user<|end_header_id|>Hello<|start_header_id|>assistant<|end_header_id|>Hi"
        markers = _find_role_markers(text)
        assert len(markers) == 2
        assert markers[0][2] == "user"
        assert markers[1][2] == "assistant"

    def test_llama3_ipython_maps_to_tool(self):
        """Llama 3 ipython role maps to tool."""
        text = "<|start_header_id|>ipython<|end_header_id|>result"
        markers = _find_role_markers(text)
        assert len(markers) == 1
        assert markers[0][2] == "tool"

    def test_deepseek_markers(self):
        """Detects DeepSeek full-width bracket markers."""
        text = "<｜User｜>Hello<｜Assistant｜>Hi"
        markers = _find_role_markers(text)
        assert len(markers) == 2
        assert markers[0][2] == "user"
        assert markers[1][2] == "assistant"

    def test_gpt_oss_markers(self):
        """Detects GPT-oss markers."""
        text = "<|start|>user\nHello<|start|>assistant\nHi"
        markers = _find_role_markers(text)
        assert len(markers) == 2
        assert markers[0][2] == "user"
        assert markers[1][2] == "assistant"

    def test_gpt_oss_developer_maps_to_system(self):
        """GPT-oss developer role maps to system."""
        text = "<|start|>developer\nInstructions"
        markers = _find_role_markers(text)
        assert len(markers) == 1
        assert markers[0][2] == "system"

    def test_no_markers_returns_empty(self):
        """Plain text without markers returns empty list."""
        markers = _find_role_markers("just plain text")
        assert markers == []

    def test_marker_positions_are_correct(self):
        """Marker char positions match actual positions in text."""
        text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n"
        markers = _find_role_markers(text)
        # First marker header_start at position 0
        assert markers[0][0] == 0
        # Second marker starts after first message
        assert text[markers[1][0]:].startswith("<|im_start|>assistant")
        # First marker content_start lands on "Hello"
        assert text[markers[0][1]:].startswith("Hello")


# ---------------------------------------------------------------------------
# _compute_message_boundaries tests
# ---------------------------------------------------------------------------


class TestComputeMessageBoundaries:
    """Tests for _compute_message_boundaries()."""

    def test_basic_boundaries(self):
        """Boundaries cover all tokens for a simple conversation."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        full_ids, boundaries, content_starts = _compute_message_boundaries(
            messages, tokenizer
        )

        assert len(full_ids) > 0
        assert len(boundaries) == len(messages) + 1
        assert len(content_starts) == len(messages)
        assert boundaries[0] == 0
        assert boundaries[-1] == len(full_ids)
        # content_start of each message must be >= its header_start
        for i, cs in enumerate(content_starts):
            assert boundaries[i] <= cs <= boundaries[i + 1]

    def test_boundaries_cover_full_sequence(self):
        """First boundary starts at 0, last boundary is len(full_ids)."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        full_ids, boundaries, _ = _compute_message_boundaries(messages, tokenizer)

        assert boundaries[0] == 0
        assert boundaries[-1] == len(full_ids)

    def test_boundaries_are_monotonic(self):
        """Boundaries are non-decreasing."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Welcome"},
        ]
        _, boundaries, content_starts = _compute_message_boundaries(messages, tokenizer)

        for i in range(len(boundaries) - 1):
            assert boundaries[i] <= boundaries[i + 1], (
                f"boundaries not monotonic at {i}: {boundaries[i]} > {boundaries[i+1]}"
            )
        # content_starts also monotonic and within their message range
        for i, cs in enumerate(content_starts):
            assert boundaries[i] <= cs <= boundaries[i + 1]


# ---------------------------------------------------------------------------
# tokenize_with_hf_template tests
# ---------------------------------------------------------------------------


class TestTokenizeWithHfTemplate:
    """Tests for tokenize_with_hf_template()."""

    def test_basic_tokenize(self):
        """Basic tokenization produces input_ids and labels."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        input_ids, labels = tokenize_with_hf_template(messages, tokenizer)

        assert len(input_ids) > 0
        assert len(input_ids) == len(labels)

    def test_user_masked_assistant_not(self):
        """User tokens masked (IGNORE_INDEX), assistant tokens have real labels."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        input_ids, labels = tokenize_with_hf_template(messages, tokenizer)

        # At least some labels should be IGNORE_INDEX (user part)
        assert any(lbl == IGNORE_INDEX for lbl in labels)
        # At least some labels should be real (assistant part)
        assert any(lbl != IGNORE_INDEX for lbl in labels)

    def test_train_on_inputs_all_labels(self):
        """train_on_inputs=True sets all labels to real values."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        input_ids, labels = tokenize_with_hf_template(
            messages, tokenizer, train_on_inputs=True
        )

        assert all(lbl != IGNORE_INDEX for lbl in labels)

    def test_explicit_weight_zero_masks(self):
        """weight=0 on assistant masks those tokens."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!", "weight": 0},
        ]
        input_ids, labels = tokenize_with_hf_template(messages, tokenizer)

        # All labels should be IGNORE_INDEX since both user (default 0) and assistant (explicit 0)
        assert all(lbl == IGNORE_INDEX for lbl in labels)

    def test_explicit_weight_one_on_user(self):
        """weight=1 on user includes user CONTENT tokens in loss.

        Note: After HF-aligned masking, role markers (`<|im_start|>role\\n`,
        `<|im_end|>\\n`) themselves stay structural and don't get loss even when
        weight=1. Only the message body participates.
        """
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hello", "weight": 1},
            {"role": "assistant", "content": "Hi!"},
        ]
        input_ids, labels = tokenize_with_hf_template(messages, tokenizer)

        # User content "Hello" tokens must be unmasked (since weight=1)
        text = "".join(chr(i) for i in input_ids)
        hello_start = text.find("Hello")
        for i in range(hello_start, hello_start + len("Hello")):
            assert labels[i] != IGNORE_INDEX, (
                f"user content position {i} should have loss when weight=1"
            )
        # Assistant content "Hi!" must also be unmasked
        hi_start = text.find("Hi!")
        for i in range(hi_start, hi_start + len("Hi!")):
            assert labels[i] != IGNORE_INDEX, (
                f"assistant content position {i} should have loss"
            )

    def test_system_injection(self):
        """system parameter injects system message with weight=0."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        input_ids, labels = tokenize_with_hf_template(
            messages, tokenizer, system="Be helpful"
        )

        # Should have tokens (system + user + assistant)
        assert len(input_ids) > 0
        # System + user should be masked, only assistant has real labels
        assert any(lbl == IGNORE_INDEX for lbl in labels)
        assert any(lbl != IGNORE_INDEX for lbl in labels)

    def test_max_length_truncation(self):
        """max_length truncates sequences."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hello world, this is a long message"},
            {"role": "assistant", "content": "This is also quite long"},
        ]
        max_len = 20
        input_ids, labels = tokenize_with_hf_template(
            messages, tokenizer, max_length=max_len
        )

        assert len(input_ids) <= max_len
        assert len(labels) <= max_len

    def test_empty_messages(self):
        """Empty messages list returns empty."""
        tokenizer = MockChatMLTokenizer()
        input_ids, labels = tokenize_with_hf_template([], tokenizer)
        assert input_ids == []
        assert labels == []

    def test_multi_turn_conversation(self):
        """Multi-turn conversation: only assistant turns have loss."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "What about 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        input_ids, labels = tokenize_with_hf_template(messages, tokenizer)

        # Both masked and unmasked tokens should exist
        num_masked = sum(1 for lbl in labels if lbl == IGNORE_INDEX)
        num_loss = sum(1 for lbl in labels if lbl != IGNORE_INDEX)
        assert num_masked > 0
        assert num_loss > 0

    def test_labels_match_input_ids_where_not_masked(self):
        """Non-IGNORE_INDEX labels equal their corresponding input_ids."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        input_ids, labels = tokenize_with_hf_template(messages, tokenizer)

        for i, (iid, label) in enumerate(zip(input_ids, labels, strict=False)):
            if label != IGNORE_INDEX:
                assert label == iid, f"Label mismatch at pos {i}: {label} != {iid}"

    def test_rich_content_blocks(self):
        """Rich content blocks (thinking + text) handled correctly."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Explain gravity"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Physics question..."},
                    {"type": "text", "text": "Gravity is a fundamental force."},
                ],
            },
        ]
        input_ids, labels = tokenize_with_hf_template(messages, tokenizer)

        assert len(input_ids) > 0
        # Assistant part should have loss tokens
        assert any(lbl != IGNORE_INDEX for lbl in labels)

    def test_debug_mode_runs_without_error(self, capsys):
        """debug=True prints output without crashing."""
        tokenizer = MockChatMLTokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        input_ids, labels = tokenize_with_hf_template(
            messages, tokenizer, debug=True
        )

        captured = capsys.readouterr()
        assert "DEBUG" in captured.out
        assert len(input_ids) > 0


# ---------------------------------------------------------------------------
# return_weights=True + type_overrides (per-token loss weighting)
# ---------------------------------------------------------------------------

QWEN3_TOKENIZER = "tests/fixtures/tokenizers/qwen3"


class TestTypeOverridesThinking:
    """Tests for test type overrides thinking."""

    @staticmethod
    def _load_qwen3():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    @staticmethod
    def _messages_with_thinking():
        return [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Adding two and two together"},
                    {"type": "text", "text": "The answer is 4."},
                ],
            },
        ]

    def test_default_return_2tuple_unchanged(self):
        """Verify default return 2tuple unchanged."""
        tok = self._load_qwen3()
        msgs = self._messages_with_thinking()
        result = tokenize_with_hf_template(msgs, tok)
        assert isinstance(result, tuple) and len(result) == 2
        input_ids, labels = result
        assert len(input_ids) == len(labels) > 0

    def test_return_weights_no_loss_config(self):
        """Verify return weights no loss config."""
        tok = self._load_qwen3()
        msgs = self._messages_with_thinking()
        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, return_weights=True
        )
        assert len(ids) == len(labels) > 0
        assert w is None

    def test_return_weights_empty_type_overrides(self):
        """Verify return weights empty type overrides."""
        tok = self._load_qwen3()
        msgs = self._messages_with_thinking()
        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, return_weights=True,
            loss_config={"type_overrides": {}},
        )
        assert w is None

    def test_thinking_scale_partial(self):
        """Verify thinking scale partial."""
        tok = self._load_qwen3()
        msgs = self._messages_with_thinking()
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        assert w is not None and len(w) == len(ids)

        open_pos = ids.index(open_id)
        close_pos = ids.index(close_id)
        assert open_pos < close_pos

        assert w[open_pos] == 1.0, f"open marker should be 1.0, got {w[open_pos]}"
        assert w[close_pos] == 1.0, f"close marker should be 1.0, got {w[close_pos]}"

        for j in range(open_pos + 1, close_pos):
            assert abs(w[j] - 0.3) < 1e-9, f"inner pos {j} should be 0.3, got {w[j]}"

        assert all(labels[j] != IGNORE_INDEX for j in range(open_pos + 1, close_pos))

    def test_thinking_scale_zero_uses_binary_path(self):
        """Verify thinking scale zero uses binary path."""
        tok = self._load_qwen3()
        msgs = self._messages_with_thinking()
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.0}},
        )
        assert w is None, "binary path should return None loss_weights for memory efficiency"

        open_pos = ids.index(open_id)
        close_pos = ids.index(close_id)

        for j in range(open_pos + 1, close_pos):
            assert labels[j] == IGNORE_INDEX, f"pos {j} should be masked, got {labels[j]}"

        assert labels[open_pos] == open_id
        assert labels[close_pos] == close_id

    def test_qwen3_auto_injected_empty_thinking_is_scaled(self):
        """Verify qwen3 auto injected empty thinking is scaled."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ]
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        open_pos = ids.index(open_id)
        close_pos = ids.index(close_id)
        if close_pos > open_pos + 1:
            assert w is not None
            for j in range(open_pos + 1, close_pos):
                assert abs(w[j] - 0.3) < 1e-9, f"auto-injected inner pos {j} should be 0.3"
        else:
            assert w is None

    def test_qwen3_collapses_multiple_thinking_blocks(self):
        """Verify qwen3 collapses multiple thinking blocks."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Two-step"},
            {
                "role": "assistant",
                "content": (
                    "<think>\nStep 1: parse\n</think>\n\n"
                    "First reading.\n"
                    "<think>\nStep 2: compute\n</think>\n\n"
                    "Result: 42."
                ),
            },
        ]
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.5}},
        )
        assert w is not None
        opens = [i for i, t in enumerate(ids) if t == open_id]
        closes = [i for i, t in enumerate(ids) if t == close_id]
        assert len(opens) == 1 and len(closes) == 1, (
            "Qwen3 模板已知会合并多段 thinking；如此 assert 失败说明模板行为改变，"
            "需重新评估 type_overrides 算法在多段场景的覆盖"
        )
        op, cl = opens[0], closes[0]
        for j in range(op + 1, cl):
            assert abs(w[j] - 0.5) < 1e-9

    def test_loss_weights_length_consistency(self):
        """Verify loss weights length consistency."""
        tok = self._load_qwen3()
        msgs = self._messages_with_thinking()
        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, max_length=4096, return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        assert len(ids) == len(labels)
        if w is not None:
            assert len(w) == len(ids)


class TestMaskEmptyThinking:
    """Tests for test mask empty thinking."""

    @staticmethod
    def _load_qwen3():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    def test_empty_thinking_masked(self):
        """Verify empty thinking masked."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        ids, labels = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True,
        )

        open_pos = ids.index(open_id)
        close_pos = ids.index(close_id)
        assert open_pos < close_pos

        for j in range(open_pos, close_pos + 1):
            assert labels[j] == IGNORE_INDEX, (
                f"pos {j} (token={ids[j]}) should be masked for empty thinking"
            )

        hello_region_start = close_pos + 1
        has_loss_after_think = any(
            labels[j] != IGNORE_INDEX
            for j in range(hello_region_start, len(labels))
        )
        assert has_loss_after_think, "Answer tokens after empty thinking should have loss"

    def test_real_thinking_not_masked(self):
        """Verify real thinking not masked."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Adding two and two"},
                    {"type": "text", "text": "4"},
                ],
            },
        ]
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        ids, labels = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True,
        )

        open_pos = ids.index(open_id)
        close_pos = ids.index(close_id)

        inner_has_loss = any(
            labels[j] != IGNORE_INDEX
            for j in range(open_pos + 1, close_pos)
        )
        assert inner_has_loss, "Real thinking content should NOT be masked"

    def test_disabled_no_masking(self):
        """Verify disabled no masking."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        open_id = tok.encode("<think>", add_special_tokens=False)[0]

        ids, labels = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=False,
        )

        open_pos = ids.index(open_id)
        assert labels[open_pos] != IGNORE_INDEX, (
            "With mask_empty_thinking=False, open marker should have loss"
        )

    def test_mixed_thinking_and_non_thinking(self):
        """Verify mixed thinking and non thinking."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Adding numbers"},
                    {"type": "text", "text": "4"},
                ],
            },
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": [{"type": "text", "text": "You're welcome!"}]},
        ]
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        ids, labels = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True,
        )

        open_positions = [i for i, t in enumerate(ids) if t == open_id]
        close_positions = [i for i, t in enumerate(ids) if t == close_id]

        for op, cl in zip(open_positions, close_positions, strict=False):
            for j in range(op, cl + 1):
                assert labels[j] == IGNORE_INDEX, (
                    f"Empty thinking at pos {op}-{cl}: pos {j} should be masked"
                )

        answer_ids = tok.encode("4", add_special_tokens=False)
        if answer_ids:
            first_answer_pos = ids.index(answer_ids[0])
            assert labels[first_answer_pos] != IGNORE_INDEX, (
                "First assistant's answer '4' should have loss"
            )

    def test_mask_empty_thinking_with_thinking_scale(self):
        """Verify mask empty thinking with thinking scale."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Two plus two equals four"},
                    {"type": "text", "text": "4"},
                ],
            },
        ]
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True, return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )

        open_pos = ids.index(open_id)
        close_pos = ids.index(close_id)

        inner_has_loss = any(
            labels[j] != IGNORE_INDEX for j in range(open_pos + 1, close_pos)
        )
        assert inner_has_loss, "Real thinking should NOT be masked"

        if w is not None:
            for j in range(open_pos + 1, close_pos):
                if labels[j] != IGNORE_INDEX:
                    assert abs(w[j] - 0.3) < 1e-9, (
                        f"Real thinking at pos {j} should be scaled to 0.3"
                    )

    def test_default_is_false(self):
        """Verify default is false."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]

        ids_default, labels_default = tokenize_with_hf_template(msgs, tok)
        ids_off, labels_off = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=False,
        )

        assert labels_default == labels_off, (
            "Default behavior should be identical to mask_empty_thinking=False"
        )


# ---------------------------------------------------------------------------
# Encode → Decode → Observe masking  (text-level verification)
# ---------------------------------------------------------------------------


def _decode_mask_segments(
    tokenizer, input_ids: list[int], labels: list[int],
) -> list[tuple[str, str]]:
    """Decode mask segments."""
    if not input_ids:
        return []
    segments: list[tuple[str, str]] = []
    cur_ids: list[int] = [input_ids[0]]
    cur_tag = "LOSS" if labels[0] != IGNORE_INDEX else "MASKED"

    for i in range(1, len(input_ids)):
        tag = "LOSS" if labels[i] != IGNORE_INDEX else "MASKED"
        if tag == cur_tag:
            cur_ids.append(input_ids[i])
        else:
            segments.append((tokenizer.decode(cur_ids), cur_tag))
            cur_ids = [input_ids[i]]
            cur_tag = tag
    segments.append((tokenizer.decode(cur_ids), cur_tag))
    return segments


def _decode_weight_segments(
    tokenizer,
    input_ids: list[int],
    labels: list[int],
    loss_weights: list[float] | None,
) -> list[tuple[str, str, float]]:
    """Decode weight segments."""
    if not input_ids:
        return []

    def _tag_and_weight(idx: int) -> tuple[str, float]:
        tag = "LOSS" if labels[idx] != IGNORE_INDEX else "MASKED"
        if loss_weights is not None:
            w = loss_weights[idx]
        else:
            w = 1.0 if tag == "LOSS" else 0.0
        return tag, w

    segments: list[tuple[str, str, float]] = []
    cur_ids: list[int] = [input_ids[0]]
    cur_tag, cur_w = _tag_and_weight(0)

    for i in range(1, len(input_ids)):
        tag, w = _tag_and_weight(i)
        if tag == cur_tag and abs(w - cur_w) < 1e-9:
            cur_ids.append(input_ids[i])
        else:
            segments.append((tokenizer.decode(cur_ids), cur_tag, cur_w))
            cur_ids = [input_ids[i]]
            cur_tag, cur_w = tag, w
    segments.append((tokenizer.decode(cur_ids), cur_tag, cur_w))
    return segments


class TestDecodeMaskingVisual:
    """Tests for test decode masking visual."""

    @staticmethod
    def _load(name: str):
        from transformers import AutoTokenizer
        path = f"tests/fixtures/tokenizers/{name}"
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # ---- Qwen3: basic SFT ----

    def test_qwen3_basic_sft(self):
        """Verify qwen3 basic sft."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok)
        segs = _decode_mask_segments(tok, ids, labels)

        tags = [s[1] for s in segs]
        assert tags[-1] == "LOSS", f"Last segment must be LOSS, got {tags}"
        assert any(t == "MASKED" for t in tags), "Must have at least one MASKED segment"

        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")
        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")

        assert "Hi" in masked_text, "User content should be MASKED"
        assert "Hello!" in loss_text, "Assistant content should have LOSS"
        assert "Hello!" not in masked_text

    # ---- Qwen3: empty thinking masking ----

    def test_qwen3_empty_thinking_decode(self):
        """Verify qwen3 empty thinking decode."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, mask_empty_thinking=True)
        segs = _decode_mask_segments(tok, ids, labels)

        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")
        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")

        assert "<think>" in masked_text, "Empty <think> marker should be MASKED"
        assert "</think>" in masked_text, "Empty </think> marker should be MASKED"
        assert "Hello!" in loss_text, "Answer should have LOSS"
        assert "<think>" not in loss_text, "Empty <think> should NOT have LOSS"

    def test_qwen3_empty_thinking_not_masked_by_default(self):
        """Verify qwen3 empty thinking not masked by default."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, mask_empty_thinking=False)
        segs = _decode_mask_segments(tok, ids, labels)

        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        assert "<think>" in loss_text, "Without mask_empty_thinking, <think> should have LOSS"

    # ---- Qwen3: real thinking ----

    def test_qwen3_real_thinking_decode(self):
        """Verify qwen3 real thinking decode."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Adding two numbers"},
                    {"type": "text", "text": "The answer is 4"},
                ],
            },
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, mask_empty_thinking=True)
        segs = _decode_mask_segments(tok, ids, labels)

        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")

        assert "Adding two numbers" in loss_text, "Thinking content should have LOSS"
        assert "The answer is 4" in loss_text, "Answer should have LOSS"
        assert "What is 2+2?" in masked_text, "User content should be MASKED"

    # ---- Qwen3: multi-turn mixed ----

    def test_qwen3_multiturn_mixed_decode(self):
        """Verify qwen3 multiturn mixed decode."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Simple addition"},
                    {"type": "text", "text": "4"},
                ],
            },
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": [{"type": "text", "text": "Welcome!"}]},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, mask_empty_thinking=True)
        segs = _decode_mask_segments(tok, ids, labels)

        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")

        assert "4" in loss_text, "First assistant answer '4' should have LOSS"
        assert "Welcome!" in loss_text, "Second assistant answer should have LOSS"
        assert "What is 2+2?" in masked_text, "User content should be MASKED"
        assert "Thanks" in masked_text, "User content should be MASKED"

    # ---- Qwen3: train_on_inputs ----

    def test_qwen3_train_on_inputs_decode(self):
        """Verify qwen3 train on inputs decode."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, train_on_inputs=True)
        segs = _decode_mask_segments(tok, ids, labels)

        tags = set(s[1] for s in segs)
        assert tags == {"LOSS"}, f"train_on_inputs should make everything LOSS, got {tags}"

    # ---- Qwen3: thinking_scale ----

    def test_qwen3_thinking_scale_zero_decode(self):
        """Verify qwen3 thinking scale zero decode."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Adding numbers together"},
                    {"type": "text", "text": "The answer is 4"},
                ],
            },
        ]
        ids, labels, w = tokenize_with_hf_template(
            msgs, tok, return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.0}},
        )
        segs = _decode_mask_segments(tok, ids, labels)

        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")

        assert "Adding numbers together" in masked_text, (
            "thinking_scale=0 should MASK thinking content"
        )
        assert "The answer is 4" in loss_text, "Answer should have LOSS"
        assert "<think>" in loss_text, "Marker should have LOSS (structural signal)"

    # ---- Qwen2.5: no thinking injection ----

    def test_qwen2_5_basic_sft_decode(self):
        """Verify qwen2 5 basic sft decode."""
        tok = self._load("qwen2_5")
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok)
        segs = _decode_mask_segments(tok, ids, labels)

        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")

        assert "Hi there!" in loss_text
        assert "Hello" in masked_text
        assert "<think>" not in loss_text, "Qwen2.5 should not inject <think>"
        assert "<think>" not in masked_text

    # ---- DeepSeek: basic SFT ----

    def test_deepseek_v3_basic_decode(self):
        """Verify deepseek v3 basic decode."""
        tok = self._load("deepseek_v3")
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok)
        segs = _decode_mask_segments(tok, ids, labels)

        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")

        assert "Hi!" in loss_text, "Assistant should have LOSS"
        assert "Hello" in masked_text, "User should be MASKED"

    # ---- System prompt ----

    def test_system_prompt_masked_decode(self):
        """Verify system prompt masked decode."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok)
        segs = _decode_mask_segments(tok, ids, labels)

        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")
        assert "helpful assistant" in masked_text, "System prompt should be MASKED"

    # ---- Roundtrip: decoded text covers full sequence ----

    def test_decode_roundtrip_covers_full_sequence(self):
        """Verify decode roundtrip covers full sequence."""
        tok = self._load("qwen3")
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is 1+1?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Simple math"},
                    {"type": "text", "text": "2"},
                ],
            },
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, mask_empty_thinking=True)
        segs = _decode_mask_segments(tok, ids, labels)

        reconstructed = "".join(s[0] for s in segs)
        full_decode = tok.decode(ids)
        assert reconstructed == full_decode, (
            f"Segments must reconstruct full decode.\n"
            f"Reconstructed: {reconstructed!r}\n"
            f"Full decode:   {full_decode!r}"
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestTypeOverridesQwen3DecodePipeline:
    """Tests for test type overrides qwen3 decode pipeline."""

    @staticmethod
    def _load_qwen3():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    @staticmethod
    def _msgs_with_thinking():
        return [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Two plus two equals four"},
                    {"type": "text", "text": "The answer is 4"},
                ],
            },
        ]

    @staticmethod
    def _msgs_no_thinking():
        return [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]


    def test_scale_0_3_thinking_content_weighted(self):
        """Verify scale 0 3 thinking content weighted."""
        tok = self._load_qwen3()
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_with_thinking(), tok,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        segs = _decode_weight_segments(tok, ids, labels, w)

        found_thinking = False
        for text, tag, weight in segs:
            if "Two plus two" in text:
                found_thinking = True
                assert tag == "LOSS", f"Thinking content should be LOSS, got {tag}"
                assert abs(weight - 0.3) < 1e-9, f"Thinking weight should be 0.3, got {weight}"
            if "The answer is 4" in text:
                assert tag == "LOSS"
                assert abs(weight - 1.0) < 1e-9, f"Answer weight should be 1.0, got {weight}"

        assert found_thinking, "Should find thinking content segment"

    def test_scale_0_5_markers_full_weight(self):
        """Verify scale 0 5 markers full weight."""
        tok = self._load_qwen3()
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_with_thinking(), tok,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.5}},
        )
        segs = _decode_weight_segments(tok, ids, labels, w)

        for text, tag, weight in segs:
            if "<think>" in text and "Two plus two" not in text:
                assert abs(weight - 1.0) < 1e-9, (
                    f"<think> marker weight should be 1.0, got {weight}"
                )
            if "</think>" in text and "Two plus two" not in text:
                assert abs(weight - 1.0) < 1e-9, (
                    f"</think> marker weight should be 1.0, got {weight}"
                )

    def test_scale_0_binary_path(self):
        """Verify scale 0 binary path."""
        tok = self._load_qwen3()
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_with_thinking(), tok,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.0}},
        )
        assert w is None, "Binary path: loss_weights should be None"

        segs = _decode_mask_segments(tok, ids, labels)
        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")
        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")

        assert "Two plus two" in masked_text, "Thinking content should be MASKED"
        assert "The answer is 4" in loss_text, "Answer should have LOSS"
        assert "<think>" in loss_text, "Marker should have LOSS"

    def test_scale_1_0_no_effect(self):
        """Verify scale 1 0 no effect."""
        tok = self._load_qwen3()
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_with_thinking(), tok,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 1.0}},
        )
        assert w is None, "thinking=1.0 degenerates to binary path (all 0/1)"

        segs = _decode_mask_segments(tok, ids, labels)
        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        assert "Two plus two" in loss_text, "Thinking should have LOSS at scale 1.0"
        assert "The answer is 4" in loss_text


    def test_empty_thinking_masked_then_scale_is_noop(self):
        """Verify empty thinking masked then scale is noop."""
        tok = self._load_qwen3()
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_no_thinking(), tok,
            mask_empty_thinking=True,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        segs = _decode_weight_segments(tok, ids, labels, w)

        for text, tag, weight in segs:
            if "<think>" in text or "</think>" in text:
                assert tag == "MASKED", (
                    f"Empty thinking should be MASKED even with scale, got {tag}: {text!r}"
                )

        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        assert "Hello!" in loss_text, "Answer should still have LOSS"

    def test_empty_thinking_not_masked_scale_applied(self):
        """Verify empty thinking not masked scale applied."""
        tok = self._load_qwen3()
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_no_thinking(), tok,
            mask_empty_thinking=False,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        segs = _decode_weight_segments(tok, ids, labels, w)

        for text, tag, weight in segs:
            if "<think>" in text and "\n" not in text:
                assert abs(weight - 1.0) < 1e-9, "Open marker should be 1.0"
            if "</think>" in text and "\n" not in text:
                assert abs(weight - 1.0) < 1e-9, "Close marker should be 1.0"

    def test_mixed_real_and_empty_thinking_with_scale(self):
        """Verify mixed real and empty thinking with scale."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Simple addition"},
                    {"type": "text", "text": "4"},
                ],
            },
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": [{"type": "text", "text": "Welcome!"}]},
        ]
        ids, labels, w = tokenize_with_hf_template(
            msgs, tok,
            mask_empty_thinking=True,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.5}},
        )
        segs = _decode_weight_segments(tok, ids, labels, w)

        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        assert "4" in loss_text, "First answer should have LOSS"
        assert "Welcome!" in loss_text, "Second answer should have LOSS"

        for text, tag, weight in segs:
            if "<think>" in text and tag == "MASKED":
                pass
            if "Welcome!" in text:
                assert abs(weight - 1.0) < 1e-9, "Non-thinking answer should be 1.0"


    def test_train_on_inputs_with_scale(self):
        """Verify train on inputs with scale."""
        tok = self._load_qwen3()
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_with_thinking(), tok,
            train_on_inputs=True,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        segs_mask = _decode_mask_segments(tok, ids, labels)
        tags = set(s[1] for s in segs_mask)
        assert tags == {"LOSS"}, "train_on_inputs should make everything LOSS"

        segs_w = _decode_weight_segments(tok, ids, labels, w)
        found_scaled = False
        for text, tag, weight in segs_w:
            if "Two plus two" in text:
                found_scaled = True
                assert abs(weight - 0.3) < 1e-9, (
                    f"Thinking content should still be scaled to 0.3 "
                    f"under train_on_inputs, got {weight}"
                )
        assert found_scaled


    def test_loss_weights_length_all_scenarios(self):
        """Verify loss weights length all scenarios."""
        tok = self._load_qwen3()
        for scale in [0.0, 0.3, 0.5, 1.0]:
            for mask in [True, False]:
                ids, labels, w = tokenize_with_hf_template(
                    self._msgs_with_thinking(), tok,
                    mask_empty_thinking=mask,
                    return_weights=True,
                    loss_config={"type_overrides": {"thinking": scale}},
                )
                assert len(ids) == len(labels), (
                    f"scale={scale}, mask={mask}: ids/labels length mismatch"
                )
                if w is not None:
                    assert len(w) == len(ids), (
                        f"scale={scale}, mask={mask}: loss_weights length mismatch"
                    )


    def test_weight_segments_cover_full_sequence(self):
        """Verify weight segments cover full sequence."""
        tok = self._load_qwen3()
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_with_thinking(), tok,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        segs = _decode_weight_segments(tok, ids, labels, w)
        reconstructed = "".join(s[0] for s in segs)
        full_decode = tok.decode(ids)
        assert reconstructed == full_decode, (
            f"Weight segments must reconstruct full decode.\n"
            f"Reconstructed: {reconstructed!r}\n"
            f"Full decode:   {full_decode!r}"
        )


    def test_no_type_overrides_no_weights(self):
        """Verify no type overrides no weights."""
        tok = self._load_qwen3()
        for lc in [None, {}, {"type_overrides": {}}]:
            ids, labels, w = tokenize_with_hf_template(
                self._msgs_with_thinking(), tok,
                return_weights=True,
                loss_config=lc,
            )
            assert w is None, f"loss_config={lc}: loss_weights should be None"


    def test_per_sample_loss_config_dict(self):
        """Verify per sample loss config dict."""
        tok = self._load_qwen3()
        sample_lc = {"type_overrides": {"thinking": 0.2}}
        ids, labels, w = tokenize_with_hf_template(
            self._msgs_with_thinking(), tok,
            return_weights=True,
            loss_config=sample_lc,
        )
        segs = _decode_weight_segments(tok, ids, labels, w)

        for text, tag, weight in segs:
            if "Two plus two" in text:
                assert abs(weight - 0.2) < 1e-9, f"Expected 0.2, got {weight}"


# ---------------------------------------------------------------------------
# Qwen3 multi-turn thinking asymmetry tests
# ---------------------------------------------------------------------------


class TestQwen3MultiTurnThinkingAsymmetry:
    """Verify Qwen3 Jinja template's multi-turn thinking behavior.

    Qwen3's HF Jinja template (chat_template.jinja) has deliberate asymmetry:
    - Historical assistant turns (before last_query_index): thinking content
      is STRIPPED from rendered text, regardless of inline <think> tags or
      reasoning_content field.
    - Only the last turn(s) after last_query_index get <think> tags rendered.
    - Empty <think>\\n\\n</think> is injected only for the LAST assistant
      turn when it has no reasoning_content.

    These tests verify titaninfra's pipeline correctly handles this behavior
    and that thinking_scale / mask_empty_thinking have the expected scope.
    """

    @staticmethod
    def _load_qwen3():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    def test_historical_thinking_dropped_by_jinja(self):
        """Historical assistant's thinking content is silently dropped by Qwen3 Jinja.

        In a [user, assistant_with_thinking, user, assistant] conversation,
        the first assistant's <think> content is stripped from rendered text.
        """
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "I need to add two numbers"},
                    {"type": "text", "text": "The answer is 4"},
                ],
            },
            {"role": "user", "content": "What about 3+3?"},
            {"role": "assistant", "content": [{"type": "text", "text": "That's 6"}]},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, mask_empty_thinking=True)
        full_text = tok.decode(ids)

        assert "I need to add two numbers" not in full_text, (
            "Historical thinking content should be DROPPED by Qwen3 Jinja template "
            "(only last-turn thinking is preserved)"
        )
        assert "The answer is 4" in full_text, "Historical answer should be preserved"
        assert "That's 6" in full_text, "Last turn answer should be preserved"

    def test_historical_thinking_dropped_reasoning_content_field(self):
        """reasoning_content field is also dropped for historical turns."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4",
                "reasoning_content": "I need to add two numbers",
            },
            {"role": "user", "content": "What about 3+3?"},
            {"role": "assistant", "content": "That's 6"},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, mask_empty_thinking=True)
        full_text = tok.decode(ids)

        assert "I need to add two numbers" not in full_text, (
            "Historical reasoning_content should be DROPPED by Qwen3 Jinja template"
        )
        assert "The answer is 4" in full_text

    def test_only_last_turn_gets_empty_thinking_injection(self):
        """Qwen3 only injects empty <think> on the last assistant turn, not historical ones."""
        tok = self._load_qwen3()
        open_id = tok.encode("<think>", add_special_tokens=False)[0]
        close_id = tok.encode("</think>", add_special_tokens=False)[0]

        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye"},
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok)

        opens = [i for i, t in enumerate(ids) if t == open_id]
        closes = [i for i, t in enumerate(ids) if t == close_id]

        assert len(opens) == 1, (
            f"Expected exactly 1 <think> (on last turn), got {len(opens)}"
        )
        assert len(closes) == 1, (
            f"Expected exactly 1 </think> (on last turn), got {len(closes)}"
        )

        goodbye_tokens = tok.encode("Goodbye", add_special_tokens=False)
        goodbye_start = None
        for i in range(len(ids) - len(goodbye_tokens) + 1):
            if ids[i : i + len(goodbye_tokens)] == goodbye_tokens:
                goodbye_start = i
                break
        assert goodbye_start is not None
        assert opens[0] < goodbye_start, (
            "<think> should appear BEFORE 'Goodbye' (injected for last turn)"
        )

    def test_thinking_scale_only_affects_last_turn(self):
        """thinking_scale only applies to last turn's thinking (historical is dropped)."""
        tok = self._load_qwen3()
        open_id = tok.encode("<think>", add_special_tokens=False)[0]

        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "First calculation"},
                    {"type": "text", "text": "4"},
                ],
            },
            {"role": "user", "content": "What is 3+3?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Second calculation"},
                    {"type": "text", "text": "6"},
                ],
            },
        ]
        ids, labels, w = tokenize_with_hf_template(
            msgs, tok,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.3}},
        )
        assert w is not None

        full_text = tok.decode(ids)
        assert "First calculation" not in full_text, (
            "Historical thinking dropped by Jinja"
        )
        assert "Second calculation" in full_text, (
            "Last turn thinking should be preserved"
        )

        opens = [i for i, t in enumerate(ids) if t == open_id]
        assert len(opens) == 1, "Only last turn should have <think>"

        segs = _decode_weight_segments(tok, ids, labels, w)
        for text, tag, weight in segs:
            if "Second calculation" in text:
                assert abs(weight - 0.3) < 1e-9, (
                    f"Last turn thinking should be scaled to 0.3, got {weight}"
                )
            if "4" in text and tag == "LOSS":
                assert abs(weight - 1.0) < 1e-9, (
                    f"First assistant answer should be full weight, got {weight}"
                )

    def test_mask_empty_thinking_multiturn_last_only(self):
        """mask_empty_thinking masks the auto-injected empty <think> on last turn only."""
        tok = self._load_qwen3()

        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye"},
        ]
        ids, labels = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True,
        )
        segs = _decode_mask_segments(tok, ids, labels)

        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")
        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")

        assert "<think>" in masked_text, (
            "Empty <think> on last turn should be MASKED"
        )
        assert "Goodbye" in loss_text, "Last turn answer should have LOSS"
        assert "Hello" in loss_text, "First turn answer should have LOSS"

    def test_multiturn_thinking_in_all_turns(self):
        """When all assistant turns have thinking, only the last turn's is preserved."""
        tok = self._load_qwen3()
        open_id = tok.encode("<think>", add_special_tokens=False)[0]

        msgs = [
            {"role": "user", "content": "Step 1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Planning step 1"},
                    {"type": "text", "text": "Done step 1"},
                ],
            },
            {"role": "user", "content": "Step 2"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Planning step 2"},
                    {"type": "text", "text": "Done step 2"},
                ],
            },
            {"role": "user", "content": "Step 3"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Planning step 3"},
                    {"type": "text", "text": "Done step 3"},
                ],
            },
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok, mask_empty_thinking=True)
        full_text = tok.decode(ids)

        assert "Planning step 1" not in full_text
        assert "Planning step 2" not in full_text
        assert "Planning step 3" in full_text, "Only last turn thinking should survive"

        opens = [i for i, t in enumerate(ids) if t == open_id]
        assert len(opens) == 1, (
            f"Only 1 <think> should exist (last turn), got {len(opens)}"
        )

        segs = _decode_mask_segments(tok, ids, labels)
        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")
        assert "Done step 1" in loss_text
        assert "Done step 2" in loss_text
        assert "Done step 3" in loss_text

    def test_single_turn_thinking_preserved(self):
        """Single-turn (most common SFT case): thinking is fully preserved."""
        tok = self._load_qwen3()
        open_id = tok.encode("<think>", add_special_tokens=False)[0]

        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "I need to add 2 and 2"},
                    {"type": "text", "text": "The answer is 4"},
                ],
            },
        ]
        ids, labels = tokenize_with_hf_template(msgs, tok)
        full_text = tok.decode(ids)

        assert "I need to add 2 and 2" in full_text, (
            "Single-turn thinking should be preserved"
        )
        opens = [i for i, t in enumerate(ids) if t == open_id]
        assert len(opens) == 1

    def test_multiturn_thinking_scale_with_mask_empty(self):
        """Multi-turn: real thinking on T1 (dropped), empty thinking on T2 (masked),
        thinking_scale only affects tokens that actually survive in the rendered text."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Solve 2+2"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Adding numbers"},
                    {"type": "text", "text": "4"},
                ],
            },
            {"role": "user", "content": "Good"},
            {"role": "assistant", "content": [{"type": "text", "text": "Thanks!"}]},
        ]
        ids, labels, w = tokenize_with_hf_template(
            msgs, tok,
            mask_empty_thinking=True,
            return_weights=True,
            loss_config={"type_overrides": {"thinking": 0.5}},
        )
        full_text = tok.decode(ids)

        assert "Adding numbers" not in full_text, "T1 thinking dropped by Jinja"

        segs = _decode_weight_segments(tok, ids, labels, w)
        masked_text = " ".join(s[0] for s in segs if s[1] == "MASKED")
        loss_text = " ".join(s[0] for s in segs if s[1] == "LOSS")

        assert "4" in loss_text, "T1 answer should have LOSS"
        assert "Thanks!" in loss_text, "T2 answer should have LOSS"
        assert "<think>" in masked_text, "T2 empty thinking should be MASKED"

        for text, tag, weight in segs:
            if "Thanks!" in text:
                assert abs(weight - 1.0) < 1e-9, "Non-thinking answer should be 1.0"


# ---------------------------------------------------------------------------
# Custom thinking_markers (DeepSeek R1 style)
# ---------------------------------------------------------------------------


class TestCustomThinkingMarkers:
    """Verify thinking_markers parameter works for non-Qwen3 templates.

    Since we don't have a DeepSeek R1 tokenizer fixture with dedicated
    <｜begin of think｜> tokens, we test the mechanism using Qwen3's tokenizer
    with explicit thinking_markers=("<think>", "</think>") to prove:
    1. The parameter flows through correctly
    2. Default (None) behaves identically to explicit Qwen3 markers
    3. Wrong markers → thinking_scale becomes a no-op (safety)
    """

    @staticmethod
    def _load_qwen3():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    @staticmethod
    def _messages_with_thinking():
        return [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Let me calculate"},
                    {"type": "text", "text": "The answer is 4"},
                ],
            },
        ]

    def test_explicit_markers_match_default(self):
        """Explicit ("<think>", "</think>") produces identical output to None."""
        tok = self._load_qwen3()
        msgs = self._messages_with_thinking()
        lc = {"type_overrides": {"thinking": 0.3}}

        ids_default, lab_default, w_default = tokenize_with_hf_template(
            msgs, tok, return_weights=True, loss_config=lc,
            thinking_markers=None,
        )
        ids_explicit, lab_explicit, w_explicit = tokenize_with_hf_template(
            msgs, tok, return_weights=True, loss_config=lc,
            thinking_markers=("<think>", "</think>"),
        )

        assert ids_default == ids_explicit
        assert lab_default == lab_explicit
        assert w_default == w_explicit

    def test_wrong_markers_no_scaling(self):
        """Wrong markers → no thinking ranges found → no scaling applied."""
        tok = self._load_qwen3()
        msgs = self._messages_with_thinking()
        lc = {"type_overrides": {"thinking": 0.3}}

        _, _, w_wrong = tokenize_with_hf_template(
            msgs, tok, return_weights=True, loss_config=lc,
            thinking_markers=("<<FAKE_OPEN>>", "<<FAKE_CLOSE>>"),
        )
        # Wrong markers can't find any thinking ranges → all weights should be 0 or 1
        if w_wrong is not None:
            for w in w_wrong:
                assert w in (0.0, 1.0), (
                    f"Wrong markers should produce no fractional weights, got {w}"
                )

    def test_mask_empty_thinking_with_explicit_markers(self):
        """mask_empty_thinking works with explicit markers too."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        ids_none, lab_none = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True,
            thinking_markers=None,
        )
        ids_explicit, lab_explicit = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True,
            thinking_markers=("<think>", "</think>"),
        )
        assert ids_none == ids_explicit
        assert lab_none == lab_explicit

    def test_mask_empty_with_wrong_markers_no_masking(self):
        """Wrong markers → empty thinking block not detected → no masking."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        # With correct markers, empty <think>\n\n</think> gets masked
        _, lab_correct = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True,
            thinking_markers=("<think>", "</think>"),
        )
        # With wrong markers, it's unmasked
        _, lab_wrong = tokenize_with_hf_template(
            msgs, tok, mask_empty_thinking=True,
            thinking_markers=("<<FAKE>>", "<</FAKE>>"),
        )
        correct_loss_count = sum(1 for x in lab_correct if x != IGNORE_INDEX)
        wrong_loss_count = sum(1 for x in lab_wrong if x != IGNORE_INDEX)
        assert wrong_loss_count >= correct_loss_count, (
            "Wrong markers should not mask empty thinking → more LOSS tokens"
        )


# ---------------------------------------------------------------------------
# Transformers alignment validation
# ---------------------------------------------------------------------------


class TestTransformersAlignment:
    """Verify titaninfra's tokenize_with_hf_template produces tokens that exactly
    align with HF Transformers apply_chat_template.

    This catches template drift: if HF updates the Jinja template or titaninfra's
    boundary detection diverges, these tests fail.

    Alignment checks:
    1. Token sequence identity: titaninfra's input_ids == HF's tokenized output
    2. Loss mask coverage: LOSS tokens are a subset of assistant content
    3. No phantom tokens: titaninfra doesn't inject extra tokens beyond the template
    """

    @staticmethod
    def _load_qwen3():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    def test_single_turn_token_alignment(self):
        """Single-turn: titaninfra tokens == HF apply_chat_template tokens."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        titaninfra_ids, _ = tokenize_with_hf_template(msgs, tok)
        hf_ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )["input_ids"]

        assert titaninfra_ids == hf_ids, (
            f"Token drift detected!\n"
            f"titaninfra len={len(titaninfra_ids)}, HF len={len(hf_ids)}\n"
            f"First diff at index {next((i for i, (a, b) in enumerate(zip(titaninfra_ids, hf_ids, strict=False)) if a != b), 'N/A')}"
        )

    def test_multi_turn_token_alignment(self):
        """Multi-turn: token sequence stays aligned across turns."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]

        titaninfra_ids, _ = tokenize_with_hf_template(msgs, tok)
        hf_ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )["input_ids"]

        assert titaninfra_ids == hf_ids, (
            f"Multi-turn token drift: titaninfra={len(titaninfra_ids)}, HF={len(hf_ids)}"
        )

    def test_thinking_turn_token_alignment(self):
        """Thinking turn: titaninfra tokens match HF even with <think> blocks."""
        tok = self._load_qwen3()
        # Use reasoning_content field (HF standard for Qwen3 thinking)
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Let me think step by step",
                "content": "The answer is 4.",
            },
        ]

        titaninfra_ids, _ = tokenize_with_hf_template(msgs, tok)
        hf_ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )["input_ids"]

        assert titaninfra_ids == hf_ids, (
            f"Thinking token drift: titaninfra={len(titaninfra_ids)}, HF={len(hf_ids)}"
        )

    def test_system_prompt_alignment(self):
        """System prompt doesn't cause token divergence."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        titaninfra_ids, _ = tokenize_with_hf_template(msgs, tok)
        hf_ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )["input_ids"]

        assert titaninfra_ids == hf_ids, (
            f"System prompt token drift: titaninfra={len(titaninfra_ids)}, HF={len(hf_ids)}"
        )

    def test_loss_mask_only_covers_assistant(self):
        """All LOSS tokens must fall within assistant message boundaries."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        ids, labels = tokenize_with_hf_template(msgs, tok)

        # Decode LOSS region
        loss_ids = [t for t, lb in zip(ids, labels, strict=False) if lb != IGNORE_INDEX]
        loss_text = tok.decode(loss_ids)

        # LOSS text must be a substring of assistant's rendered content
        assert "Python is a programming language." in loss_text, (
            f"Assistant answer missing from LOSS region: {loss_text!r}"
        )
        # User prompt must NOT be in LOSS text
        assert "What is Python?" not in loss_text, (
            f"User prompt leaked into LOSS region: {loss_text!r}"
        )

    def test_multiturn_thinking_alignment_with_hf(self):
        """Multi-turn thinking: verify Jinja asymmetry is consistent with HF.

        This is the critical alignment test: titaninfra must produce the same
        tokens as HF even when Jinja silently drops historical thinking.
        """
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Simple addition",
                "content": "4",
            },
            {"role": "user", "content": "And 3+3?"},
            {
                "role": "assistant",
                "reasoning_content": "Another addition",
                "content": "6",
            },
        ]

        titaninfra_ids, labels = tokenize_with_hf_template(msgs, tok)
        hf_ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )["input_ids"]

        assert titaninfra_ids == hf_ids, (
            f"Multi-turn thinking alignment drift!\n"
            f"titaninfra={len(titaninfra_ids)}, HF={len(hf_ids)}"
        )

        # Verify the asymmetry is reflected: "Simple addition" should be dropped
        full_text = tok.decode(titaninfra_ids)
        assert "Simple addition" not in full_text, (
            "Historical thinking should be dropped by Jinja"
        )
        assert "Another addition" in full_text, (
            "Last turn thinking should be preserved"
        )

    def test_empty_thinking_alignment(self):
        """Non-thinking sample: empty <think> injection matches HF exactly."""
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        titaninfra_ids, _ = tokenize_with_hf_template(msgs, tok)
        hf_ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )["input_ids"]

        assert titaninfra_ids == hf_ids, (
            "Empty thinking injection should match HF template"
        )

        # Verify <think>\n\n</think> is present (Qwen3 auto-injects it)
        full_text = tok.decode(titaninfra_ids)
        assert "<think>" in full_text, "Qwen3 should auto-inject empty <think>"


class TestValidateHfAlignment:
    """Unit tests for the validate_hf_alignment() public function."""

    @staticmethod
    def _load_qwen3():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

    def test_valid_alignment_passes(self):
        """Normal conversation: validate_hf_alignment raises no error."""
        from titaninfra.data.sft.alignment_check import validate_hf_alignment
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        ids, _ = tokenize_with_hf_template(msgs, tok)
        validate_hf_alignment(msgs, ids, tok)

    def test_drift_detected(self):
        """Tampered input_ids: validate_hf_alignment raises AssertionError."""
        from titaninfra.data.sft.alignment_check import validate_hf_alignment
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        ids, _ = tokenize_with_hf_template(msgs, tok)
        tampered = list(ids)
        tampered[3] = 99999
        with pytest.raises(AssertionError, match="Template alignment drift"):
            validate_hf_alignment(msgs, tampered, tok)

    def test_truncation_only_compares_prefix(self):
        """When titaninfra truncates, only the prefix is compared (no false alarm)."""
        from titaninfra.data.sft.alignment_check import validate_hf_alignment
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "Tell me a long story about dragons."},
            {"role": "assistant", "content": "Once upon a time " * 50},
        ]
        ids, _ = tokenize_with_hf_template(msgs, tok, max_length=64)
        validate_hf_alignment(msgs, ids, tok, max_length=64)

    def test_multi_turn_alignment(self):
        """Multi-turn thinking: validates successfully."""
        from titaninfra.data.sft.alignment_check import validate_hf_alignment
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "reasoning_content": "Simple math", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "reasoning_content": "Easy", "content": "6"},
        ]
        ids, _ = tokenize_with_hf_template(msgs, tok)
        validate_hf_alignment(msgs, ids, tok)

    def test_rich_blocks_format(self):
        """Messages with content blocks list (rich format) don't crash."""
        from titaninfra.data.sft.alignment_check import validate_hf_alignment
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Explain gravity"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Physics question"},
                {"type": "text", "text": "Gravity is a fundamental force."},
            ]},
        ]
        ids, _ = tokenize_with_hf_template(msgs, tok)
        validate_hf_alignment(msgs, ids, tok)

    def test_reasoning_content_only(self):
        """Messages with reasoning_content and no content key handled correctly."""
        from titaninfra.data.sft.alignment_check import validate_hf_alignment
        tok = self._load_qwen3()
        msgs = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "reasoning_content": "Simple addition", "content": "2"},
        ]
        ids, _ = tokenize_with_hf_template(msgs, tok)
        validate_hf_alignment(msgs, ids, tok)


# ---------------------------------------------------------------------------
# Gap-filling: boundary fallback path + edge cases
# ---------------------------------------------------------------------------

FIXTURES_DIR = "tests/fixtures/tokenizers"
DEEPSEEK_V3_TOKENIZER = f"{FIXTURES_DIR}/deepseek_v3"
LLAMA3_TOKENIZER = f"{FIXTURES_DIR}/llama3"
LLAMA4_TOKENIZER = f"{FIXTURES_DIR}/llama4"
QWEN2_5_TOKENIZER = f"{FIXTURES_DIR}/qwen2_5"
GEMMA4_TOKENIZER = f"{FIXTURES_DIR}/gemma4"
GPT_OSS_TOKENIZER = f"{FIXTURES_DIR}/gpt_oss"


class TestBoundaryFallbackPath:
    """Tests for _compute_message_boundaries fallback path.

    When a message role has no dedicated marker in the template text
    (e.g., DeepSeek system role → no <｜System｜> marker), the code
    falls back to inheriting the previous boundary position. These tests
    verify that path doesn't corrupt boundaries or create phantom loss.
    """

    def test_deepseek_system_no_marker(self):
        """DeepSeek: system role has no dedicated marker, falls back correctly."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(DEEPSEEK_V3_TOKENIZER, trust_remote_code=True)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        full_ids, boundaries, content_starts = _compute_message_boundaries(msgs, tok)

        assert len(boundaries) == 4
        assert len(content_starts) == 3
        assert boundaries[0] == 0
        assert boundaries[-1] == len(full_ids)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] <= boundaries[i + 1]
        for i in range(len(content_starts)):
            assert boundaries[i] <= content_starts[i] <= len(full_ids)

    def test_deepseek_system_loss_mask_correct(self):
        """DeepSeek system: system tokens must be masked, assistant gets loss."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(DEEPSEEK_V3_TOKENIZER, trust_remote_code=True)
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "The answer is 2."},
        ]

        ids, labels = tokenize_with_hf_template(msgs, tok)

        assert any(lb != IGNORE_INDEX for lb in labels), "Assistant should have loss"

        loss_ids = [t for t, lb in zip(ids, labels, strict=False) if lb != IGNORE_INDEX]
        loss_text = tok.decode(loss_ids)
        assert "The answer is 2." in loss_text
        assert "You are helpful" not in loss_text

    def test_role_without_marker_inherits_boundary(self):
        """Mock: role not in marker_roles gets cp=0 (first msg) or inherits prev."""
        tokenizer = MockChatMLTokenizer()
        msgs = [
            {"role": "developer", "content": "Dev instructions"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        full_ids, boundaries, content_starts = _compute_message_boundaries(
            msgs, tokenizer
        )

        assert len(boundaries) == 4
        assert boundaries[0] == 0
        assert boundaries[-1] == len(full_ids)
        assert content_starts[0] == 0


class TestConsecutiveSameRole:
    """Tests for consecutive messages with the same role.

    This is an edge case that can confuse marker-to-message alignment when
    there are more messages of a role than markers (shouldn't happen with
    well-formed templates, but must not crash).
    """

    def test_two_user_messages_chatml(self):
        """Two consecutive user messages: each gets its own boundary."""
        tokenizer = MockChatMLTokenizer()
        msgs = [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Answer to both"},
        ]

        full_ids, boundaries, content_starts = _compute_message_boundaries(
            msgs, tokenizer
        )

        assert len(boundaries) == 4
        assert len(content_starts) == 3
        assert boundaries[0] < boundaries[1] < boundaries[2]
        assert boundaries[-1] == len(full_ids)

    def test_consecutive_assistant_messages_loss(self):
        """Two consecutive assistant messages: both should have loss by default."""
        tokenizer = MockChatMLTokenizer()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "First reply"},
            {"role": "assistant", "content": "Second reply"},
        ]

        ids, labels = tokenize_with_hf_template(msgs, tokenizer)

        loss_positions = [i for i, lb in enumerate(labels) if lb != IGNORE_INDEX]
        assert len(loss_positions) > 0

    def test_consecutive_assistant_with_real_tokenizer(self):
        """Real Qwen3: consecutive assistant messages boundary alignment."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "assistant", "content": "How can I help?"},
        ]

        full_ids, boundaries, content_starts = _compute_message_boundaries(msgs, tok)

        assert len(boundaries) == 4
        assert boundaries[-1] == len(full_ids)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] <= boundaries[i + 1]


class TestMultiTokenizerTransformersAlignment:
    """Extend TransformersAlignment beyond Qwen3 to DeepSeek, Llama3, Llama4, Gemma4.

    Verifies that titaninfra's input_ids == HF apply_chat_template tokens across
    multiple tokenizer families, catching pattern-specific drift.
    """

    _TOKENIZER_PARAMS = [
        pytest.param(QWEN3_TOKENIZER, id="qwen3"),
        pytest.param(DEEPSEEK_V3_TOKENIZER, id="deepseek_v3"),
        pytest.param(LLAMA3_TOKENIZER, id="llama3"),
        pytest.param(LLAMA4_TOKENIZER, id="llama4"),
        pytest.param(QWEN2_5_TOKENIZER, id="qwen2_5"),
        pytest.param(GEMMA4_TOKENIZER, id="gemma4"),
        pytest.param(GPT_OSS_TOKENIZER, id="gpt_oss"),
    ]

    @staticmethod
    def _load(path):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    @pytest.mark.parametrize("tok_path", _TOKENIZER_PARAMS)
    def test_single_turn_alignment(self, tok_path):
        """Single turn: titaninfra tokens == HF tokens."""
        tok = self._load(tok_path)
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        titaninfra_ids, _ = tokenize_with_hf_template(msgs, tok)
        hf_result = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )
        hf_ids = hf_result["input_ids"] if not isinstance(hf_result, list) else hf_result

        assert titaninfra_ids == hf_ids, (
            f"Token drift on {tok_path}!\n"
            f"titaninfra len={len(titaninfra_ids)}, HF len={len(hf_ids)}\n"
            f"First diff at {next((i for i, (a, b) in enumerate(zip(titaninfra_ids, hf_ids, strict=False)) if a != b), 'N/A')}"
        )

    @pytest.mark.parametrize("tok_path", _TOKENIZER_PARAMS)
    def test_multi_turn_alignment(self, tok_path):
        """Multi-turn: token sequence stays aligned across turns."""
        tok = self._load(tok_path)
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]

        titaninfra_ids, _ = tokenize_with_hf_template(msgs, tok)
        hf_result = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )
        hf_ids = hf_result["input_ids"] if not isinstance(hf_result, list) else hf_result

        assert titaninfra_ids == hf_ids, (
            f"Multi-turn drift on {tok_path}: "
            f"titaninfra={len(titaninfra_ids)}, HF={len(hf_ids)}"
        )

    @pytest.mark.parametrize("tok_path", [
        pytest.param(DEEPSEEK_V3_TOKENIZER, id="deepseek_v3"),
        pytest.param(LLAMA3_TOKENIZER, id="llama3"),
        pytest.param(LLAMA4_TOKENIZER, id="llama4"),
    ])
    def test_system_prompt_alignment(self, tok_path):
        """System prompt doesn't cause token divergence.

        Note: Gemma4 excluded — its template rejects system role.
        """
        tok = self._load(tok_path)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        titaninfra_ids, _ = tokenize_with_hf_template(msgs, tok)
        hf_result = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )
        hf_ids = hf_result["input_ids"] if not isinstance(hf_result, list) else hf_result

        assert titaninfra_ids == hf_ids, (
            f"System prompt drift on {tok_path}: "
            f"titaninfra={len(titaninfra_ids)}, HF={len(hf_ids)}"
        )

    @pytest.mark.parametrize("tok_path", _TOKENIZER_PARAMS)
    def test_loss_mask_only_covers_assistant(self, tok_path):
        """All LOSS tokens fall within assistant content, not user/system."""
        tok = self._load(tok_path)
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        ids, labels = tokenize_with_hf_template(msgs, tok)

        loss_ids = [t for t, lb in zip(ids, labels, strict=False) if lb != IGNORE_INDEX]
        loss_text = tok.decode(loss_ids)

        assert "programming language" in loss_text, (
            f"Assistant content missing from loss on {tok_path}: {loss_text!r}"
        )
        assert "What is Python" not in loss_text, (
            f"User prompt leaked into loss on {tok_path}: {loss_text!r}"
        )
