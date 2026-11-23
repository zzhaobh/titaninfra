"""Unit tests for templates."""

import pytest

from titaninfra.data.templates import (
    IGNORE_INDEX,
    ChatTemplate,
    Role,
    apply_template,
    get_template,
    infer_template_from_model_name,
    list_templates,
    register_template,
    remove_thought,
)


class TestTemplateRegistry:
    def test_get_known_templates(self):
        """Verify get known templates."""
        for name in ["qwen3", "llama3", "llama2", "mistral", "chatml", "plain"]:
            t = get_template(name)
            assert t.name is not None

    def test_case_insensitive(self):
        """Verify case insensitive."""
        t1 = get_template("qwen3")
        t2 = get_template("QWEN3")
        t3 = get_template("Qwen3")
        assert t1.name == t2.name == t3.name

    def test_aliases(self):
        """Verify aliases."""
        assert get_template("qwen").name == "qwen3"
        assert get_template("qwen2").name == "qwen2_5"
        assert get_template("llama").name == "llama3"
        assert get_template("r1").name == "deepseek_r1"

    def test_unknown_template_raises(self):
        """Verify unknown template raises."""
        with pytest.raises(ValueError, match="未知模板"):
            get_template("totally_unknown_template_xyz")

    def test_list_templates_includes_builtins(self):
        """Verify list templates includes builtins."""
        available = list_templates()
        for name in ["qwen3", "llama3", "mistral", "chatml"]:
            assert name in available

    def test_register_custom_template(self):
        """Verify register custom template."""
        custom = ChatTemplate(
            name="test_custom_xyz",
            system_role=Role(prefix="[SYS]", suffix="[/SYS]\n"),
            user_role=Role(prefix="[USER]", suffix="[/USER]\n"),
            assistant_role=Role(prefix="[ASST]", suffix="[/ASST]\n"),
        )
        register_template(custom)
        retrieved = get_template("test_custom_xyz")
        assert retrieved.name == "test_custom_xyz"


class TestApplyTemplate:
    """Tests for test apply template."""

    def setup_method(self):
        self.template = get_template("chatml")
        self.messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_basic_formatting(self):
        """Verify basic formatting."""
        turns = apply_template(self.template, self.messages)
        assert len(turns) > 0
        for text, is_loss in turns:
            assert isinstance(text, str)
            assert isinstance(is_loss, bool)

    def test_assistant_content_has_loss(self):
        """Verify assistant content has loss."""
        turns = apply_template(self.template, self.messages)
        loss_text = "".join(t for t, il in turns if il)
        assert "Hi there!" in loss_text

    def test_user_content_no_loss(self):
        """Verify user content no loss."""
        turns = apply_template(self.template, self.messages)
        no_loss_text = "".join(t for t, il in turns if not il)
        assert "Hello" in no_loss_text

    def test_system_content_no_loss(self):
        """Verify system content no loss."""
        turns = apply_template(self.template, self.messages)
        no_loss_text = "".join(t for t, il in turns if not il)
        assert "You are helpful." in no_loss_text

    def test_generation_prompt(self):
        """Verify generation prompt."""
        turns_no_gen = apply_template(self.template, self.messages, add_generation_prompt=False)
        turns_with_gen = apply_template(self.template, self.messages, add_generation_prompt=True)
        assert len(turns_with_gen) > len(turns_no_gen)

    def test_default_system_injected(self):
        """Verify default system injected."""
        messages_no_sys = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        template = get_template("qwen3")
        turns = apply_template(template, messages_no_sys)
        all_text = "".join(t for t, _ in turns)
        assert "Qwen" in all_text

    def test_plain_template(self):
        """Verify plain template."""
        template = get_template("plain")
        messages = [{"role": "assistant", "content": "hello world"}]
        turns = apply_template(template, messages)
        loss_text = "".join(t for t, il in turns if il)
        assert "hello world" in loss_text

    def test_multi_turn_conversation(self):
        """Verify multi turn conversation."""
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        turns = apply_template(self.template, messages)
        loss_text = "".join(t for t, il in turns if il)
        assert "A1" in loss_text
        assert "A2" in loss_text


class TestInferTemplate:
    """Tests for test infer template."""

    def test_infer_qwen3(self):
        t = infer_template_from_model_name("qwen3/8B")
        assert t is not None
        assert t.name == "qwen3"

    def test_infer_llama3(self):
        t = infer_template_from_model_name("llama3/8B")
        assert t is not None
        assert t.name == "llama3"

    def test_infer_unknown_returns_none(self):
        t = infer_template_from_model_name("unknown_family/8B")
        assert t is None

    def test_infer_empty_returns_none(self):
        t = infer_template_from_model_name("")
        assert t is None


class TestEncodeWithTemplate:
    """Tests for test encode with template."""

    def test_encode_produces_correct_labels(self):
        """Verify encode produces correct labels."""
        pytest.importorskip("transformers")
        from unittest.mock import MagicMock


        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text.split())))

        from titaninfra.data.templates import encode_with_template
        template = get_template("chatml")
        messages = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
        ]

        input_ids, labels = encode_with_template(tokenizer, template, messages)
        assert len(input_ids) == len(labels)
        assert IGNORE_INDEX in labels
        assert any(lbl != IGNORE_INDEX for lbl in labels)

    def test_train_on_inputs_all_labels(self):
        """Verify train on inputs all labels."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]

        from titaninfra.data.templates import encode_with_template
        template = get_template("chatml")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        input_ids, labels = encode_with_template(
            tokenizer, template, messages, train_on_inputs=True
        )
        assert len(input_ids) == len(labels)
        assert IGNORE_INDEX not in labels


class TestReasoningTemplate:
    """Tests for test reasoning template."""

    def test_remove_thought_basic(self):
        """Verify remove thought basic."""
        content = "<think>\n这是思考过程\n</think>\n\n这是最终答案"
        thought_words = ("<think>\n", "\n</think>\n\n")
        result = remove_thought(content, thought_words)
        assert result == "这是最终答案"

    def test_remove_thought_empty(self):
        """Verify remove thought empty."""
        content = "这是答案"
        thought_words = ("<think>\n", "\n</think>\n\n")
        result = remove_thought(content, thought_words)
        assert result == "这是答案"

    def test_remove_thought_multiple(self):
        """Verify remove thought multiple."""
        content = "<think>\n思考1\n</think>\n\n文本1<think>\n思考2\n</think>\n\n文本2"
        thought_words = ("<think>\n", "\n</think>\n\n")
        result = remove_thought(content, thought_words)
        assert result == "文本1文本2"

    def test_get_qwen3_reasoning_template(self):
        """Verify get qwen3 reasoning template."""
        template = get_template("qwen3_reasoning")
        assert template.name == "qwen3"
        assert template.thought_words is not None
        assert template.enable_thinking is True

    def test_get_deepseek_r1_reasoning_template(self):
        """Verify get deepseek r1 reasoning template."""
        template = get_template("deepseek_r1_reasoning")
        assert template.name == "deepseek_r1_reasoning"
        assert template.thought_words is not None
        assert template.enable_thinking is True

    def test_reasoning_template_aliases(self):
        """Verify reasoning template aliases."""
        assert get_template("qwen3_thinking").name == "qwen3"
        assert get_template("deepseek_r1_thinking").name == "deepseek_r1_reasoning"
        assert get_template("r1_thinking").name == "deepseek_r1_reasoning"

    def test_reasoning_with_thought_in_content(self):
        """Verify reasoning with thought in content."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "<think>\n计算1+1\n</think>\n\n答案是2"},
        ]
        turns = apply_template(template, messages, enable_thinking=True)
        loss_text = "".join(t for t, il in turns if il)
        assert "计算1+1" in loss_text

    def test_reasoning_without_thought_in_loss(self):
        """Verify reasoning without thought in loss."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "<think>\n计算1+1\n</think>\n\n答案是2"},
        ]
        turns = apply_template(template, messages, enable_thinking=False)
        loss_text = "".join(t for t, il in turns if il)
        assert "计算1+1" not in loss_text
        assert "答案是2" in loss_text

    def test_reasoning_history_thought_removal(self):
        """Verify reasoning history thought removal."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "<think>\n第一次思考\n</think>\n\n答案是2"},
            {"role": "user", "content": "2+2=?"},
            {"role": "assistant", "content": "<think>\n第二次思考\n</think>\n\n答案是4"},
        ]
        turns = apply_template(template, messages, enable_thinking=True)

        all_text = "".join(t for t, _ in turns)
        assert "第一次思考" not in all_text
        assert "第二次思考" in all_text

    def test_custom_thought_words(self):
        """Verify custom thought words."""
        custom_template = ChatTemplate(
            name="custom_reasoning",
            system_role=Role(prefix="[SYS]", suffix="[/SYS]\n"),
            user_role=Role(prefix="[USER]", suffix="[/USER]\n"),
            assistant_role=Role(prefix="[ASST]", suffix="[/ASST]\n"),
            thought_words=("<think>", "</think>"),
            enable_thinking=True,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<think>思考</think>回答"},
        ]
        turns = apply_template(custom_template, messages, enable_thinking=True)
        loss_text = "".join(t for t, il in turns if il)
        assert "思考" in loss_text
        assert "回答" in loss_text

    def test_no_thought_field_no_empty_tags(self):
        """Verify no thought field no empty tags."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "答案是2"},
        ]
        turns = apply_template(template, messages, enable_thinking=True)

        loss_text = "".join(t for t, il in turns if il)
        assert "答案是2" in loss_text

        start_tag, end_tag = template.thought_words
        all_text = "".join(t for t, _ in turns)
        assert start_tag.strip() not in all_text
        assert end_tag.strip() not in all_text

    def test_multi_turn_no_thought(self):
        """Verify multi turn no thought."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "答案是2"},
            {"role": "user", "content": "2+2=?"},
            {"role": "assistant", "content": "答案是4"},
        ]
        turns = apply_template(template, messages, enable_thinking=True)

        loss_text = "".join(t for t, il in turns if il)
        assert "答案是2" in loss_text
        assert "答案是4" in loss_text

        start_tag, end_tag = template.thought_words
        all_text = "".join(t for t, _ in turns)
        assert start_tag.strip() not in all_text
        assert end_tag.strip() not in all_text


class TestThinkingFieldHandling:
    """Tests for test thinking field handling."""

    def test_independent_think_field_with_value(self):
        """Verify independent think field with value."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "答案是2", "think": "计算1+1"},
        ]
        turns = apply_template(template, messages, enable_thinking=True)

        loss_text = "".join(t for t, il in turns if il)
        assert "计算1+1" in loss_text
        assert "答案是2" in loss_text

    def test_independent_think_field_empty_string(self):
        """Verify independent think field empty string."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "答案是2", "think": ""},
        ]
        turns = apply_template(template, messages, enable_thinking=True)

        loss_text = "".join(t for t, il in turns if il)
        assert "答案是2" in loss_text

        start_tag, end_tag = template.thought_words
        all_text = "".join(t for t, _ in turns)
        assert start_tag.strip() not in all_text

    def test_independent_think_field_none(self):
        """Verify independent think field none."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "答案是2", "think": None},
        ]
        turns = apply_template(template, messages, enable_thinking=True)

        loss_text = "".join(t for t, il in turns if il)
        assert "答案是2" in loss_text

        start_tag, end_tag = template.thought_words
        all_text = "".join(t for t, _ in turns)
        assert start_tag.strip() not in all_text

    def test_mixed_think_no_think_data(self):
        """Verify mixed think no think data."""
        template = get_template("qwen3_reasoning")

        messages1 = [
            {"role": "user", "content": "伦敦奥运会"},
            {"role": "assistant", "content": "伦敦奥运会时间线..."},
        ]
        turns1 = apply_template(template, messages1, enable_thinking=True)
        loss_text1 = "".join(t for t, il in turns1 if il)
        assert "伦敦奥运会时间线" in loss_text1

        messages2 = [
            {"role": "user", "content": "What is the source..."},
            {"role": "assistant", "content": "The factor of 1/2...", "think": "We are to identify..."},
        ]
        turns2 = apply_template(template, messages2, enable_thinking=True)
        loss_text2 = "".join(t for t, il in turns2 if il)
        assert "We are to identify" in loss_text2
        assert "The factor of 1/2" in loss_text2

        messages3 = [
            {"role": "user", "content": "Another question"},
            {"role": "assistant", "content": "Another answer...", "think": None},
        ]
        turns3 = apply_template(template, messages3, enable_thinking=True)
        loss_text3 = "".join(t for t, il in turns3 if il)
        assert "Another answer" in loss_text3

    def test_think_field_priority_over_content_tags(self):
        """Verify think field priority over content tags."""
        template = get_template("qwen3_reasoning")

        messages = [
            {"role": "user", "content": "1+1=?"},
            {
                "role": "assistant",
                "content": "<tool_call>\\ncontent中的思考\\n}\\n\\n答案是2",
                "think": "独立think字段",
            },
        ]
        turns = apply_template(template, messages, enable_thinking=True)

        loss_text = "".join(t for t, il in turns if il)
        assert "独立think字段" in loss_text
        assert "content中的思考" in loss_text
        assert "答案是2" in loss_text

    def test_enable_thinking_false_with_think_field(self):
        """Verify enable thinking false with think field."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "答案是2", "think": "计算1+1"},
        ]
        turns = apply_template(template, messages, enable_thinking=False)

        loss_text = "".join(t for t, il in turns if il)
        assert "计算1+1" not in loss_text
        assert "答案是2" in loss_text

    def test_multi_turn_with_mixed_think_fields(self):
        """Verify multi turn with mixed think fields."""
        template = get_template("qwen3_reasoning")
        messages = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "答案是2"},
            {"role": "user", "content": "2+2=?"},
            {"role": "assistant", "content": "答案是4", "think": "计算2+2"},
        ]
        turns = apply_template(template, messages, enable_thinking=True)

        loss_text = "".join(t for t, il in turns if il)
        assert "答案是2" in loss_text
        assert "计算2+2" in loss_text
        assert "答案是4" in loss_text


class TestValidateMessages:
    """Tests for test validate messages."""

    def test_valid_messages(self):
        """Verify valid messages."""
        from titaninfra.data.templates import validate_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = validate_messages(messages)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_empty_messages(self):
        """Verify empty messages."""
        from titaninfra.data.templates import validate_messages

        result = validate_messages([])

        assert result.is_valid is False
        assert "messages 为空" in result.errors[0]

    def test_missing_user_message(self):
        """Verify missing user message."""
        from titaninfra.data.templates import validate_messages

        messages = [
            {"role": "assistant", "content": "Hi!"},
        ]
        result = validate_messages(messages)

        assert result.is_valid is False
        assert any("user" in err.lower() for err in result.errors)

    def test_missing_assistant_message(self):
        """Verify missing assistant message."""
        from titaninfra.data.templates import validate_messages

        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = validate_messages(messages)

        assert result.is_valid is False
        assert any("assistant" in err.lower() for err in result.errors)

    def test_missing_role_field(self):
        """Verify missing role field."""
        from titaninfra.data.templates import validate_messages

        messages = [
            {"content": "Hello"},
        ]
        result = validate_messages(messages)

        assert result.is_valid is False
        assert any("role" in err for err in result.errors)


class TestToolsAndImagesHandling:
    """Tests for test tools and images handling."""

    def test_tool_message_with_tool_role(self):
        """Verify tool message with tool role."""
        from titaninfra.data.templates import ChatTemplate, Role, apply_template

        template = ChatTemplate(
            name="test_tool",
            system_role=Role(prefix="[SYS]", suffix="[/SYS]\n"),
            user_role=Role(prefix="[USER]", suffix="[/USER]\n"),
            assistant_role=Role(prefix="[ASST]", suffix="[/ASST]\n"),
            tool_role=Role(prefix="[TOOL]", suffix="[/TOOL]\n"),
        )

        messages = [
            {"role": "user", "content": "调用工具"},
            {"role": "assistant", "content": "好的"},
            {"role": "tool", "content": "工具返回结果"},
        ]
        turns = apply_template(template, messages)

        all_text = "".join(t for t, _ in turns)
        assert "[TOOL]" in all_text
        assert "工具返回结果" in all_text

    def test_tool_message_fallback_to_user_role(self):
        """Verify tool message fallback to user role."""
        template = get_template("chatml")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "tool", "content": "Tool result"},
        ]
        turns = apply_template(template, messages)

        all_text = "".join(t for t, _ in turns)
        assert "Tool result" in all_text

    def test_user_message_with_images(self):
        """Verify user message with images."""
        from titaninfra.data.templates import ChatTemplate, Role, apply_template

        template = ChatTemplate(
            name="test_image",
            system_role=Role(prefix="[SYS]", suffix="[/SYS]\n"),
            user_role=Role(prefix="[USER]", suffix="[/USER]\n"),
            assistant_role=Role(prefix="[ASST]", suffix="[/ASST]\n"),
            image_token="<IMAGE>",
        )

        messages = [
            {"role": "user", "content": "描述这张图", "images": ["img1.jpg", "img2.jpg"]},
            {"role": "assistant", "content": "这是一张..."},
        ]
        turns = apply_template(template, messages)

        all_text = "".join(t for t, _ in turns)
        assert "<IMAGE><IMAGE>" in all_text
        assert "描述这张图" in all_text

    def test_assistant_message_with_tool_calls(self):
        """Verify assistant message with tool calls."""
        from titaninfra.data.templates import ChatTemplate, Role, apply_template

        template = ChatTemplate(
            name="test_tool_call",
            system_role=Role(prefix="[SYS]", suffix="[/SYS]\n"),
            user_role=Role(prefix="[USER]", suffix="[/USER]\n"),
            assistant_role=Role(prefix="[ASST]", suffix="[/ASST]\n"),
            tool_call_prefix="[TOOL_CALLS]",
        )

        messages = [
            {"role": "user", "content": "调用工具"},
            {
                "role": "assistant",
                "content": "好的",
                "tool_calls": [{"name": "weather", "args": {"city": "Beijing"}}],
            },
        ]
        turns = apply_template(template, messages)

        all_text = "".join(t for t, _ in turns)
        assert "[TOOL_CALLS]" in all_text
        assert "weather" in all_text


class TestDebugMode:
    """Tests for test debug mode."""

    def test_encode_with_template_debug_true(self, capsys):
        """Verify encode with template debug true."""
        from unittest.mock import MagicMock

        from titaninfra.data.templates import encode_with_template, get_template

        tokenizer = MagicMock()
        tokenizer.encode = lambda text, **kw: [ord(c) % 100 for c in text[:20]]
        tokenizer.decode = lambda ids, **kw: f"decoded_{len(ids)}"
        tokenizer.pad_token_id = 0

        template = get_template("chatml")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        input_ids, labels = encode_with_template(
            tokenizer=tokenizer,
            template=template,
            messages=messages,
            max_length=100,
            debug=True,
        )

        captured = capsys.readouterr()

        assert "[DEBUG]" in captured.out
        assert "[配置]" in captured.out
        assert "[原始消息]" in captured.out
        assert "[模板处理后]" in captured.out
        assert "[Tokenize 结果]" in captured.out

    def test_encode_with_template_debug_false(self, capsys):
        """Verify encode with template debug false."""
        from unittest.mock import MagicMock

        from titaninfra.data.templates import encode_with_template, get_template

        tokenizer = MagicMock()
        tokenizer.encode = lambda text, **kw: [ord(c) % 100 for c in text[:20]]
        tokenizer.decode = lambda ids, **kw: f"decoded_{len(ids)}"
        tokenizer.pad_token_id = 0

        template = get_template("chatml")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        input_ids, labels = encode_with_template(
            tokenizer=tokenizer,
            template=template,
            messages=messages,
            max_length=100,
            debug=False,
        )

        captured = capsys.readouterr()

        assert "[DEBUG]" not in captured.out

    def test_tokenize_conversation_debug_parameter(self, capsys):
        """Verify tokenize conversation debug parameter."""
        from unittest.mock import MagicMock

        from titaninfra.data.sft_utils import tokenize_conversation
        from titaninfra.data.templates import get_template

        tokenizer = MagicMock()
        tokenizer.encode = lambda text, **kw: [ord(c) % 100 for c in text[:20]]
        tokenizer.decode = lambda ids, **kw: f"decoded_{len(ids)}"
        tokenizer.pad_token_id = 0

        template = get_template("chatml")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        input_ids, labels = tokenize_conversation(
            messages=messages,
            tokenizer=tokenizer,
            template=template,
            debug=True,
        )

        captured = capsys.readouterr()

        assert "[DEBUG]" in captured.out
