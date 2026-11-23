"""Unit tests for template registry."""

from pathlib import Path

import pytest

from titaninfra.data.template_registry import (
    DEEPSEEK_V3,
    GEMMA4,
    GPT_OSS,
    LLAMA4,
    QWEN3,
    BlockTag,
    RoleTag,
    TemplateConfig,
    get_template,
    list_templates,
    register_template,
)

TOKENIZERS_DIR = Path(__file__).parent / "fixtures" / "tokenizers"


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestGetTemplate:

    def test_qwen3(self):
        t = get_template("qwen3")
        assert t.name == "qwen3"

    def test_deepseek_v3(self):
        t = get_template("deepseek_v3")
        assert t.name == "deepseek_v3"

    def test_llama4(self):
        t = get_template("llama4")
        assert t.name == "llama4"

    def test_case_insensitive(self):
        assert get_template("Qwen3").name == "qwen3"
        assert get_template("LLAMA4").name == "llama4"

    def test_alias_qwen(self):
        assert get_template("qwen").name == "qwen3"
        assert get_template("chatml").name == "qwen3"

    def test_alias_deepseek(self):
        assert get_template("deepseek").name == "deepseek_v3"
        assert get_template("deepseek_r1").name == "deepseek_v3"

    def test_alias_llama(self):
        assert get_template("llama").name == "llama4"
        assert get_template("llama3").name == "llama4"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="未知模板"):
            get_template("nonexistent_model")

    def test_list_templates(self):
        names = list_templates()
        assert "qwen3" in names
        assert "deepseek_v3" in names
        assert "llama4" in names
        assert "chatml" in names  # alias


class TestRegisterTemplate:

    def test_register_custom(self):
        custom = TemplateConfig(
            name="test_custom",
            roles={"user": RoleTag("[U]", "[/U]"), "assistant": RoleTag("[A]", "[/A]")},
            blocks={"text": BlockTag()},
        )
        register_template(custom)
        assert get_template("test_custom").name == "test_custom"


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestFrozen:

    def test_template_config_frozen(self):
        t = get_template("qwen3")
        with pytest.raises(AttributeError):
            t.name = "hacked"

    def test_role_tag_frozen(self):
        r = RoleTag(start="a", end="b")
        with pytest.raises(AttributeError):
            r.start = "hacked"

    def test_block_tag_frozen(self):
        b = BlockTag(start="a", end="b")
        with pytest.raises(AttributeError):
            b.start = "hacked"


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestQwen3Template:

    def test_roles_complete(self):
        t = QWEN3
        assert set(t.roles.keys()) >= {"system", "user", "assistant", "tool"}

    def test_assistant_role(self):
        r = QWEN3.roles["assistant"]
        assert r.start == "<|im_start|>assistant\n"
        assert r.end == "<|im_end|>\n"

    def test_thinking_block(self):
        b = QWEN3.blocks["thinking"]
        assert b.start == "<think>\n"
        assert "</think>" in b.end

    def test_tool_use_block(self):
        b = QWEN3.blocks["tool_use"]
        assert "<tool_call>" in b.start

    def test_tool_role_wraps_response(self):
        r = QWEN3.roles["tool"]
        assert "<tool_response>" in r.start
        assert "</tool_response>" in r.end


class TestDeepSeekV3Template:

    def test_assistant_role(self):
        r = DEEPSEEK_V3.roles["assistant"]
        assert "Assistant" in r.start
        assert "end▁of▁sentence" in r.end

    def test_bos(self):
        assert "begin▁of▁sentence" in DEEPSEEK_V3.bos

    def test_tool_role(self):
        r = DEEPSEEK_V3.roles["tool"]
        assert r.start == ""


class TestLlama4Template:

    def test_assistant_role(self):
        r = LLAMA4.roles["assistant"]
        assert "<|header_start|>assistant<|header_end|>" in r.start
        assert r.end == "<|eot|>"

    def test_bos(self):
        assert LLAMA4.bos == "<|begin_of_text|>"

    def test_tool_role_is_ipython(self):
        r = LLAMA4.roles["tool"]
        assert "ipython" in r.start


class TestGemma4Template:

    def test_assistant_role(self):
        r = GEMMA4.roles["assistant"]
        assert "<|turn>model" in r.start
        assert r.end == "<turn|>\n"

    def test_eos(self):
        assert GEMMA4.eos == "<turn|>"

    def test_thinking_block(self):
        bt = GEMMA4.blocks["thinking"]
        assert "thought" in bt.start
        assert "<channel|>" in bt.end

    def test_tool_use_block(self):
        bt = GEMMA4.blocks["tool_use"]
        assert "<|tool_call>" in bt.start
        assert "<tool_call|>" in bt.end


class TestGptOssTemplate:

    def test_assistant_role_uses_final_channel(self):
        r = GPT_OSS.roles["assistant"]
        assert "<|channel|>final" in r.start
        assert r.end == "<|end|>"

    def test_eos(self):
        assert GPT_OSS.eos == "<|end|>"

    def test_thinking_uses_analysis_channel(self):
        bt = GPT_OSS.blocks["thinking"]
        assert "<|channel|>analysis" in bt.start
        assert bt.end == "<|end|>"

    def test_tool_role_renders_as_user(self):
        r = GPT_OSS.roles["tool"]
        assert "<|start|>user" in r.start


# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestCrossValidateWithTokenizer:
    """Tests for test cross validate with tokenizer."""

    @pytest.fixture(params=["qwen3", "qwen2_5", "deepseek_v3", "llama4", "gemma4", "gpt_oss"])
    def model_name(self, request):
        return request.param

    @pytest.fixture
    def tokenizer(self, model_name):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            str(TOKENIZERS_DIR / model_name), trust_remote_code=True
        )

    @pytest.fixture
    def template(self, model_name):
        return get_template(model_name)

    def test_role_start_encodable(self, tokenizer, template):
        """Verify role start encodable."""
        for role_name, role_tag in template.roles.items():
            if role_tag.start:
                ids = tokenizer.encode(role_tag.start, add_special_tokens=False)
                assert len(ids) > 0, f"{template.name} role={role_name} start encode 为空"

    def test_role_end_encodable(self, tokenizer, template):
        """Verify role end encodable."""
        for role_name, role_tag in template.roles.items():
            if role_tag.end:
                ids = tokenizer.encode(role_tag.end, add_special_tokens=False)
                assert len(ids) > 0, f"{template.name} role={role_name} end encode 为空"

    def test_thinking_tag_roundtrip(self, tokenizer, template):
        """Verify thinking tag roundtrip."""
        bt = template.blocks.get("thinking")
        if bt and bt.start:
            ids = tokenizer.encode(bt.start, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            assert bt.start.strip() in decoded.strip()

    def test_basic_message_structure(self, tokenizer, template):
        """Verify basic message structure."""
        text = (
            template.bos
            + template.roles["user"].start + "hello" + template.roles["user"].end
            + template.roles["assistant"].start + "hi" + template.roles["assistant"].end
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert "hello" in decoded
        assert "hi" in decoded
