"""Unit tests for refactored modules."""

import pytest

from tests.conftest import make_test_train_config


class TestTemplatesModuleRefactoring:
    """Tests for test templates module refactoring."""

    def test_import_from_new_module(self):
        """Verify import from new module."""
        from titaninfra.data.templates import (
            IGNORE_INDEX,
            apply_template,
            encode_with_template,
            get_template,
            infer_template_from_model_name,
            list_templates,
            register_template,
            remove_thought,
            validate_messages,
        )

        assert callable(get_template)
        assert callable(apply_template)
        assert callable(encode_with_template)
        assert callable(validate_messages)
        assert callable(remove_thought)
        assert callable(infer_template_from_model_name)
        assert callable(list_templates)
        assert callable(register_template)
        assert IGNORE_INDEX == -100

    def test_module_structure(self):
        """Verify module structure."""
        import titaninfra.data.templates as templates

        assert "__init__.py" in templates.__file__

        from titaninfra.data.templates import definition, formatters, inference, registry, thinking, validation

        assert hasattr(definition, "ChatTemplate")
        assert hasattr(registry, "get_template")
        assert hasattr(formatters, "apply_template")
        assert hasattr(thinking, "remove_thought")
        assert hasattr(validation, "validate_messages")
        assert hasattr(inference, "infer_template_from_model_name")

    def test_get_template_works(self):
        """Verify get template works."""
        from titaninfra.data.templates import get_template

        template = get_template("qwen3")
        assert template is not None
        assert template.name == "qwen3"

    def test_apply_template_works(self):
        """Verify apply template works."""
        from titaninfra.data.templates import apply_template, get_template

        template = get_template("qwen3")
        messages = [
            {"role": "user", "content": "测试问题"},
            {"role": "assistant", "content": "测试回答"},
        ]

        turns = apply_template(template, messages)
        assert len(turns) > 0
        assert all(isinstance(turn, tuple) and len(turn) == 2 for turn in turns)


class TestSFTModuleRefactoring:
    """Tests for test sftmodule refactoring."""

    def test_module_structure(self):
        """Verify module structure."""
        import titaninfra.data.sft as sft

        assert "__init__.py" in sft.__file__


class TestConfigLayerRefactoring:
    """Tests for test config layer refactoring."""

    def test_train_config(self):
        """Verify train config."""

        config = make_test_train_config()
        assert hasattr(config, "local_batch_size")
        assert hasattr(config, "seq_len")
        assert hasattr(config, "neftune_noise_alpha")

    def test_stream_config(self):
        """Verify stream config."""
        from titaninfra.config import StreamConfig

        config = StreamConfig(tokenizer_name="qwen3", local="/data/train")
        assert hasattr(config, "local")
        assert hasattr(config, "data_type")
        assert hasattr(config, "proportion")
        assert hasattr(config, "family")
        assert hasattr(config, "pack_strategy")


class TestThinkingStrategy:
    """Tests for test thinking strategy."""

    def test_strategy_enum(self):
        """Verify strategy enum."""
        from titaninfra.data.templates.thinking import ThinkingStrategy

        assert hasattr(ThinkingStrategy, "THINK_FIELD")
        assert hasattr(ThinkingStrategy, "CONTENT_TAG")
        assert hasattr(ThinkingStrategy, "NO_THOUGHT")

    def test_strategy_detection(self):
        """Verify strategy detection."""
        from titaninfra.data.templates import get_template
        from titaninfra.data.templates.thinking import ThinkingStrategy, _detect_thinking_strategy

        template = get_template("qwen3_reasoning")

        strategy = _detect_thinking_strategy(template, "答案", "思考过程")
        assert strategy == ThinkingStrategy.THINK_FIELD

        strategy = _detect_thinking_strategy(template, "答案", None)
        assert strategy == ThinkingStrategy.NO_THOUGHT


class TestAPIDocumentation:
    """Tests for test apidocumentation."""

    def test_templates_module_docstring(self):
        """Verify templates module docstring."""
        import titaninfra.data.templates as templates

        assert templates.__doc__ is not None
        assert "get_template" in templates.__doc__
        assert "apply_template" in templates.__doc__
        assert "encode_with_template" in templates.__doc__
        assert "apply_template" in templates.__doc__

    def test_all_exports_documented(self):
        """Verify all exports documented."""
        from titaninfra.data.templates import __all__

        assert __all__ is not None
        assert len(__all__) > 0

        assert "get_template" in __all__
        assert "apply_template" in __all__
        assert "encode_with_template" in __all__
        assert "validate_messages" in __all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
