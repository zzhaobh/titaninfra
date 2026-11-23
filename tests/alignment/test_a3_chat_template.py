"""Unit tests for a3 chat template."""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

TOKENIZERS_DIR = Path(__file__).parent.parent / "fixtures" / "tokenizers"


def _load_tokenizer(model_name: str = "qwen3"):
    return AutoTokenizer.from_pretrained(
        str(TOKENIZERS_DIR / model_name), trust_remote_code=True
    )



SINGLE_TURN = [
    {"role": "user", "content": "计算 2+2"},
    {"role": "assistant", "content": "2+2 的结果是 4。"},
]

MULTI_TURN = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    {"role": "user", "content": "帮我写一个 Python hello world"},
    {"role": "assistant", "content": 'print("Hello, world!")'},
]

WITH_SYSTEM = [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是AI助手。"},
]

LONG_RESPONSE = [
    {"role": "user", "content": "写一首关于春天的诗"},
    {"role": "assistant", "content": (
        "春风拂面花满枝，\n"
        "燕舞莺歌柳色新。\n"
        "碧水东流云卷舒，\n"
        "万物生长共此春。\n"
        "\n"
        "春天是一年中最美好的季节，万物复苏，生机勃勃。"
    )},
]

CODE_RESPONSE = [
    {"role": "user", "content": "Write a fibonacci function in Python"},
    {"role": "assistant", "content": (
        "```python\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n"
        "```"
    )},
]


class TestA3ChatTemplate:
    """Tests for test a3 chat template."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _load_tokenizer()


    @pytest.mark.parametrize("messages,name", [
        (SINGLE_TURN, "single_turn"),
        (MULTI_TURN, "multi_turn"),
        (WITH_SYSTEM, "with_system"),
        (LONG_RESPONSE, "long_response"),
        (CODE_RESPONSE, "code_response"),
    ])
    def test_rendered_text_matches_hf(self, tokenizer, messages, name):
        """Verify rendered text matches hf."""
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        assert isinstance(rendered, str)
        assert len(rendered) > 0

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            assert f"<|im_start|>{role}" in rendered, (
                f"渲染文本缺少 <|im_start|>{role} marker"
            )
            assert content in rendered, (
                f"渲染文本缺少消息内容: {content[:30]}..."
            )

    @pytest.mark.parametrize("messages,name", [
        (SINGLE_TURN, "single_turn"),
        (MULTI_TURN, "multi_turn"),
        (WITH_SYSTEM, "with_system"),
    ])
    def test_token_ids_match_hf(self, tokenizer, messages, name):
        """Verify token ids match hf."""
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        ids_direct = result["input_ids"] if hasattr(result, "keys") else result

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        ids_text_then_encode = tokenizer.encode(text, add_special_tokens=False)

        assert ids_direct == ids_text_then_encode, (
            f"两种 tokenize 路径不一致:\n"
            f"  direct:     {ids_direct[:10]}...\n"
            f"  text+encode: {ids_text_then_encode[:10]}...\n"
            f"  长度: {len(ids_direct)} vs {len(ids_text_then_encode)}"
        )


    @pytest.mark.parametrize("messages,name", [
        (SINGLE_TURN, "single_turn"),
        (MULTI_TURN, "multi_turn"),
        (WITH_SYSTEM, "with_system"),
        (LONG_RESPONSE, "long_response"),
        (CODE_RESPONSE, "code_response"),
    ])
    def test_titaninfra_tokenize_input_ids_match(self, tokenizer, messages, name):
        """Verify titaninfra tokenize input ids match."""
        from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

        input_ids, labels = tokenize_with_hf_template(
            messages=messages,
            tokenizer=tokenizer,
            max_length=4096,
        )

        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        hf_ids = result["input_ids"] if hasattr(result, "keys") else result

        assert input_ids == hf_ids, (
            f"titaninfra input_ids 与 HF 不一致:\n"
            f"  titaninfra: {input_ids[:10]}...{input_ids[-5:]}\n"
            f"  HF:      {hf_ids[:10]}...{hf_ids[-5:]}\n"
            f"  长度: {len(input_ids)} vs {len(hf_ids)}"
        )


    def test_message_boundaries_count(self, tokenizer):
        """Verify message boundaries count."""
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        full_ids, boundaries, content_starts = _compute_message_boundaries(
            MULTI_TURN, tokenizer
        )
        assert len(boundaries) == len(MULTI_TURN) + 1, (
            f"boundaries 数量不对: {len(boundaries)} != {len(MULTI_TURN) + 1}"
        )
        assert len(content_starts) == len(MULTI_TURN)

    def test_boundaries_cover_all_tokens(self, tokenizer):
        """Verify boundaries cover all tokens."""
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        full_ids, boundaries, content_starts = _compute_message_boundaries(
            SINGLE_TURN, tokenizer
        )
        assert boundaries[0] == 0
        assert boundaries[-1] == len(full_ids)
        for i, cs in enumerate(content_starts):
            assert boundaries[i] <= cs <= boundaries[i + 1]

    def test_boundaries_monotonically_increasing(self, tokenizer):
        """Verify boundaries monotonically increasing."""
        from tests.oracle._legacy._boundary import _compute_message_boundaries

        full_ids, boundaries, _ = _compute_message_boundaries(
            MULTI_TURN, tokenizer
        )
        for i in range(1, len(boundaries)):
            assert boundaries[i] >= boundaries[i - 1], (
                f"boundary[{i}]={boundaries[i]} < boundary[{i-1}]={boundaries[i-1]}"
            )


    def test_chatml_structure(self, tokenizer):
        """Verify chatml structure."""
        text = tokenizer.apply_chat_template(
            SINGLE_TURN, tokenize=False, add_generation_prompt=False
        )

        assert text.startswith("<|im_start|>user")
        assert "<|im_start|>assistant" in text
        assert text.count("<|im_end|>") == 2

    def test_system_message_position(self, tokenizer):
        """Verify system message position."""
        text = tokenizer.apply_chat_template(
            WITH_SYSTEM, tokenize=False, add_generation_prompt=False
        )
        system_pos = text.find("<|im_start|>system")
        user_pos = text.find("<|im_start|>user")
        assert system_pos < user_pos, (
            f"system 应在 user 之前: system@{system_pos}, user@{user_pos}"
        )
