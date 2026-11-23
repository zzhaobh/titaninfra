"""Unit tests for a2 tokenization."""

from pathlib import Path

import pytest

pytest.importorskip("torchtitan", reason="torchtitan not installed; A2 needs both tokenizers")

TOKENIZERS_DIR = Path(__file__).parent.parent / "fixtures" / "tokenizers"

TEXT_CASES = [
    pytest.param("Hello, world!", id="english_simple"),
    pytest.param("你好，世界！", id="chinese_simple"),
    pytest.param(
        "The quick brown fox jumps over the lazy dog.",
        id="english_pangram",
    ),
    pytest.param(
        "今天天气真好，我们一起去公园散步吧。",
        id="chinese_sentence",
    ),
    pytest.param(
        "Hello 你好 こんにちは مرحبا",
        id="multilingual",
    ),
    pytest.param("🎉🚀💻🔥", id="emoji"),
    pytest.param(
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        id="code_python",
    ),
    pytest.param("a " * 500, id="repetitive_500_tokens"),
    pytest.param(" ", id="single_space"),
    pytest.param("  \n\t  ", id="whitespace_only"),
    pytest.param(
        "2 + 2 = 4, 3.14159, 1e-10, 0xFF",
        id="numbers_and_formats",
    ),
    pytest.param(
        '<html><body class="main">Hello &amp; world</body></html>',
        id="html_entities",
    ),
]


def _load_hf_tokenizer(model_name: str = "qwen3"):
    """Load hf tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(TOKENIZERS_DIR / model_name), trust_remote_code=True
    )


def _load_titan_tokenizer(model_name: str = "qwen3"):
    """Load titan tokenizer."""
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    return HuggingFaceTokenizer(tokenizer_path=str(TOKENIZERS_DIR / model_name))


class TestA2Tokenization:
    """Tests for test a2 tokenization."""

    @pytest.fixture(scope="class")
    def hf_tok(self):
        return _load_hf_tokenizer()

    @pytest.fixture(scope="class")
    def titan_tok(self):
        return _load_titan_tokenizer()


    @pytest.mark.parametrize("text", TEXT_CASES)
    def test_encode_bitexact(self, hf_tok, titan_tok, text):
        """Verify encode bitexact."""
        hf_ids = hf_tok.encode(text, add_special_tokens=False)
        titan_ids = titan_tok.tokenizer.encode(text).ids

        assert hf_ids == titan_ids, (
            f"Token IDs 不一致:\n"
            f"  HF:    {hf_ids[:20]}{'...' if len(hf_ids) > 20 else ''}\n"
            f"  Titan: {titan_ids[:20]}{'...' if len(titan_ids) > 20 else ''}\n"
            f"  长度: HF={len(hf_ids)}, Titan={len(titan_ids)}"
        )

    @pytest.mark.parametrize("text", TEXT_CASES)
    def test_pretrain_encode_with_bos_eos(self, hf_tok, titan_tok, text):
        """Verify pretrain encode with bos eos."""
        titan_ids = titan_tok.encode(text, add_bos=True, add_eos=True)

        hf_ids = hf_tok.encode(text, add_special_tokens=False)
        if hf_tok.bos_token is not None:
            hf_ids = [hf_tok.bos_token_id] + hf_ids
        if hf_tok.eos_token is not None:
            hf_ids = hf_ids + [hf_tok.eos_token_id]

        assert titan_ids == hf_ids, (
            f"Pretrain encode 不一致:\n"
            f"  Titan: {titan_ids[:10]}...{titan_ids[-3:]}\n"
            f"  HF:    {hf_ids[:10]}...{hf_ids[-3:]}\n"
            f"  长度: Titan={len(titan_ids)}, HF={len(hf_ids)}"
        )


    def test_qwen3_no_bos(self, hf_tok, titan_tok):
        """Verify qwen3 no bos."""
        assert hf_tok.bos_token is None, f"HF bos_token 应为 None, got {hf_tok.bos_token}"
        assert titan_tok.bos_id is None, f"Titan bos_id 应为 None, got {titan_tok.bos_id}"

        ids_with = titan_tok.encode("test", add_bos=True, add_eos=False)
        ids_without = titan_tok.encode("test", add_bos=False, add_eos=False)
        assert ids_with == ids_without, "add_bos=True 不应改变 Qwen3 的输出"

    def test_qwen3_eos_is_im_end(self, hf_tok, titan_tok):
        """Verify qwen3 eos is im end."""
        assert hf_tok.eos_token == "<|im_end|>"
        assert titan_tok.eos_token == "<|im_end|>"
        assert hf_tok.eos_token_id == titan_tok.eos_id

        ids_with = titan_tok.encode("test", add_bos=False, add_eos=True)
        ids_without = titan_tok.encode("test", add_bos=False, add_eos=False)
        assert len(ids_with) == len(ids_without) + 1
        assert ids_with[-1] == titan_tok.eos_id


    def test_special_token_ids_match(self, hf_tok, titan_tok):
        """Verify special token ids match."""
        special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        for token in special_tokens:
            hf_id = hf_tok.convert_tokens_to_ids(token)
            titan_id = titan_tok.token_to_id(token)
            assert hf_id == titan_id, (
                f"特殊 token '{token}' ID 不一致: HF={hf_id}, Titan={titan_id}"
            )

    def test_vocab_size_match(self, hf_tok, titan_tok):
        """Verify vocab size match."""
        hf_size = hf_tok.vocab_size
        titan_size = titan_tok.vocab_size
        hf_full_size = len(hf_tok)
        titan_full_size = titan_tok.get_vocab_size()
        assert hf_full_size == titan_full_size, (
            f"完整词表大小不一致: HF={hf_full_size}, Titan={titan_full_size}\n"
            f"  (HF vocab_size={hf_size}, Titan vocab_size={titan_size})"
        )


    @pytest.mark.parametrize("text", TEXT_CASES[:6])
    def test_decode_roundtrip(self, hf_tok, titan_tok, text):
        """Verify decode roundtrip."""
        hf_ids = hf_tok.encode(text, add_special_tokens=False)

        hf_decoded = hf_tok.decode(hf_ids, skip_special_tokens=False)
        titan_decoded = titan_tok.decode(hf_ids, skip_special_tokens=False)

        assert hf_decoded == titan_decoded, (
            f"解码不一致:\n  HF:    '{hf_decoded}'\n  Titan: '{titan_decoded}'"
        )


    def test_empty_string(self, hf_tok, titan_tok):
        """Verify empty string."""
        hf_ids = hf_tok.encode("", add_special_tokens=False)
        titan_ids = titan_tok.tokenizer.encode("").ids
        assert hf_ids == titan_ids, f"空串编码不一致: HF={hf_ids}, Titan={titan_ids}"

    def test_long_text(self, hf_tok, titan_tok):
        """Verify long text."""
        long_text = "这是一段测试文本。" * 500
        hf_ids = hf_tok.encode(long_text, add_special_tokens=False)
        titan_ids = titan_tok.tokenizer.encode(long_text).ids

        assert len(hf_ids) == len(titan_ids), (
            f"长文本长度不一致: HF={len(hf_ids)}, Titan={len(titan_ids)}"
        )
        assert hf_ids == titan_ids, "长文本 token IDs 不一致"

    def test_single_char_tokens(self, hf_tok, titan_tok):
        """Verify single char tokens."""
        chars = list("abcABC你我他123!@#")
        for ch in chars:
            hf_ids = hf_tok.encode(ch, add_special_tokens=False)
            titan_ids = titan_tok.tokenizer.encode(ch).ids
            assert hf_ids == titan_ids, (
                f"字符 '{ch}' 编码不一致: HF={hf_ids}, Titan={titan_ids}"
            )
