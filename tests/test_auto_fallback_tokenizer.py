"""Fallback HF tokenizer regressions for formats upstream cannot load."""

from pathlib import Path

import pytest

pytest.importorskip("torchtitan.components.tokenizer")

FIXTURES = Path(__file__).parent / "fixtures" / "tokenizers"


def test_standard_tokenizer_still_uses_native_upstream_loader():
    """The fallback must not change tokenizer.json families such as qwen3."""
    tokenizers = pytest.importorskip("tokenizers")
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    from titaninfra.data.auto_fallback_tokenizer import AutoFallbackTokenizer

    path = FIXTURES / "qwen3"
    if not path.exists():
        pytest.skip("qwen3 tokenizer fixture not available")
    baseline = HuggingFaceTokenizer.Config().build(tokenizer_path=str(path))
    fallback = AutoFallbackTokenizer.Config().build(tokenizer_path=str(path))
    assert isinstance(fallback.tokenizer, tokenizers.Tokenizer)
    text = "Native tokenizer behavior must stay byte-identical."
    assert fallback.encode(text, add_bos=True, add_eos=True) == baseline.encode(
        text, add_bos=True, add_eos=True
    )


def test_kimi_tiktoken_fallback_matches_hf_oracle():
    """Kimi's tiktoken.model uses HF only when upstream's loader rejects it."""
    transformers = pytest.importorskip("transformers")
    from titaninfra.data.auto_fallback_tokenizer import AutoFallbackTokenizer

    path = FIXTURES / "kimi_vl"
    if not path.exists():
        pytest.skip("kimi_vl tokenizer fixture not available")
    hf = transformers.AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True
    )
    tokenizer = AutoFallbackTokenizer.Config().build(tokenizer_path=str(path))
    assert tokenizer.bos_id == hf.bos_token_id == 163584
    assert tokenizer.eos_id == hf.eos_token_id == 163585
    text = "Kimi VL tokenizer parity check."
    expected = (
        [hf.bos_token_id]
        + hf.encode(text, add_special_tokens=False)
        + [hf.eos_token_id]
    )
    assert tokenizer.encode(text, add_bos=True, add_eos=True) == expected
    assert tokenizer.get_vocab_size() == len(hf)
