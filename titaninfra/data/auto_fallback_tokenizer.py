"""Fallback for HF tokenizer formats unsupported by upstream's native loader."""

from dataclasses import dataclass
from types import SimpleNamespace

from torchtitan.components.tokenizer import HuggingFaceTokenizer


class _HFAutoTokenizerAdapter:
    """Expose the small ``tokenizers.Tokenizer`` surface upstream consumes."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text, add_special_tokens=True):
        ids = self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        )
        return SimpleNamespace(ids=ids)

    def decode(self, token_ids, **kwargs):
        return self._tokenizer.decode(token_ids, **kwargs)

    def get_vocab_size(self):
        return len(self._tokenizer)

    def get_vocab(self):
        return self._tokenizer.get_vocab()

    def token_to_id(self, token):
        return self._tokenizer.convert_tokens_to_ids(token)

    def id_to_token(self, token_id):
        return self._tokenizer.convert_ids_to_tokens(token_id)

    def add_tokens(self, new_tokens):
        return self._tokenizer.add_tokens(new_tokens)


class AutoFallbackTokenizer(HuggingFaceTokenizer):
    """Use upstream's native formats first, then trusted HF AutoTokenizer."""

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        pass

    def _load_tokenizer_from_path(self, tokenizer_path: str):
        try:
            return super()._load_tokenizer_from_path(tokenizer_path)
        except FileNotFoundError:
            # Kimi-VL ships tiktoken.model plus tokenization_moonshot.py,
            # which upstream's tokenizer.json/BPE loader does not support.
            # Keep BOS/EOS inference and encode semantics in the upstream
            # wrapper; only the low-level tokenizer implementation changes.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
            return _HFAutoTokenizerAdapter(tokenizer)
