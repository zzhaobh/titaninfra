"""tokenize_conversation must route through dispatch, not the Jinja module."""

import inspect

from titaninfra.data import sft_utils


def test_tokenize_conversation_uses_dispatch():
    src = inspect.getsource(sft_utils.tokenize_conversation)
    assert "dispatch.tokenize_sample" in src or "tokenize_sample" in src
    assert "tokenize_with_hf_template" not in src
