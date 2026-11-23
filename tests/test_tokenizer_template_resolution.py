"""
Regression: tokenize_with_hf_template must not raise on unknown template_name.

The template lookup at the top of tokenize_with_hf_template only reads
optional behaviors (thinking_markers, mask_empty_thinking) from
ChatTemplate. When template_name is a valid tool_use_adapter name but
not registered in ChatTemplate (e.g. "deepseek_v3"), get_template raises
ValueError. The original except clause caught only (KeyError, ImportError),
so users hit a bare ValueError instead of falling back gracefully.

This test pins the expected behavior:
- Unknown template_name → silent fallback (no exception), warning logged.
- HF apply_chat_template still runs because tokenizer has its own chat_template.
"""

from __future__ import annotations

import logging
import pathlib

import pytest

_FIXTURES = pathlib.Path("tests/fixtures/tokenizers/deepseek_v3")
_TOKENIZER_AVAILABLE = (_FIXTURES / "tokenizer.json").exists()

pytestmark = pytest.mark.skipif(
    not _TOKENIZER_AVAILABLE,
    reason="deepseek_v3 tokenizer fixture not downloaded",
)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(_FIXTURES), trust_remote_code=True)


def test_unknown_template_name_does_not_raise(tok, caplog):
    """ChatTemplate registry miss must not propagate as ValueError."""
    from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    with caplog.at_level(logging.WARNING):
        input_ids, labels = tokenize_with_hf_template(
            messages=messages,
            tokenizer=tok,
            template_name="definitely_not_a_real_template_xyz",
        )

    assert len(input_ids) > 0
    assert len(input_ids) == len(labels)
    # Warning surfaces the typo without crashing
    assert any(
        "definitely_not_a_real_template_xyz" in r.getMessage()
        for r in caplog.records
    ), "expected a warning naming the unknown template"


def test_adapter_only_template_name_resolves(tok):
    """Verify adapter only template name resolves."""
    from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

    messages = [
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "2"},
    ]
    input_ids, labels = tokenize_with_hf_template(
        messages=messages,
        tokenizer=tok,
        template_name="deepseek_v3",
    )
    assert len(input_ids) > 0
    assert len(input_ids) == len(labels)
