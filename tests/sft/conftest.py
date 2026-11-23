"""Shared SFT-test fixtures.

Mirrors the per-family tokenizer fixtures from ``tests/oracle/conftest.py``
so ``tests/sft/`` tests don't depend on import-time side effects from a
sibling test package. Session-scoped to avoid reloading tokenizer files
per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_TOKENIZER_ROOT: Path = _REPO_ROOT / "tests" / "fixtures" / "tokenizers"


def _load(family: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(_TOKENIZER_ROOT / family))


@pytest.fixture(scope="session")
def qwen3_tokenizer():
    return _load("qwen3")


@pytest.fixture(scope="session")
def qwen2_5_tokenizer():
    return _load("qwen2_5")


@pytest.fixture(scope="session")
def llama3_tokenizer():
    return _load("llama3")


@pytest.fixture(scope="session")
def deepseek_v3_tokenizer():
    return _load("deepseek_v3")


@pytest.fixture(scope="session")
def deepseek_v4_tokenizer():
    return _load("deepseek_v4")


@pytest.fixture(scope="session")
def gpt_oss_tokenizer():
    return _load("gpt_oss")


@pytest.fixture(scope="session")
def gemma4_tokenizer():
    return _load("gemma4")
