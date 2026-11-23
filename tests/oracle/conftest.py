"""Pytest fixtures for the oracle harness.

Provides:

* ``load_oracle_fixture`` — JSON-fixture loader callable. Reads a
  fixture file under ``tests/fixtures/sft/<family>/`` and exposes its
  contents as a typed namespace so tests don't reach into raw dicts.
* ``identity_qwen3_encoder`` — registers ``IdentityEncoder`` under
  the test-only name ``_identity_qwen3`` for the duration of one
  test, then tears down. Function-scoped so the global encoder
  registry stays clean across tests.
* ``qwen3_tokenizer`` — session-scoped HF tokenizer loaded from
  ``tests/fixtures/tokenizers/qwen3``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

# Repo root: tests/oracle/conftest.py → tests/oracle → tests → repo
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_FIXTURE_ROOT: Path = _REPO_ROOT / "tests" / "fixtures" / "sft"


@dataclass(frozen=True)
class OracleFixture:
    """In-memory view of one JSON fixture file."""

    path: Path
    description: str
    family: str
    tokenizer_path: Path
    sample: dict
    max_length: int | None
    expected: dict


def _load_fixture(relpath: str) -> OracleFixture:
    """Resolve ``relpath`` (relative to ``tests/fixtures/sft/``) and parse."""
    abs_path = _FIXTURE_ROOT / relpath
    if not abs_path.is_file():
        raise FileNotFoundError(
            f"Oracle fixture not found: {abs_path}. "
            f"All oracle fixtures live under {_FIXTURE_ROOT}/."
        )
    raw = json.loads(abs_path.read_text(encoding="utf-8"))
    return OracleFixture(
        path=abs_path,
        description=raw.get("_description", ""),
        family=raw["family"],
        tokenizer_path=_REPO_ROOT / raw["tokenizer_path"],
        sample=raw["sample"],
        max_length=raw.get("max_length"),
        expected=raw.get("expected", {}),
    )


@pytest.fixture
def load_oracle_fixture():
    """Return a function that loads a fixture by relative path.

    Use as::

        def test_x(load_oracle_fixture):
            fx = load_oracle_fixture("qwen3/01_single_turn_plain.json")
            assert fx.expected["oracle_ok"] is True
    """
    return _load_fixture


@pytest.fixture(scope="session")
def qwen3_tokenizer():
    """Session-scoped Qwen3 tokenizer loaded from the offline fixture."""
    from transformers import AutoTokenizer

    tok_dir = _REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "qwen3"
    return AutoTokenizer.from_pretrained(str(tok_dir))


@pytest.fixture(scope="session")
def qwen2_5_tokenizer():
    """Session-scoped Qwen2.5 tokenizer loaded from the offline fixture."""
    from transformers import AutoTokenizer

    tok_dir = _REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "qwen2_5"
    return AutoTokenizer.from_pretrained(str(tok_dir))


@pytest.fixture(scope="session")
def llama3_tokenizer():
    """Session-scoped Llama3 tokenizer loaded from the offline fixture."""
    from transformers import AutoTokenizer

    tok_dir = _REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "llama3"
    return AutoTokenizer.from_pretrained(str(tok_dir))


@pytest.fixture(scope="session")
def gemma4_tokenizer():
    """Session-scoped Gemma4 tokenizer loaded from the offline fixture."""
    from transformers import AutoTokenizer

    tok_dir = _REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "gemma4"
    return AutoTokenizer.from_pretrained(str(tok_dir))


@pytest.fixture(scope="session")
def gpt_oss_tokenizer():
    """Session-scoped GPT-OSS tokenizer loaded from the offline fixture."""
    from transformers import AutoTokenizer

    tok_dir = _REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "gpt_oss"
    return AutoTokenizer.from_pretrained(str(tok_dir))


@pytest.fixture(scope="session")
def deepseek_v3_tokenizer():
    """Session-scoped DeepSeek-V3 tokenizer loaded from the offline fixture."""
    from transformers import AutoTokenizer

    tok_dir = _REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "deepseek_v3"
    return AutoTokenizer.from_pretrained(str(tok_dir))


@pytest.fixture(autouse=True)
def _gemma4_oracle_adapter_registered():
    """Install the legacy Gemma4 adapter into the production registry for oracle tests.

    PR-C (2026-05-28) deleted the production ``Gemma4Adapter``; the native encoder
    is now the source of truth. Any oracle test that exercises the legacy
    ``tokenize_with_hf_template`` path with ``template_name="gemma4"`` still needs
    the family-specific boundary logic (tool-use samples otherwise hit fail-loud
    on continue-same-turn assistants when ``DefaultAdapter`` is used).

    This fixture restores the legacy adapter under the ``gemma4`` template name
    for every oracle test, then removes it on teardown. Production runtime never
    imports this adapter — only oracle tests do, and only for the duration of a
    test. The fixture is a no-op if importing the legacy adapter fails (e.g. the
    test file was deleted in some later PR), so it never blocks the test suite.
    """
    from tests.oracle._legacy.tool_use_adapters import _REGISTRY

    try:
        from tests.oracle._legacy_adapters.gemma4 import Gemma4OracleAdapter
    except ImportError:
        yield
        return

    sentinel = object()
    previous = _REGISTRY.get("gemma4", sentinel)
    _REGISTRY["gemma4"] = Gemma4OracleAdapter
    try:
        yield
    finally:
        if previous is sentinel:
            _REGISTRY.pop("gemma4", None)
        else:
            _REGISTRY["gemma4"] = previous


@pytest.fixture
def identity_qwen3_encoder():
    """Register ``IdentityEncoder`` for the test, unregister on teardown.

    Yields the registry name (``_identity_qwen3``) so the test can pass
    it as ``family=...`` to ``dispatch.tokenize_sample``.
    """
    from tests.oracle._identity_encoder import IDENTITY_NAME, IdentityEncoder
    from titaninfra.data.sft.encoders import register, unregister

    register(IDENTITY_NAME)(IdentityEncoder)
    try:
        yield IDENTITY_NAME
    finally:
        unregister(IDENTITY_NAME)
