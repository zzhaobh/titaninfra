"""Gemma4 oracle parity — smoke + 8 fixture parity tests + tamper control.

Post jinja-purge: ``tokenize_sample`` is the single runtime path; the
oracle baseline (``render_with_jinja``) is the independent Jinja
reference used for regression detection.
"""
from __future__ import annotations

import pytest

from tests.oracle._jinja_oracle._jinja_baseline import render_with_jinja
from tests.oracle._jinja_oracle.diff import compare_encoded
from titaninfra.data.sft.dispatch import tokenize_sample


def test_gemma4_tokenizer_loads(gemma4_tokenizer):
    """The session-scoped Gemma4 tokenizer fixture is reachable."""
    assert gemma4_tokenizer is not None
    ids = gemma4_tokenizer.encode("hello", add_special_tokens=False)
    assert isinstance(ids, list) and len(ids) > 0


def test_gemma4_fixture_loads(load_oracle_fixture):
    """The first gemma4 fixture parses into an OracleFixture."""
    fx = load_oracle_fixture("gemma4/01_single_turn_plain.json")
    assert fx.family == "gemma4"
    assert fx.sample["messages"][0]["role"] == "user"


# ----------------------------------------------------------------------
# Oracle parity — 8 fixtures: native (tokenize_sample) vs Jinja baseline.
# ----------------------------------------------------------------------

GEMMA4_FIXTURES = [
    "gemma4/01_single_turn_plain.json",
    "gemma4/02_with_system.json",
    "gemma4/03_thinking_last_assistant.json",
    "gemma4/04_thinking_history_drop.json",
    "gemma4/05_tool_call.json",
    "gemma4/06_tool_response.json",
    "gemma4/07_tools_top_level.json",
    "gemma4/08_full_tool_loop.json",
    "gemma4/09_math_tools_real_world.json",
]


@pytest.mark.parametrize("fixture_path", GEMMA4_FIXTURES)
def test_gemma4_native_matches_jinja(
    gemma4_tokenizer, load_oracle_fixture, fixture_path
):
    """Native encoder must produce the same EncodedSample as the Jinja oracle.

    If the fixture's expected.oracle_ok is true, both L1 (input_ids) and L2
    (labels) must match exactly. If false, the recorded first_divergence
    must hit (documented deviation, not a silent drift).
    """
    fx = load_oracle_fixture(fixture_path)
    jinja_out = render_with_jinja(
        fx.sample, gemma4_tokenizer,
        max_length=fx.max_length, template_name="gemma4",
    )
    native_out = tokenize_sample(
        fx.sample, gemma4_tokenizer, family="gemma4",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="gemma4", fixture=str(fx.path),
    )

    if fx.expected["oracle_ok"]:
        assert report["ok"] is True, (
            f"{fixture_path}: expected oracle_ok=true but got divergence "
            f"at index {report['first_divergence']}: {report}"
        )
        assert report["first_divergence"] is None
        assert report["levels"]["L1"]["ok"] is True
        assert report["levels"]["L2"]["ok"] is True
    else:
        assert report["ok"] is False
        assert report["first_divergence"] == fx.expected["first_divergence"]


def test_gemma4_oracle_catches_tampered_tokens(
    gemma4_tokenizer, load_oracle_fixture
):
    """Negative control: hand-flip one token, oracle must report ok=False.

    Without this test, a no-op compare_encoded (always returns ok=True)
    would silently pass every parity assertion above.
    """
    from titaninfra.data.sft.types import EncodedSample

    fx = load_oracle_fixture("gemma4/01_single_turn_plain.json")
    baseline = render_with_jinja(
        fx.sample, gemma4_tokenizer,
        max_length=fx.max_length, template_name="gemma4",
    )
    assert len(baseline.input_ids) >= 4

    tamper_idx = len(baseline.input_ids) // 2
    tampered_ids = list(baseline.input_ids)
    tampered_ids[tamper_idx] = (tampered_ids[tamper_idx] + 1) % 100_000
    tampered = EncodedSample(
        input_ids=tuple(tampered_ids),
        labels=baseline.labels,
        positions=baseline.positions,
    )

    report = compare_encoded(
        native=tampered, jinja=baseline,
        family="gemma4", fixture=None,
    )
    assert report["ok"] is False
    assert report["first_divergence"] == tamper_idx
