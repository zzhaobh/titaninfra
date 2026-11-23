"""Cross-family ``type_overrides`` consumption parity.

Asserts every family declaring ``FEATURE_TYPE_OVERRIDES`` actually
scales thinking-block loss when ``loss_config.type_overrides.thinking``
is set. Prevents the declaration-vs-implementation drift discovered
2026-06-01 (gpt_oss/gemma4 implemented but undeclared; dsv4 declared
but raised ``NotImplementedError``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from titaninfra.data.sft.capabilities import FEATURE_TYPE_OVERRIDES
from titaninfra.data.sft.dispatch import tokenize_sample

FIXTURES = Path(__file__).parent / "fixtures" / "tokenizers"

# Families whose SUPPORTED_FEATURES include FEATURE_TYPE_OVERRIDES.
# tok_dir is the directory under tests/fixtures/tokenizers/.
SUPPORTING = [
    ("qwen3", "qwen3"),
    ("gpt_oss", "gpt_oss"),
    ("gemma4", "gemma4"),
    ("deepseek_v4", "deepseek_v4"),
]

# pytest-xdist groups: pin every test row that loads tokenizer ``T`` to the
# worker handling group ``tok_T``. Without this, with high worker counts
# (e.g. 100+ on the GPU host) parallel ``AutoTokenizer.from_pretrained(...,
# trust_remote_code=True)`` on the same fixture races against the existing
# ``test_tool_use_pipeline.py`` cases (which already use ``tok_<family>``
# groups) and crashes one of the workers with an xdist INTERNALERROR.
# Same convention as ``test_tool_use_pipeline.py:765``.
_SUPPORTING_PARAMS = [
    pytest.param(
        family, tok_dir,
        id=family,
        marks=pytest.mark.xdist_group(f"tok_{family}"),
    )
    for family, tok_dir in SUPPORTING
]

# Sample with a single thinking block in the assistant turn.
SAMPLE: dict = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "blocks": [
            {"type": "thinking", "text": "Let me think... 2 plus 2 equals 4."},
            {"type": "text", "text": "4"},
        ]},
    ],
}


def _sum_weight(sample) -> float:
    """Sum of per-token loss weight; falls back to non-IGNORE_INDEX count."""
    if sample.loss_weights is not None:
        return float(sum(sample.loss_weights))
    return float(sum(1 for lb in sample.labels if lb != -100))


def _load_tokenizer(tok_dir: str):
    from transformers import AutoTokenizer
    path = FIXTURES / tok_dir
    if not path.exists():
        pytest.skip(f"tokenizer fixture missing: {path}")
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def _fresh_sample() -> dict:
    return json.loads(json.dumps(SAMPLE))


@pytest.mark.parametrize("family,tok_dir", _SUPPORTING_PARAMS)
def test_thinking_scale_half_reduces_thinking_loss(family, tok_dir):
    """``type_overrides.thinking=0.5`` must lower thinking-token weight vs baseline."""
    tok = _load_tokenizer(tok_dir)

    base = tokenize_sample(_fresh_sample(), tok, family=family)
    scaled = tokenize_sample(
        _fresh_sample(), tok, family=family,
        loss_config={"type_overrides": {"thinking": 0.5}},
    )

    assert base.input_ids == scaled.input_ids, f"[{family}] input_ids drifted"
    assert _sum_weight(scaled) < _sum_weight(base), (
        f"[{family}] thinking_scale=0.5 did NOT reduce loss weight "
        f"(base={_sum_weight(base)}, scaled={_sum_weight(scaled)})"
    )


@pytest.mark.parametrize("family,tok_dir", _SUPPORTING_PARAMS)
def test_thinking_scale_zero_masks_thinking(family, tok_dir):
    """``type_overrides.thinking=0.0`` must fully mask thinking tokens."""
    tok = _load_tokenizer(tok_dir)

    base = tokenize_sample(_fresh_sample(), tok, family=family)
    zeroed = tokenize_sample(
        _fresh_sample(), tok, family=family,
        loss_config={"type_overrides": {"thinking": 0.0}},
    )
    assert _sum_weight(zeroed) < _sum_weight(base), (
        f"[{family}] thinking_scale=0.0 did NOT mask thinking tokens"
    )


def test_capability_declaration_matches_supporting_list():
    """Guard: the test's ``SUPPORTING`` list must equal the set of families
    declaring ``FEATURE_TYPE_OVERRIDES``. Any drift (a new family declares
    it, or an existing one drops it) trips this assertion so the test
    matrix can't silently fall behind reality.
    """
    from titaninfra.data.sft.encoders import deepseek_v4, gemma4, gpt_oss, qwen3
    declared = {
        name for name, mod in [
            ("qwen3", qwen3),
            ("gpt_oss", gpt_oss),
            ("gemma4", gemma4),
            ("deepseek_v4", deepseek_v4),
        ]
        if FEATURE_TYPE_OVERRIDES in mod.SUPPORTED_FEATURES
    }
    supporting_names = {f for f, _ in SUPPORTING}
    assert declared == supporting_names, (
        f"Declared families {declared} != test-covered {supporting_names}. "
        f"Update SUPPORTING list or fix encoder declarations."
    )
