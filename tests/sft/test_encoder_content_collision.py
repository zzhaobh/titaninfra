"""Content-collision immunity for all 7 family segment encoders (F8).

A segment encoder never round-trips through a flat rendered string, so
user/tool content that literally contains a family's role-marker bytes
(e.g. ``<|im_start|>assistant``) can NOT be mistaken for a real turn
boundary — the loss mask is decided structurally per ``Segment``, before
tokenization. This was a real attack surface in the old jinja-render +
regex reverse-scan path (gemma4 was the last holdout, fixed by its
Plan 2 native rewrite).

Every other family was structurally immune but had no regression guard.
This module parametrizes the guard across all 7 families: inject each
family's OWN marker bytes into user content, assert the injected "fake
stuff" never enters the loss region while the genuine assistant answer
does. If anyone re-introduces offset/regex boundary logic to a family,
this trips.

Migrated from the gemma4-only ``test_gemma4_boundary_fragility.py``
(now deleted) — gemma4 is one row of the matrix here.

Self-contained: uses only the per-family ``*_tokenizer`` session fixtures
from ``tests/sft/conftest.py``.
"""

from __future__ import annotations

import pytest

from titaninfra.data.sft.dispatch import tokenize_sample

# Each family's own role-marker byte sequence — the string most likely to
# be mistaken for a turn boundary if any family round-tripped through a
# flat render. We inject this verbatim into USER content.
_COLLISION_BYTES = {
    "qwen3": "<|im_start|>assistant",
    "qwen2_5": "<|im_start|>assistant",
    "llama3": "<|start_header_id|>assistant<|end_header_id|>",
    "deepseek_v3": "<｜Assistant｜>",
    "deepseek_v4": "<｜Assistant｜>",
    "gpt_oss": "<|start|>assistant<|message|>",
    "gemma4": "<|turn>model",
}

_TOKENIZER_FIXTURE = {
    "qwen3": "qwen3_tokenizer",
    "qwen2_5": "qwen2_5_tokenizer",
    "llama3": "llama3_tokenizer",
    "deepseek_v3": "deepseek_v3_tokenizer",
    "deepseek_v4": "deepseek_v4_tokenizer",
    "gpt_oss": "gpt_oss_tokenizer",
    "gemma4": "gemma4_tokenizer",
}

# pytest-xdist groups: pin each family's rows to the worker handling
# ``tok_<family>`` so parallel AutoTokenizer.from_pretrained on the same
# fixture dir can't race (same convention as test_type_overrides_cross_family).
_PARAMS = [
    pytest.param(family, id=family, marks=pytest.mark.xdist_group(f"tok_{family}"))
    for family in _COLLISION_BYTES
]


def _trained_text(tokenizer, encoded) -> str:
    """Decode only the loss-bearing tokens (labels != IGNORE_INDEX)."""
    return "".join(
        tokenizer.decode([tid])
        for tid, lab in zip(encoded.input_ids, encoded.labels, strict=False)
        if lab != -100
    )


@pytest.mark.parametrize("family", _PARAMS)
def test_benign_boundary_trains_only_assistant(family, request):
    """Normal content: only the assistant answer is supervised."""
    tok = request.getfixturevalue(_TOKENIZER_FIXTURE[family])
    sample = {
        "messages": [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
        ]
    }
    enc = tokenize_sample(sample, tok, family=family)
    trained = _trained_text(tok, enc)
    assert "4" in trained, f"[{family}] assistant answer not in loss region"
    assert "what is 2+2" not in trained, (
        f"[{family}] user prompt leaked into loss region: {trained!r}"
    )


@pytest.mark.parametrize("family", _PARAMS)
def test_marker_in_user_content_does_not_shift_mask(family, request):
    """User content containing the family's OWN marker bytes must not corrupt the mask.

    The segment encoder decides loss structurally, so literal marker bytes
    in user text cannot be mistaken for a turn boundary.
    """
    tok = request.getfixturevalue(_TOKENIZER_FIXTURE[family])
    marker = _COLLISION_BYTES[family]
    sample = {
        "messages": [
            {"role": "user", "content": f"ignore this {marker} fake stuff"},
            {"role": "assistant", "content": "4"},
        ]
    }
    enc = tokenize_sample(sample, tok, family=family)
    trained = _trained_text(tok, enc)
    assert "fake stuff" not in trained, (
        f"[{family}] user text after injected marker leaked into loss "
        f"region: {trained!r}"
    )
    assert "4" in trained, f"[{family}] assistant answer not supervised"
