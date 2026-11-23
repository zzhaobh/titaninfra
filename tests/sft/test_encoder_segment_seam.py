"""Segment-seam BPE-stability guard for all 7 family encoders (F9).

A segment encoder splits one turn into many ``Segment``s; ``assemble()``
tokenizes each independently then concatenates. If a segment boundary
lands mid-BPE-merge, piecewise tokenize != one-shot tokenize → input_ids
drift → oracle parity breaks. ``_repair_seams`` was deleted (dead code,
F4), so segment boundaries MUST be naturally BPE-stable, healed only by
``_merge_adjacent_segments``.

This pins it for EVERY family × EVERY fixture: the concatenation of
per-segment token streams must equal tokenizing the joined wire string
in one pass. qwen2_5's leading-``\\n`` relocation comment flags it as the
likeliest drift site — it is covered here.

Migrated from the gemma4-only ``test_gemma4_segment_seam.py`` (now
deleted) — gemma4 is one family in the matrix here.

Self-contained fixture loading (the oracle ``load_oracle_fixture``
fixture lives under tests/oracle/conftest.py and is not visible here);
uses only the per-family ``*_tokenizer`` session fixtures from
tests/sft/conftest.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from titaninfra.data.sft.encoders import get_encoder
from titaninfra.data.sft.normalize import from_sft_sample

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "sft"

_TOKENIZER_FIXTURE = {
    "qwen3": "qwen3_tokenizer",
    "qwen2_5": "qwen2_5_tokenizer",
    "llama3": "llama3_tokenizer",
    "deepseek_v3": "deepseek_v3_tokenizer",
    "deepseek_v4": "deepseek_v4_tokenizer",
    "gpt_oss": "gpt_oss_tokenizer",
    "gemma4": "gemma4_tokenizer",
}


def _fixture_params():
    """Build (family, fixture_name) params for every fixture on disk."""
    params = []
    for family in _TOKENIZER_FIXTURE:
        fdir = _FIXTURE_ROOT / family
        if not fdir.is_dir():
            continue
        for p in sorted(fdir.glob("*.json")):
            params.append(
                pytest.param(
                    family, p.name,
                    id=f"{family}/{p.name}",
                    marks=pytest.mark.xdist_group(f"tok_{family}"),
                )
            )
    return params


def _load_sample(family: str, fixture_name: str) -> dict:
    raw = json.loads((_FIXTURE_ROOT / family / fixture_name).read_text(encoding="utf-8"))
    return raw["sample"]


@pytest.mark.parametrize("family,fixture_name", _fixture_params())
def test_segments_are_bpe_stable(family, fixture_name, request):
    """Per-segment tokenize, concatenated, equals one-shot tokenize of the wire.

    Only text-bearing segments participate: a segment carrying pre-baked
    ``token_ids`` skips BPE by construction, so it acts as a hard boundary
    that splits the wire into independently-stable runs (we compare each
    run separately).
    """
    tok = request.getfixturevalue(_TOKENIZER_FIXTURE[family])
    enc = get_encoder(family, tok)
    normalized = from_sft_sample(_load_sample(family, fixture_name))
    segments = enc.encode_segments(normalized)

    # Partition into maximal runs of text segments separated by any
    # token_ids segment. Each run must be internally BPE-stable.
    runs: list[list[str]] = [[]]
    for seg in segments:
        if seg.text is None:
            # token_ids segment: hard boundary, start a fresh run.
            if runs[-1]:
                runs.append([])
            continue
        runs[-1].append(seg.text)

    for run in runs:
        if len(run) < 2:
            continue  # single segment can't drift against itself
        split_ids: list[int] = []
        for text in run:
            split_ids.extend(tok.encode(text, add_special_tokens=False))
        joint_ids = tok.encode("".join(run), add_special_tokens=False)
        assert split_ids == joint_ids, (
            f"{family}/{fixture_name}: segment boundary caused BPE drift. "
            f"split={len(split_ids)} tokens, joint={len(joint_ids)} tokens. "
            "A segment boundary fell mid-merge; relocate it to a protocol "
            "marker (the seam-stability invariant all families share)."
        )
