"""Phase 4 — Assembler segment-merge pass invariants.

Covers ``docs/superpowers/specs/2026-06-02-encoder-layer-refactor.md`` §4:

* Adjacent text segments with matching (kind, role, loss_scale,
  token_class, is_generation, msg_index) collapse into one — tokenize
  runs once over the merged text, so BPE boundaries across the seam
  are naturally captured.
* Segments with mismatched merge keys stay separate.
* Pre-tokenized ``token_ids`` segments never merge (merging would skip
  BPE on the seam, defeating the point).
* Tokens after merge are IDENTICAL to tokens before merge when the
  tokenizer is BPE-stable across the seam (the char-tokenizer used
  here is trivially stable, so identity is checkable).

The qwen3 fixture-09 bug (3-vs-2 \\n between ``</think>`` and
``<tool_call>``) is structurally impossible in the new design because:

* If body and tool_call share metadata they merge into one
  tokenize call — no encoder-planted linker means no off-by-one
  linker count.
* If body and tool_call have different ``kind`` (which they do —
  THINKING vs TOOL_CALL) they stay separate, and the natural
  ``</think>\\n\\n<tool_call>`` concatenation is what the encoder
  emits (no extra linker required).
"""

from __future__ import annotations

import pytest

from titaninfra.data.sft.assembler import _merge_adjacent_segments, assemble
from titaninfra.data.sft.segment import SegKind, Segment, TokenClass


class _CharTokenizer:
    """Char-codepoint tokenizer for stable seam behavior."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(c) for c in text]


@pytest.fixture
def tok():
    return _CharTokenizer()


def _seg(
    text=None,
    *,
    ids=None,
    kind=SegKind.CONTENT,
    role="assistant",
    scale=1.0,
    gen=False,
    midx=None,
    cls=TokenClass.DECODED,
) -> Segment:
    return Segment(
        text=text,
        token_ids=ids,
        kind=kind,
        role=role,
        scale_factor=scale,
        is_generation=gen,
        token_class=cls,
        msg_index=midx,
    )


# ----------------------------------------------------------------------
# _merge_adjacent_segments — unit
# ----------------------------------------------------------------------
def test_merge_collapses_identical_metadata_runs() -> None:
    segs = [
        _seg(text="<think>\n"),
        _seg(text="body content"),
        _seg(text="\n</think>\n\n"),
    ]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 1
    assert merged[0].text == "<think>\nbody content\n</think>\n\n"


def test_merge_keeps_different_kind_separate() -> None:
    # The qwen3 fixture-09 scenario: thinking → tool_call.
    segs = [
        _seg(text="<think>\n</think>\n\n", kind=SegKind.THINKING),
        _seg(text="<tool_call>{...}</tool_call>", kind=SegKind.TOOL_CALL),
    ]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 2
    assert merged[0].kind == SegKind.THINKING
    assert merged[1].kind == SegKind.TOOL_CALL


def test_merge_keeps_different_role_separate() -> None:
    segs = [
        _seg(text="user msg", role="user", scale=0.0),
        _seg(text="asst msg", role="assistant", scale=1.0),
    ]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 2


def test_merge_keeps_different_loss_scale_separate() -> None:
    segs = [
        _seg(text="zero", scale=0.0),
        _seg(text="one", scale=1.0),
    ]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 2


def test_merge_keeps_different_token_class_separate() -> None:
    # METADATA must stay loss_scale=0 (invariant); use that.
    segs = [
        _seg(text="proto", scale=0.0, cls=TokenClass.PROTOCOL),
        _seg(text="meta", scale=0.0, cls=TokenClass.METADATA),
    ]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 2


def test_merge_keeps_different_is_generation_separate() -> None:
    segs = [
        _seg(text="prompt", scale=0.0, gen=False),
        _seg(text="response", scale=1.0, gen=True),
    ]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 2


def test_merge_keeps_different_msg_index_separate() -> None:
    # Two assistant turns with same kind/role/scale — different turns
    # must stay separate so per-message diagnostics survive.
    segs = [_seg(text="turn1", midx=1), _seg(text="turn2", midx=3)]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 2


def test_merge_skips_pre_tokenized_segments() -> None:
    # Pre-tokenized segments must NOT merge — merging would lose the
    # raw token IDs and require re-tokenization on the seam.
    segs = [
        _seg(ids=(1, 2, 3), kind=SegKind.MARKER, scale=0.0),
        _seg(ids=(4, 5, 6), kind=SegKind.MARKER, scale=0.0),
    ]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 2


def test_merge_handles_three_or_more_in_a_run() -> None:
    segs = [
        _seg(text="a"),
        _seg(text="b"),
        _seg(text="c"),
        _seg(text="d"),
    ]
    merged = _merge_adjacent_segments(segs)
    assert len(merged) == 1
    assert merged[0].text == "abcd"


def test_merge_empty_list_returns_empty() -> None:
    assert _merge_adjacent_segments([]) == []


def test_merge_single_segment_unchanged() -> None:
    seg = _seg(text="only")
    merged = _merge_adjacent_segments([seg])
    assert merged == [seg]


# ----------------------------------------------------------------------
# End-to-end: assemble() output is equivalent pre/post merge on a
# stable tokenizer (character-level codepoint encoder)
# ----------------------------------------------------------------------
def test_assemble_merged_run_tokens_equivalent(tok) -> None:
    # Three mergeable segments — assemble's output should match a
    # single segment carrying the concatenated text. With a char
    # tokenizer the BPE seam is trivially stable, so equality is
    # well-defined.
    pre_merge = assemble(
        [_seg(text="hello "), _seg(text="world "), _seg(text="!")],
        tok,
        mode="sft",
    )
    post_merge = assemble(
        [_seg(text="hello world !")],
        tok,
        mode="sft",
    )
    assert pre_merge.input_ids == post_merge.input_ids
    assert pre_merge.labels == post_merge.labels
    assert pre_merge.positions == post_merge.positions


def test_assemble_runs_merge_after_rl_prompt_cut(tok) -> None:
    # rl_prompt drops everything from the first is_generation onward.
    # The merge pass runs AFTER the cut so prompt and response never
    # collapse into one block.
    segs = [
        _seg(text="sys ", role="system", scale=0.0, midx=0),
        _seg(text="user ", role="user", scale=0.0, midx=1),
        _seg(text="asst", role="assistant", scale=1.0, gen=True, midx=2),
    ]
    out = assemble(segs, tok, mode="rl_prompt")
    # Only sys+user survived (different roles → not merged), asst was
    # cut. Two segments after the cut would normally merge if same
    # metadata; here they differ in role → 2 tokenize calls.
    assert out.input_ids == tuple(ord(c) for c in "sys user ")
