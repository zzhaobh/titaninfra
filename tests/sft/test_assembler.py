"""Assembler -- segment list -> EncodedSample (M-beta.0 RED then GREEN).

These tests pin the contract that Layer 2 encoders rely on:

* text vs token_ids paths both work and bypass tokenize for the latter
* loss_scale = 0.0   -> labels = IGNORE_INDEX
* loss_scale = 1.0   -> labels = token_id, loss_weights = None (binary fast path)
* loss_scale in (0,1) -> labels = token_id, loss_weights[t] = scale
* positions = 0..N-1 (single document; packer handles cross-sample reset)
* mode="rl_prompt" -> truncate at first is_generation=True segment
* mode="rl_full"   -> emit full sequence; labels masked on prompt side
* max_length truncation: segment-aligned, never mid-segment
* split_prompt_response: structural-only; raises when no generation
"""

from __future__ import annotations

import pytest

from titaninfra.data.sft.assembler import (
    TruncatedToEmptyError,
    assemble,
    split_prompt_response,
)
from titaninfra.data.sft.segment import SegKind, Segment
from titaninfra.data.sft.types import IGNORE_INDEX, EncodedSample


# ----------------------------------------------------------------------
# A toy tokenizer keeps these tests independent of any HF fixture.
# encode("abc") == [ord('a'), ord('b'), ord('c')]; sufficient for
# checking offsets, lengths, label masking, loss_weight emission.
# ----------------------------------------------------------------------
class _CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False, "Assembler must pass add_special_tokens=False"
        return [ord(c) for c in text]


@pytest.fixture
def tok() -> _CharTokenizer:
    return _CharTokenizer()


def _seg(text=None, *, ids=None, kind=SegKind.CONTENT, role="assistant",
         scale=1.0, gen=False, midx=None, token_class=None) -> Segment:
    # Phase 1.4: token_class is required. Default policy for tests:
    # loss>0 → DECODED (the model emits these), loss==0 → PROTOCOL
    # (framework markers). Explicit `token_class=` overrides.
    if token_class is None:
        from titaninfra.data.sft.segment import TokenClass

        token_class = TokenClass.DECODED if scale > 0 else TokenClass.PROTOCOL
    return Segment(
        text=text, token_ids=ids, kind=kind, role=role,
        scale_factor=scale, is_generation=gen,
        token_class=token_class, msg_index=midx,
    )


# ----------------------------------------------------------------------
# Output structural invariants
# ----------------------------------------------------------------------
class TestStructure:
    def test_returns_encoded_sample(self, tok):
        out = assemble([_seg(text="hi", scale=1.0)], tok, mode="sft")
        assert isinstance(out, EncodedSample)

    def test_input_ids_length_matches_total_segment_tokens(self, tok):
        segs = [_seg(text="ab", scale=0.0), _seg(text="cd", scale=1.0)]
        out = assemble(segs, tok, mode="sft")
        assert len(out.input_ids) == 4
        assert out.input_ids == (ord("a"), ord("b"), ord("c"), ord("d"))

    def test_positions_are_sequential_from_zero(self, tok):
        out = assemble([_seg(text="hello", scale=1.0)], tok, mode="sft")
        assert out.positions == tuple(range(5))

    def test_labels_and_positions_aligned(self, tok):
        out = assemble([_seg(text="hello", scale=1.0)], tok, mode="sft")
        n = len(out.input_ids)
        assert len(out.labels) == n
        assert len(out.positions) == n

    def test_empty_segments_rejected(self, tok):
        with pytest.raises(ValueError):
            assemble([], tok, mode="sft")


# ----------------------------------------------------------------------
# text vs token_ids paths
# ----------------------------------------------------------------------
class TestTokenSources:
    def test_text_path_calls_tokenizer(self, tok):
        out = assemble([_seg(text="abc", scale=1.0)], tok, mode="sft")
        assert out.input_ids == (ord("a"), ord("b"), ord("c"))

    def test_token_ids_path_bypasses_tokenizer(self, tok):
        # token_ids path must not be re-encoded; produce exact ids verbatim.
        out = assemble([_seg(ids=(151644, 872, 198), scale=0.0)], tok, mode="sft")
        assert out.input_ids == (151644, 872, 198)

    def test_mixed_text_and_token_ids(self, tok):
        segs = [
            _seg(ids=(100,), scale=0.0, kind=SegKind.MARKER, role="assistant"),
            _seg(text="hi", scale=1.0),
            _seg(ids=(200,), scale=0.0, kind=SegKind.MARKER, role="assistant"),
        ]
        out = assemble(segs, tok, mode="sft")
        assert out.input_ids == (100, ord("h"), ord("i"), 200)


# ----------------------------------------------------------------------
# loss_scale: 0 / 1 / continuous
# ----------------------------------------------------------------------
class TestLossScale:
    def test_zero_masks_with_ignore_index(self, tok):
        out = assemble([_seg(text="ab", scale=0.0)], tok, mode="sft")
        assert out.labels == (IGNORE_INDEX, IGNORE_INDEX)
        assert out.loss_weights is None  # binary fast path

    def test_one_emits_token_id_label(self, tok):
        out = assemble([_seg(text="ab", scale=1.0)], tok, mode="sft")
        assert out.labels == (ord("a"), ord("b"))
        assert out.loss_weights is None  # binary fast path

    def test_mixed_binary_segments_stay_binary_fast_path(self, tok):
        segs = [_seg(text="ab", scale=0.0), _seg(text="cd", scale=1.0)]
        out = assemble(segs, tok, mode="sft")
        assert out.labels == (IGNORE_INDEX, IGNORE_INDEX, ord("c"), ord("d"))
        assert out.loss_weights is None

    def test_continuous_scale_emits_loss_weights(self, tok):
        segs = [
            _seg(text="ab", scale=1.0),
            _seg(text="cd", scale=0.3),  # forces non-binary path
        ]
        out = assemble(segs, tok, mode="sft")
        assert out.labels == (ord("a"), ord("b"), ord("c"), ord("d"))
        assert out.loss_weights == (1.0, 1.0, 0.3, 0.3)

    def test_continuous_scale_zero_segment_still_ignore_index(self, tok):
        segs = [
            _seg(text="ab", scale=0.5),  # non-binary forces loss_weights
            _seg(text="cd", scale=0.0),  # this one still gets IGNORE_INDEX
        ]
        out = assemble(segs, tok, mode="sft")
        assert out.labels == (ord("a"), ord("b"), IGNORE_INDEX, IGNORE_INDEX)
        assert out.loss_weights == (0.5, 0.5, 0.0, 0.0)


# ----------------------------------------------------------------------
# RL modes
# ----------------------------------------------------------------------
class TestRLModes:
    def test_rl_prompt_truncates_at_first_generation(self, tok):
        segs = [
            _seg(text="prompt ", scale=0.0, role="user"),
            _seg(text="ans", scale=1.0, role="assistant", gen=True),  # first gen
            _seg(text="more", scale=1.0, role="assistant", gen=True),
        ]
        out = assemble(segs, tok, mode="rl_prompt")
        assert out.input_ids == tuple(ord(c) for c in "prompt ")
        assert all(lab == IGNORE_INDEX for lab in out.labels)

    def test_rl_prompt_raises_when_no_generation_segment(self, tok):
        with pytest.raises(ValueError):
            assemble([_seg(text="x", scale=0.0)], tok, mode="rl_prompt")

    def test_rl_full_emits_full_sequence_prompt_masked(self, tok):
        segs = [
            _seg(text="pr", scale=0.0, role="user"),
            _seg(text="ans", scale=1.0, role="assistant", gen=True),
        ]
        out = assemble(segs, tok, mode="rl_full")
        # prompt side masked regardless of declared scale
        assert out.labels[:2] == (IGNORE_INDEX, IGNORE_INDEX)
        # response side carries its tokens
        assert out.labels[2:] == tuple(ord(c) for c in "ans")


# ----------------------------------------------------------------------
# max_length truncation -- segment-aligned, never mid-segment
# ----------------------------------------------------------------------
class TestMaxLength:
    def test_max_length_drops_trailing_segments_that_overflow(self, tok):
        # Use distinct kinds so the Phase 4 merge pass does not collapse
        # these two segments into a single "abcd" segment.
        # Audit #4 (2026-06-06): truncation that keeps ONLY the scale=0
        # prompt MARKER (dropping the sole loss-bearing CONTENT) would yield
        # an all-IGNORE_INDEX sample (zero training signal). That must now
        # raise TruncatedToEmptyError (caught+skipped by streaming_dataset),
        # NOT silently return a no-loss sample.
        segs = [
            _seg(text="ab", kind=SegKind.MARKER, scale=0.0),
            _seg(text="cd", kind=SegKind.CONTENT, scale=1.0),
        ]
        with pytest.raises(TruncatedToEmptyError):
            assemble(segs, tok, mode="sft", max_length=2)

    def test_max_length_keeps_sample_when_loss_survives_truncation(self, tok):
        # Truncation that retains at least one loss-bearing segment is fine.
        # Distinct kinds prevent the Phase-4 merge pass from collapsing them.
        segs = [
            _seg(text="ab", kind=SegKind.CONTENT, scale=1.0, role="assistant"),
            _seg(text="cd", kind=SegKind.MARKER, scale=0.0),
        ]
        out = assemble(segs, tok, mode="sft", max_length=2)
        assert out.input_ids == (ord("a"), ord("b"))
        assert any(lbl != IGNORE_INDEX for lbl in out.labels)

    def test_truncate_at_last_segment_to_zero_loss_raises(self, tok):
        # Opt-in mid-segment truncation that lands before the first
        # loss-bearing segment → all-IGNORE → must raise (audit #4).
        segs = [
            _seg(text="abcd", kind=SegKind.MARKER, scale=0.0),
            _seg(text="efgh", kind=SegKind.CONTENT, scale=1.0, role="assistant"),
        ]
        with pytest.raises(TruncatedToEmptyError):
            assemble(segs, tok, mode="sft", max_length=6,
                     truncate_at_last_segment=True)

    def test_max_length_equal_to_total_keeps_everything(self, tok):
        out = assemble([_seg(text="abcd", scale=1.0)], tok, mode="sft", max_length=4)
        assert len(out.input_ids) == 4

    def test_max_length_in_middle_of_segment_raises(self, tok):
        # M-alpha spec: truncation must not split a segment -- caller must
        # rebudget or pre-trim. Better fail-loud than emit a malformed
        # half-segment whose role-marker/content boundary is broken.
        segs = [_seg(text="abcd", scale=1.0)]
        with pytest.raises(ValueError):
            assemble(segs, tok, mode="sft", max_length=2)


# ----------------------------------------------------------------------
# Mode validation
# ----------------------------------------------------------------------
class TestMode:
    def test_invalid_mode_rejected(self, tok):
        with pytest.raises(ValueError):
            assemble([_seg(text="x", scale=1.0)], tok, mode="bogus")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# split_prompt_response -- pure structural
# ----------------------------------------------------------------------
class TestSplitPromptResponse:
    def test_split_at_first_generation(self):
        s0 = _seg(text="a", scale=0.0, role="user")
        s1 = _seg(text="b", scale=1.0, role="assistant", gen=True)
        s2 = _seg(text="c", scale=1.0, role="assistant", gen=True)
        prompt, response = split_prompt_response([s0, s1, s2])
        assert prompt == [s0]
        assert response == [s1, s2]

    def test_split_raises_when_no_generation(self):
        with pytest.raises(ValueError):
            split_prompt_response([_seg(text="a", scale=0.0)])


# ----------------------------------------------------------------------
# add_generation_prompt -- vLLM-aligned interception
# ----------------------------------------------------------------------
def tok_decode(ids) -> str:
    """Round-trip _CharTokenizer ids -> chars for byte-level assertions."""
    return "".join(chr(i) for i in ids)


class TestAddGenerationPrompt:
    """The flag lets RL prompt builders skip jinja yet stay byte-aligned
    with vLLM/SGLang's ``apply_chat_template(add_generation_prompt=True)``
    output. Marker-inclusive semantics: the first ``is_generation=True``
    segment (the assistant role marker) is appended to the prompt cut
    with ``loss_scale=0``.
    """

    def _build_assistant_turn_segments(self):
        # Models a user turn followed by an assistant turn split into
        # role marker (gen, mask=0) + content (gen, mask=1) + trailing
        # EOS STOP_MARKER (gen, mask=1 -- the model's own trained terminator).
        # Mirrors how qwen2_5 / qwen3 / deepseek_v3 emit segments.
        return [
            _seg(text="USR:hi;", scale=0.0, role="user"),
            _seg(text="AST:", scale=0.0, role="assistant", gen=True, kind=SegKind.MARKER),
            _seg(text="response", scale=1.0, role="assistant", gen=True),
            _seg(text="<eot>", scale=1.0, role="assistant", gen=True, kind=SegKind.STOP_MARKER),
        ]

    def test_default_false_preserves_existing_rl_prompt_behavior(self, tok):
        segs = self._build_assistant_turn_segments()
        out = assemble(segs, tok, mode="rl_prompt")
        # Prompt-side only: 'USR:hi;' = 7 chars
        assert len(out.input_ids) == len("USR:hi;")
        assert all(lbl == IGNORE_INDEX for lbl in out.labels)
        assert tok_decode(out.input_ids) == "USR:hi;"

    def test_true_appends_first_generation_marker(self, tok):
        segs = self._build_assistant_turn_segments()
        out = assemble(segs, tok, mode="rl_prompt", add_generation_prompt=True)
        # Prompt + assistant role marker: 'USR:hi;' + 'AST:'
        assert tok_decode(out.input_ids) == "USR:hi;AST:"
        # Every label still IGNORE_INDEX (the marker is class-A protocol
        # framing and never trains).
        assert all(lbl == IGNORE_INDEX for lbl in out.labels)
        assert len(out.input_ids) == 11

    def test_true_coerces_marker_loss_scale_to_zero(self, tok):
        # A STOP_MARKER legitimately carries loss in SFT (it's the model's
        # own trained terminator), but on the RL PROMPT side it is framing,
        # not generation -- add_generation_prompt must coerce its loss to 0
        # so the returned labels never leak training signal into the prompt.
        # (A plain framework MARKER with loss is now unconstructible -- assert-1
        # in Segment.__post_init__ forbids it -- so STOP_MARKER is the only way
        # a loss-bearing marker can reach this path.)
        segs = [
            _seg(text="USR:hi;", scale=0.0, role="user"),
            _seg(text="AST:", scale=1.0, role="assistant", gen=True,
                 kind=SegKind.STOP_MARKER),
            _seg(text="body", scale=1.0, role="assistant", gen=True),
        ]
        out = assemble(segs, tok, mode="rl_prompt", add_generation_prompt=True)
        assert tok_decode(out.input_ids) == "USR:hi;AST:"
        assert all(lbl == IGNORE_INDEX for lbl in out.labels)

    def test_true_rejects_sft_mode(self, tok):
        # sft mode emits the FULL conversation -- the assistant marker
        # is already in the segment list. Setting the flag here would
        # be a no-op at best and a double-emit if someone added a
        # synthesis path later. Fail loud instead.
        segs = self._build_assistant_turn_segments()
        with pytest.raises(ValueError, match="requires mode='rl_prompt'"):
            assemble(segs, tok, mode="sft", add_generation_prompt=True)

    def test_true_rejects_rl_full_mode(self, tok):
        segs = self._build_assistant_turn_segments()
        with pytest.raises(ValueError, match="requires mode='rl_prompt'"):
            assemble(segs, tok, mode="rl_full", add_generation_prompt=True)

    def test_true_fails_loud_when_no_generation_segment(self, tok):
        # Encoder forgot to mark any segment as is_generation. The
        # base rl_prompt validation catches this -- the new flag must
        # not weaken it (silent fallback would defeat the whole point).
        segs = [_seg(text="abc", scale=0.0, role="user")]
        with pytest.raises(ValueError, match="no segment with is_generation"):
            assemble(segs, tok, mode="rl_prompt", add_generation_prompt=True)

    def test_byte_identity_against_concat_render(self, tok):
        # The cut + append must reproduce the wire bytes you'd get from
        # concatenating prompt segments with the marker text directly.
        # This is the contract that lets vLLM consume titaninfra's RL
        # prompts unchanged.
        segs = self._build_assistant_turn_segments()
        out = assemble(segs, tok, mode="rl_prompt", add_generation_prompt=True)
        expected_text = "USR:hi;" + "AST:"
        assert tok_decode(out.input_ids) == expected_text
