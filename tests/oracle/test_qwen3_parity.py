"""Qwen3 oracle parity — full coverage from plain text through tool-use.

Post jinja-purge: ``tokenize_sample`` is the single runtime path; the
oracle baseline (``render_with_jinja``) is the independent Jinja
reference used for regression detection. A divergence between the two
means either the encoder drifted or the legacy Jinja template changed.

Test groups:

* Dispatch smoke + tamper control — proves the harness reacts to real
  divergence (no-op compare would silently pass everything else).
* Plain-text / system / thinking parity — locks the M1.x delegation.
* History thinking-drop (Quirk B) — documents a *known* divergence.
* Step B tool-use parity (fixtures 05/06/07/08) — exercises each
  conditional in the Qwen3 jinja template's tool-use logic.
* Tool-role boundary regression — sanity bounds on loss-bearing tokens.
"""

from __future__ import annotations

import pytest

from tests.oracle._jinja_oracle._jinja_baseline import render_with_jinja
from tests.oracle._jinja_oracle.diff import compare_encoded
from titaninfra.data.sft.dispatch import tokenize_sample
from titaninfra.data.sft.types import EncodedSample


def test_dispatch_smoke(qwen3_tokenizer, load_oracle_fixture):
    """``tokenize_sample`` emits a well-formed ``EncodedSample``."""
    fx = load_oracle_fixture("qwen3/01_single_turn_plain.json")
    out = tokenize_sample(
        fx.sample,
        qwen3_tokenizer,
        family="qwen3",
        max_length=fx.max_length,
    )
    assert isinstance(out, EncodedSample)
    assert len(out.input_ids) > 0
    assert len(out.input_ids) == len(out.labels) == len(out.positions)
    assert out.positions == tuple(range(len(out.input_ids)))


def test_oracle_diff_catches_tampered_tokens(qwen3_tokenizer, load_oracle_fixture):
    """Negative control: oracle must flip ``ok=False`` on a forced delta.

    Without this test, a no-op ``compare_encoded`` (e.g. always
    returning ok=True) would pass every parity assertion by accident.
    """
    fx = load_oracle_fixture("qwen3/01_single_turn_plain.json")
    baseline = render_with_jinja(
        fx.sample,
        qwen3_tokenizer,
        max_length=fx.max_length,
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
        native=tampered,
        jinja=baseline,
        family="qwen3",
        fixture=None,
    )
    assert report["ok"] is False
    assert report["first_divergence"] == tamper_idx
    assert report["levels"]["L1"]["ok"] is False
    assert report["levels"]["L1"]["first_idx"] == tamper_idx
    assert report["levels"]["L2"]["ok"] is True


def test_identity_encoder_rejects_unsupported_features(
    qwen3_tokenizer,
    identity_qwen3_encoder,
):
    """``IdentityEncoder`` must fail loudly on thinking blocks.

    The point of the shim is to round-trip plain text faithfully.
    Silently dropping a thinking block would yield a "valid" but
    semantically wrong baseline — exactly the failure mode we want
    a real native encoder to surface, never to mask.
    """
    sample = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "blocks": [
                    {"type": "thinking", "text": "let me think"},
                    {"type": "text", "text": "ok"},
                ],
            },
        ]
    }
    with pytest.raises(NotImplementedError, match="thinking"):
        tokenize_sample(
            sample,
            qwen3_tokenizer,
            family=identity_qwen3_encoder,
            max_length=256,
        )


# ----------------------------------------------------------------------
# Real Qwen3Encoder (registered as "qwen3") — plain / system / thinking
# ----------------------------------------------------------------------
def test_qwen3_native_matches_jinja_oracle_ok(qwen3_tokenizer, load_oracle_fixture):
    """Headline: ``Qwen3Encoder`` reports ``ok=True`` vs Jinja oracle.

    The encoder delegates to ``tokenize_with_hf_template`` in M1.x so
    this is structurally tautological — both paths share the renderer.
    The test exists to prove the **registry + dispatch wiring**.
    """
    fx = load_oracle_fixture("qwen3/01_single_turn_plain.json")
    jinja_out = render_with_jinja(
        fx.sample,
        qwen3_tokenizer,
        max_length=fx.max_length,
        template_name="qwen3",
    )
    native_out = tokenize_sample(
        fx.sample,
        qwen3_tokenizer,
        family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out,
        jinja=jinja_out,
        family="qwen3",
        fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert report["levels"]["L1"]["ok"] is True
    assert report["levels"]["L2"]["ok"] is True


def test_qwen3_with_system_oracle_ok(qwen3_tokenizer, load_oracle_fixture):
    """System prompt flows through dispatch → encoder → renderer.

    No known Jinja quirk for plain system+user+assistant; native must
    match jinja bit-for-bit (including the empty ``<think>\\n\\n</think>``
    that Jinja injects on the last assistant — that injection is the
    M2-preserved baseline for plain text, distinct from Quirk B).
    """
    fx = load_oracle_fixture("qwen3/02_with_system.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer,
        max_length=fx.max_length, template_name="qwen3",
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert fx.expected["oracle_ok"] is True


def test_qwen3_last_asst_thinking_oracle_ok(qwen3_tokenizer, load_oracle_fixture):
    """Single-turn with thinking on the last assistant.

    The NormalizedSample carries thinking as a ``thinking`` block;
    ``normalize_content`` embeds it into ``msg.text`` as
    ``<think>\\n{text}\\n</think>\\n\\n{rest}``. Jinja preserves this
    envelope unchanged on the *last* assistant turn — sidestepping
    Quirk A (empty-think injection that fires only when content has
    no ``<think>`` tag at all).
    """
    fx = load_oracle_fixture("qwen3/03_thinking_last_assistant.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer, max_length=fx.max_length,
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert fx.expected["oracle_ok"] is True


def test_qwen3_type_overrides_thinking_loss_weights_parity(
    qwen3_tokenizer, load_oracle_fixture
):
    """Native must emit the same ``loss_weights`` as the Jinja oracle.

    Both modes are passed ``loss_config.type_overrides.thinking = 0.5``.
    The Jinja oracle baseline requests ``return_weights=True`` whenever
    ``type_overrides`` is set; the ``Qwen3Encoder.encode`` path mirrors
    that policy. Verifying both paths return identical ``loss_weights``
    tuples (not just both non-None) catches the silent failure mode where
    native flips back to ``return_weights=False`` and silently drops the
    thinking-scale multiplier.

    Fixture 03 is single-turn with thinking on the last assistant, so the
    thinking marker range is unambiguous and non-empty.
    """
    fx = load_oracle_fixture("qwen3/03_thinking_last_assistant.json")
    loss_config = {"type_overrides": {"thinking": 0.5}}

    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer,
        max_length=fx.max_length,
        loss_config=loss_config,
        return_weights=True,
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
        loss_config=loss_config,
    )

    assert jinja_out.loss_weights is not None, (
        "jinja path failed to emit loss_weights — type_overrides plumbing broken"
    )
    assert native_out.loss_weights is not None, (
        "native path failed to emit loss_weights — A.2 regression"
    )
    assert native_out.loss_weights == jinja_out.loss_weights, (
        f"loss_weights divergence: native={native_out.loss_weights[:10]}... "
        f"jinja={jinja_out.loss_weights[:10]}..."
    )
    # Sanity: at least one token has the 0.5 scale applied, otherwise the
    # test fixture has no thinking and the assertion above is vacuous.
    assert any(0.0 < w < 1.0 for w in native_out.loss_weights), (
        "loss_weights had no scaled tokens — fixture or thinking-range logic broken"
    )


def test_qwen3_history_thinking_drop(qwen3_tokenizer, load_oracle_fixture):
    """Multi-turn with historical thinking — Quirk B contract enforced.

    Jinja silently strips ``<think>...</think>`` from historical assistant
    turns. The token-first renderer preserves them, producing a
    documented divergence at token index 10 (native emits ``<think>`` /
    151667, Jinja emits ``a`` / 64). Asserting the exact index ensures
    the encoder fixes the *documented* quirk and not something else.
    """
    fx = load_oracle_fixture("qwen3/04_thinking_history_drop.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer, max_length=fx.max_length,
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    assert report["ok"] is False
    assert report["first_divergence"] == fx.expected["first_divergence"]


# ----------------------------------------------------------------------
# Step B — tool-use parity (fixtures 05/06/07/08)
# ----------------------------------------------------------------------
# Each test asserts byte-exact L1 (input_ids) + L2 (labels) equality
# between the M3+Step B native renderer and the jinja baseline on a
# fixture covering a different tool-use branch. Together they exercise
# every conditional in the Qwen3 jinja template's tool-use logic.


def test_qwen3_tool_call_oracle_ok(qwen3_tokenizer, load_oracle_fixture):
    """Step B fixture 05: assistant text + single tool_call.

    Pins the (a) Quirk A injection on a tool-call-bearing last assistant,
    (b) leading-\\n rule when ``loop.first AND content`` is truthy, and
    (c) ``<tool_call>`` block ordering between text content and
    ``<|im_end|>\\n``. Native must produce ``...Let me check.\\n<tool_call>
    \\n{...}\\n</tool_call><|im_end|>\\n`` to match jinja byte-for-byte.
    """
    fx = load_oracle_fixture("qwen3/05_tool_call.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer,
        max_length=fx.max_length, template_name="qwen3",
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert report["levels"]["L1"]["ok"] is True
    assert report["levels"]["L2"]["ok"] is True
    assert fx.expected["oracle_ok"] is True


def test_qwen3_tool_response_oracle_ok(qwen3_tokenizer, load_oracle_fixture):
    """Step B fixture 06: empty-content asst + tool_call + tool response.

    Pins three contracts a string-equality test on a simpler fixture would
    miss: (a) empty content + ``loop.first`` ⇒ NO leading ``\\n`` before
    ``<tool_call>``; (b) tool-role wraps as ``<|im_start|>user
    \\n<tool_response>\\n...\\n</tool_response><|im_end|>\\n``; (c)
    ``last_query_index`` must NOT regress to point at the wrapped tool
    response — endswith check is load-bearing.
    """
    fx = load_oracle_fixture("qwen3/06_tool_response.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer,
        max_length=fx.max_length, template_name="qwen3",
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert report["levels"]["L1"]["ok"] is True
    assert report["levels"]["L2"]["ok"] is True
    assert fx.expected["oracle_ok"] is True


def test_qwen3_tools_top_level_oracle_ok(qwen3_tokenizer, load_oracle_fixture):
    """Step B fixture 07: top-level tools + plain system + user/asst.

    Locks the tools-header coalescing contract: a single ``<|im_start|>system
    \\n`` block contains the system text joined with ``\\n\\n``, then
    ``# Tools\\n\\n...<tools>``, then ``\\n`` + jinja-compatible
    ``tojson(tool)`` per tool (``sort_keys=True, ensure_ascii=True``),
    closed by the suffix and ``<|im_end|>\\n``. Drift in tojson flags
    (sort_keys / ensure_ascii) would surface here as an L1 divergence
    inside the header bytes.
    """
    fx = load_oracle_fixture("qwen3/07_tools_top_level.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer,
        max_length=fx.max_length, template_name="qwen3",
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None
    assert report["levels"]["L1"]["ok"] is True
    assert report["levels"]["L2"]["ok"] is True
    assert fx.expected["oracle_ok"] is True


def test_qwen3_full_tool_loop_oracle_ok(qwen3_tokenizer, load_oracle_fixture):
    """Step B fixture 08: end-to-end loop exercising every branch in one shot.

    Combines tools header, multi-call assistant (``loop.first AND content``
    plus second-call always-\\n), two consecutive tool responses merged
    into a single ``<|im_start|>user`` block with one ``<|im_end|>\\n``
    only on the last in the chain, and final-asst Quirk A injection. If
    any tool-use branch regresses, this catches it even when the
    single-branch fixtures (05/06/07) pass.
    """
    fx = load_oracle_fixture("qwen3/08_full_tool_loop.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer,
        max_length=fx.max_length, template_name="qwen3",
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    assert report["ok"] is True
    assert report["first_divergence"] is None


def test_qwen3_math_tools_real_world_oracle_ok(qwen3_tokenizer, load_oracle_fixture):
    """Step B fixture 09: real-world math TIR sample from data/math_tools/.

    UARA shape: user -> assistant(thinking + tool_use) -> tool(tool_result) ->
    assistant(thinking + final_text). First fixture with thinking+tool_use on
    the same assistant message. Fixes _build_hf_messages to extract thinking
    to reasoning_content + strip <think> wrapper from content, mirroring the
    gemma4 fix. Now oracle_ok=True.
    """
    fx = load_oracle_fixture("qwen3/09_math_tools_real_world.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer,
        max_length=fx.max_length, template_name="qwen3",
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    if fx.expected["oracle_ok"]:
        assert report["ok"] is True
        assert report["first_divergence"] is None
    else:
        assert report["ok"] is False
        assert report["first_divergence"] == fx.expected["first_divergence"]
    assert report["levels"]["L1"]["ok"] is True
    assert report["levels"]["L2"]["ok"] is True
    assert fx.expected["oracle_ok"] is True


# ----------------------------------------------------------------------
# Tool-role boundary regression (Step B follow-up fix)
# ----------------------------------------------------------------------
# Qwen3 wraps ``role="tool"`` messages as
# ``<|im_start|>user\n<tool_response>...</tool_response>`` so the rendered
# text has no ``tool`` marker. Before the fix the fallback branch in
# ``_compute_message_boundaries`` set ``cp = header_chars[-1]`` (the
# preceding asst's header), so ``boundaries[asst+1]`` collapsed back to
# ``asst_header`` and the asst's loss range
# ``[content_starts[asst], boundaries[asst+1])`` became empty/negative —
# all labels masked even though weight=1.


def _asst_indices(messages):
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def test_qwen3_asst_before_tool_has_nonempty_loss_range(
    qwen3_tokenizer, load_oracle_fixture
):
    """Fixture 06: asst pure tool_call is followed by a tool message.

    Before the fix the asst's labels were all ``IGNORE_INDEX`` (-100)
    because the tool-role fallback collapsed its boundary. After the
    fix the asst's ``<tool_call>...</tool_call>`` tokens must appear
    in ``labels``.
    """
    fx = load_oracle_fixture("qwen3/06_tool_response.json")
    assert _asst_indices(fx.sample["messages"]), "fixture must contain asst"

    out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    non_masked = [lab for lab in out.labels if lab != -100]
    assert non_masked, (
        "asst before tool has no loss-bearing tokens — tool-role boundary "
        "fallback collapsed its range (Step B follow-up regression)."
    )


def test_qwen3_full_loop_both_assts_have_loss(
    qwen3_tokenizer, load_oracle_fixture
):
    """Fixture 08: asst(text+2 tool_calls) → 2 tools → asst(final text).

    Sanity bound: total loss-bearing tokens must exceed the final asst's
    plain text length, proving the first asst's tool_call tokens (which
    only exist when its boundary is preserved) actually land in labels.
    """
    fx = load_oracle_fixture("qwen3/08_full_tool_loop.json")
    asst_idxs = _asst_indices(fx.sample["messages"])
    assert len(asst_idxs) == 2, "fixture 08 expects exactly 2 assistants"

    out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    final_text = fx.sample["messages"][asst_idxs[-1]]["content"]
    final_ids = qwen3_tokenizer.encode(final_text, add_special_tokens=False)
    non_masked_count = sum(1 for lab in out.labels if lab != -100)
    assert non_masked_count > len(final_ids), (
        f"total loss-bearing tokens ({non_masked_count}) does not exceed "
        f"final-asst text length ({len(final_ids)}); first asst's tool_call "
        f"tokens are missing — tool-role boundary bug regression."
    )


def test_qwen3_user_after_tool_chain_keeps_its_marker(qwen3_tokenizer):
    """Tool chain followed by a real user must not steal the user's marker.

    Constructs user → asst(tool_call) → tool → tool → user(real). The
    renderer emits TWO ``<|im_start|>user`` markers (one wrapping the
    tool chain, one for the real user). The fix advances ``marker_idx``
    exactly once at the chain start (not on mid-chain tools), so the
    trailing real user still consumes its own marker. Native vs Jinja
    oracle byte equality keeps both paths in lock-step.
    """
    sample = {
        "messages": [
            {"role": "user", "content": "trigger the tool"},
            {
                "role": "assistant",
                "blocks": [
                    {"type": "tool_use", "name": "f", "input": {"x": 1}, "id": "c0"}
                ],
            },
            {"role": "tool", "content": "r0", "tool_call_id": "c0"},
            {"role": "tool", "content": "r1", "tool_call_id": "c0"},
            {"role": "user", "content": "now answer in words"},
        ]
    }
    jinja_out = render_with_jinja(
        sample, qwen3_tokenizer,
        max_length=512, template_name="qwen3",
    )
    native_out = tokenize_sample(
        sample, qwen3_tokenizer, family="qwen3",
        max_length=512,
    )
    assert jinja_out.input_ids == native_out.input_ids
    assert jinja_out.labels == native_out.labels


def test_qwen3_plain_system_path_unaffected_by_wrapped_fallback(qwen3_tokenizer):
    """Sentinel: refactor must not perturb the ``role in marker_roles`` path.

    Qwen3 has a ``system`` marker, so the wrapped-role branch is never
    entered for this sample. Native vs Jinja oracle must stay byte-exact.
    """
    sample = {
        "system": "You are concise.",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
        ],
    }
    jinja_out = render_with_jinja(
        sample, qwen3_tokenizer,
        max_length=128, template_name="qwen3",
    )
    native_out = tokenize_sample(
        sample, qwen3_tokenizer, family="qwen3",
        max_length=128,
    )
    assert jinja_out.input_ids == native_out.input_ids
    assert jinja_out.labels == native_out.labels


def test_qwen3_assistant_text_leading_newline_oracle_ok(
    qwen3_tokenizer, load_oracle_fixture,
):
    """Regression: assistant TextBlock starting with ``\\n`` must mirror jinja.

    Real-world data hit this in ``data/sft_test/mixed/train.jsonl``
    samples 87 + 130 (GPU SFT Case 17, 2026-06-03). Encoder splits the
    assistant turn into MARKER + CONTENT segments with different ``kind``;
    the Phase 4 merge pass therefore cannot collapse the seam, and a
    BPE merge across ``assistant\\n`` + ``\\nBody...`` produces 1 fewer
    token in the joint render than in two-piece tokenization. Jinja
    line 45's ``content.lstrip('\\n')`` neutralises this; the encoder
    must mirror it. The fixture exercises both the plain CONTENT path
    and the quirk-A injected-empty-think path (single-turn, no thinking
    block ⇒ quirk-A active).
    """
    fx = load_oracle_fixture("qwen3/10_assistant_text_leading_newline.json")
    jinja_out = render_with_jinja(
        fx.sample, qwen3_tokenizer,
        max_length=fx.max_length, template_name="qwen3",
    )
    native_out = tokenize_sample(
        fx.sample, qwen3_tokenizer, family="qwen3",
        max_length=fx.max_length,
    )
    report = compare_encoded(
        native=native_out, jinja=jinja_out,
        family="qwen3", fixture=str(fx.path),
    )
    assert report["ok"] is True, report
    assert report["first_divergence"] is None
    assert fx.expected["oracle_ok"] is True



def test_qwen3_mode_off_consecutive_assistants_only_last_injects_envelope(
    qwen3_tokenizer, load_oracle_fixture
):
    """Audit #5 (2026-06-06): with thinking_intent.mode='off' and CONSECUTIVE
    post-query assistant turns (agentic multi-step: user->asst->asst, both
    after_query=True), the empty-``<think>`` envelope must be injected ONLY on
    the LAST assistant turn — mirroring jinja ``apply_chat_template(
    enable_thinking=False)``, which injects on the final turn only.

    The old ``quirk_a = after_query and not has_think and (is_last or
    mode_off)`` fired on EVERY post-query turn when mode_off, so the non-last
    assistant of a consecutive pair got a phantom ``<think>\\n\\n</think>\\n\\n``
    jinja never emits (L1 drift) plus a trained trailing ``\\n\\n`` (L2 drift).
    Note the ``after_query`` precondition means a NORMAL user/asst/user/asst
    convo never triggered this (intermediate asst is historical,
    after_query=False); the bug only bites consecutive post-query assistants.
    """
    sample = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "text": "q1"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "a1"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "a2"}]},
        ],
        "thinking_intent": {"mode": "off"},
    }
    jinja_out = render_with_jinja(
        sample, qwen3_tokenizer, max_length=None, template_name="qwen3",
    )
    native_out = tokenize_sample(sample, qwen3_tokenizer, family="qwen3")
    report = compare_encoded(
        native=native_out, jinja=jinja_out, family="qwen3",
        fixture="inline:mode_off_consecutive_assistants",
    )
    assert report["levels"]["L1"]["ok"] is True, (
        f"L1 (input_ids) must match jinja — only the last consecutive asst "
        f"turn gets the empty-think envelope under mode_off: {report}"
    )
    assert report["levels"]["L2"]["ok"] is True, (
        f"L2 (labels) must match jinja — no phantom trained \\n\\n on the "
        f"non-last consecutive asst turn: {report}"
    )
