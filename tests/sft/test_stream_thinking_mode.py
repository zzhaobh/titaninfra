"""F6: StreamConfig.thinking_mode overrides per-sample thinking_intent.mode.

``thinking_scale`` already had both a per-sample knob and a StreamConfig
override; ``thinking_intent.mode`` only had the per-sample path. F6 adds
``StreamConfig.thinking_mode`` (on/off/None) so a recipe can force the
thinking branch for a whole stream without editing the data.

Priority (documented): StreamConfig > per-sample. None = fall through to
per-sample / family default.

streaming_dataset imports torchtitan, so this module skips on the macOS
CPU box and runs on the GPU node.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torchtitan", reason="streaming_dataset requires torchtitan")

from titaninfra.config.data import StreamConfig  # noqa: E402
from titaninfra.data.streaming_dataset import _make_messages_doc_iter  # noqa: E402

# A sample with a thinking block so mode on/off changes the wire.
_THINKING_SAMPLE = json.dumps(
    [
        {"role": "user", "content": "what is 2+2"},
        {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "text": "two plus two is four"},
                {"type": "text", "text": "4"},
            ],
        },
    ]
)


def _cfg(family: str, thinking_mode):
    return StreamConfig(
        tokenizer_name=family,
        family=family,
        local="/tmp/unused",
        data_type="messages",
        add_bos_eos=False,
        thinking_mode=thinking_mode,
    )


def _encode(samples, tok, cfg):
    return list(_make_messages_doc_iter(samples, tok, cfg, seq_len=256))


@pytest.mark.parametrize(
    "family,fixture",
    [
        pytest.param(
            "qwen3", "qwen3_tokenizer",
            marks=[
                pytest.mark.xdist_group("tok_qwen3"),
                pytest.mark.xfail(
                    reason="qwen3 dynamic-thinking: think tokens always present, "
                    "on/off wire identical by design",
                    strict=True,
                ),
            ],
        ),
        pytest.param("gpt_oss", "gpt_oss_tokenizer", marks=pytest.mark.xdist_group("tok_gpt_oss")),
    ],
)
def test_thinking_mode_on_off_changes_wire(family, fixture, request):
    """thinking_mode on vs off must produce different input_ids for the same sample."""
    tok = request.getfixturevalue(fixture)
    samples = [{"messages": _THINKING_SAMPLE}]

    on_docs = _encode([dict(samples[0])], tok, _cfg(family, "on"))
    off_docs = _encode([dict(samples[0])], tok, _cfg(family, "off"))

    on_ids = [t[0] for t in on_docs[0]]
    off_ids = [t[0] for t in off_docs[0]]
    assert on_ids != off_ids, (
        f"[{family}] thinking_mode on/off produced identical wire — "
        "the override did not reach the encoder"
    )


@pytest.mark.parametrize(
    "family,fixture",
    [
        pytest.param("qwen3", "qwen3_tokenizer", marks=pytest.mark.xdist_group("tok_qwen3")),
        pytest.param("gpt_oss", "gpt_oss_tokenizer", marks=pytest.mark.xdist_group("tok_gpt_oss")),
    ],
)
def test_thinking_mode_none_matches_no_override(family, fixture, request):
    """thinking_mode=None must equal the no-thinking_intent baseline (no override)."""
    tok = request.getfixturevalue(fixture)
    samples = [{"messages": _THINKING_SAMPLE}]

    none_docs = _encode([dict(samples[0])], tok, _cfg(family, None))
    baseline_docs = _encode([dict(samples[0])], tok, _cfg(family, None))
    assert [t[0] for t in none_docs[0]] == [t[0] for t in baseline_docs[0]]


@pytest.mark.xdist_group("tok_qwen3")
def test_stream_thinking_mode_overrides_per_sample(qwen3_tokenizer):
    """StreamConfig.thinking_mode wins over a per-sample thinking_intent.mode."""
    tok = qwen3_tokenizer
    # Per-sample says ON; StreamConfig forces OFF → must match a fresh OFF.
    sample_on_in_data = {
        "messages": _THINKING_SAMPLE,
        "thinking_intent": json.dumps({"mode": "on"}),
    }
    forced_off = _encode([dict(sample_on_in_data)], tok, _cfg("qwen3", "off"))

    plain_off = _encode([{"messages": _THINKING_SAMPLE}], tok, _cfg("qwen3", "off"))
    assert [t[0] for t in forced_off[0]] == [t[0] for t in plain_off[0]], (
        "StreamConfig.thinking_mode=off did not override per-sample mode=on"
    )
