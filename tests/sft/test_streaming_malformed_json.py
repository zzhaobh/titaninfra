"""F7: malformed per-sample JSON in the messages doc-iter must warn, not swallow.

``_make_messages_doc_iter`` reads ``loss_config`` and ``tools`` as JSON
strings off each MDS sample. A malformed value previously hit a silent
``except: pass`` — a bad loss_config would make thinking_scale /
type_overrides vanish with zero signal. The runtime contract is
"data is valid by the time it reaches here" (validate-at-converter), so
we do NOT raise; but we must warn-loud so the drift is visible.

This feeds a tiny in-memory iterable (the function only does
``for sample in streaming_ds``) plus a real qwen3 tokenizer fixture, and
asserts the warning fires while encoding still succeeds.
"""

from __future__ import annotations

import json
import logging

import pytest

# streaming_dataset.py imports torchtitan (ParallelAwareDataloader) at module
# load; skip the whole module on hosts without it (macOS CPU box). Runs on the
# GPU node where torchtitan is installed.
pytest.importorskip("torchtitan", reason="streaming_dataset requires torchtitan")

from titaninfra.config.data import StreamConfig  # noqa: E402
from titaninfra.data.streaming_dataset import _make_messages_doc_iter  # noqa: E402

_GOOD_MESSAGES = json.dumps(
    [
        {"role": "user", "content": "what is 2+2"},
        {"role": "assistant", "content": "4"},
    ]
)


def _stream_cfg() -> StreamConfig:
    return StreamConfig(
        tokenizer_name="qwen3",
        family="qwen3",
        local="/tmp/unused",
        data_type="messages",
        add_bos_eos=False,
    )


@pytest.mark.xdist_group("tok_qwen3")
def test_malformed_loss_config_warns_and_continues(qwen3_tokenizer, caplog):
    """Bad loss_config JSON → warn-loud, sample still encodes (loss_config dropped)."""
    samples = [
        {"messages": _GOOD_MESSAGES, "loss_config": "{not valid json"},
    ]
    with caplog.at_level(logging.WARNING, logger="titaninfra.data.streaming_dataset"):
        docs = list(
            _make_messages_doc_iter(samples, qwen3_tokenizer, _stream_cfg(), seq_len=128)
        )
    assert any(
        "malformed loss_config JSON" in r.message for r in caplog.records
    ), f"expected loss_config warn-loud, got: {[r.message for r in caplog.records]}"
    # Encoding still produced the document (loss_config simply ignored).
    assert len(docs) == 1
    assert len(docs[0]) > 0


@pytest.mark.xdist_group("tok_qwen3")
def test_malformed_tools_warns_and_continues(qwen3_tokenizer, caplog):
    """Bad tools JSON → warn-loud, sample still encodes (tools dropped)."""
    samples = [
        {"messages": _GOOD_MESSAGES, "tools": "[{broken"},
    ]
    with caplog.at_level(logging.WARNING, logger="titaninfra.data.streaming_dataset"):
        docs = list(
            _make_messages_doc_iter(samples, qwen3_tokenizer, _stream_cfg(), seq_len=128)
        )
    assert any(
        "malformed tools JSON" in r.message for r in caplog.records
    ), f"expected tools warn-loud, got: {[r.message for r in caplog.records]}"
    assert len(docs) == 1
    assert len(docs[0]) > 0


@pytest.mark.xdist_group("tok_qwen3")
def test_valid_loss_config_does_not_warn(qwen3_tokenizer, caplog):
    """Well-formed loss_config → no warning."""
    samples = [
        {
            "messages": _GOOD_MESSAGES,
            "loss_config": json.dumps({"type_overrides": {"thinking": 0.5}}),
        },
    ]
    with caplog.at_level(logging.WARNING, logger="titaninfra.data.streaming_dataset"):
        docs = list(
            _make_messages_doc_iter(samples, qwen3_tokenizer, _stream_cfg(), seq_len=128)
        )
    assert not any("malformed" in r.message for r in caplog.records)
    assert len(docs) == 1
