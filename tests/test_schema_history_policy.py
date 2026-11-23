"""Phase 1.2 — msgspec round-trip for HistoryPolicy + HistorySummaryBlock.

Covers ``docs/superpowers/specs/2026-06-02-encoder-layer-refactor.md`` §3.2-3.3:

* HistoryAction / HistoryPolicy decode + default semantics
* HistorySummaryBlock decode under each of the three styles
* SFTSample.history_policy default = None (back-compat with on-disk MDS)
* Block-union discrimination: a ``"type": "history_summary"`` block
  inside a Message decodes to ``HistorySummaryBlock`` (not the catch-all)
"""

from __future__ import annotations

import msgspec
import pytest

from titaninfra.data.schema import (
    HistoryAction,
    HistoryPolicy,
    HistorySummaryBlock,
    decode_sample,
)


# ----------------------------------------------------------------------
# HistoryAction
# ----------------------------------------------------------------------
def test_history_action_keep_no_template() -> None:
    ha = msgspec.json.decode(b'{"action": "keep"}', type=HistoryAction)
    assert ha.action == "keep"
    assert ha.summary_template is None


def test_history_action_summarize_with_template() -> None:
    raw = b'{"action": "summarize", "summary_template": "[\\u5386\\u53f2\\u63a8\\u7406\\u7701\\u7565]"}'
    ha = msgspec.json.decode(raw, type=HistoryAction)
    assert ha.action == "summarize"
    assert ha.summary_template == "[历史推理省略]"


def test_history_action_drop() -> None:
    ha = msgspec.json.decode(b'{"action": "drop"}', type=HistoryAction)
    assert ha.action == "drop"


def test_history_action_invalid_literal_rejected() -> None:
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"action": "rewrite"}', type=HistoryAction)


# ----------------------------------------------------------------------
# HistoryPolicy
# ----------------------------------------------------------------------
def test_history_policy_default_empty() -> None:
    hp = msgspec.json.decode(b"{}", type=HistoryPolicy)
    assert hp.per_type == {}
    assert hp.insert_summary is False


def test_history_policy_per_type_decoded() -> None:
    raw = b"""{
        "per_type": {
            "thinking":    {"action": "keep"},
            "tool_use":    {"action": "summarize", "summary_template": "[ran {name}]"},
            "tool_result": {"action": "drop"}
        }
    }"""
    hp = msgspec.json.decode(raw, type=HistoryPolicy)
    assert set(hp.per_type) == {"thinking", "tool_use", "tool_result"}
    assert hp.per_type["thinking"].action == "keep"
    assert hp.per_type["tool_use"].action == "summarize"
    assert hp.per_type["tool_use"].summary_template == "[ran {name}]"
    assert hp.per_type["tool_result"].action == "drop"
    assert hp.insert_summary is False


def test_history_policy_insert_summary_flag() -> None:
    hp = msgspec.json.decode(b'{"insert_summary": true}', type=HistoryPolicy)
    assert hp.insert_summary is True
    assert hp.per_type == {}


# ----------------------------------------------------------------------
# HistorySummaryBlock
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "style",
    ["latest_reminder", "system_prefix", "ephemeral_block"],
)
def test_history_summary_block_each_style(style: str) -> None:
    raw = (
        b'{"type": "history_summary", "text": "older convo summary", "style": "'
        + style.encode()
        + b'"}'
    )
    blk = msgspec.json.decode(raw, type=HistorySummaryBlock)
    assert blk.text == "older convo summary"
    assert blk.style == style


def test_history_summary_block_default_style_system_prefix() -> None:
    raw = b'{"type": "history_summary", "text": "x"}'
    blk = msgspec.json.decode(raw, type=HistorySummaryBlock)
    assert blk.style == "system_prefix"
    assert blk.compressed_turn_count == 0


def test_history_summary_block_compressed_turn_count() -> None:
    raw = b'{"type": "history_summary", "text": "y", "compressed_turn_count": 5}'
    blk = msgspec.json.decode(raw, type=HistorySummaryBlock)
    assert blk.compressed_turn_count == 5


# ----------------------------------------------------------------------
# SFTSample integration
# ----------------------------------------------------------------------
# 2026-06-06). The former ``test_sft_sample_history_policy_defaults_none``
# and ``test_sft_sample_history_policy_decoded`` are removed — SFTSample no
# longer carries the field. StreamConfig-level coverage lives in
# tests/sft/test_history_policy_v2.py.
def test_sft_sample_history_summary_block_in_message() -> None:
    # The Block union must discriminate ``history_summary`` correctly.
    raw = b"""{
        "messages": [
            {"role": "system", "blocks": [
                {"type": "history_summary", "text": "summary", "style": "system_prefix"}
            ]},
            {"role": "user", "blocks": [{"type": "text", "text": "next"}]}
        ]
    }"""
    sample = decode_sample(raw)
    blk0 = sample.messages[0].blocks[0]
    assert isinstance(blk0, HistorySummaryBlock)
    assert blk0.text == "summary"
    assert blk0.style == "system_prefix"
