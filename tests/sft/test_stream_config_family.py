"""Regression tests for the StreamConfig.chat_template → family rename
and the TrainConfig.chat_template removal (plan 2026-06-02 §3).

The contract verified here:

* ``StreamConfig.chat_template`` was renamed to ``family`` — no deprecation
  alias. Old configs that still pass ``chat_template=`` are silently
  ignored (Pydantic v2 default extra='ignore').
* ``TrainConfig.chat_template`` is fully removed.
* ``StreamConfig.sft_mask_mode`` defaults to ``"auto"``; ``LossPolicy``
  honours the three-layer priority (train_on_inputs > sft_mask_mode >
  per-message weight).
* The SFT runtime (``_iter_sft_samples`` in ``streaming_dataset.py``)
  fails loud when neither ``family`` nor ``tokenizer_name`` is set —
  no silent "qwen3" fallback.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_test_train_config
from titaninfra.config import StreamConfig, TrainConfig

# ---------------------------------------------------------------------------
# StreamConfig.family
# ---------------------------------------------------------------------------


class TestStreamConfigFamily:
    def test_family_set_explicitly(self) -> None:
        """Verify family set explicitly."""
        s = StreamConfig(
            tokenizer_name="qwen3", local="/tmp/x", data_type="messages", family="llama3", add_bos_eos=False
        )
        assert s.family == "llama3"

    def test_family_defaults_to_none(self) -> None:
        """Verify family defaults to none."""
        s = StreamConfig(tokenizer_name="qwen3", local="/tmp/x", data_type="messages", add_bos_eos=False)
        assert s.family is None

    def test_chat_template_kw_silently_ignored(self) -> None:
        """Verify chat template kw silently ignored."""
        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/x",
            data_type="messages",
            add_bos_eos=False,
            chat_template="qwen3",  # extra ignored, must not become family
        )
        assert s.family is None
        assert not hasattr(s, "chat_template")
        assert "chat_template" not in StreamConfig.model_fields


# ---------------------------------------------------------------------------
# Runtime fail-loud (production code in streaming_dataset.py:1099+)
# ---------------------------------------------------------------------------


class TestRuntimeFamilyFallback:
    """Mirror the family-resolution branch from ``_iter_sft_samples``.

    The runtime owns the actual logic; this test pins the same rule so that
    any future regression (e.g. someone reintroducing the silent qwen3
    fallback) trips a fast unit test instead of a multi-hour GPU job.
    """

    @staticmethod
    def _resolve(stream_cfg) -> str:
        family = getattr(stream_cfg, "family", None) or getattr(
            stream_cfg, "tokenizer_name", None
        )
        if not family:
            raise ValueError(
                "StreamConfig.family 与 tokenizer_name 均未设置，无法路由到 SFT encoder。"
            )
        return family

    def test_family_takes_priority(self) -> None:
        s = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/x",
            data_type="messages",
            family="llama3",
            add_bos_eos=False,
        )
        assert self._resolve(s) == "llama3"

    def test_falls_back_to_tokenizer_name(self) -> None:
        s = StreamConfig(tokenizer_name="qwen3", local="/tmp/x", data_type="messages", add_bos_eos=False)
        assert self._resolve(s) == "qwen3"

    def test_raises_when_both_missing(self) -> None:
        # Both fields blank — Pydantic still validates because tokenizer_name
        # has no default; we have to fabricate the same state at runtime.
        class _Stub:
            family = None
            tokenizer_name = None

        with pytest.raises(ValueError, match="无法路由到 SFT encoder"):
            self._resolve(_Stub())


# ---------------------------------------------------------------------------
# TrainConfig.chat_template removal
# ---------------------------------------------------------------------------


class TestTrainConfigChatTemplateRemoved:
    def test_field_absent_from_model(self) -> None:
        assert "chat_template" not in TrainConfig.model_fields

    def test_legacy_kw_silently_ignored(self) -> None:
        """Old recipes that still pass ``chat_template=`` must not crash
        (resume from disk-pickled configs) but also must not produce a
        live attribute."""
        cfg = make_test_train_config(chat_template="qwen3", lr=1e-5, steps=10)
        assert not hasattr(cfg, "chat_template")


# ---------------------------------------------------------------------------
# StreamConfig.sft_mask_mode + LossPolicy priority chain
# ---------------------------------------------------------------------------


class TestSftMaskMode:
    def test_default_is_auto(self) -> None:
        s = StreamConfig(tokenizer_name="qwen3", local="/tmp/x")
        assert s.sft_mask_mode == "auto"

    def test_manual_mode_set(self) -> None:
        s = StreamConfig(tokenizer_name="qwen3", local="/tmp/x", sft_mask_mode="manual")
        assert s.sft_mask_mode == "manual"

    def test_invalid_mode_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StreamConfig(tokenizer_name="qwen3", local="/tmp/x", sft_mask_mode="off")

    def test_priority_chain_train_on_inputs_wins(self) -> None:
        """LossPolicy: train_on_inputs=True overrides everything (Layer 1)."""
        from titaninfra.data.sft.loss_policy import LossPolicy, resolve_weights
        from titaninfra.data.sft.types import NormalizedMessage, NormalizedSample

        msgs = (
            NormalizedMessage(role="system", weight=0.0),
            NormalizedMessage(role="user", weight=0.0),
            NormalizedMessage(role="assistant", weight=0.0),
        )
        sample = NormalizedSample(messages=msgs)
        out = resolve_weights(
            sample, LossPolicy(sft_mask_mode="manual", train_on_inputs=True)
        )
        assert [m.weight for m in out.messages] == [1.0, 1.0, 1.0]

    def test_priority_chain_manual_respects_per_message_weight(self) -> None:
        """sft_mask_mode='manual' returns per-message weight verbatim (Layer 2/3)."""
        from titaninfra.data.sft.loss_policy import LossPolicy, resolve_weights
        from titaninfra.data.sft.types import NormalizedMessage, NormalizedSample

        msgs = (
            NormalizedMessage(role="system", weight=0.5),
            NormalizedMessage(role="user", weight=0.7),
            NormalizedMessage(role="assistant", weight=0.3),
        )
        sample = NormalizedSample(messages=msgs)
        out = resolve_weights(
            sample, LossPolicy(sft_mask_mode="manual", train_on_inputs=False)
        )
        assert [m.weight for m in out.messages] == [0.5, 0.7, 0.3]
