"""Renderers oracle baseline — thin wrapper around ``renderers.build_training_sample``.

Mirrors the role ``_jinja_oracle/_jinja_baseline.py`` plays for the Jinja
path: one chokepoint that turns a raw ``sample_json`` (the same shape the
``tests/fixtures/sft/<family>/*.json`` fixtures and the native encoder both
consume) into an ``EncodedSample`` using the ``renderers`` library instead of
HF's ``apply_chat_template``.

Qwen3-specific mapping decisions (see ``docs/renderers_oracle_feasibility_report.md``
for the full rationale):

* ``sample["thinking_intent"]["mode"] == "off"`` maps to
  ``Qwen3RendererConfig(enable_thinking=False)``; anything else (including
  absent) maps to the library default ``enable_thinking=True``. Native's own
  per-family default mirrors ``enable_thinking=False`` for plain samples with
  no explicit ``thinking_intent`` (see ``encoders/qwen3.py``), but this
  mapping only matters for the empty-think envelope's exact wire bytes
  (which is one of the two known deviation patterns below), not for
  anything this adapter needs to get "right" beyond faithfully invoking
  the library — see the feasibility report for why.
* ``normalize_to_hf_messages`` already adapts titaninfra's rich ``blocks``
  format to the ``{role, content, tool_calls, reasoning_content}`` shape
  ``renderers.Message`` expects (same adapter the Jinja oracle reuses), so
  this wrapper never re-implements that translation.
* No ``role_to_mask`` override is passed to ``build_training_sample`` — the
  library's own ``sampled_mask`` is used directly for the loss mask, per its
  documented "recommended default" (narrowing via ``role_to_mask`` cannot
  change the two known deviations below; see the feasibility report).

Two systematic, fully-explained label-mask deviations exist between this
oracle and both titaninfra's native encoder AND the Jinja oracle (confirmed
by exhaustive per-position classification across every ``test_qwen3_parity.py``
fixture + inline sample during the 2026-11 feasibility investigation):

* **Pattern A** (``renderers`` trains, native+Jinja mask): the empty
  ``<think>\\n\\n</think>`` envelope (titaninfra's "Quirk A", documented in
  ``CLAUDE.md`` as deliberately class-A/PROTOCOL, never trained) is marked
  ``is_sampled=True`` by ``renderers``' ``Qwen3Renderer`` whenever it emits
  that body — this is a hard-coded choice in the library
  (``renderers/qwen3.py``), not a config knob.
* **Pattern B** (``renderers`` masks, native trains): the single ``\\n``
  immediately following an assistant's closing ``<|im_end|>`` is emitted
  with ``is_sampled=False`` by ``renderers`` ("template-appended between
  turns and never sampled" per its own source comment), while titaninfra
  bundles it into the assistant's trainable ``STOP_MARKER`` segment.

``classify_label_divergence`` in ``diff.py`` detects exactly these two
patterns programmatically so tests can assert "no OTHER divergence" without
hand-listing token positions per fixture.
"""

from __future__ import annotations

from typing import Any

from renderers import Qwen3RendererConfig, build_training_sample, create_renderer

from titaninfra.data.sft.message_converter import normalize_to_hf_messages
from titaninfra.data.sft.types import IGNORE_INDEX, EncodedSample

_RENDERER_CACHE: dict[bool, Any] = {}


def _get_qwen3_renderer(tokenizer, *, enable_thinking: bool):
    """Cache one renderer instance per ``enable_thinking`` value per tokenizer.

    ``create_renderer`` does nontrivial setup (special-token id resolution);
    reusing instances across fixtures in the same test session avoids
    redundant work. Keyed only on ``enable_thinking`` since all callers in
    this test suite share one tokenizer fixture.
    """
    key = bool(enable_thinking)
    if key not in _RENDERER_CACHE:
        _RENDERER_CACHE[key] = create_renderer(
            tokenizer, Qwen3RendererConfig(enable_thinking=enable_thinking)
        )
    return _RENDERER_CACHE[key]


def render_with_renderers(
    sample_json: dict,
    tokenizer,
    *,
    max_length: int | None = None,
) -> EncodedSample:
    """Render ``sample_json`` through the ``renderers`` qwen3 path.

    Returns an ``EncodedSample`` with ``positions = range(N)``, matching the
    contract ``render_with_jinja`` and ``dispatch._to_encoded`` both use, so
    callers can feed the result straight into
    ``tests.oracle._jinja_oracle.diff.compare_encoded``.
    """
    messages = sample_json.get("messages", [])
    system = sample_json.get("system")
    tools = sample_json.get("tools")
    mode = (sample_json.get("thinking_intent") or {}).get("mode")
    enable_thinking = mode != "off"

    renderer = _get_qwen3_renderer(tokenizer, enable_thinking=enable_thinking)
    hf_messages, _weights = normalize_to_hf_messages(messages, system=system)
    rendered = build_training_sample(renderer, hf_messages, tools=tools)

    ids = rendered.token_ids
    loss_mask = rendered.loss_mask
    if max_length is not None:
        ids = ids[:max_length]
        loss_mask = loss_mask[:max_length]

    labels = tuple(tid if keep else IGNORE_INDEX for tid, keep in zip(ids, loss_mask, strict=True))
    return EncodedSample(
        input_ids=tuple(ids),
        labels=labels,
        positions=tuple(range(len(ids))),
    )


__all__ = ["render_with_renderers"]
