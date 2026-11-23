"""
Jinja oracle baseline — thin wrapper around the legacy renderer.

Wrapping ``tokenize_with_hf_template`` here (rather than calling it
directly in oracle code) gives us one chokepoint to:

* Swap the legacy renderer for a frozen-version reference once
  ``_legacy/tokenizer.py`` is moved (M3).
* Cache rendered outputs across fixtures during offline diffs
  without polluting the legacy module.

The wrapper returns an ``EncodedSample`` — same contract as
``dispatch._to_encoded`` — so oracle callers never touch
``(input_ids, labels)`` tuples directly.
"""

from __future__ import annotations

from titaninfra.data.sft.types import EncodedSample


def render_with_jinja(
    sample_json: dict,
    tokenizer,
    *,
    max_length: int | None = None,
    **legacy_kwargs,
) -> EncodedSample:
    """Render ``sample_json`` through the legacy Jinja path.

    The returned ``EncodedSample`` has ``positions = range(N)``
    (single-document view), matching ``dispatch._to_encoded``.

    When the caller supplies ``loss_config.type_overrides`` we
    auto-flip ``return_weights=True`` so the downstream consumer
    never silently loses per-type loss multipliers — mirroring the
    contract the legacy dispatcher used to enforce.
    """
    from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template

    messages = sample_json.get("messages", [])
    system = sample_json.get("system")
    tools = sample_json.get("tools")

    legacy_kwargs.setdefault("system", system)
    legacy_kwargs.setdefault("tools", tools)
    legacy_kwargs["max_length"] = max_length

    lc = legacy_kwargs.get("loss_config")
    if lc and isinstance(lc, dict) and lc.get("type_overrides"):
        # Byte-match the deleted `_run_jinja` baseline: unconditional assign,
        # not setdefault. type_overrides without weights would silently lose
        # the per-type multipliers downstream.
        legacy_kwargs["return_weights"] = True

    result = tokenize_with_hf_template(messages, tokenizer, **legacy_kwargs)
    if len(result) == 3:
        input_ids, labels, loss_weights = result
    else:
        input_ids, labels = result[0], result[1]
        loss_weights = None
    ids = tuple(int(t) for t in input_ids)
    lbl = tuple(int(t) for t in labels)
    weights = tuple(float(w) for w in loss_weights) if loss_weights is not None else None
    return EncodedSample(
        input_ids=ids,
        labels=lbl,
        positions=tuple(range(len(ids))),
        loss_weights=weights,
    )


__all__ = ["render_with_jinja"]
