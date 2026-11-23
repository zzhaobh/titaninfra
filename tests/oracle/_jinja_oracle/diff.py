"""
Three-level diff between native and Jinja ``EncodedSample`` outputs.

M0 ships a minimal-but-correct implementation:

* **L1** — ``input_ids`` exact match
* **L2** — ``labels`` exact match
* **L3** — message-span boundaries (start/end token index per
  role). M0 leaves L3 as a TODO because span detection requires
  the per-encoder role-marker registry that lands with M1's
  Qwen3 PoC. The L3 entry is present in the report so dashboards
  can already render the schema.

The diff is deliberately read-only. It returns a JSON-friendly
dict that offline CI fixtures and oracle parity tests consume.
"""

from __future__ import annotations

from typing import Any

from titaninfra.data.sft.types import EncodedSample

_CONTEXT_WINDOW: int = 8


def compare_encoded(
    *,
    native: EncodedSample,
    jinja: EncodedSample,
    family: str,
    fixture: str | None = None,
) -> dict[str, Any]:
    """Return a structured diff report.

    Report keys
    -----------
    ``ok``                  — True iff L1 and L2 both green
    ``family``              — passed through for log aggregation
    ``fixture``             — optional fixture identifier
    ``levels``              — per-level result dict (``L1``, ``L2``, ``L3``)
    ``first_divergence``    — earliest token index where ids or labels
                              disagree; None if identical

    The exact schema mirrors ``docs/sft/pipeline_refactor_plan.md`` §4.3
    so the same JSON can flow into offline CI reports and runtime
    shadow logs.
    """
    l1 = _diff_sequence(native.input_ids, jinja.input_ids, label="input_ids")
    l2 = _diff_sequence(native.labels, jinja.labels, label="labels")
    # L3 stub — populated by M1 with role-marker spans per family.
    l3: dict[str, Any] = {"ok": True, "note": "L3 span diff not implemented in M0"}

    first_divergence = _pick_first(l1.get("first_idx"), l2.get("first_idx"))

    return {
        "ok": bool(l1["ok"] and l2["ok"]),
        "family": family,
        "fixture": fixture,
        "first_divergence": first_divergence,
        "levels": {"L1": l1, "L2": l2, "L3": l3},
    }


def _diff_sequence(
    native_seq: tuple[int, ...],
    jinja_seq: tuple[int, ...],
    *,
    label: str,
) -> dict[str, Any]:
    if native_seq == jinja_seq:
        return {"ok": True, "label": label, "len": len(native_seq)}

    first_idx = _first_divergence_index(native_seq, jinja_seq)
    lo = max(0, first_idx - _CONTEXT_WINDOW)
    hi_native = min(len(native_seq), first_idx + _CONTEXT_WINDOW)
    hi_jinja = min(len(jinja_seq), first_idx + _CONTEXT_WINDOW)

    return {
        "ok": False,
        "label": label,
        "len_native": len(native_seq),
        "len_jinja": len(jinja_seq),
        "first_idx": first_idx,
        "context_native": list(native_seq[lo:hi_native]),
        "context_jinja": list(jinja_seq[lo:hi_jinja]),
    }


def _first_divergence_index(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def _pick_first(*candidates: int | None) -> int | None:
    valid = [c for c in candidates if c is not None]
    return min(valid) if valid else None


__all__ = ["compare_encoded"]
