"""
Encoder registry — the single source of truth for ``family → encoder``.

Why a registry instead of an ``if/elif`` chain in ``dispatch.py``:

* **Adding a family is a one-file change.** Drop
  ``titaninfra/data/sft/encoders/<family>.py``, decorate the class with
  ``@register("<family>")``, and the dispatcher discovers it without
  any edit elsewhere. This is the M3 acceptance criterion ("new
  family = 1 encoder file + 1 fixture set + 1 test file").
* **Re-registration is fail-loud.** Two encoders silently shadowing
  the same name was the failure mode that produced the legacy
  ``tool_use_adapters`` patchwork; ``register()`` raises on collision.
* **Backward compatibility for the pre-M0 caller.** ``deepseek_v4``
  predates this module and still exposes the old
  ``encode(messages, tools=..., max_length=..., train_on_inputs=...,
  loss_config=...) -> (input_ids, labels)`` Protocol. We lazy-import
  it and return it unchanged so existing tests
  (``tests/test_dsv4_encoder.py``, ``tests/alignment/test_template_decoupling.py``)
  keep passing while M2 migrates the rest of the families.

The dispatcher is the only runtime caller. Tests and tooling MAY
``get_encoder(name, tokenizer)`` directly, but production paths go
through ``titaninfra.data.sft.dispatch.tokenize_sample``.
"""

from __future__ import annotations

from collections.abc import Callable

from titaninfra.data.sft.encoders.base import SFTEncoder
from titaninfra.data.sft.types import IGNORE_INDEX

# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
# ``_REGISTRY`` maps family name → encoder *class* (NOT instance). A new
# instance is constructed per ``get_encoder()`` call because encoders
# bind to a tokenizer; we cannot share instances across tokenizers.
_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    """Decorator: register ``cls`` as the encoder for ``name``.

    Example::

        @register("qwen3")
        class Qwen3Encoder:
            name = "qwen3"
            def __init__(self, tokenizer): ...
            def encode(self, sample, *, max_length=None): ...

    Re-registering an existing name is a programmer error (two
    encoders shadowing one family); raises ``ValueError`` immediately
    so the conflict surfaces at import time, not at the first batch.
    """

    def _decorator(cls: type) -> type:
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            raise ValueError(
                f"Encoder name {name!r} already registered to {existing!r}; "
                f"cannot re-register {cls!r}. Pick a unique name or call "
                f"unregister({name!r}) first."
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def unregister(name: str) -> None:
    """Remove ``name`` from the registry. Tests use this to isolate.

    No-op if ``name`` is absent; raising would force every teardown to
    guard with ``in _REGISTRY`` and offers no defensive value here.
    """
    _REGISTRY.pop(name, None)


def list_encoders() -> list[str]:
    """Return all registered family names. Used by diagnostics / CLIs."""
    return sorted(_REGISTRY)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def get_encoder(encoder_name: str, tokenizer) -> SFTEncoder:
    """Construct the encoder for ``encoder_name`` bound to ``tokenizer``.

    Resolution order:

    1. Registered encoders (``@register("...")``). This is the new
       Protocol — ``encode(sample, *, max_length=None) -> EncodedSample``.
    2. Pre-M0 ``deepseek_v4`` lazy import. Old Protocol
       ``encode(messages, *, tools=..., max_length=..., train_on_inputs=...,
       loss_config=...) -> (input_ids, labels)``. Kept until M2 migrates it.

    Order matters: a future ``@register("deepseek_v4")`` overrides the
    lazy import. That is the intended migration path — the new
    encoder replaces the legacy one in the same PR that ships it.
    """
    cls = _REGISTRY.get(encoder_name)
    if cls is not None:
        return cls(tokenizer)

    # Lazy-import families that register at module load. The import has
    # the side effect of running ``@register("...")``, after which a
    # second registry lookup hits the fast path on subsequent calls.
    if encoder_name == "qwen3":
        from titaninfra.data.sft.encoders import qwen3 as _qwen3  # noqa: F401

        return _REGISTRY["qwen3"](tokenizer)

    if encoder_name == "qwen3_5":
        from titaninfra.data.sft.encoders import qwen3_5 as _qwen3_5  # noqa: F401

        return _REGISTRY["qwen3_5"](tokenizer)

    if encoder_name == "qwen2_5":
        from titaninfra.data.sft.encoders import qwen2_5 as _qwen2_5  # noqa: F401

        return _REGISTRY["qwen2_5"](tokenizer)

    if encoder_name == "llama3":
        from titaninfra.data.sft.encoders import llama3 as _llama3  # noqa: F401

        return _REGISTRY["llama3"](tokenizer)

    if encoder_name == "gemma4":
        from titaninfra.data.sft.encoders import gemma4 as _gemma4  # noqa: F401

        return _REGISTRY["gemma4"](tokenizer)

    if encoder_name == "deepseek_v3":
        from titaninfra.data.sft.encoders import deepseek_v3 as _deepseek_v3  # noqa: F401

        return _REGISTRY["deepseek_v3"](tokenizer)

    if encoder_name == "deepseek_v4":
        from titaninfra.data.sft.encoders.deepseek_v4 import DeepSeekV4Encoder

        return DeepSeekV4Encoder(tokenizer)

    if encoder_name == "gpt_oss":
        from titaninfra.data.sft.encoders import gpt_oss as _gpt_oss  # noqa: F401

        return _REGISTRY["gpt_oss"](tokenizer)

    available = sorted({
        *_REGISTRY, "qwen3", "qwen3_5", "qwen2_5", "llama3", "gemma4",
        "deepseek_v3", "deepseek_v4", "gpt_oss",
    })
    raise ValueError(
        f"Unknown encoder: {encoder_name!r}. Available: {available!r}"
    )


__all__ = [
    "IGNORE_INDEX",
    "SFTEncoder",
    "register",
    "unregister",
    "list_encoders",
    "get_encoder",
]
