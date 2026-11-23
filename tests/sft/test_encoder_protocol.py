"""Every registered family encoder must satisfy the SFTEncoder Protocol.

F5: ``encode_segments`` is the canonical method (RL rollout calls it
directly; ``encode`` delegates to it). It was previously absent from the
``SFTEncoder`` Protocol in ``encoders/base.py``, so a new encoder could
omit it, pass an ``encode``-only structural check, and only blow up on the
RL path at runtime. This test pins that all 7 families expose BOTH methods
and structurally satisfy the (now two-method) Protocol.
"""

from __future__ import annotations

import inspect

import pytest

from titaninfra.data.sft.encoders import get_encoder
from titaninfra.data.sft.encoders.base import SFTEncoder

_FAMILIES = [
    "qwen3",
    "qwen2_5",
    "llama3",
    "deepseek_v3",
    "deepseek_v4",
    "gpt_oss",
    "gemma4",
]

_TOKENIZER_FIXTURE = {
    "qwen3": "qwen3_tokenizer",
    "qwen2_5": "qwen2_5_tokenizer",
    "llama3": "llama3_tokenizer",
    "deepseek_v3": "deepseek_v3_tokenizer",
    "deepseek_v4": "deepseek_v4_tokenizer",
    "gpt_oss": "gpt_oss_tokenizer",
    "gemma4": "gemma4_tokenizer",
}


@pytest.mark.parametrize("family", _FAMILIES)
def test_encoder_satisfies_protocol(family, request):
    """Each family's encoder structurally satisfies SFTEncoder (both methods)."""
    tok = request.getfixturevalue(_TOKENIZER_FIXTURE[family])
    enc = get_encoder(family, tok)

    # runtime_checkable Protocol: verifies method *presence* (not signature).
    assert isinstance(enc, SFTEncoder), (
        f"{family} encoder does not satisfy the SFTEncoder Protocol"
    )
    # Explicit method presence — the whole point of F5 is that BOTH exist.
    assert callable(getattr(enc, "encode", None)), f"{family} missing encode"
    assert callable(getattr(enc, "encode_segments", None)), (
        f"{family} missing encode_segments — RL rollout would crash"
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_encode_segments_signature(family, request):
    """encode_segments accepts the canonical (sample, *, type_overrides) shape."""
    tok = request.getfixturevalue(_TOKENIZER_FIXTURE[family])
    enc = get_encoder(family, tok)
    sig = inspect.signature(enc.encode_segments)
    params = sig.parameters
    # First positional param is the sample.
    assert "sample" in params or len(params) >= 1, (
        f"{family}.encode_segments must take a sample argument"
    )
    # type_overrides must be accepted as a keyword (Protocol declares it).
    assert "type_overrides" in params, (
        f"{family}.encode_segments must accept type_overrides keyword "
        "(declared on the SFTEncoder Protocol)"
    )
    assert params["type_overrides"].kind == inspect.Parameter.KEYWORD_ONLY, (
        f"{family}.encode_segments type_overrides must be keyword-only"
    )
