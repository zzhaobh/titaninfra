"""Per-family generation-prompt trailers — reference table.

VLLM and SGLang call ``tokenizer.apply_chat_template(...,
add_generation_prompt=True)`` when building an inference prompt.
That call appends a short, family-specific suffix (the
"generation prompt") that signals to the model "now produce the
next assistant turn".

titaninfra's runtime production path produces these bytes structurally
(each family encoder marks one ``Segment`` with ``is_generation=True``;
``assembler.assemble(mode='rl_prompt', add_generation_prompt=True)``
cuts at that marker and includes it). This module is kept as the
**audit-only single source of truth** for the trailer bytes — useful
for documentation, drift checks against future HF tokenizer updates,
and the **only** runtime consumer remaining: ``Gemma4Encoder``, which
still routes through ``apply_chat_template`` + boundary scan and
therefore has no structural marker to assemble. When gemma4 is
segment-ised (Step C of the dual-pipeline plan), this module becomes
purely a reference table.

Why a static table beats probing the tokenizer at runtime:

* **No template instability**. Probing ``apply_chat_template`` at init
  re-runs jinja and can pick up dynamic injections (Current-date,
  enable_thinking-conditional headers). A static table is byte-exact
  regardless of how the upstream template changes.
* **Audit-friendly**. The exact bytes that train and inference agree
  on are visible in one place; a reviewer can grep ``generation_prompt_for``
  and read the truth.

Per-family stripping policy (the "class C tokens" rule):

* Qwen3 ``enable_thinking=False`` would inject ``<think>\n\n</think>\n\n``
  -- this is the documented Quirk A and IS legitimate training signal
  for the mixed-thinking recipe, so SFT handles it inside encode_segments
  but the inference trailer stays the bare role header (vLLM default).
* Gemma4 jinja injects ``<|turn>model\n<|channel>thought\n<channel|>``
  on ``add_generation_prompt=True`` even when no thinking is enabled.
  The ``<|channel>thought\n<channel|>`` portion is template formatting
  the LLM does NOT emit autonomously; stripping it aligns inference
  with vLLM/SGLang's expected token-by-token decode.
* Llama3 jinja already emits the trailing assistant header
  unconditionally (the "trailing-header quirk" — see encoders/llama3.py).
  ``add_generation_prompt=True`` adds NOTHING extra, so the trailer is
  the empty string and the structural marker on the segment side already
  matches.
"""

from __future__ import annotations

# Family -> the literal byte string vLLM/SGLang would have appended
# when the user calls apply_chat_template(add_generation_prompt=True).
# Empty string means "the family's regular flow already includes the
# generation prompt as part of its trailing structure" (Llama3).
_GENERATION_PROMPT_TRAILERS: dict[str, str] = {
    "qwen3": "<|im_start|>assistant\n",
    "qwen2_5": "<|im_start|>assistant\n",
    "llama3": "",
    "deepseek_v3": "<｜Assistant｜>",
    "deepseek_v4": "<｜Assistant｜>",
    "gemma4": "<|turn>model\n",
    # gpt_oss landed later -- its trailer is also static
    # (``<|start|>assistant``) but the encoder doesn't exist yet.
    # Adding the entry here so the dispatch path doesn't have to
    # special-case the absent encoder.
    "gpt_oss": "<|start|>assistant",
}


def generation_prompt_for(family: str) -> str:
    """Return the literal trailer bytes for ``family``.

    Raises ``KeyError`` for unknown families so a caller can decide
    whether to fall back (e.g. unknown family during legacy migration)
    or surface the misconfiguration immediately.
    """
    return _GENERATION_PROMPT_TRAILERS[family]


def supports_generation_prompt(family: str) -> bool:
    return family in _GENERATION_PROMPT_TRAILERS


__all__ = [
    "generation_prompt_for",
    "supports_generation_prompt",
]
