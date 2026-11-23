"""Runtime alignment check: titaninfra encoder output vs HF apply_chat_template.

This opt-in validator (gated by ``StreamConfig.validate_template_alignment``)
re-renders each batch sample through HF ``apply_chat_template`` and asserts
that titaninfra's encoded ``input_ids`` match token-for-token within the
comparable prefix. Used to catch template / chat-template drift early.

Lives in its own module (not ``_jinja_renderer``) so that ``streaming_dataset``
can import it without coupling the runtime hot path to the legacy Jinja
renderer that is slated for deletion in Phase 2.
"""

from __future__ import annotations

from titaninfra.data.sft.message_converter import normalize_to_hf_messages


def validate_hf_alignment(
    messages: list[dict],
    input_ids: list[int],
    tokenizer,
    *,
    max_length: int | None = None,
) -> None:
    """Verify titaninfra input_ids match HF apply_chat_template token-for-token.

    Accepts raw messages (same format as streaming_dataset passes through
    the encoder). Internally normalizes to HF format before calling
    apply_chat_template.

    Raises AssertionError on any token mismatch within the comparable prefix.
    Handles both list and BatchEncoding return types from apply_chat_template.
    """
    hf_messages, _ = normalize_to_hf_messages(messages)
    hf_result = tokenizer.apply_chat_template(
        hf_messages, tokenize=True, add_generation_prompt=False,
    )
    if hasattr(hf_result, "input_ids"):
        hf_ids = hf_result["input_ids"]
    elif isinstance(hf_result, list):
        hf_ids = hf_result
    else:
        hf_ids = list(hf_result)

    # Length drift is itself a template-misalignment signal — surface it
    # explicitly rather than silently truncating to ``min(...)``. Skip when
    # ``max_length`` is in effect, since the caller is intentionally
    # comparing only a prefix.
    if max_length is None and len(input_ids) != len(hf_ids):
        raise AssertionError(
            f"Template alignment length mismatch: "
            f"titaninfra_len={len(input_ids)}, HF_len={len(hf_ids)}"
        )

    compare_len = min(len(input_ids), len(hf_ids))
    if max_length is not None:
        compare_len = min(compare_len, max_length)

    # Element-wise comparison: ``input_ids`` is a tuple from EncodedSample
    # while ``hf_ids`` is a list — ``tuple != list`` is always True even
    # when contents match, so direct slice comparison would false-positive
    # on every sample.
    first_diff = next(
        (i for i in range(compare_len) if input_ids[i] != hf_ids[i]), None
    )
    if first_diff is not None:
        raise AssertionError(
            f"Template alignment drift at token {first_diff}! "
            f"titaninfra[{first_diff}]={input_ids[first_diff]}, "
            f"HF[{first_diff}]={hf_ids[first_diff]}, "
            f"titaninfra_len={len(input_ids)}, HF_len={len(hf_ids)}"
        )


__all__ = ["validate_hf_alignment"]
