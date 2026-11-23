"""Dispatch entry point is single-path — no ``mode`` kwarg, no fallback."""

import inspect

from titaninfra.data.sft import dispatch


def test_no_mode_kwarg():
    sig = inspect.signature(dispatch.tokenize_sample)
    assert "mode" not in sig.parameters


def test_signature_is_minimal():
    sig = inspect.signature(dispatch.tokenize_sample)
    params = list(sig.parameters)
    # Required positional: sample_json, tokenizer, family
    assert params[:3] == ["sample_json", "tokenizer", "family"]
    # Keyword-only allowed: length policy (max_length, truncate_at_last_
    # segment), loss_config, and the boundary-builder training policy
    # (sft_mask_mode, train_on_inputs).
    #
    # SFT dispatch is the SFT-only path; the add_generation_prompt flag
    # lives on assembler.assemble(mode='rl_prompt', ...) because exposing
    # it here would let a SFT caller pollute training data with an
    # unfinished assistant header. truncate_at_last_segment is a length
    # policy (same family as max_length: "what to do when too long"), not
    # a control-flow branch — see c795b98.
    #
    # sft_mask_mode / train_on_inputs are the boundary-builder inputs
    # (LossPolicy.resolve_weights is invoked here, between normalize and
    # encode — see encoders/base.py "boundary builder (upstream)"). They
    # collapse the masking policy into NormalizedMessage.weight; they are
    # training POLICY, not a mode-style encoder fallback. Anything OTHER
    # than these creeping in means someone reintroduced a mode switch.
    #
    # history_policy (history v2, 2026-06-06) is the same family of input:
    # a recipe-level data-preparation policy from StreamConfig.history_policy,
    # resolved here (dispatch knows the family) and injected into sample_json
    # before normalize. It is training POLICY, not a mode switch.
    #
    # resolution_policy / spatial_merge_size (VLM SFT, 2026-07-28) are the
    # same family again: the image pixel budget the assembler needs to expand
    # IMAGE segments into vision tokens. Threaded down to assemble() only when
    # non-None (text families' encode signatures are unaffected). Data-prep
    # policy, not a mode switch.
    kwonly = {n for n, p in sig.parameters.items() if p.kind is p.KEYWORD_ONLY}
    assert kwonly <= {
        "max_length",
        "loss_config",
        "truncate_at_last_segment",
        "sft_mask_mode",
        "train_on_inputs",
        "history_policy",
        "resolution_policy",
        "spatial_merge_size",
        "decoded_images",
    }


def test_unknown_family_raises():
    import pytest

    with pytest.raises(ValueError):  # from get_encoder
        dispatch.tokenize_sample(
            {"messages": [{"role": "user", "content": "hi"}]},
            tokenizer=None,
            family="not_a_real_family",
        )
