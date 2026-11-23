"""DeepSeek-V4 segment-pipeline parity (dispatch path).

The legacy 22-case suite (``tests/test_dsv4_encoder.py``) covers the
pre-M0 ``encode(messages, tools=...)`` API. This suite covers the NEW
dispatch path: raw SFTSample dict -> ``dispatch.tokenize_sample(family=
'deepseek_v4')`` -> ``EncodedSample``. The decoded token stream must
byte-match the same gold text files the legacy suite uses, proving the
segment pipeline produces identical wire bytes.

We also lightly exercise the encoder's ``encode_segments`` direct entry
to confirm it accepts ``NormalizedSample`` and emits at least one
segment (the BOS prelude).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from titaninfra.data.sft.dispatch import tokenize_sample

_FIXTURES = pathlib.Path("tests/fixtures/tokenizers/deepseek_v4")
_ENCODING_TESTS = _FIXTURES / "encoding" / "tests"
_TOKENIZER_AVAILABLE = (_FIXTURES / "tokenizer.json").exists()

pytestmark = pytest.mark.skipif(
    not _TOKENIZER_AVAILABLE,
    reason="deepseek_v4 tokenizer fixture not downloaded; run tests/fixtures/tokenizers/download_all.py",
)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(_FIXTURES), trust_remote_code=True)


def _gold(name: str) -> str:
    return (_ENCODING_TESTS / name).read_text(encoding="utf-8").rstrip("\n")


def _load_raw(name: str) -> dict:
    raw = json.loads((_ENCODING_TESTS / name).read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return raw


@pytest.mark.parametrize(
    ("case_json", "gold_txt"),
    [
        ("titaninfra_case_1.json", "test_output_1.txt"),
        ("titaninfra_case_2.json", "test_output_2.txt"),
        ("titaninfra_case_3.json", "test_output_3.txt"),
        ("titaninfra_case_4.json", "test_output_4.txt"),
    ],
)
def test_dispatch_byte_matches_legacy_gold(tok, case_json, gold_txt):
    """Segment pipeline via dispatch.tokenize_sample matches the legacy gold."""
    sample_json = _load_raw(case_json)
    enc = tokenize_sample(sample_json, tok, family="deepseek_v4")
    decoded = tok.decode(enc.input_ids, skip_special_tokens=False)
    assert decoded == _gold(gold_txt)


def test_encode_segments_emits_bos_prelude(tok):
    """encode_segments(NormalizedSample) returns at least the BOS marker."""
    from titaninfra.data.sft import normalize
    from titaninfra.data.sft.encoders import get_encoder
    from titaninfra.data.sft.segment import SegKind

    sample = normalize.from_sft_sample(_load_raw("titaninfra_case_2.json"))
    enc = get_encoder("deepseek_v4", tok)
    segs = enc.encode_segments(sample)
    assert segs, "encode_segments must produce at least one segment"
    assert segs[0].kind == SegKind.MARKER
    assert segs[0].role == "system"
    assert segs[0].text == "<｜begin▁of▁sentence｜>"
    assert segs[0].loss_scale == 0.0


def test_legacy_encode_rejects_positional_tools_with_sample(tok):
    """When called with NormalizedSample, ``tools=...`` is reserved-error."""
    from titaninfra.data.sft import normalize
    from titaninfra.data.sft.encoders import get_encoder

    sample = normalize.from_sft_sample(_load_raw("titaninfra_case_1.json"))
    enc = get_encoder("deepseek_v4", tok)
    with pytest.raises(ValueError, match="tools must live on"):
        enc.encode(sample, tools=[{"name": "x", "description": "", "input_schema": {}}])


# ---------------------------------------------------------------------------
# P0 fixture matrix: tests/fixtures/sft/deepseek_v4/*.json
# DSv4 has no jinja template, so these are self-locking golden snapshots —
# the expected.decoded field is the encoder's own output frozen at fixture
# creation time. A divergence means the encoder drifted.
# ---------------------------------------------------------------------------

_DSV4_FIXTURE_DIR = pathlib.Path("tests/fixtures/sft/deepseek_v4")


@pytest.mark.parametrize(
    "fixture_name",
    [
        "01_single_turn_plain",
        "02_with_system",
        "03_multi_turn",
        "04_thinking_assistant",
        "05_tool_call_result_loop",
        "06_multi_turn_history_thinking",
        "07_type_overrides_thinking",
        "08_thinking_intent_off",
    ],
)
def test_dsv4_fixture_golden(tok, fixture_name):
    """Encoder output matches the frozen golden snapshot in the P0 fixture."""
    fx = json.loads(
        (_DSV4_FIXTURE_DIR / f"{fixture_name}.json").read_text(encoding="utf-8")
    )
    enc = tokenize_sample(
        fx["sample"], tok, family="deepseek_v4", max_length=fx["max_length"],
        loss_config=fx.get("loss_config"),
    )

    expected = fx["expected"]
    assert len(enc.input_ids) == expected["n_tokens"], (
        f"{fixture_name}: encoder emitted {len(enc.input_ids)} tokens, "
        f"fixture expects {expected['n_tokens']}"
    )

    n_loss = sum(1 for label in enc.labels if label != -100)
    assert n_loss == expected["n_loss_tokens"], (
        f"{fixture_name}: encoder marked {n_loss} loss tokens, "
        f"fixture expects {expected['n_loss_tokens']}"
    )

    decoded = tok.decode(enc.input_ids, skip_special_tokens=False)
    assert decoded == expected["decoded"], (
        f"{fixture_name}: decoded wire string drifted from golden snapshot"
    )

    # Fixtures carrying type_overrides also pin the continuous loss-weight sum.
    if "loss_weight_sum" in expected:
        assert enc.loss_weights is not None, (
            f"{fixture_name}: expected continuous loss_weights, got None"
        )
        got = round(sum(enc.loss_weights), 4)
        assert got == expected["loss_weight_sum"], (
            f"{fixture_name}: loss_weight_sum {got} != golden "
            f"{expected['loss_weight_sum']}"
        )
