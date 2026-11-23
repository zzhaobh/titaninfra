"""
DeepSeek-V3 SFT alignment tests (Jinja path).

DSV3 has no custom encoder — it routes through `tokenize_with_hf_template`
(HF apply_chat_template + adapter preprocessing + boundary-based loss mask).
Mirrors the DSV4 alignment fixtures: titaninfra_case_{1..4}.json + test_output_{1..4}.txt.

Covers:
- P0 bit-exact: titaninfra SFT pipeline reproduces the gold rendered string.
- P0 labels: assistant content carries loss; system/user/tool markers are IGNORE.
- weight=0 history masking (case 4): masked assistant tokens are IGNORE, final
  assistant carries loss.
- train_on_inputs=True parametrized: token stream unchanged, all non-BOS labels
  equal input_ids (BOS stays IGNORE per encoder contract).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from titaninfra.data.sft.encoders import IGNORE_INDEX

_FIXTURES = pathlib.Path("tests/fixtures/tokenizers/deepseek_v3")
_ENCODING_TESTS = _FIXTURES / "encoding" / "tests"
_TOKENIZER_AVAILABLE = (_FIXTURES / "tokenizer.json").exists()

pytestmark = pytest.mark.skipif(
    not _TOKENIZER_AVAILABLE,
    reason="deepseek_v3 tokenizer fixture not downloaded; run tests/fixtures/tokenizers/download_all.py",
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(_FIXTURES), trust_remote_code=True)


def _load_raw(name: str) -> dict:
    """Load titaninfra SFTSample as raw dict (normalize_to_hf_messages expects dicts, not Pydantic)."""
    return json.loads((_ENCODING_TESTS / name).read_text(encoding="utf-8"))


def _gold(name: str) -> str:
    return (_ENCODING_TESTS / name).read_text(encoding="utf-8").rstrip("\n")


# ── P0: alignment tests ────────────────────────────────────────────────────

def test_align_case_1(tok):
    """Single-turn thinking + text: titaninfra SFT pipeline matches gold render."""
    sample = _load_raw("titaninfra_case_1.json")
    input_ids, _ = tokenize_with_hf_template(
        messages=sample["messages"],
        tokenizer=tok,
        template_name="deepseek_r1",
        tools=sample.get("tools"),
    )
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    assert decoded == _gold("test_output_1.txt")


def test_align_case_2(tok):
    """system + 2 assistant turns no-thinking: BOS+system rendering + multi-turn boundary."""
    sample = _load_raw("titaninfra_case_2.json")
    input_ids, _ = tokenize_with_hf_template(
        messages=sample["messages"],
        tokenizer=tok,
        template_name="deepseek_r1",
        tools=sample.get("tools"),
    )
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    assert decoded == _gold("test_output_2.txt")


def test_align_case_3(tok):
    """Tool-result round-trip (no assistant tool_calls — DSV3 Jinja drops them)."""
    sample = _load_raw("titaninfra_case_3.json")
    input_ids, _ = tokenize_with_hf_template(
        messages=sample["messages"],
        tokenizer=tok,
        template_name="deepseek_r1",
        tools=sample.get("tools"),
    )
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    assert decoded == _gold("test_output_3.txt")


def test_align_case_4(tok):
    """weight=0 history masking: token stream identical to ungated render."""
    sample = _load_raw("titaninfra_case_4.json")
    input_ids, _ = tokenize_with_hf_template(
        messages=sample["messages"],
        tokenizer=tok,
        template_name="deepseek_r1",
        tools=sample.get("tools"),
    )
    decoded = tok.decode(input_ids, skip_special_tokens=False)
    assert decoded == _gold("test_output_4.txt")


# ── P0: label correctness ──────────────────────────────────────────────────

def test_align_case_1_labels(tok):
    """user tokens are IGNORE; assistant content (incl. thinking) carries loss."""
    sample = _load_raw("titaninfra_case_1.json")
    input_ids, labels = tokenize_with_hf_template(
        messages=sample["messages"],
        tokenizer=tok,
        template_name="deepseek_r1",
    )
    assert len(input_ids) == len(labels)

    asst_id = tok.encode("<｜Assistant｜>", add_special_tokens=False)[0]
    first_asst_pos = input_ids.index(asst_id)
    # Everything up to and including <｜Assistant｜> marker must be IGNORE
    assert all(lbl == IGNORE_INDEX for lbl in labels[: first_asst_pos + 1])
    # At least some assistant content must carry loss
    assert any(lbl != IGNORE_INDEX for lbl in labels[first_asst_pos + 1 :])


def test_align_case_4_weight0_history_masking(tok):
    """weight=0 first assistant turn: tokens are IGNORE; final assistant carries loss."""
    sample = _load_raw("titaninfra_case_4.json")
    input_ids, labels = tokenize_with_hf_template(
        messages=sample["messages"],
        tokenizer=tok,
        template_name="deepseek_r1",
    )
    assert len(input_ids) == len(labels)

    asst_id = tok.encode("<｜Assistant｜>", add_special_tokens=False)[0]
    eos_id = tok.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)[0]

    # First <｜Assistant｜>...<｜end▁of▁sentence｜> span = "Bonjour" (weight=0 → IGNORE)
    first_asst_pos = input_ids.index(asst_id)
    first_eos_pos = input_ids.index(eos_id, first_asst_pos)
    assert all(
        lbl == IGNORE_INDEX for lbl in labels[first_asst_pos : first_eos_pos + 1]
    ), "weight=0 assistant span must be fully masked"

    # Final assistant span = "Hola" (default weight=1 → loss)
    assert any(
        lbl != IGNORE_INDEX for lbl in labels[first_eos_pos + 1 :]
    ), "final assistant span must carry loss"


# ── train_on_inputs sanity (parametrized over all 4 cases) ────────────────

@pytest.mark.parametrize(
    "case_name",
    ["titaninfra_case_1.json", "titaninfra_case_2.json", "titaninfra_case_3.json", "titaninfra_case_4.json"],
)
def test_align_cases_train_on_inputs_full_loss(tok, case_name):
    """train_on_inputs=True: token stream unchanged; labels == input_ids over full sequence.

    Guards against accidental masking when users opt into full-sequence loss.
    weight=0 messages must also become loss-carrying when train_on_inputs overrides.
    """
    sample = _load_raw(case_name)
    ids_default, _ = tokenize_with_hf_template(
        messages=sample["messages"],
        tokenizer=tok,
        template_name="deepseek_r1",
        tools=sample.get("tools"),
    )
    ids_train_in, labels_train_in = tokenize_with_hf_template(
        messages=sample["messages"],
        tokenizer=tok,
        template_name="deepseek_r1",
        tools=sample.get("tools"),
        train_on_inputs=True,
    )
    # Token stream identical regardless of label policy
    assert ids_default == ids_train_in
    # train_on_inputs=True → labels equal input_ids over full sequence (Jinja path contract)
    assert labels_train_in == ids_train_in
