"""Unit tests for a limo multi tokenizer."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from titaninfra.data.sft.types import IGNORE_INDEX

TOKENIZERS_DIR = Path(__file__).parent.parent / "fixtures" / "tokenizers"
LIMO_JSONL = (
    Path(__file__).parent.parent.parent
    / "data"
    / "limo_qwen25"
    / "sft"
    / "train"
    / "limo.jsonl"
)

#   ChatML       qwen2_5, qwen3   <|im_start|>role\n...<|im_end|>
#   Llama 3      llama3           <|start_header_id|>role<|end_header_id|>\n\n
#   Llama 4      llama4           <|header_start|>role<|header_end|>\n\n
#
TOKENIZER_FAMILIES = [
    "qwen2_5",
    "qwen3",
    "llama3",
    "llama4",
    "deepseek_v3",
]

pytestmark = pytest.mark.skipif(
    not LIMO_JSONL.exists(),
    reason="LIMO JSONL not found locally (run convert_limo_qwen25.py first)",
)


def _load_tokenizer(name: str):
    """Load tokenizer."""
    base = TOKENIZERS_DIR / name
    cfg = json.loads((base / "tokenizer_config.json").read_text())
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        chat_tpl_path = base / "chat_template.jinja"
        chat_tpl = chat_tpl_path.read_text() if chat_tpl_path.exists() else None
        kwargs = {
            k: v
            for k, v in cfg.items()
            if k not in ("tokenizer_class", "auto_map", "chat_template", "processor_class")
        }
        tok = PreTrainedTokenizerFast(tokenizer_file=str(base / "tokenizer.json"), **kwargs)
        if chat_tpl:
            tok.chat_template = chat_tpl
        return tok
    return AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)


def _hf_tokenize_with_mask(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Hf tokenize with mask."""
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    assert full_ids[: len(prompt_ids)] == prompt_ids, (
        "HF 模板异常：full_ids 不以 prompt_ids 开头"
    )
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


@pytest.fixture(scope="module")
def limo_samples():
    """Limo samples."""
    samples = []
    with open(LIMO_JSONL) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            samples.append(json.loads(line)["messages"])
    return samples


@pytest.fixture(scope="module", params=TOKENIZER_FAMILIES)
def tokenizer_pair(request):
    """Tokenizer pair."""
    return request.param, _load_tokenizer(request.param)


class TestLimoMultiTokenizerAlignment:
    """Tests for test limo multi tokenizer alignment."""

    def test_input_ids_bit_exact(self, tokenizer_pair, limo_samples):
        """Verify input ids bit exact."""
        family, tokenizer = tokenizer_pair
        for i, messages in enumerate(limo_samples):
            hf_ids, _ = _hf_tokenize_with_mask(messages, tokenizer)
            jd_ids, _ = tokenize_with_hf_template(
                messages=messages, tokenizer=tokenizer, max_length=131072,
            )
            assert hf_ids == jd_ids, (
                f"[{family}] 样本 {i}: input_ids 不一致 "
                f"(hf_len={len(hf_ids)}, jd_len={len(jd_ids)}, "
                f"first_diff_pos="
                f"{next((j for j, (a, b) in enumerate(zip(hf_ids, jd_ids, strict=False)) if a != b), 'N/A')})"
            )

    def test_labels_bit_exact(self, tokenizer_pair, limo_samples):
        """Verify labels bit exact."""
        family, tokenizer = tokenizer_pair
        for i, messages in enumerate(limo_samples):
            hf_ids, hf_labels = _hf_tokenize_with_mask(messages, tokenizer)
            jd_ids, jd_labels = tokenize_with_hf_template(
                messages=messages, tokenizer=tokenizer, max_length=131072,
            )
            assert len(hf_labels) == len(jd_labels), (
                f"[{family}] 样本 {i}: labels 长度不一致"
            )
            for pos in range(len(hf_labels)):
                assert jd_labels[pos] == hf_labels[pos], (
                    f"[{family}] 样本 {i} pos {pos}: "
                    f"hf={hf_labels[pos]}, jd={jd_labels[pos]}, "
                    f"token={jd_ids[pos]}"
                )

    def test_loss_count_matches(self, tokenizer_pair, limo_samples):
        """Verify loss count matches."""
        family, tokenizer = tokenizer_pair
        for i, messages in enumerate(limo_samples):
            _, hf_labels = _hf_tokenize_with_mask(messages, tokenizer)
            _, jd_labels = tokenize_with_hf_template(
                messages=messages, tokenizer=tokenizer, max_length=131072,
            )
            hf_loss = sum(1 for v in hf_labels if v != IGNORE_INDEX)
            jd_loss = sum(1 for v in jd_labels if v != IGNORE_INDEX)
            assert hf_loss == jd_loss, (
                f"[{family}] 样本 {i}: loss token 数不一致 "
                f"(hf={hf_loss}, jd={jd_loss})"
            )

    def test_prompt_region_fully_masked(self, tokenizer_pair, limo_samples):
        """Verify prompt region fully masked."""
        family, tokenizer = tokenizer_pair
        messages = limo_samples[0]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        _, jd_labels = tokenize_with_hf_template(
            messages=messages, tokenizer=tokenizer, max_length=131072,
        )
        prompt_labels = jd_labels[:prompt_len]
        non_masked = [v for v in prompt_labels if v != IGNORE_INDEX]
        assert non_masked == [], (
            f"[{family}] prompt 区间存在 {len(non_masked)} 个非 -100 label"
        )
