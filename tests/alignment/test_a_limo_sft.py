"""Unit tests for a limo sft."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from titaninfra.data.sft.message_converter import normalize_to_hf_messages
from titaninfra.data.sft.types import IGNORE_INDEX

TOKENIZERS_DIR = Path(__file__).parent.parent / "fixtures" / "tokenizers"
LIMO_JSONL = Path(__file__).parent.parent.parent / "data" / "limo_qwen25" / "sft" / "train" / "limo.jsonl"
QWEN25_TOKENIZER = str(TOKENIZERS_DIR / "qwen2_5")

pytestmark = pytest.mark.skipif(
    not LIMO_JSONL.exists(),
    reason="LIMO JSONL not found locally (run convert_limo_qwen25.py first)",
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        QWEN25_TOKENIZER, trust_remote_code=True
    )


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
        "prompt_ids 不是 full_ids 的前缀，模板或 tokenize 行为异常"
    )

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


class TestLimoSFTAlignment:
    """Tests for test limo sftalignment."""

    def test_input_ids_bit_exact(self, tokenizer, limo_samples):
        """Verify input ids bit exact."""
        for i, messages in enumerate(limo_samples):
            hf_ids, _ = _hf_tokenize_with_mask(messages, tokenizer)
            jd_ids, _ = tokenize_with_hf_template(
                messages=messages, tokenizer=tokenizer,
                max_length=32768,
            )
            assert hf_ids == jd_ids, (
                f"样本 {i}: input_ids 不一致，"
                f"HF len={len(hf_ids)}, titaninfra len={len(jd_ids)}, "
                f"首个差异位置={next((j for j, (a, b) in enumerate(zip(hf_ids, jd_ids, strict=False)) if a != b), '长度不同')}"
            )

    def test_labels_bit_exact(self, tokenizer, limo_samples):
        """Verify labels bit exact."""
        for i, messages in enumerate(limo_samples):
            hf_ids, hf_labels = _hf_tokenize_with_mask(messages, tokenizer)
            jd_ids, jd_labels = tokenize_with_hf_template(
                messages=messages, tokenizer=tokenizer,
                max_length=32768,
            )
            assert len(hf_labels) == len(jd_labels), (
                f"样本 {i}: labels 长度不一致 HF={len(hf_labels)}, jd={len(jd_labels)}"
            )
            for pos in range(len(hf_labels)):
                assert jd_labels[pos] == hf_labels[pos], (
                    f"样本 {i} pos {pos}: labels 不一致 "
                    f"(hf={hf_labels[pos]}, jd={jd_labels[pos]}, token={jd_ids[pos]})"
                )

    def test_user_tokens_masked(self, tokenizer, limo_samples):
        """Verify user tokens masked."""
        messages = limo_samples[0]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        _, jd_labels = tokenize_with_hf_template(
            messages=messages, tokenizer=tokenizer,
            max_length=32768,
        )
        prompt_labels = jd_labels[:prompt_len]
        non_masked = [v for v in prompt_labels if v != IGNORE_INDEX]
        assert non_masked == [], (
            f"prompt 区间（含 assistant role marker）存在非 -100 label "
            f"（{len(non_masked)} 个），mask 逻辑未与 HF 对齐"
        )

    def test_assistant_tokens_have_loss(self, tokenizer, limo_samples):
        """Verify assistant tokens have loss."""
        messages = limo_samples[0]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        jd_ids, jd_labels = tokenize_with_hf_template(
            messages=messages, tokenizer=tokenizer,
            max_length=32768,
        )
        asst_labels = jd_labels[prompt_len:]
        loss_count = sum(1 for v in asst_labels if v != IGNORE_INDEX)
        assert loss_count > 100, (
            f"assistant 区间 loss token 数过少（{loss_count}），solution 未正确纳入 loss"
        )
        for pos, (tid, label) in enumerate(zip(jd_ids[prompt_len:], asst_labels, strict=False)):
            if label != IGNORE_INDEX:
                assert label == tid, (
                    f"assistant token[{pos}]: label={label} != token_id={tid}"
                )

    def test_long_sample_fits_seq_len(self, tokenizer, limo_samples):
        """Verify long sample fits seq len."""
        longest = max(limo_samples, key=lambda m: len(m[1]["content"]))
        jd_ids, jd_labels = tokenize_with_hf_template(
            messages=longest, tokenizer=tokenizer,
            max_length=32768,
        )
        assert len(jd_ids) < 32768, "样本超出 max_length=32768，需要调大 seq_len"
        assert len(jd_ids) == len(jd_labels)

    def test_loss_ratio_reasonable(self, tokenizer, limo_samples):
        """Verify loss ratio reasonable."""
        for i, messages in enumerate(limo_samples):
            jd_ids, jd_labels = tokenize_with_hf_template(
                messages=messages, tokenizer=tokenizer,
                max_length=32768,
            )
            loss_count = sum(1 for v in jd_labels if v != IGNORE_INDEX)
            ratio = loss_count / max(len(jd_ids), 1)
            assert 0.5 <= ratio <= 0.995, (
                f"样本 {i}: loss ratio={ratio:.2%} 异常 "
                f"（loss={loss_count}, total={len(jd_ids)}）"
            )


# ── Layer 2: MDS Pipeline Alignment ─────────────────────────




def _first_diff(a: list[int], b: list[int]) -> str:
    """First diff."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return f"pos {i}: a={x}, b={y}"
    if len(a) != len(b):
        return f"lengths differ: {len(a)} vs {len(b)}"
    return "identical"


class TestLimoMDSPipelineAlignment:
    """Tests for test limo mdspipeline alignment."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_dir = str(tmp_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN25_TOKENIZER, trust_remote_code=True
        )

    def _create_mds_from_limo(self, samples: list[dict], num_shards: int = 2) -> str:
        """Create mds from limo."""
        from titaninfra.data.converter import convert_messages_to_mds

        jsonl_path = str(Path(self.tmp_dir) / "limo_subset.jsonl")
        mds_out = str(Path(self.tmp_dir) / "mds_out")
        with open(jsonl_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        convert_messages_to_mds(
            source=jsonl_path,
            output=mds_out,
            tokenizer_path=QWEN25_TOKENIZER,
            family="qwen2_5",
            num_shards=num_shards,
            overwrite=True,
        )
        return mds_out

    def _load_mds_messages(self, mds_path: str) -> list[list[dict]]:
        """Load mds messages."""
        from tests._mds_utils import load_mds_messages

        return load_mds_messages(mds_path)

    @staticmethod
    def _user_content(messages: list[dict]) -> str:
        """User content."""
        for m in messages:
            if m.get("role") != "user":
                continue
            if "content" in m and m["content"] is not None:
                content = m["content"]
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(b.get("text", ""))
                    return "".join(parts)
            blocks = m.get("blocks") or []
            parts2: list[str] = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts2.append(b.get("text", ""))
            return "".join(parts2)
        raise ValueError("no user message found")

    @classmethod
    def _build_lookup(cls, samples: list[dict]) -> dict[str, list[dict]]:
        """Build lookup."""
        return {cls._user_content(s["messages"]): s["messages"] for s in samples}


    def test_mds_roundtrip_preserves_messages(self):
        """Verify mds roundtrip preserves messages."""
        raw_samples = []
        with open(LIMO_JSONL) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                raw_samples.append(json.loads(line))

        mds_path = self._create_mds_from_limo(raw_samples)
        loaded = self._load_mds_messages(mds_path)
        orig_lookup = self._build_lookup(raw_samples)

        assert len(loaded) == len(raw_samples)
        for mds_msgs in loaded:
            key = self._user_content(mds_msgs)
            assert key in orig_lookup, f"MDS 中出现未知样本: {key[:60]}..."
            orig_msgs = orig_lookup[key]
            assert len(mds_msgs) == len(orig_msgs), (
                f"消息数不一致 (orig={len(orig_msgs)}, mds={len(mds_msgs)})"
            )
            for j, (om, lm) in enumerate(zip(orig_msgs, mds_msgs, strict=False)):
                assert om["role"] == lm["role"], f"消息 {j}: role 不一致"
                orig_text = om.get("content", "") if isinstance(om.get("content"), str) else ""
                mds_text_parts = [
                    b.get("text", "")
                    for b in lm.get("blocks", [])
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                mds_text = "".join(mds_text_parts)
                assert orig_text == mds_text, (
                    f"消息 {j}: 文本内容不一致 "
                    f"(orig len={len(orig_text)}, mds len={len(mds_text)})"
                )


    def test_mds_input_ids_match_hf(self):
        """Verify mds input ids match hf."""
        raw_samples = []
        with open(LIMO_JSONL) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                raw_samples.append(json.loads(line))

        mds_path = self._create_mds_from_limo(raw_samples, num_shards=4)
        mds_all_msgs = self._load_mds_messages(mds_path)
        orig_lookup = self._build_lookup(raw_samples)

        for mds_msgs in mds_all_msgs:
            key = self._user_content(mds_msgs)
            orig_msgs = orig_lookup[key]

            jd_ids, _ = tokenize_with_hf_template(
                messages=mds_msgs, tokenizer=self.tokenizer, max_length=32768,
            )
            hf_messages, _ = normalize_to_hf_messages(orig_msgs)
            hf_ids = self.tokenizer.apply_chat_template(
                hf_messages, tokenize=True, add_generation_prompt=False,
                return_dict=False,
            )[:32768]

            assert jd_ids == hf_ids, (
                f"input_ids 不一致! question={key[:60]}... "
                f"jd_len={len(jd_ids)}, hf_len={len(hf_ids)}, "
                f"first diff: {_first_diff(jd_ids, hf_ids)}"
            )


    def test_mds_labels_match_hf(self):
        """Verify mds labels match hf."""
        raw_samples = []
        with open(LIMO_JSONL) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                raw_samples.append(json.loads(line))

        mds_path = self._create_mds_from_limo(raw_samples, num_shards=4)
        mds_all_msgs = self._load_mds_messages(mds_path)
        orig_lookup = self._build_lookup(raw_samples)

        for mds_msgs in mds_all_msgs:
            key = self._user_content(mds_msgs)
            orig_msgs = orig_lookup[key]

            _, jd_labels = tokenize_with_hf_template(
                messages=mds_msgs, tokenizer=self.tokenizer, max_length=32768,
            )
            _, hf_labels = _hf_tokenize_with_mask(orig_msgs, self.tokenizer)

            assert len(jd_labels) == len(hf_labels), "labels 长度不一致"
            for pos in range(len(hf_labels)):
                assert jd_labels[pos] == hf_labels[pos], (
                    f"q={key[:50]}... pos {pos}: "
                    f"hf={hf_labels[pos]}, jd={jd_labels[pos]}"
                )


    def test_mds_all_817_samples_input_ids_match_hf(self):
        """Verify mds all 817 samples input ids match hf."""
        raw_samples = []
        with open(LIMO_JSONL) as f:
            for line in f:
                raw_samples.append(json.loads(line))

        mds_path = self._create_mds_from_limo(raw_samples, num_shards=16)
        mds_all_msgs = self._load_mds_messages(mds_path)
        orig_lookup = self._build_lookup(raw_samples)

        assert len(mds_all_msgs) == len(raw_samples), (
            f"MDS 样本数 ({len(mds_all_msgs)}) != JSONL 样本数 ({len(raw_samples)})"
        )

        mismatches = []
        for mds_msgs in mds_all_msgs:
            key = self._user_content(mds_msgs)
            orig_msgs = orig_lookup[key]

            jd_ids, _ = tokenize_with_hf_template(
                messages=mds_msgs, tokenizer=self.tokenizer, max_length=32768,
            )
            hf_messages, _ = normalize_to_hf_messages(orig_msgs)
            hf_ids = self.tokenizer.apply_chat_template(
                hf_messages, tokenize=True, add_generation_prompt=False,
                return_dict=False,
            )[:32768]

            if jd_ids != hf_ids:
                mismatches.append(
                    f"  q={key[:50]}... jd_len={len(jd_ids)}, hf_len={len(hf_ids)}, "
                    f"diff={_first_diff(jd_ids, hf_ids)}"
                )

        assert not mismatches, (
            f"全量对齐失败！{len(mismatches)}/{len(raw_samples)} 条不一致:\n"
            + "\n".join(mismatches[:20])
        )


    def test_mds_loss_ratio_matches_direct(self):
        """Verify mds loss ratio matches direct."""
        raw_samples = []
        with open(LIMO_JSONL) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                raw_samples.append(json.loads(line))

        mds_path = self._create_mds_from_limo(raw_samples, num_shards=4)
        mds_all_msgs = self._load_mds_messages(mds_path)
        orig_lookup = self._build_lookup(raw_samples)

        for mds_msgs in mds_all_msgs:
            key = self._user_content(mds_msgs)
            orig_msgs = orig_lookup[key]

            _, jd_labels_mds = tokenize_with_hf_template(
                messages=mds_msgs, tokenizer=self.tokenizer, max_length=32768,
            )
            _, jd_labels_direct = tokenize_with_hf_template(
                messages=orig_msgs, tokenizer=self.tokenizer, max_length=32768,
            )

            mds_loss = sum(1 for v in jd_labels_mds if v != IGNORE_INDEX)
            direct_loss = sum(1 for v in jd_labels_direct if v != IGNORE_INDEX)
            assert mds_loss == direct_loss, (
                f"MDS loss_tokens={mds_loss} != direct loss_tokens={direct_loss} "
                f"q={key[:50]}..."
            )
