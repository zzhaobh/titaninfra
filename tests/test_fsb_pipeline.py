"""Unit tests for fsb pipeline."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tests._mds_utils import load_mds_samples
from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from titaninfra.data.converter import convert_messages_to_mds
from titaninfra.data.sft.message_converter import normalize_to_hf_messages
from titaninfra.data.sft.types import IGNORE_INDEX

TOKENIZERS_DIR = Path(__file__).parent / "fixtures" / "tokenizers"
QWEN3_TOKENIZER = str(TOKENIZERS_DIR / "qwen3")
FSB_DATA_PATH = Path(__file__).parent / "fixtures" / "datasets" / "fsb.json"


def _parse_fsb_line(line: str) -> list[dict]:
    """Parse a single FSB line (concatenated JSON objects) into a message list."""
    decoder = json.JSONDecoder()
    msgs = []
    idx = 0
    while idx < len(line):
        while idx < len(line) and line[idx] in " \t\n\r,":
            idx += 1
        if idx >= len(line):
            break
        obj, end_idx = decoder.raw_decode(line, idx)
        msgs.append(obj)
        idx = end_idx
    return msgs


def _load_fsb_samples() -> list[list[dict]]:
    """Load all FSB samples from data/fsb.json."""
    with open(FSB_DATA_PATH, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [_parse_fsb_line(line) for line in lines]


def _wrap_as_titaninfra_jsonl(messages: list[dict]) -> dict:
    """Wrap parsed messages into titaninfra JSONL record format."""
    return {"messages": messages}


def _write_jsonl(path: str, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _extract_user_text(messages: list[dict]) -> str:
    """Extract user text from messages (either raw block format or HF format)."""
    for m in messages:
        if m.get("role") != "user":
            continue
        blocks = m.get("blocks")
        if blocks:
            return blocks[0].get("text", "")
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for b in content:
                if b.get("type") == "text":
                    return b.get("text", "")
    return ""


def _find_matching_raw(mds_messages: list[dict], raw_samples: list[list[dict]]) -> list[dict] | None:
    """Find the raw sample whose user text matches the MDS-loaded sample."""
    mds_user = _extract_user_text(mds_messages)
    for raw_msgs in raw_samples:
        if _extract_user_text(raw_msgs) == mds_user:
            return raw_msgs
    return None


def _create_mds(samples: list[dict], tmp_dir: str, num_shards: int = 2) -> str:
    jsonl_path = str(Path(tmp_dir) / "input.jsonl")
    mds_out = str(Path(tmp_dir) / "mds_out")
    _write_jsonl(jsonl_path, samples)
    convert_messages_to_mds(
        source=jsonl_path,
        output=mds_out,
        tokenizer_path=QWEN3_TOKENIZER,
        family="qwen3",
        num_shards=num_shards,
        overwrite=True,
    )
    return mds_out


@pytest.fixture(scope="module")
def fsb_raw_samples():
    if not FSB_DATA_PATH.exists():
        pytest.skip(f"FSB data not found: {FSB_DATA_PATH}")
    return _load_fsb_samples()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)


class TestFSBRawParsing:
    """Tests for test fsbraw parsing."""

    def test_sample_count(self, fsb_raw_samples):
        assert len(fsb_raw_samples) == 10

    def test_all_samples_are_single_turn(self, fsb_raw_samples):
        for i, msgs in enumerate(fsb_raw_samples):
            assert len(msgs) == 2, f"Sample {i}: expected 2 messages, got {len(msgs)}"
            assert msgs[0]["role"] == "user"
            assert msgs[1]["role"] == "assistant"

    def test_user_has_text_block(self, fsb_raw_samples):
        for i, msgs in enumerate(fsb_raw_samples):
            user_blocks = msgs[0].get("blocks", [])
            assert len(user_blocks) >= 1, f"Sample {i}: user has no blocks"
            assert user_blocks[0]["type"] == "text"
            assert len(user_blocks[0]["text"]) > 0

    def test_assistant_has_thinking_and_text_blocks(self, fsb_raw_samples):
        for i, msgs in enumerate(fsb_raw_samples):
            asst_blocks = msgs[1].get("blocks", [])
            assert len(asst_blocks) == 2, f"Sample {i}: expected 2 assistant blocks, got {len(asst_blocks)}"
            assert asst_blocks[0]["type"] == "thinking"
            assert asst_blocks[1]["type"] == "text"

    def test_thinking_blocks_have_content(self, fsb_raw_samples):
        for i, msgs in enumerate(fsb_raw_samples):
            thinking = msgs[1]["blocks"][0]
            assert len(thinking["text"].strip()) > 0, f"Sample {i}: empty thinking block"

    def test_no_think_tags_in_block_text(self, fsb_raw_samples):
        """ThinkingBlock text should NOT contain <think>/<\\/think> tags —
        those are added at render time by the template."""
        for i, msgs in enumerate(fsb_raw_samples):
            for block in msgs[1]["blocks"]:
                text = block.get("text", "")
                assert "<think>" not in text, f"Sample {i}: raw block text contains <think> tag"
                assert "</think>" not in text, f"Sample {i}: raw block text contains </think> tag"


class TestFSBMDSConversion:
    """Tests for test fsbmdsconversion."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, fsb_raw_samples, tokenizer):
        self.tmp_dir = str(tmp_path)
        self.tokenizer = tokenizer
        titaninfra_records = [_wrap_as_titaninfra_jsonl(msgs) for msgs in fsb_raw_samples]
        self.mds_path = _create_mds(titaninfra_records, self.tmp_dir, num_shards=2)
        self.loaded = load_mds_samples(self.mds_path)
        self.raw_samples = fsb_raw_samples

    def test_sample_count_preserved(self):
        assert len(self.loaded) == 10

    def test_messages_parseable(self):
        for i, sample in enumerate(self.loaded):
            msgs = sample["messages"]
            assert isinstance(msgs, list), f"Sample {i}: messages not a list"
            assert len(msgs) >= 2, f"Sample {i}: fewer than 2 messages"

    def test_thinking_block_survives_roundtrip(self):
        """ThinkingBlock must survive MDS roundtrip as blocks format."""
        for i, sample in enumerate(self.loaded):
            msgs = sample["messages"]
            asst_msgs = [m for m in msgs if m["role"] == "assistant"]
            assert len(asst_msgs) >= 1, f"Sample {i}: no assistant message"
            asst = asst_msgs[0]
            content = asst.get("content") or asst.get("blocks")
            assert isinstance(content, list), (
                f"Sample {i}: assistant content should be blocks list, got {type(content)}"
            )
            block_types = [b.get("type") for b in content]
            assert "thinking" in block_types, (
                f"Sample {i}: thinking block lost after MDS roundtrip. "
                f"Block types: {block_types}"
            )

    def test_token_count_column_positive(self):
        for i, sample in enumerate(self.loaded):
            assert sample["token_count"] > 0, f"Sample {i}: token_count <= 0"

    def test_stream_id_column_present(self):
        for i, sample in enumerate(self.loaded):
            assert "stream_id" in sample


class TestFSBTokenization:
    """Tests for test fsbtokenization."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, fsb_raw_samples, tokenizer):
        self.tmp_dir = str(tmp_path)
        self.tokenizer = tokenizer
        titaninfra_records = [_wrap_as_titaninfra_jsonl(msgs) for msgs in fsb_raw_samples]
        self.mds_path = _create_mds(titaninfra_records, self.tmp_dir, num_shards=2)
        self.loaded = load_mds_samples(self.mds_path)
        self.raw_samples = fsb_raw_samples

    def test_input_ids_match_hf_apply_chat_template(self):
        """Verify input ids match hf apply chat template."""
        for i, sample in enumerate(self.loaded):
            mds_messages = sample["messages"]
            input_ids_titaninfra, _ = tokenize_with_hf_template(
                messages=mds_messages,
                tokenizer=self.tokenizer,
                max_length=32768,
            )

            raw_msgs = _find_matching_raw(mds_messages, self.raw_samples)
            assert raw_msgs is not None, f"Sample {i}: no matching raw sample found"
            hf_messages, _ = normalize_to_hf_messages(raw_msgs)
            input_ids_hf = self.tokenizer.apply_chat_template(
                hf_messages, tokenize=True, add_generation_prompt=False,
                return_dict=False,
            )

            assert input_ids_titaninfra == input_ids_hf, (
                f"Sample {i}: input_ids mismatch! "
                f"titaninfra len={len(input_ids_titaninfra)}, hf len={len(input_ids_hf)}"
            )

    def test_think_tags_in_tokenized_text(self):
        """Tokenized text must contain <think> and </think> tags."""
        for i, sample in enumerate(self.loaded):
            input_ids, _ = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            assert "<think>" in decoded, f"Sample {i}: <think> tag missing from decoded text"
            assert "</think>" in decoded, f"Sample {i}: </think> tag missing from decoded text"

    def test_thinking_content_present_in_tokenized_text(self):
        """The thinking content from ThinkingBlock must appear in the final tokenized sequence."""
        for i, sample in enumerate(self.loaded):
            input_ids, _ = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)

            raw_msgs = _find_matching_raw(sample["messages"], self.raw_samples)
            assert raw_msgs is not None, f"Sample {i}: no matching raw sample found"
            original_thinking = raw_msgs[1]["blocks"][0]["text"]
            snippet = original_thinking[:50]
            assert snippet in decoded, (
                f"Sample {i}: thinking content not found in tokenized text. "
                f"Expected snippet: '{snippet}'"
            )

    def test_think_tag_order(self):
        """<think> must appear before </think> in the decoded sequence."""
        for i, sample in enumerate(self.loaded):
            input_ids, _ = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            think_pos = decoded.find("<think>")
            end_think_pos = decoded.find("</think>")
            assert think_pos < end_think_pos, (
                f"Sample {i}: <think> at {think_pos} but </think> at {end_think_pos}"
            )

    def test_response_text_after_think_close(self):
        """The text block content must appear AFTER </think> in the final sequence."""
        for i, sample in enumerate(self.loaded):
            input_ids, _ = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            end_think_pos = decoded.find("</think>")

            raw_msgs = _find_matching_raw(sample["messages"], self.raw_samples)
            assert raw_msgs is not None, f"Sample {i}: no matching raw sample found"
            original_text = raw_msgs[1]["blocks"][1]["text"]
            snippet = original_text[:50]
            text_pos = decoded.find(snippet)
            assert text_pos > end_think_pos, (
                f"Sample {i}: response text at {text_pos} but </think> at {end_think_pos}"
            )


class TestFSBLossMask:
    """Tests for test fsbloss mask."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, fsb_raw_samples, tokenizer):
        self.tmp_dir = str(tmp_path)
        self.tokenizer = tokenizer
        titaninfra_records = [_wrap_as_titaninfra_jsonl(msgs) for msgs in fsb_raw_samples]
        self.mds_path = _create_mds(titaninfra_records, self.tmp_dir, num_shards=2)
        self.loaded = load_mds_samples(self.mds_path)
        self.raw_samples = fsb_raw_samples

    def test_user_tokens_all_masked(self):
        """User message tokens must all have labels = IGNORE_INDEX."""
        for i, sample in enumerate(self.loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            hf_messages, weights = normalize_to_hf_messages(sample["messages"])
            user_text = hf_messages[0]["content"]
            user_tokens = self.tokenizer.encode(user_text, add_special_tokens=False)

            first_user_token_in_seq = None
            for j in range(len(input_ids) - len(user_tokens) + 1):
                if input_ids[j : j + len(user_tokens)] == user_tokens:
                    first_user_token_in_seq = j
                    break

            if first_user_token_in_seq is not None:
                user_labels = labels[first_user_token_in_seq : first_user_token_in_seq + len(user_tokens)]
                assert all(lbl == IGNORE_INDEX for lbl in user_labels), (
                    f"Sample {i}: user tokens have loss! "
                    f"Non-masked positions: {[j for j, lbl in enumerate(user_labels) if lbl != IGNORE_INDEX]}"
                )

    def test_assistant_tokens_have_loss(self):
        """Assistant message tokens (thinking + text) must have loss > 0 total."""
        for i, sample in enumerate(self.loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            loss_token_count = sum(1 for lbl in labels if lbl != IGNORE_INDEX)
            assert loss_token_count > 0, f"Sample {i}: no tokens have loss"

    def test_thinking_tokens_participate_in_loss(self):
        """By default, <think>...</think> content tokens should have loss (not masked)."""
        for i, sample in enumerate(self.loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            think_open_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
            think_close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)

            from tests.oracle._jinja_oracle._jinja_renderer import _find_thinking_ranges
            ranges = _find_thinking_ranges(input_ids, 0, len(input_ids), think_open_ids, think_close_ids)
            assert len(ranges) >= 1, f"Sample {i}: no thinking ranges found"

            inner_start, inner_end = ranges[0]
            assert inner_end > inner_start, f"Sample {i}: empty thinking range"

            thinking_labels = labels[inner_start:inner_end]
            loss_count = sum(1 for lbl in thinking_labels if lbl != IGNORE_INDEX)
            assert loss_count == len(thinking_labels), (
                f"Sample {i}: thinking tokens partially masked! "
                f"{loss_count}/{len(thinking_labels)} have loss"
            )

    def test_loss_mask_no_gap_in_assistant_content(self):
        """Within the assistant's loss region, there should be no unexpected masked gaps."""
        for i, sample in enumerate(self.loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            first_loss = None
            last_loss = None
            for j, lbl in enumerate(labels):
                if lbl != IGNORE_INDEX:
                    if first_loss is None:
                        first_loss = j
                    last_loss = j

            if first_loss is not None and last_loss is not None:
                region = labels[first_loss : last_loss + 1]
                gaps = [j for j, lbl in enumerate(region) if lbl == IGNORE_INDEX]
                assert len(gaps) == 0, (
                    f"Sample {i}: {len(gaps)} masked gaps in assistant loss region "
                    f"[{first_loss}:{last_loss + 1}]. Gap positions: {gaps[:10]}"
                )

    def test_loss_ratio_reasonable(self):
        """Loss tokens should be a significant portion (>20%) of total tokens."""
        for i, sample in enumerate(self.loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            total = len(labels)
            loss_count = sum(1 for lbl in labels if lbl != IGNORE_INDEX)
            ratio = loss_count / total if total > 0 else 0
            assert ratio > 0.2, (
                f"Sample {i}: loss ratio too low ({ratio:.2%}). "
                f"Only {loss_count}/{total} tokens have loss"
            )


class TestFSBThinkingScaling:
    """Tests for test fsbthinking scaling."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, fsb_raw_samples, tokenizer):
        self.tmp_dir = str(tmp_path)
        self.tokenizer = tokenizer
        titaninfra_records = [_wrap_as_titaninfra_jsonl(msgs) for msgs in fsb_raw_samples]
        self.mds_path = _create_mds(titaninfra_records, self.tmp_dir, num_shards=2)
        self.loaded = load_mds_samples(self.mds_path)

    def test_thinking_scale_zero_masks_thinking_content(self):
        """type_overrides={"thinking": 0.0} should mask thinking content but keep markers."""
        sample = self.loaded[0]
        input_ids, labels, loss_weights = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            max_length=32768,
            loss_config={"type_overrides": {"thinking": 0.0}},
            return_weights=True,
        )
        think_open_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
        think_close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)

        from tests.oracle._jinja_oracle._jinja_renderer import _find_thinking_ranges
        ranges = _find_thinking_ranges(input_ids, 0, len(input_ids), think_open_ids, think_close_ids)
        assert len(ranges) >= 1

        inner_start, inner_end = ranges[0]
        thinking_labels = labels[inner_start:inner_end]
        assert all(lbl == IGNORE_INDEX for lbl in thinking_labels), (
            "thinking content should be fully masked with scale=0.0"
        )

        think_open_start = inner_start - len(think_open_ids)
        marker_labels = labels[think_open_start : think_open_start + len(think_open_ids)]
        assert any(lbl != IGNORE_INDEX for lbl in marker_labels), (
            "<think> marker should keep loss even with thinking scale=0"
        )

    def test_thinking_scale_half_produces_weights(self):
        """type_overrides={"thinking": 0.5} should produce non-None loss_weights."""
        sample = self.loaded[0]
        input_ids, labels, loss_weights = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            max_length=32768,
            loss_config={"type_overrides": {"thinking": 0.5}},
            return_weights=True,
        )
        assert loss_weights is not None, "loss_weights should not be None with scale=0.5"

        think_open_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
        think_close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)

        from tests.oracle._jinja_oracle._jinja_renderer import _find_thinking_ranges
        ranges = _find_thinking_ranges(input_ids, 0, len(input_ids), think_open_ids, think_close_ids)
        inner_start, inner_end = ranges[0]

        thinking_weights = loss_weights[inner_start:inner_end]
        assert all(abs(w - 0.5) < 1e-6 for w in thinking_weights), (
            f"thinking content weights should be 0.5, got: {set(thinking_weights)}"
        )

    def test_no_scaling_produces_no_weights(self):
        """Without type_overrides, loss_weights should be None (fast path)."""
        sample = self.loaded[0]
        input_ids, labels, loss_weights = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            max_length=32768,
            return_weights=True,
        )
        assert loss_weights is None, "loss_weights should be None without type_overrides"


class TestFSBConversionScript:
    """Tests for test fsbconversion script."""

    def test_parse_assistant_blocks_with_think_tags(self):
        """_parse_assistant_blocks correctly splits <think>...</think> + text."""
        from scripts.data.convert_fsb_to_mds import _parse_assistant_blocks

        content = "<think>Let me analyze this problem.\nStep 1: ...</think>Here is the answer."
        blocks = _parse_assistant_blocks(content)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "thinking"
        assert blocks[1]["type"] == "text"
        assert "<think>" not in blocks[0]["text"]
        assert "</think>" not in blocks[0]["text"]
        assert "Let me analyze" in blocks[0]["text"]
        assert "Here is the answer" in blocks[1]["text"]

    def test_parse_assistant_blocks_no_think_tags(self):
        """Content without <think> tags becomes a single TextBlock."""
        from scripts.data.convert_fsb_to_mds import _parse_assistant_blocks

        content = "Direct answer without thinking."
        blocks = _parse_assistant_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Direct answer without thinking."

    def test_convert_row_basic(self):
        """_convert_row produces correct titaninfra message structure."""
        from scripts.data.convert_fsb_to_mds import _convert_row

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>Simple math</think>4"},
        ]
        result = _convert_row(messages)
        assert result is not None
        assert "messages" in result
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["blocks"][0]["type"] == "thinking"
        assert msgs[1]["blocks"][1]["type"] == "text"

    def test_sanitize_strips_foreign_tokens(self):
        """Foreign special tokens from other models are stripped."""
        from scripts.data.convert_fsb_to_mds import _sanitize_text

        text = "Hello <|begin_of_text|>world<|end_of_text|>"
        cleaned = _sanitize_text(text, "test")
        assert "<|begin_of_text|>" not in cleaned
        assert "<|end_of_text|>" not in cleaned
        assert "Hello " in cleaned
        assert "world" in cleaned


class TestFSBAllSamplesE2E:
    """Tests for test fsball samples e2 e."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, fsb_raw_samples, tokenizer):
        self.tmp_dir = str(tmp_path)
        self.tokenizer = tokenizer
        titaninfra_records = [_wrap_as_titaninfra_jsonl(msgs) for msgs in fsb_raw_samples]
        self.mds_path = _create_mds(titaninfra_records, self.tmp_dir, num_shards=2)
        self.loaded = load_mds_samples(self.mds_path)
        self.raw_samples = fsb_raw_samples

    def test_all_samples_tokenize_without_error(self):
        for i, sample in enumerate(self.loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            assert len(input_ids) > 0, f"Sample {i}: empty input_ids"
            assert len(labels) == len(input_ids), f"Sample {i}: labels length mismatch"

    def test_all_samples_have_think_tags(self):
        for i, sample in enumerate(self.loaded):
            input_ids, _ = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            assert "<think>" in decoded, f"Sample {i}: missing <think>"
            assert "</think>" in decoded, f"Sample {i}: missing </think>"

    def test_all_samples_loss_mask_valid(self):
        for i, sample in enumerate(self.loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            loss_count = sum(1 for lbl in labels if lbl != IGNORE_INDEX)
            masked_count = sum(1 for lbl in labels if lbl == IGNORE_INDEX)
            assert loss_count > 0, f"Sample {i}: no loss tokens"
            assert masked_count > 0, f"Sample {i}: no masked tokens"

    def test_all_samples_thinking_in_loss(self):
        """Verify thinking content has loss for all 10 samples."""
        think_open_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
        think_close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        from tests.oracle._jinja_oracle._jinja_renderer import _find_thinking_ranges

        for i, sample in enumerate(self.loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            ranges = _find_thinking_ranges(
                input_ids, 0, len(input_ids), think_open_ids, think_close_ids
            )
            assert len(ranges) >= 1, f"Sample {i}: no thinking range found"
            inner_start, inner_end = ranges[0]
            thinking_labels = labels[inner_start:inner_end]
            masked_in_thinking = sum(1 for lbl in thinking_labels if lbl == IGNORE_INDEX)
            assert masked_in_thinking == 0, (
                f"Sample {i}: {masked_in_thinking}/{len(thinking_labels)} "
                f"thinking tokens masked (should be 0)"
            )

    def test_token_lengths_vary_across_samples(self):
        """FSB samples have varying lengths — verify the pipeline handles all sizes."""
        lengths = []
        for sample in self.loaded:
            input_ids, _ = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=32768,
            )
            lengths.append(len(input_ids))

        assert min(lengths) > 50, f"Shortest sample suspiciously short: {min(lengths)}"
        assert max(lengths) > 500, f"Longest sample suspiciously short: {max(lengths)}"
        assert max(lengths) > 2 * min(lengths), (
            f"Length distribution too uniform: min={min(lengths)}, max={max(lengths)}"
        )
