"""Unit tests for a8 e2e chain."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tests._mds_utils import load_mds_samples
from tests.oracle._jinja_oracle._jinja_renderer import tokenize_with_hf_template
from titaninfra.data.converter import convert_messages_to_mds
from titaninfra.data.sft.message_converter import normalize_to_hf_messages
from titaninfra.data.sft.types import IGNORE_INDEX

TOKENIZERS_DIR = Path(__file__).parent.parent / "fixtures" / "tokenizers"
QWEN3_TOKENIZER = str(TOKENIZERS_DIR / "qwen3")



SINGLE_TURN_SAMPLE = {
    "messages": [
        {"role": "user", "content": "计算 2+2 等于多少？"},
        {"role": "assistant", "content": "2+2 的结果是 4。"},
    ]
}

MULTI_TURN_SAMPLE = {
    "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        {"role": "user", "content": "帮我写一个 Python hello world"},
        {"role": "assistant", "content": 'print("Hello, world!")'},
    ]
}

WITH_SYSTEM_SAMPLE = {
    "messages": [
        {"role": "system", "content": "你是一个数学助手，回答要简洁。"},
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
    ]
}

BLOCKS_FORMAT_SAMPLE = {
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "解释量子纠缠"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "text": "用户想了解量子纠缠的概念"},
                {"type": "text", "text": "量子纠缠是两个粒子之间的关联现象。"},
            ],
        },
    ]
}

WITH_LOSS_CONFIG_SAMPLE = {
    "messages": [
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支。"},
    ],
    "loss_config": {"type_overrides": {"thinking": 0.5}},
}

ALL_SAMPLES = [
    SINGLE_TURN_SAMPLE,
    MULTI_TURN_SAMPLE,
    WITH_SYSTEM_SAMPLE,
    BLOCKS_FORMAT_SAMPLE,
    WITH_LOSS_CONFIG_SAMPLE,
]



def _write_jsonl(path: str, records: list[dict]):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _create_mds_from_samples(
    samples: list[dict], tmp_dir: str, num_shards: int = 2
) -> str:
    """Create mds from samples."""
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


def _first_diff(a: list[int], b: list[int]) -> str:
    """First diff."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return f"pos {i}: titaninfra={x}, hf={y}"
    if len(a) != len(b):
        return f"lengths differ: {len(a)} vs {len(b)}"
    return "identical"



class TestA8E2EChain:
    """Tests for test a8 e2 echain."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_dir = str(tmp_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN3_TOKENIZER, trust_remote_code=True
        )


    def test_mds_roundtrip_messages_parseable(self):
        """Verify mds roundtrip messages parseable."""
        mds_path = _create_mds_from_samples(ALL_SAMPLES, self.tmp_dir)
        loaded = load_mds_samples(mds_path)

        assert len(loaded) == len(ALL_SAMPLES)
        for sample in loaded:
            msgs = sample["messages"]
            assert isinstance(msgs, list)
            assert len(msgs) >= 2
            for msg in msgs:
                assert "role" in msg
                assert "blocks" in msg, "on-disk messages must be IR (blocks)"
                assert isinstance(msg["blocks"], list) and msg["blocks"]


    @pytest.mark.parametrize(
        "sample,name",
        [
            (SINGLE_TURN_SAMPLE, "single_turn"),
            (MULTI_TURN_SAMPLE, "multi_turn"),
            (WITH_SYSTEM_SAMPLE, "with_system"),
        ],
        ids=["single_turn", "multi_turn", "with_system"],
    )
    def test_input_ids_match_hf(self, sample, name):
        """Verify input ids match hf."""
        mds_path = _create_mds_from_samples([sample], self.tmp_dir, num_shards=1)
        loaded = load_mds_samples(mds_path)
        assert len(loaded) == 1

        mds_messages = loaded[0]["messages"]
        input_ids_titaninfra, _ = tokenize_with_hf_template(
            messages=mds_messages,
            tokenizer=self.tokenizer,
            max_length=4096,
        )

        hf_messages, _ = normalize_to_hf_messages(sample["messages"])
        input_ids_hf = self.tokenizer.apply_chat_template(
            hf_messages, tokenize=True, add_generation_prompt=False,
            return_dict=False,
        )

        assert input_ids_titaninfra == input_ids_hf, (
            f"[{name}] input_ids 不一致!\n"
            f"titaninfra len={len(input_ids_titaninfra)}, hf len={len(input_ids_hf)}\n"
            f"first diff at: {_first_diff(input_ids_titaninfra, input_ids_hf)}"
        )


    def test_loss_mask_user_masked_assistant_unmasked(self):
        """Verify loss mask user masked assistant unmasked."""
        mds_path = _create_mds_from_samples(
            [WITH_SYSTEM_SAMPLE], self.tmp_dir, num_shards=1
        )
        loaded = load_mds_samples(mds_path)
        mds_messages = loaded[0]["messages"]

        input_ids, labels = tokenize_with_hf_template(
            messages=mds_messages,
            tokenizer=self.tokenizer,
            max_length=4096,
        )

        assert len(input_ids) == len(labels)
        loss_tokens = [v for v in labels if v != IGNORE_INDEX]
        assert len(loss_tokens) > 0, "没有任何 token 有 loss"
        masked_tokens = [v for v in labels if v == IGNORE_INDEX]
        assert len(masked_tokens) > 0, "没有任何 token 被 mask"

    def test_loss_mask_multi_turn_all_assistants(self):
        """Verify loss mask multi turn all assistants."""
        mds_path = _create_mds_from_samples(
            [MULTI_TURN_SAMPLE], self.tmp_dir, num_shards=1
        )
        loaded = load_mds_samples(mds_path)
        mds_messages = loaded[0]["messages"]

        input_ids, labels = tokenize_with_hf_template(
            messages=mds_messages,
            tokenizer=self.tokenizer,
            max_length=4096,
        )

        hf_messages, _ = normalize_to_hf_messages(MULTI_TURN_SAMPLE["messages"])
        for msg in hf_messages:
            if msg["role"] == "assistant":
                content_ids = self.tokenizer.encode(
                    msg["content"], add_special_tokens=False
                )
                assert len(content_ids) > 0

        loss_count = sum(1 for v in labels if v != IGNORE_INDEX)
        assert loss_count > 0


    def test_blocks_format_preserved_through_mds(self):
        """Verify blocks format preserved through mds."""
        mds_path = _create_mds_from_samples(
            [BLOCKS_FORMAT_SAMPLE], self.tmp_dir, num_shards=1
        )
        loaded = load_mds_samples(mds_path)
        mds_messages = loaded[0]["messages"]

        assistant_msg = [m for m in mds_messages if m["role"] == "assistant"][0]
        blocks = assistant_msg["blocks"]
        assert isinstance(blocks, list), "IR blocks 应该是 list"
        assert any(b.get("type") == "thinking" for b in blocks), "thinking block 丢失"

        input_ids, labels = tokenize_with_hf_template(
            messages=mds_messages,
            tokenizer=self.tokenizer,
            max_length=4096,
        )
        assert len(input_ids) > 0


    def test_token_count_column_is_content_estimate(self):
        """Verify token count column is content estimate."""
        mds_path = _create_mds_from_samples(
            [SINGLE_TURN_SAMPLE], self.tmp_dir, num_shards=1
        )
        loaded = load_mds_samples(mds_path)
        sample = loaded[0]

        input_ids, _ = tokenize_with_hf_template(
            messages=sample["messages"],
            tokenizer=self.tokenizer,
            max_length=4096,
        )

        assert sample["token_count"] == len(input_ids), (
            f"token_count({sample['token_count']}) 应该等于 "
            f"完整序列长度({len(input_ids)})（encoder-truth）"
        )


    def test_loss_config_survives_mds_roundtrip(self):
        """Verify loss config survives mds roundtrip."""
        mds_path = _create_mds_from_samples(
            [WITH_LOSS_CONFIG_SAMPLE], self.tmp_dir, num_shards=1
        )
        loaded = load_mds_samples(
            mds_path, columns=["messages", "stream_id", "token_count", "loss_config"]
        )
        loss_config = loaded[0]["loss_config"]
        assert "type_overrides" in loss_config
        assert loss_config["type_overrides"]["thinking"] == 0.5


    def test_batch_all_samples_tokenize_successfully(self):
        """Verify batch all samples tokenize successfully."""
        mds_path = _create_mds_from_samples(ALL_SAMPLES, self.tmp_dir)
        loaded = load_mds_samples(mds_path)

        for i, sample in enumerate(loaded):
            input_ids, labels = tokenize_with_hf_template(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_length=4096,
            )
            assert len(input_ids) > 0, f"sample {i}: 空 input_ids"
            assert len(labels) == len(input_ids), (
                f"sample {i}: labels 长度不匹配"
            )


    @pytest.mark.parametrize(
        "dataset_name,mds_subdir",
        [
            ("basic", "data/sft_test/basic/mds/train"),
            ("with_system", "data/sft_test/with_system/mds/train"),
            ("multiturn", "data/sft_test/multiturn/mds/train"),
            ("thinking", "data/sft_test/thinking/mds/train"),
            ("mixed", "data/sft_test/mixed/mds/train"),
        ],
    )
    def test_real_mds_input_ids_match_hf(self, dataset_name, mds_subdir):
        """Verify real mds input ids match hf."""
        repo_root = Path(__file__).parent.parent.parent
        mds_path = str(repo_root / mds_subdir)
        if not Path(mds_path).is_dir():
            pytest.skip(f"MDS 目录不存在: {mds_path}（需先生成 MDS）")

        loaded = load_mds_samples(mds_path)
        n_samples = min(20, len(loaded))
        assert n_samples > 0, f"{dataset_name}: MDS 为空"

        for i in range(n_samples):
            mds_messages = loaded[i]["messages"]

            input_ids_titaninfra, _ = tokenize_with_hf_template(
                messages=mds_messages,
                tokenizer=self.tokenizer,
                max_length=8192,
            )

            hf_messages, _ = normalize_to_hf_messages(mds_messages)
            input_ids_hf = self.tokenizer.apply_chat_template(
                hf_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            input_ids_hf = input_ids_hf[:8192]

            assert input_ids_titaninfra == input_ids_hf, (
                f"[{dataset_name} sample {i}] input_ids 不一致!\n"
                f"titaninfra len={len(input_ids_titaninfra)}, hf len={len(input_ids_hf)}\n"
                f"first diff at: {_first_diff(input_ids_titaninfra, input_ids_hf)}"
            )


    @pytest.mark.parametrize(
        "dataset_name,mds_subdir",
        [
            ("basic", "data/sft_test/basic/mds/train"),
            ("multiturn", "data/sft_test/multiturn/mds/train"),
            ("thinking", "data/sft_test/thinking/mds/train"),
        ],
    )
    def test_real_mds_loss_mask_consistency(self, dataset_name, mds_subdir):
        """Verify real mds loss mask consistency."""
        repo_root = Path(__file__).parent.parent.parent
        mds_path = str(repo_root / mds_subdir)
        if not Path(mds_path).is_dir():
            pytest.skip(f"MDS 目录不存在: {mds_path}")

        loaded = load_mds_samples(mds_path)
        n_samples = min(10, len(loaded))
        assert n_samples > 0

        for i in range(n_samples):
            input_ids, labels = tokenize_with_hf_template(
                messages=loaded[i]["messages"],
                tokenizer=self.tokenizer,
                max_length=8192,
            )
            assert len(input_ids) == len(labels), (
                f"[{dataset_name} sample {i}] labels 长度不匹配"
            )
            loss_count = sum(1 for v in labels if v != IGNORE_INDEX)
            mask_count = sum(1 for v in labels if v == IGNORE_INDEX)
            assert loss_count > 0, (
                f"[{dataset_name} sample {i}] 无任何 token 有 loss"
            )
            assert mask_count > 0, (
                f"[{dataset_name} sample {i}] 无任何 token 被 mask"
            )


    def test_real_mds_aime_sft_input_ids_match_hf(self):
        """Verify real mds aime sft input ids match hf."""
        repo_root = Path(__file__).parent.parent.parent
        mds_path = str(repo_root / "data" / "aime" / "mds_sft" / "train")
        if not Path(mds_path).is_dir():
            pytest.skip(f"MDS 目录不存在: {mds_path}")

        loaded = load_mds_samples(mds_path)
        n_samples = min(10, len(loaded))
        assert n_samples > 0

        for i in range(n_samples):
            mds_messages = loaded[i]["messages"]

            input_ids_titaninfra, _ = tokenize_with_hf_template(
                messages=mds_messages,
                tokenizer=self.tokenizer,
                max_length=16384,
            )

            hf_messages, _ = normalize_to_hf_messages(mds_messages)
            input_ids_hf = self.tokenizer.apply_chat_template(
                hf_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )[:16384]

            assert input_ids_titaninfra == input_ids_hf, (
                f"[aime sample {i}] input_ids 不一致!\n"
                f"titaninfra len={len(input_ids_titaninfra)}, hf len={len(input_ids_hf)}\n"
                f"first diff at: {_first_diff(input_ids_titaninfra, input_ids_hf)}"
            )
