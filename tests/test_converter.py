"""Unit tests for converter."""

from pathlib import Path

import pytest


class TestDataConvertConfig:
    """Tests for test data convert config."""

    def test_config_defaults(self):
        """Verify config defaults."""
        from titaninfra.data import DataConvertConfig

        config = DataConvertConfig(
            source="s3://bucket/data.parquet",
            output="./output",
        )

        assert config.num_workers == 4
        assert config.batch_size == 1000
        assert config.compression is None
        assert config.text_key == "text"
        assert config.stream_id == ""
        assert config.stream_type == "pretrain"

    def test_config_custom_values(self):
        """Verify config custom values."""
        from titaninfra.data import DataConvertConfig

        config = DataConvertConfig(
            source="s3://bucket/data/",
            output="./output",
            tokenizer_path="Qwen/Qwen2.5-7B",
            num_workers=8,
        )

        assert config.tokenizer_path == "Qwen/Qwen2.5-7B"
        assert config.num_workers == 8




class TestModuleImport:
    """Tests for test module import."""

    def test_import_from_data(self):
        """Verify import from data."""
        from titaninfra.data import (
            DataConvertConfig,
            DataConverter,
            convert_to_mds,
        )

        assert DataConverter is not None
        assert DataConvertConfig is not None
        assert convert_to_mds is not None

    def test_lazy_import(self):
        """Verify lazy import."""
        from titaninfra.data import DataConvertConfig

        config = DataConvertConfig(
            source="s3://bucket/data.parquet",
            output="./output",
        )
        assert config.source == "s3://bucket/data.parquet"


# ============================================================================
# ============================================================================


class TestShardValidation:
    """Tests for test shard validation."""

    @staticmethod
    def _write_index(output_dir: Path, shards: list[dict]) -> None:
        """Write index."""
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        index = {"version": 2, "shards": shards}
        (output_dir / "index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )

    @staticmethod
    def _write_data_file(output_dir: Path, filename: str, content: bytes) -> None:
        """Write data file."""
        (output_dir / filename).write_bytes(content)


    def test_valid_shards_pass(self, tmp_path):
        """Verify valid shards pass."""
        from titaninfra.data.converter import validate_mds_shards

        output_dir = tmp_path / "output"
        data_content = b"fake_mds_data_1234567890"

        self._write_index(output_dir, [
            {
                "samples": 100,
                "raw_data": {
                    "basename": "shard.00000.mds",
                    "bytes": len(data_content),
                },
            },
        ])
        self._write_data_file(output_dir, "shard.00000.mds", data_content)

        result = validate_mds_shards(str(output_dir))

        assert result["passed"] is True
        assert result["total_shards"] == 1
        assert result["total_samples"] == 100

    def test_multiple_shards_pass(self, tmp_path):
        """Verify multiple shards pass."""
        from titaninfra.data.converter import validate_mds_shards

        output_dir = tmp_path / "output"
        data_a = b"shard_a_data"
        data_b = b"shard_b_data"

        self._write_index(output_dir, [
            {
                "samples": 50,
                "raw_data": {"basename": "shard.00000.mds", "bytes": len(data_a)},
            },
            {
                "samples": 70,
                "raw_data": {"basename": "shard.00001.mds", "bytes": len(data_b)},
            },
        ])
        self._write_data_file(output_dir, "shard.00000.mds", data_a)
        self._write_data_file(output_dir, "shard.00001.mds", data_b)

        result = validate_mds_shards(str(output_dir))

        assert result["passed"] is True
        assert result["total_shards"] == 2
        assert result["total_samples"] == 120


    def test_missing_index_json(self, tmp_path):
        """Verify missing index json."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(ShardValidationError, match="index.json 不存在"):
            validate_mds_shards(str(output_dir))

    def test_corrupted_index_json(self, tmp_path):
        """Verify corrupted index json."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "index.json").write_text("{{invalid json", encoding="utf-8")

        with pytest.raises(ShardValidationError, match="index.json 解析失败"):
            validate_mds_shards(str(output_dir))

    def test_empty_shards_list(self, tmp_path):
        """Verify empty shards list."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        self._write_index(output_dir, [])

        with pytest.raises(ShardValidationError, match="shards 列表为空"):
            validate_mds_shards(str(output_dir))


    def test_missing_data_file(self, tmp_path):
        """Verify missing data file."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        self._write_index(output_dir, [
            {
                "samples": 10,
                "raw_data": {"basename": "shard.00000.mds", "bytes": 100},
            },
        ])

        with pytest.raises(ShardValidationError, match="数据文件缺失"):
            validate_mds_shards(str(output_dir))

    def test_file_size_mismatch(self, tmp_path):
        """Verify file size mismatch."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        data_content = b"short"

        self._write_index(output_dir, [
            {
                "samples": 10,
                "raw_data": {"basename": "shard.00000.mds", "bytes": 99999},
            },
        ])
        self._write_data_file(output_dir, "shard.00000.mds", data_content)

        with pytest.raises(ShardValidationError, match="大小不匹配"):
            validate_mds_shards(str(output_dir))


    def test_hash_mismatch(self, tmp_path):
        """Verify hash mismatch."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        data_content = b"real_data_here"

        self._write_index(output_dir, [
            {
                "samples": 5,
                "raw_data": {
                    "basename": "shard.00000.mds",
                    "bytes": len(data_content),
                    "hashes": {"sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
                },
            },
        ])
        self._write_data_file(output_dir, "shard.00000.mds", data_content)

        with pytest.raises(ShardValidationError, match="哈希不匹配"):
            validate_mds_shards(str(output_dir))

    def test_hash_match_passes(self, tmp_path):
        """Verify hash match passes."""
        import hashlib as hl

        from titaninfra.data.converter import validate_mds_shards

        output_dir = tmp_path / "output"
        data_content = b"exact_data_for_hash"
        expected_sha256 = hl.sha256(data_content).hexdigest()

        self._write_index(output_dir, [
            {
                "samples": 3,
                "raw_data": {
                    "basename": "shard.00000.mds",
                    "bytes": len(data_content),
                    "hashes": {"sha256": expected_sha256},
                },
            },
        ])
        self._write_data_file(output_dir, "shard.00000.mds", data_content)

        result = validate_mds_shards(str(output_dir))
        assert result["passed"] is True


    def test_zero_total_samples(self, tmp_path):
        """Verify zero total samples."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        data_content = b"empty_shard"

        self._write_index(output_dir, [
            {
                "samples": 0,
                "raw_data": {"basename": "shard.00000.mds", "bytes": len(data_content)},
            },
        ])
        self._write_data_file(output_dir, "shard.00000.mds", data_content)

        with pytest.raises(ShardValidationError, match="样本总数为 0"):
            validate_mds_shards(str(output_dir))

    def test_schema_inconsistency(self, tmp_path):
        """Verify schema inconsistency."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        data_a = b"aaa"
        data_b = b"bbb"

        self._write_index(output_dir, [
            {
                "samples": 10,
                "column_encodings": {"text": "str"},
                "raw_data": {"basename": "shard.00000.mds", "bytes": len(data_a)},
            },
            {
                "samples": 10,
                "column_encodings": {"text": "str", "label": "int"},
                "raw_data": {"basename": "shard.00001.mds", "bytes": len(data_b)},
            },
        ])
        self._write_data_file(output_dir, "shard.00000.mds", data_a)
        self._write_data_file(output_dir, "shard.00001.mds", data_b)

        with pytest.raises(ShardValidationError, match="column schema.*不一致"):
            validate_mds_shards(str(output_dir))


    def test_error_contains_error_list(self, tmp_path):
        """Verify error contains error list."""
        from titaninfra.data.converter import ShardValidationError, validate_mds_shards

        output_dir = tmp_path / "output"
        self._write_index(output_dir, [
            {
                "samples": 10,
                "raw_data": {"basename": "shard.00000.mds", "bytes": 100},
            },
            {
                "samples": 10,
                "raw_data": {"basename": "shard.00001.mds", "bytes": 200},
            },
        ])

        with pytest.raises(ShardValidationError) as exc_info:
            validate_mds_shards(str(output_dir))

        assert len(exc_info.value.errors) == 2
        assert all("数据文件缺失" in e for e in exc_info.value.errors)


    def test_config_validation_params(self):
        """Verify config validation params."""
        from titaninfra.data.converter import DataConvertConfig

        config = DataConvertConfig(source="dummy", output="dummy")
        assert config.validate_after_convert is True
        assert config.validate_sample_ratio == 0.01
        assert config.validate_min_samples == 2


# ============================================================================
# ============================================================================


class TestGreedyBinPack:
    """Tests for test greedy bin pack."""

    def test_even_distribution(self):
        """Verify even distribution."""
        from titaninfra.data.converter import _greedy_bin_pack

        items = [(f"text_{i}", 100) for i in range(12)]
        bins, bin_tokens = _greedy_bin_pack(items, 4)

        assert len(bins) == 4
        assert all(len(b) == 3 for b in bins)
        assert all(t == 300 for t in bin_tokens)

    def test_skewed_distribution(self):
        """Verify skewed distribution."""
        from titaninfra.data.converter import _greedy_bin_pack

        items = [(f"t_{i}", 10 + i * 50) for i in range(100)]
        bins, bin_tokens = _greedy_bin_pack(items, 8)

        avg = sum(bin_tokens) / 8
        max_dev = max(abs(t - avg) / avg * 100 for t in bin_tokens)
        assert max_dev < 5.0, f"最大偏差 {max_dev:.1f}% 超过 5%"

    def test_single_bin(self):
        """Verify single bin."""
        from titaninfra.data.converter import _greedy_bin_pack

        items = [("a", 10), ("b", 20), ("c", 30)]
        bins, bin_tokens = _greedy_bin_pack(items, 1)

        assert len(bins) == 1
        assert len(bins[0]) == 3
        assert bin_tokens[0] == 60

    def test_more_bins_than_items(self):
        """Verify more bins than items."""
        from titaninfra.data.converter import _greedy_bin_pack

        items = [("a", 10), ("b", 20)]
        bins, bin_tokens = _greedy_bin_pack(items, 5)

        assert len(bins) == 5
        non_empty = [b for b in bins if b]
        assert len(non_empty) == 2


class TestBalancedShardsConfig:
    """Tests for test balanced shards config."""

    def test_balanced_shards_fields(self):
        """Verify balanced shards fields."""
        from titaninfra.data.converter import DataConvertConfig

        config = DataConvertConfig(source="dummy", output="dummy")
        assert config.num_shards == 16
        assert config.store_token_count is False


class TestReaderWorker:
    """Tests for test reader worker."""

    def test_reader_puts_batches(self, tmp_path):
        """Verify reader puts batches."""
        import multiprocessing as mp
        import time

        from titaninfra.data.converter import _generic_reader_worker

        data_file = tmp_path / "data.jsonl"
        lines = [f'{{"text": "sample text {i}"}}' for i in range(25)]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        q: mp.Queue = mp.Queue()
        _generic_reader_worker([str(data_file)], q, batch_size=10)

        time.sleep(0.1)

        batches = []
        while not q.empty():
            batches.append(q.get())

        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5
        assert isinstance(batches[0][0], dict)
        assert "text" in batches[0][0]


    def test_reader_skips_bad_file(self, tmp_path):
        """Verify reader skips bad file."""
        import multiprocessing as mp
        import time

        from titaninfra.data.converter import _generic_reader_worker

        good1 = tmp_path / "good1.jsonl"
        good1.write_text('{"text": "a"}\n{"text": "b"}', encoding="utf-8")
        bad = tmp_path / "bad.parquet"
        bad.write_text("this is not a parquet file", encoding="utf-8")
        good2 = tmp_path / "good2.jsonl"
        good2.write_text('{"text": "c"}', encoding="utf-8")

        q: mp.Queue = mp.Queue()
        _generic_reader_worker([str(good1), str(bad), str(good2)], q, batch_size=100)

        time.sleep(0.1)
        batches = []
        while not q.empty():
            batches.append(q.get())

        total_records = sum(len(b) for b in batches)
        assert total_records == 3


class TestPreload:
    """Tests for test preload."""

    def test_preload_pipeline(self, tmp_path):
        """Verify preload pipeline."""
        pytest.importorskip("streaming")


        for i in range(3):
            f = tmp_path / "input" / f"data_{i}.jsonl"
            f.parent.mkdir(exist_ok=True)
            lines = [f'{{"text": "file {i} sample {j}"}}' for j in range(10)]
            f.write_text("\n".join(lines), encoding="utf-8")

        files = sorted(str(p) for p in (tmp_path / "input").glob("*.jsonl"))

        from titaninfra.data.converter import (
            _read_records_from_file,
        )

        bad = tmp_path / "input" / "bad.parquet"
        bad.write_text("corrupt", encoding="utf-8")
        all_files = files + [str(bad)]

        total_records = 0
        failed = 0
        for fp in all_files:
            try:
                records = _read_records_from_file(fp)
                total_records += len(records)
            except Exception:
                failed += 1

        assert total_records == 30  # 3 files × 10 records
        assert failed == 1  # bad.parquet

    def test_preload_config_default(self):
        """Verify preload config default."""
        from titaninfra.data.converter import DataConvertConfig

        config = DataConvertConfig(source="dummy", output="dummy")
        assert config.preload is False

    def test_resume_without_checkpoint_guards_existing_data(self, tmp_path):
        """Verify resume without checkpoint guards existing data."""
        pytest.importorskip("streaming")

        from titaninfra.data.converter import DataConvertConfig, DataConverter

        output = tmp_path / "output"
        output.mkdir()
        (output / "_batch_000" / "shard_00000").mkdir(parents=True)
        (output / "_batch_001" / "shard_00000").mkdir(parents=True)

        source = tmp_path / "src"
        source.mkdir()
        (source / "data.jsonl").write_text('{"text": "hello"}', encoding="utf-8")

        converter = DataConverter(DataConvertConfig(
            source=str(source),
            output=str(output),
            tokenizer_path=QWEN3_TOK,
            num_shards=1,
            checkpoint_interval=10,
            resume=True,
        ))

        with pytest.raises(FileExistsError, match="batch 目录"):
            converter.convert()


class TestStreamIdAndStats:
    """Tests for test stream id and stats."""

    def test_resolve_stream_id_explicit(self):
        """Verify resolve stream id explicit."""
        from titaninfra.data.converter import DataConvertConfig, _resolve_stream_id

        config = DataConvertConfig(
            source="/data/zh_wiki.parquet",
            output="/out",
            stream_id="my_wiki",
        )
        assert _resolve_stream_id(config) == "my_wiki"

    def test_resolve_stream_id_from_path(self):
        """Verify resolve stream id from path."""
        from titaninfra.data.converter import DataConvertConfig, _resolve_stream_id

        config = DataConvertConfig(
            source="/data/zh_wiki.parquet",
            output="/out",
        )
        assert _resolve_stream_id(config) == "zh_wiki"

    def test_write_stream_stats_v3(self, tmp_path):
        """Verify write stream stats v3."""
        import json

        from titaninfra.data.converter import _write_stream_stats

        _write_stream_stats(
            str(tmp_path), 100, 50000,
            stream_id="zh_wiki", stream_type="pretrain",
            tokenizer_name="qwen",
        )

        with open(tmp_path / "stream_stats.json") as f:
            stats = json.load(f)

        assert stats["version"] == 3
        assert stats["stream_id"] == "zh_wiki"
        assert stats["stream_type"] == "pretrain"
        assert stats["num_samples"] == 100
        assert "total_tokens" not in stats
        assert "avg_tokens_per_sample" not in stats
        assert stats["default_tokenizer"] == "qwen"
        assert "qwen" in stats["tokenizer_stats"]
        tok = stats["tokenizer_stats"]["qwen"]
        assert tok["total_tokens"] == 50000
        assert abs(tok["avg_tokens_per_sample"] - 500.0) < 1e-6
        assert "tokenizer" not in stats

    def test_write_stream_stats_explicit_tokenizer_name(self, tmp_path):
        """Verify write stream stats explicit tokenizer name."""
        import json

        from titaninfra.data.converter import _write_stream_stats

        _write_stream_stats(
            str(tmp_path), 50, 25000,
            stream_id="test", stream_type="pretrain",
            tokenizer_name="qwen3", tokenizer_path="/path/to/Qwen3-30B-A3B-Base",
        )

        with open(tmp_path / "stream_stats.json") as f:
            stats = json.load(f)

        assert stats["default_tokenizer"] == "qwen3"
        assert "qwen3" in stats["tokenizer_stats"]
        assert "Qwen3-30B-A3B-Base" not in stats["tokenizer_stats"]

    def test_config_stream_type_sft(self):
        """Verify config stream type sft."""
        from titaninfra.data.converter import DataConvertConfig

        config = DataConvertConfig(
            source="/data/alpaca.parquet",
            output="/out",
            stream_id="alpaca_sft",
            stream_type="sft",
        )
        assert config.stream_type == "sft"
        assert config.stream_id == "alpaca_sft"


# ============================================================================
# ============================================================================


class TestExtractMessagesContent:
    """Tests for test extract messages content."""

    def test_blocks_format(self):
        """Verify blocks format."""
        from titaninfra.data.converter import _extract_messages_content

        messages = [
            {"role": "user", "blocks": [{"type": "text", "text": "Hello world"}]},
            {"role": "assistant", "blocks": [
                {"type": "thinking", "text": "Let me think..."},
                {"type": "text", "text": "Hi there!"},
            ]},
        ]
        result = _extract_messages_content(messages)
        assert "Hello world" in result
        assert "Let me think..." in result
        assert "Hi there!" in result

    def test_content_format(self):
        """Verify content format."""
        from titaninfra.data.converter import _extract_messages_content

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        result = _extract_messages_content(messages)
        assert "What is 2+2?" in result
        assert "4" in result

    def test_content_list_format(self):
        """Verify content list format."""
        from titaninfra.data.converter import _extract_messages_content

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": "..."},
            ]},
        ]
        result = _extract_messages_content(messages)
        assert "Describe this image" in result

    def test_mixed_formats(self):
        """Verify mixed formats."""
        from titaninfra.data.converter import _extract_messages_content

        messages = [
            {"role": "user", "blocks": [{"type": "text", "text": "AAA"}]},
            {"role": "assistant", "content": "BBB"},
        ]
        result = _extract_messages_content(messages)
        assert "AAA" in result
        assert "BBB" in result

    def test_empty_messages(self):
        """Verify empty messages."""
        from titaninfra.data.converter import _extract_messages_content

        assert _extract_messages_content([]) == ""

    def test_no_text_content(self):
        """Verify no text content."""
        from titaninfra.data.converter import _extract_messages_content

        messages = [{"role": "system"}]
        assert _extract_messages_content(messages) == ""

    def test_content_vs_json_chars(self):
        """Verify content vs json chars."""
        import json

        from titaninfra.data.converter import _extract_messages_content

        messages = [
            {"role": "user", "blocks": [{"type": "text", "text": "short"}]},
            {"role": "assistant", "blocks": [{"type": "text", "text": "reply"}]},
        ]
        content = _extract_messages_content(messages)
        json_chars = len(json.dumps(messages, ensure_ascii=False))

        assert len(content) < json_chars
        assert "short" in content
        assert "reply" in content


QWEN3_TOKENIZER = "tests/fixtures/tokenizers/qwen3"


class TestConvertMessagesToMds:
    """Tests for test convert messages to mds."""

    def _create_test_jsonl(self, tmp_path, samples):
        """Create test jsonl."""
        import json

        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return str(path)

    def test_basic_conversion(self, tmp_path):
        """Verify basic conversion."""
        pytest.importorskip("streaming")

        from titaninfra.data.converter import convert_messages_to_mds

        samples = [
            {"messages": [
                {"role": "user", "blocks": [{"type": "text", "text": "Hi"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "Hello!"}]},
            ]}
            for _ in range(20)
        ]

        src = self._create_test_jsonl(tmp_path / "src", samples)
        out = str(tmp_path / "out")

        result = convert_messages_to_mds(
            source=src, output=out, num_shards=4, overwrite=True,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
        )

        assert result["num_samples"] == 20
        assert result["total_tokens"] > 0
        assert result["max_deviation_pct"] is not None

        import json
        stats_path = tmp_path / "out" / "stream_stats.json"
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["version"] == 3
        assert stats["stream_type"] == "messages"
        assert stats["num_samples"] == 20

    def test_bin_packing_uniformity(self, tmp_path):
        """Verify bin packing uniformity."""
        pytest.importorskip("streaming")

        from titaninfra.data.converter import convert_messages_to_mds

        samples = []
        for i in range(40):
            text_len = 100 + i * 50
            samples.append({
                "messages": [
                    {"role": "user", "blocks": [{"type": "text", "text": "Q"}]},
                    {"role": "assistant", "blocks": [{"type": "text", "text": "A" * text_len}]},
                ],
            })

        src = self._create_test_jsonl(tmp_path / "src", samples)
        out = str(tmp_path / "out")

        result = convert_messages_to_mds(
            source=src, output=out, num_shards=4, overwrite=True,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
        )

        assert result["max_deviation_pct"] < 10.0

    def test_loss_config_preserved(self, tmp_path):
        """Verify loss config preserved."""
        pytest.importorskip("streaming")

        from titaninfra.data.converter import convert_messages_to_mds

        samples = [
            {
                "messages": [
                    {"role": "user", "blocks": [{"type": "text", "text": "Hi"}]},
                    {"role": "assistant", "blocks": [{"type": "text", "text": "Hello!"}]},
                ],
                "loss_config": {"thinking": 0.5},
            }
            for _ in range(10)
        ]

        src = self._create_test_jsonl(tmp_path / "src", samples)
        out = str(tmp_path / "out")

        result = convert_messages_to_mds(
            source=src, output=out, num_shards=2, overwrite=True,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
        )
        assert result["num_samples"] == 10

    def test_stream_id_inference(self, tmp_path):
        """Verify stream id inference."""
        pytest.importorskip("streaming")
        import json

        from titaninfra.data.converter import convert_messages_to_mds

        samples = [
            {"messages": [
                {"role": "user", "blocks": [{"type": "text", "text": "Hi"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "Hello"}]},
            ]}
            for _ in range(5)
        ]

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src_path = src_dir / "my_dataset.jsonl"
        with open(src_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        out = str(tmp_path / "out")
        convert_messages_to_mds(
            source=str(src_path), output=out, num_shards=2, overwrite=True,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
        )

        with open(tmp_path / "out" / "stream_stats.json") as f:
            stats = json.load(f)
        assert stats["stream_id"] == "my_dataset"


# ============================================================================
# ============================================================================


class TestTokenizerAlignment:
    """Tests for test tokenizer alignment."""

    def _create_test_jsonl(self, tmp_path, samples):
        import json

        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return str(path)

    def _ground_truth_tokens(self, samples, tokenizer, messages_key="messages", family="qwen3"):
        """Ground truth tokens."""
        from titaninfra.data.sft.capabilities import EmptyAssistantError
        from titaninfra.data.sft.dispatch import tokenize_sample

        token_counts = []
        for sample in samples:
            sample_for_dispatch = {"messages": sample[messages_key]}
            for k in ("system", "tools", "loss_config", "thinking_intent"):
                if sample.get(k):
                    sample_for_dispatch[k] = sample[k]
            try:
                encoded = tokenize_sample(sample_for_dispatch, tokenizer, family)
                token_counts.append(max(1, len(encoded.input_ids)))
            except EmptyAssistantError:
                token_counts.append(0)
        return token_counts

    def test_chinese_data_tokenizer_vs_chars4(self, tmp_path):
        """Verify chinese data tokenizer vs chars4."""
        pytest.importorskip("streaming")
        from transformers import AutoTokenizer

        from titaninfra.data.converter import convert_messages_to_mds

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

        samples = []
        cn_texts = [
            "请解释什么是机器学习？",
            (
                "机器学习是人工智能的一个分支，"
                "它使用统计方法让计算机从数据中学习并做出决策或预测，而不需要明确地编程。"
            ),
            "深度强化学习在自动驾驶领域的应用有哪些？" * 3,
            "好的",
            "请用Python实现一个二叉搜索树，支持插入删除查找操作。" * 2,
            "什么是Transformer？",
            (
                "Transformer是一种基于自注意力机制的神经网络架构，"
                "由Vaswani等人在2017年提出。"
            ),
            "请问量子计算如何加速优化问题的求解？",
        ]
        for i, text in enumerate(cn_texts):
            samples.append({
                "messages": [
                    {"role": "user", "blocks": [{"type": "text", "text": f"问题{i + 1}"}]},
                    {"role": "assistant", "blocks": [{"type": "text", "text": text}]},
                ],
            })

        src = self._create_test_jsonl(tmp_path / "src", samples)

        out = str(tmp_path / "out")
        result = convert_messages_to_mds(
            source=src, output=out, num_shards=4, overwrite=True,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
        )

        assert result["num_samples"] == len(samples)

        gt_tokens = self._ground_truth_tokens(samples, tokenizer)
        gt_total = sum(gt_tokens)

        assert result["total_tokens"] == gt_total, (
            f"total_tokens={result['total_tokens']} != ground_truth={gt_total}"
        )

        print("\n=== 中文数据 tokenizer 对齐验证 ===")
        print(f"Ground truth total tokens: {gt_total}")
        print(f"tokenizer total_tokens:    {result['total_tokens']}")
        print(f"shard deviation:           {result['max_deviation_pct']:.1f}%")

    def test_mixed_lang_data_alignment(self, tmp_path):
        """Verify mixed lang data alignment."""
        pytest.importorskip("streaming")
        from transformers import AutoTokenizer

        from titaninfra.data.converter import convert_messages_to_mds

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

        samples = []
        conversations = [
            [
                {"role": "user", "blocks": [{"type": "text", "text": "What is 2+2?"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "4"}]},
            ],
            [
                {"role": "user", "blocks": [{"type": "text", "text": "解释梯度下降"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": (
                    "梯度下降是一种优化算法，通过计算损失函数关于模型参数的梯度，"
                    "沿着梯度的反方向更新参数，逐步最小化损失函数。"
                )}]},
            ],
            [
                {"role": "user", "blocks": [{"type": "text", "text": "Write fibonacci in Python"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": (
                    "def fibonacci(n):\n"
                    "    a, b = 0, 1\n"
                    "    for _ in range(n): a, b = b, a+b\n"
                    "    return a"
                )}]},
            ],
            [
                {"role": "user", "blocks": [{"type": "text", "text": "请介绍大语言模型的训练流程"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "训练分三阶段：预训练、SFT和RLHF。"}]},
                {"role": "user", "blocks": [{"type": "text", "text": "预训练阶段的数据如何处理？"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": (
                    "预训练数据处理包括文本清洗去重、Tokenization分词、"
                    "序列打包以提高GPU利用率。"
                )}]},
            ],
            [
                {"role": "user", "blocks": [{"type": "text", "text": "Hi"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "Hello!"}]},
            ],
        ]
        for conv in conversations * 6:
            samples.append({"messages": conv})

        src = self._create_test_jsonl(tmp_path / "src", samples)

        out = str(tmp_path / "out")
        result = convert_messages_to_mds(
            source=src, output=out, num_shards=8, overwrite=True,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
        )

        gt_tokens = self._ground_truth_tokens(samples, tokenizer)
        gt_total = sum(gt_tokens)

        assert result["total_tokens"] == gt_total

        print("\n=== 中英混合 tokenizer 对齐验证 ===")
        print(f"Samples: {len(samples)}, Shards: 8")
        print(f"Ground truth total tokens: {gt_total}")
        print(f"tokenizer total_tokens:    {result['total_tokens']}")
        print(f"shard deviation:           {result['max_deviation_pct']:.1f}%")

    def test_stream_stats_accuracy(self, tmp_path):
        """Verify stream stats accuracy."""
        pytest.importorskip("streaming")
        import json

        from transformers import AutoTokenizer

        from titaninfra.data.converter import convert_messages_to_mds

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

        samples = [
            {"messages": [
                {"role": "user", "blocks": [{"type": "text", "text": "你好"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "你好！有什么我可以帮助你的吗？"}]},
            ]}
            for _ in range(20)
        ]

        src = self._create_test_jsonl(tmp_path / "src", samples)

        out = str(tmp_path / "out")
        convert_messages_to_mds(
            source=src, output=out, num_shards=4, overwrite=True,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
        )

        with open(tmp_path / "out" / "stream_stats.json") as f:
            stats = json.load(f)

        # ground truth
        gt_tokens = self._ground_truth_tokens(samples, tokenizer)
        gt_total = sum(gt_tokens)
        gt_avg = gt_total / len(samples)

        assert stats["num_samples"] == 20
        # token stats live only in tokenizer_stats, not at top level
        assert "total_tokens" not in stats
        assert "avg_tokens_per_sample" not in stats
        tok_key = stats["default_tokenizer"]
        tok_entry = stats["tokenizer_stats"][tok_key]
        assert tok_entry["total_tokens"] == gt_total
        assert abs(tok_entry["avg_tokens_per_sample"] - gt_avg) < 1e-6

        print("\n=== stream_stats 精度验证 ===")
        print(f"avg_tokens_per_sample: stats={tok_entry['avg_tokens_per_sample']:.1f}, gt={gt_avg:.1f}")

    def test_tokenizer_path_mds_token_count_column(self, tmp_path):
        """Verify tokenizer path mds token count column."""
        pytest.importorskip("streaming")
        from transformers import AutoTokenizer

        from titaninfra.data.converter import convert_messages_to_mds
        from titaninfra.data.sft.dispatch import tokenize_sample

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

        samples = [
            {"messages": [
                {"role": "user", "blocks": [{"type": "text", "text": "短问题"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "短回答"}]},
            ]},
            {"messages": [
                {"role": "user", "blocks": [{"type": "text", "text": "请详细解释注意力机制的工作原理"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": "注意力机制通过计算QK相似度得到权重。" * 3}]},
            ]},
        ]

        src = self._create_test_jsonl(tmp_path / "src", samples)
        out = str(tmp_path / "out")
        convert_messages_to_mds(
            source=src, output=out, num_shards=1, overwrite=True,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
        )

        from tests._mds_utils import load_mds_samples

        loaded = load_mds_samples(out, columns=["messages", "stream_id", "token_count"])

        for i, mds_sample in enumerate(loaded):
            msgs_parsed = mds_sample["messages"]
            encoded = tokenize_sample({"messages": msgs_parsed}, tokenizer, "qwen3")
            expected_tc = max(1, len(encoded.input_ids))
            actual_tc = mds_sample["token_count"]
            assert actual_tc == expected_tc, (
                f"Sample {i}: MDS token_count={actual_tc} != expected={expected_tc}"
            )

        print(f"\n=== MDS token_count 列验证通过: {len(loaded)} samples ===")


class TestRowFilter:
    """Tests for test row filter."""

    def test_numeric_gt(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "> 0.5"})
        assert f({"score": "0.8"}) is True
        assert f({"score": "0.3"}) is False
        assert f({"score": "0.5"}) is False  # strictly greater

    def test_numeric_gte(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": ">= 0.5"})
        assert f({"score": "0.5"}) is True
        assert f({"score": "0.4"}) is False

    def test_numeric_range(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "0.5 ~ 0.9"})
        assert f({"score": "0.7"}) is True
        assert f({"score": "0.5"}) is True  # inclusive
        assert f({"score": "0.9"}) is True  # inclusive
        assert f({"score": "0.95"}) is False
        assert f({"score": "0.3"}) is False

    def test_exact_match(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"source": "FineWeb"})
        assert f({"source": "FineWeb"}) is True
        assert f({"source": "C4"}) is False

    def test_list_match(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"source": ["FineWeb", "C4"]})
        assert f({"source": "C4"}) is True
        assert f({"source": "FineWeb"}) is True
        assert f({"source": "Wiki"}) is False

    def test_regex(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"content": "re:^[A-Z]"})
        assert f({"content": "Hello"}) is True
        assert f({"content": "hello"}) is False

    def test_regex_case_insensitive(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"url": "re:(?i)arxiv\\.org"})
        assert f({"url": "https://Arxiv.org/paper"}) is True
        assert f({"url": "https://google.com"}) is False

    def test_multi_column_and(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "> 0.5", "source": "FineWeb"})
        assert f({"score": "0.8", "source": "FineWeb"}) is True
        assert f({"score": "0.8", "source": "C4"}) is False
        assert f({"score": "0.3", "source": "FineWeb"}) is False

    def test_missing_column_filtered_out(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "> 0.5"})
        # score missing → "" → float("") → ValueError → False
        assert f({"content": "text"}) is False

    def test_none_filter_returns_none(self):
        from titaninfra.data.converter import _compile_row_filter
        assert _compile_row_filter(None) is None
        assert _compile_row_filter({}) is None

    def test_numeric_lt(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "< 0.5"})
        assert f({"score": "0.3"}) is True
        assert f({"score": "0.5"}) is False  # strictly less
        assert f({"score": "0.8"}) is False

    def test_numeric_lte(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "<= 0.5"})
        assert f({"score": "0.5"}) is True
        assert f({"score": "0.3"}) is True
        assert f({"score": "0.6"}) is False

    def test_numeric_eq(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "== 1.0"})
        assert f({"score": "1.0"}) is True
        assert f({"score": "1"}) is True  # 1 == 1.0
        assert f({"score": "0.9"}) is False

    def test_numeric_ne(self):
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "!= 0"})
        assert f({"score": "0.5"}) is True
        assert f({"score": "0"}) is False
        assert f({"score": "0.0"}) is False  # 0.0 == 0

    def test_float_value_from_parquet(self):
        """Verify float value from parquet."""
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"score": "> 0.5"})
        assert f({"score": 0.8}) is True
        assert f({"score": 0.3}) is False
        assert f({"score": 0.30000000000000004}) is False

    def test_int_value_from_parquet(self):
        """Verify int value from parquet."""
        from titaninfra.data.converter import _compile_row_filter
        f = _compile_row_filter({"length": ">= 100"})
        assert f({"length": 200}) is True
        assert f({"length": 50}) is False
        assert f({"length": 100}) is True

    def test_e2e_with_pipeline(self, tmp_path):
        """Verify e2e with pipeline."""
        pytest.importorskip("streaming")
        import json

        jsonl_file = tmp_path / "data.jsonl"
        lines = []
        for i in range(10):
            source = "good" if i >= 5 else "bad"
            msgs = [
                {"role": "user", "blocks": [{"type": "text", "text": f"question {i}"}]},
                {"role": "assistant", "blocks": [{"type": "text", "text": f"answer {i}"}]},
            ]
            lines.append(json.dumps({"messages": msgs, "source": source}))
        jsonl_file.write_text("\n".join(lines))

        from titaninfra.data.converter import convert_messages_to_mds

        out = str(tmp_path / "mds_out")
        result = convert_messages_to_mds(
            source=str(jsonl_file),
            output=out,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
            num_shards=2,
            overwrite=True,
            row_filter={"source": "good"},
        )
        # source == "good" → 5 samples
        assert result["num_samples"] == 5

    def test_e2e_pretrain_filter(self, tmp_path):
        """Verify e2e pretrain filter."""
        pytest.importorskip("streaming")
        import json

        jsonl_file = tmp_path / "data.jsonl"
        lines = []
        for i in range(10):
            score = round(0.1 * i, 1)
            lines.append(json.dumps({"text": f"This is sample number {i} with some content.", "score": str(score)}))
        jsonl_file.write_text("\n".join(lines))

        from titaninfra.data.converter import DataConvertConfig, DataConverter

        out = str(tmp_path / "mds_out")
        config = DataConvertConfig(
            source=str(jsonl_file),
            output=out,
            tokenizer_path=QWEN3_TOKENIZER, family="qwen3",
            num_shards=2,
            overwrite=True,
            row_filter={"score": "> 0.5"},
        )
        converter = DataConverter(config)
        result = converter.convert()

        # score > 0.5 → values 0.6, 0.7, 0.8, 0.9 → 4 samples
        assert result["samples"] == 4


# ============================================================================
# ============================================================================


class TestValidateColumns:
    """Tests for test validate columns."""

    def test_wrong_text_key_raises(self, tmp_path):
        """Verify wrong text key raises."""
        import json

        from titaninfra.data.converter import _validate_columns

        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"content": "hello world", "score": 0.9}) + "\n")

        with pytest.raises(ValueError, match="数据中不存在列 'text'"):
            _validate_columns([str(f)], mode="pretrain", text_key="text")

    def test_correct_text_key_passes(self, tmp_path):
        import json

        from titaninfra.data.converter import _validate_columns

        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"text": "hello world"}) + "\n")

        _validate_columns([str(f)], mode="pretrain", text_key="text")  # no error

    def test_wrong_messages_key_raises(self, tmp_path):
        import json

        from titaninfra.data.converter import _validate_columns

        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"conversations": [{"role": "user", "content": "hi"}]}) + "\n")

        with pytest.raises(ValueError, match="数据中不存在列 'messages'"):
            _validate_columns([str(f)], mode="messages", messages_key="messages")

    def test_wrong_filter_column_raises(self, tmp_path):
        """Verify wrong filter column raises."""
        import json

        from titaninfra.data.converter import _validate_columns

        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"text": "hello", "quality": 0.8}) + "\n")

        with pytest.raises(ValueError, match="row_filter 引用的列不存在.*score"):
            _validate_columns(
                [str(f)], mode="pretrain", text_key="text",
                row_filter={"score": "> 0.5"},
            )

    def test_correct_filter_column_passes(self, tmp_path):
        import json

        from titaninfra.data.converter import _validate_columns

        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"text": "hello", "score": 0.8}) + "\n")

        _validate_columns(
            [str(f)], mode="pretrain", text_key="text",
            row_filter={"score": "> 0.5"},
        )  # no error

    def test_shows_available_columns(self, tmp_path):
        """Verify shows available columns."""
        import json

        from titaninfra.data.converter import _validate_columns

        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"content": "hi", "quality_score": 0.9}) + "\n")

        with pytest.raises(ValueError, match="content.*quality_score"):
            _validate_columns([str(f)], mode="pretrain", text_key="text")

    def test_empty_files_no_error(self, tmp_path):
        """Verify empty files no error."""
        from titaninfra.data.converter import _validate_columns

        f = tmp_path / "empty.jsonl"
        f.write_text("")

        _validate_columns([str(f)], mode="pretrain", text_key="text")  # no error


# ============================================================================
# ============================================================================

QWEN3_TOK = "tests/fixtures/tokenizers/qwen3"


def _create_jsonl_files(tmp_path, n_files=6, samples_per_file=10):
    """Create jsonl files."""
    import json

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for fid in range(n_files):
        path = data_dir / f"part-{fid:03d}.jsonl"
        with open(path, "w") as f:
            for sid in range(samples_per_file):
                line = {"text": f"File {fid} sample {sid}. " * 5}
                f.write(json.dumps(line) + "\n")
    return str(data_dir)


class TestCheckpointHelpers:
    """Tests for test checkpoint helpers."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """save → load round-trip"""
        from titaninfra.data.converter import _load_checkpoint, _save_checkpoint

        ckpt_path = str(tmp_path / "_checkpoint.json")
        data = {
            "version": 1,
            "num_shards": 4,
            "checkpoint_interval": 2,
            "completed_files": ["/a.jsonl", "/b.jsonl"],
            "shard_tokens": [100, 200, 150, 180],
            "total_samples": 50,
            "total_tokens": 630,
            "batch_count": 1,
        }
        _save_checkpoint(ckpt_path, data)
        loaded = _load_checkpoint(ckpt_path)
        assert loaded == data

    def test_load_missing_returns_none(self, tmp_path):
        from titaninfra.data.converter import _load_checkpoint

        assert _load_checkpoint(str(tmp_path / "nonexistent.json")) is None

    def test_load_corrupted_returns_none(self, tmp_path):
        from titaninfra.data.converter import _load_checkpoint

        bad = tmp_path / "_checkpoint.json"
        bad.write_text("not json {{{")
        assert _load_checkpoint(str(bad)) is None

    def test_load_wrong_version_returns_none(self, tmp_path):
        import json

        from titaninfra.data.converter import _load_checkpoint

        bad = tmp_path / "_checkpoint.json"
        bad.write_text(json.dumps({"version": 999}))
        assert _load_checkpoint(str(bad)) is None

    def test_atomic_write_leaves_no_tmp_on_success(self, tmp_path):
        import os

        from titaninfra.data.converter import _save_checkpoint

        ckpt_path = str(tmp_path / "_checkpoint.json")
        _save_checkpoint(ckpt_path, {"version": 1})
        assert not os.path.exists(ckpt_path + ".tmp")

    def test_clean_batch_dirs(self, tmp_path):
        from titaninfra.data.converter import _clean_batch_dirs

        (tmp_path / "_batch_000").mkdir()
        (tmp_path / "_batch_001").mkdir()
        (tmp_path / "shard_00000").mkdir()
        _clean_batch_dirs(str(tmp_path))
        assert not (tmp_path / "_batch_000").exists()
        assert not (tmp_path / "_batch_001").exists()
        assert (tmp_path / "shard_00000").exists()


class TestCheckpointPretrain:
    """Tests for test checkpoint pretrain."""

    def test_checkpoint_produces_same_total_as_no_checkpoint(self, tmp_path):
        """Verify checkpoint produces same total as no checkpoint."""
        pytest.importorskip("streaming")
        from titaninfra.data.converter import DataConvertConfig, DataConverter

        data_dir = _create_jsonl_files(tmp_path, n_files=6, samples_per_file=10)

        out_no_ckpt = str(tmp_path / "no_ckpt")
        c1 = DataConverter(DataConvertConfig(
            source=data_dir, output=out_no_ckpt,
            tokenizer_path=QWEN3_TOK, num_shards=2, overwrite=True,
        ))
        r1 = c1.convert()

        out_ckpt = str(tmp_path / "with_ckpt")
        c2 = DataConverter(DataConvertConfig(
            source=data_dir, output=out_ckpt,
            tokenizer_path=QWEN3_TOK, num_shards=2, overwrite=True,
            checkpoint_interval=2, resume=True,
        ))
        r2 = c2.convert()

        assert r1["samples"] == r2["samples"], (
            f"sample 数不一致: no_ckpt={r1['samples']}, ckpt={r2['samples']}"
        )
        assert r1["total_tokens"] == r2["total_tokens"], (
            f"token 数不一致: no_ckpt={r1['total_tokens']}, ckpt={r2['total_tokens']}"
        )

    def test_checkpoint_file_deleted_on_completion(self, tmp_path):
        """Verify checkpoint file deleted on completion."""
        import os

        pytest.importorskip("streaming")
        from titaninfra.data.converter import DataConvertConfig, DataConverter

        data_dir = _create_jsonl_files(tmp_path, n_files=4, samples_per_file=5)
        out = str(tmp_path / "out")
        c = DataConverter(DataConvertConfig(
            source=data_dir, output=out,
            tokenizer_path=QWEN3_TOK, num_shards=2, overwrite=True,
            checkpoint_interval=2, resume=True,
        ))
        c.convert()

        assert not os.path.exists(os.path.join(out, "_checkpoint.json"))

    def test_resume_skips_completed_files(self, tmp_path):
        """Verify resume skips completed files."""

        pytest.importorskip("streaming")
        from titaninfra.data.converter import (
            DataConvertConfig,
            DataConverter,
        )

        data_dir = _create_jsonl_files(tmp_path, n_files=6, samples_per_file=5)
        out = str(tmp_path / "out")

        c = DataConverter(DataConvertConfig(
            source=data_dir, output=out,
            tokenizer_path=QWEN3_TOK, num_shards=2, overwrite=True,
            checkpoint_interval=2, resume=True,
        ))
        r_full = c.convert()
        total_full = r_full["samples"]

        batch_dirs = list(Path(out).glob("_batch_*"))
        assert len(batch_dirs) >= 2  # 6 files / 2 = 3 batches

        c2 = DataConverter(DataConvertConfig(
            source=data_dir, output=out,
            tokenizer_path=QWEN3_TOK, num_shards=2,
            checkpoint_interval=2, resume=True,
        ))
        r2 = c2.convert()
        assert r2["samples"] == total_full

    def test_resume_num_shards_mismatch_raises(self, tmp_path):
        """Verify resume num shards mismatch raises."""
        import os

        pytest.importorskip("streaming")
        from titaninfra.data.converter import _CHECKPOINT_VERSION, _save_checkpoint

        out = str(tmp_path / "out")
        os.makedirs(out, exist_ok=True)
        data_dir = _create_jsonl_files(tmp_path, n_files=4, samples_per_file=3)

        _save_checkpoint(os.path.join(out, "_checkpoint.json"), {
            "version": _CHECKPOINT_VERSION,
            "num_shards": 4,
            "checkpoint_interval": 2,
            "completed_files": [],
            "shard_tokens": [0, 0, 0, 0],
            "total_samples": 0,
            "total_tokens": 0,
            "batch_count": 0,
        })

        from titaninfra.data.converter import DataConvertConfig, DataConverter

        c = DataConverter(DataConvertConfig(
            source=data_dir, output=out,
            tokenizer_path=QWEN3_TOK, num_shards=2,
            checkpoint_interval=2, resume=True,
        ))
        with pytest.raises(ValueError, match="num_shards"):
            c.convert()

    def test_no_checkpoint_mode_unchanged(self, tmp_path):
        """Verify no checkpoint mode unchanged."""
        import os

        pytest.importorskip("streaming")
        from titaninfra.data.converter import DataConvertConfig, DataConverter

        data_dir = _create_jsonl_files(tmp_path, n_files=4, samples_per_file=5)
        out = str(tmp_path / "out")
        c = DataConverter(DataConvertConfig(
            source=data_dir, output=out,
            tokenizer_path=QWEN3_TOK, num_shards=2, overwrite=True,
        ))
        r = c.convert()

        assert r["samples"] == 20
        assert not list(Path(out).glob("_batch_*"))
        assert list(Path(out).glob("shard_*"))
        assert not os.path.exists(os.path.join(out, "_checkpoint.json"))

    def test_checkpoint_interval_larger_than_files(self, tmp_path):
        """Verify checkpoint interval larger than files."""
        pytest.importorskip("streaming")
        from titaninfra.data.converter import DataConvertConfig, DataConverter

        data_dir = _create_jsonl_files(tmp_path, n_files=3, samples_per_file=5)
        out = str(tmp_path / "out")
        c = DataConverter(DataConvertConfig(
            source=data_dir, output=out,
            tokenizer_path=QWEN3_TOK, num_shards=2, overwrite=True,
            checkpoint_interval=100, resume=True,
        ))
        r = c.convert()
        assert r["samples"] == 15
        assert r.get("batches") == 1

    def test_stream_stats_written_in_checkpoint_mode(self, tmp_path):
        """Verify stream stats written in checkpoint mode."""
        import json
        import os

        pytest.importorskip("streaming")
        from titaninfra.data.converter import DataConvertConfig, DataConverter

        data_dir = _create_jsonl_files(tmp_path, n_files=4, samples_per_file=5)
        out = str(tmp_path / "out")
        c = DataConverter(DataConvertConfig(
            source=data_dir, output=out,
            tokenizer_path=QWEN3_TOK, num_shards=2, overwrite=True,
            checkpoint_interval=2, resume=True,
        ))
        c.convert()

        stats_path = os.path.join(out, "stream_stats.json")
        assert os.path.exists(stats_path)
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["num_samples"] == 20
        tok_entry = stats["tokenizer_stats"].get(stats["default_tokenizer"], {})
        assert tok_entry.get("total_tokens", 0) > 0



class TestParseMathDialogue:
    """Tests for test parse math dialogue."""

    def test_basic_two_speaker(self):
        """Verify basic two speaker."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        record = {"content": "Student: What is 2+2?\n\nTeacher: It equals 4."}
        result = parse_math_dialogue(record)

        assert "messages" in result
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["blocks"] == [{"type": "text", "text": "What is 2+2?"}]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["blocks"] == [{"type": "text", "text": "It equals 4."}]
        for msg in msgs:
            assert "blocks" in msg
            assert "content" not in msg

    def test_multi_word_speaker(self):
        """Verify multi word speaker."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        record = {"content": "Professor A: Explain integrals.\n\nStudent B: Like summing slices?"}
        result = parse_math_dialogue(record)

        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert "Explain integrals" in msgs[0]["blocks"][0]["text"]

    def test_three_speakers_alternating(self):
        """Verify three speakers alternating."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        record = {"content": (
            "Alice: Question one.\n\n"
            "Bob: Answer one.\n\n"
            "Charlie: Follow-up question."
        )}
        result = parse_math_dialogue(record)

        msgs = result["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"      # Alice → user
        assert msgs[1]["role"] == "assistant"  # Bob → assistant
        assert msgs[2]["role"] == "user"       # Charlie → user (3rd unique → user)

    def test_consecutive_same_role_merge(self):
        """Verify consecutive same role merge."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        record = {"content": (
            "Student: Part 1.\n\n"
            "Student: Part 2.\n\n"
            "Teacher: Combined answer."
        )}
        result = parse_math_dialogue(record)

        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert "Part 1" in msgs[0]["blocks"][0]["text"]
        assert "Part 2" in msgs[0]["blocks"][0]["text"]

    def test_single_turn_skip(self):
        """Verify single turn skip."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        record = {"content": "Student: Just one turn."}
        result = parse_math_dialogue(record)
        assert result == {}

    def test_empty_content_skip(self):
        """Verify empty content skip."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        assert parse_math_dialogue({"content": ""}) == {}
        assert parse_math_dialogue({}) == {}

    def test_markdown_bold_stripped(self):
        """Verify markdown bold stripped."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        record = {"content": "**Student**: What is **bold**?\n\n**Teacher**: It is emphasis."}
        result = parse_math_dialogue(record)

        msgs = result["messages"]
        assert len(msgs) == 2
        assert "**" not in msgs[0]["blocks"][0]["text"]
        assert "bold" in msgs[0]["blocks"][0]["text"]

    def test_newline_not_matching_speaker_boundary(self):
        """Verify newline not matching speaker boundary."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        record = {"content": "Student: Line one.\nStill student text.\n\nTeacher: Reply."}
        result = parse_math_dialogue(record)

        msgs = result["messages"]
        assert len(msgs) == 2
        assert "Line one" in msgs[0]["blocks"][0]["text"]
        assert "Still student text" in msgs[0]["blocks"][0]["text"]

    def test_output_is_blocks_format(self):
        """Verify output is blocks format."""
        from scripts.ultradata.convert_math import parse_math_dialogue

        record = {"content": "User: Hello\n\nAssistant: Hi there"}
        result = parse_math_dialogue(record)

        for msg in result["messages"]:
            assert "blocks" in msg, "should use blocks format"
            assert "content" not in msg, "should NOT use content format"
            assert isinstance(msg["blocks"], list)
            for block in msg["blocks"]:
                assert block["type"] == "text"
                assert "text" in block

    def test_picklable(self):
        """Verify picklable."""
        import pickle

        from scripts.ultradata.convert_math import parse_math_dialogue

        roundtrip = pickle.loads(pickle.dumps(parse_math_dialogue))
        record = {"content": "Student: Q?\n\nTeacher: A."}
        assert roundtrip(record) == parse_math_dialogue(record)



class TestMessageConverterFn:
    """Tests for test message converter fn."""

    def test_converter_fn_transforms_records(self, tmp_path):
        """Verify converter fn transforms records."""
        import multiprocessing as mp
        import time

        from titaninfra.data.converter import _generic_reader_worker

        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            '{"content": "Student: Hi\\n\\nTeacher: Hello"}\n'
            '{"content": "Alice: Q\\n\\nBob: A"}',
            encoding="utf-8",
        )

        from scripts.ultradata.convert_math import parse_math_dialogue

        q: mp.Queue = mp.Queue()
        _generic_reader_worker(
            [str(data_file)], q, batch_size=100,
            message_converter_fn=parse_math_dialogue,
        )

        time.sleep(0.1)
        batches = []
        while not q.empty():
            batches.append(q.get())

        records = [r for b in batches for r in b]
        assert len(records) == 2
        for r in records:
            assert "messages" in r
            assert len(r["messages"]) == 2
            assert r["messages"][0]["role"] == "user"
            assert "blocks" in r["messages"][0]

    def test_converter_fn_skips_empty_result(self, tmp_path):
        """Verify converter fn skips empty result."""
        import multiprocessing as mp
        import time

        from titaninfra.data.converter import _generic_reader_worker

        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            '{"content": "Student: Only one turn."}\n'  # < 2 turns → skip
            '{"content": "Student: Q\\n\\nTeacher: A"}',
            encoding="utf-8",
        )

        from scripts.ultradata.convert_math import parse_math_dialogue

        q: mp.Queue = mp.Queue()
        _generic_reader_worker(
            [str(data_file)], q, batch_size=100,
            message_converter_fn=parse_math_dialogue,
        )

        time.sleep(0.1)
        batches = []
        while not q.empty():
            batches.append(q.get())

        records = [r for b in batches for r in b]
        assert len(records) == 1
        assert "messages" in records[0]

    def test_converter_fn_none_passthrough(self, tmp_path):
        """Verify converter fn none passthrough."""
        import multiprocessing as mp
        import time

        from titaninfra.data.converter import _generic_reader_worker

        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            '{"text": "hello world"}',
            encoding="utf-8",
        )

        q: mp.Queue = mp.Queue()
        _generic_reader_worker(
            [str(data_file)], q, batch_size=100,
            message_converter_fn=None,
        )

        time.sleep(0.1)
        batches = []
        while not q.empty():
            batches.append(q.get())

        records = [r for b in batches for r in b]
        assert len(records) == 1
        assert records[0] == {"text": "hello world"}
