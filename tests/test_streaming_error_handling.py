"""Unit tests for streaming error handling."""

import importlib.util
from typing import Any

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TORCHTITAN = importlib.util.find_spec("torchtitan") is not None

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch 未安装")
requires_torchtitan = pytest.mark.skipif(
    not HAS_TORCHTITAN, reason="torchtitan 未安装"
)






class MockTokenizer:
    """Mock TorchTitan BaseTokenizer"""
    def encode(self, text: str, **kwargs) -> list[int]:
        """Encode."""
        return [ord(c) for c in text]


class MockStreamingDataset:
    """Mock MosaicML StreamingDataset"""
    def __init__(self, samples: list[Any]):
        self.samples = samples
        self.num_samples = len(samples)

    def __iter__(self):
        return iter(self.samples)

    def state_dict(self, **kwargs):
        return {"mock": "state"}

    def load_state_dict(self, state):
        pass


class ShardCorruptedStreamingDataset:
    """Tests for shard corrupted streaming dataset."""
    def __init__(self, samples: list[Any], corrupt_indices: set[int]):
        self.samples = samples
        self.num_samples = len(samples)
        self._corrupt_indices = corrupt_indices

    def __iter__(self):
        return self._Iterator(self.samples, self._corrupt_indices)

    def state_dict(self, **kwargs):
        return {"mock": "state"}

    def load_state_dict(self, state):
        pass

    class _Iterator:
        def __init__(self, samples, corrupt_indices):
            self._samples = samples
            self._corrupt_indices = corrupt_indices
            self._idx = 0

        def __next__(self):
            if self._idx >= len(self._samples):
                raise StopIteration
            idx = self._idx
            self._idx += 1
            if idx in self._corrupt_indices:
                raise ValueError("not enough values to unpack (expected 2, got 0)")
            return self._samples[idx]


class MultiErrorStreamingDataset:
    """Tests for multi error streaming dataset."""
    def __init__(self, samples: list[Any], error_map: dict[int, Exception]):
        """Init."""
        self.samples = samples
        self.num_samples = len(samples)
        self._error_map = error_map

    def __iter__(self):
        return self._Iterator(self.samples, self._error_map)

    def state_dict(self, **kwargs):
        return {"mock": "state"}

    def load_state_dict(self, state):
        pass

    class _Iterator:
        def __init__(self, samples, error_map):
            self._samples = samples
            self._error_map = error_map
            self._idx = 0

        def __next__(self):
            if self._idx >= len(self._samples):
                raise StopIteration
            idx = self._idx
            self._idx += 1
            if idx in self._error_map:
                raise self._error_map[idx]
            return self._samples[idx]


class ErrorTokenizer:
    """Tests for error tokenizer."""
    def __init__(self, fail_on: set[str] | None = None):
        """Init."""
        self._fail_on = fail_on

    def encode(self, text: str, **kwargs) -> list[int]:
        if self._fail_on is None or any(s in text for s in self._fail_on):
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "模拟编码失败")
        return [ord(c) for c in text]


class EmptyTokenizer:
    """Tests for empty tokenizer."""
    def __init__(self, empty_on: set[str] | None = None):
        """Init."""
        self._empty_on = empty_on

    def encode(self, text: str, **kwargs) -> list[int]:
        if self._empty_on is None or any(s in text for s in self._empty_on):
            return []
        return [ord(c) for c in text]






@requires_torch
@requires_torchtitan
def test_sample_error_skipping():
    """Verify sample error skipping."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    test_samples = [
        {"text": "正常样本 1"},
        None,
        {},
        {"text": ""},
        {"text": "正常样本 2"},
        {"text": "正常样本 3"},
        {"content": "缺少 text 字段"},
        {"text": "正常样本 4"},
    ]

    mock_ds = MockStreamingDataset(test_samples)
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    assert valid_count > 0, "应至少处理了一些样本"

    error_stats = dataset.get_error_stats()
    assert error_stats.get('null_sample', 0) == 1, "应检测到 1 个 null_sample"
    assert error_stats.get('empty_dict', 0) == 1, "应检测到 1 个 empty_dict"
    assert error_stats.get('empty_text', 0) >= 1, "应至少检测到 1 个 empty_text"

    corrupted = dataset.get_corrupted_samples()
    assert len(corrupted) >= 3, "应至少记录 3 个损坏样本"


@requires_torch
@requires_torchtitan
def test_consecutive_error_protection():
    """Verify consecutive error protection."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    bad_samples = [
        None,
        {},
        {"text": ""},
        {"text": "正常"},
    ]

    mock_ds = MockStreamingDataset(bad_samples)
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=2,
        infinite=False,
    )

    with pytest.raises(RuntimeError) as exc_info:
        for _ in dataset:
            pass

    assert "数据损坏严重" in str(exc_info.value), "应包含数据损坏提示"
    assert "连续 2" in str(exc_info.value) or "连续 3" in str(exc_info.value), \
        "应包含连续错误次数"


@requires_torch
@requires_torchtitan
def test_error_handling_disabled():
    """Verify error handling disabled."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    mock_ds = MockStreamingDataset([None, {"text": "正常"}])
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=False,
        max_consecutive_errors=10,
        infinite=False,
    )

    with pytest.raises(ValueError):
        for _ in dataset:
            pass


@requires_torch
@requires_torchtitan
def test_error_stats_reset():
    """Verify error stats reset."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    mock_ds = MockStreamingDataset([None, {"text": "正常"}])
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    for _ in dataset:
        pass

    error_stats_before = dataset.get_error_stats()
    assert len(error_stats_before) > 0, "应有错误记录"

    dataset.reset_error_stats()

    error_stats_after = dataset.get_error_stats()
    assert len(error_stats_after) == 0, "重置后应为空"

    corrupted_after = dataset.get_corrupted_samples()
    assert len(corrupted_after) == 0, "损坏样本列表也应重置"


@requires_torch
@requires_torchtitan
def test_config_parameter_passing():
    """Verify config parameter passing."""
    from titaninfra.config.data import StreamConfig
    from titaninfra.data.streaming_dataset import StreamingTextDataLoader

    config = StreamingTextDataLoader.Config(
        streams=[StreamConfig(tokenizer_name="qwen3", local="/tmp/test")],
        skip_corrupted_samples=False,
        max_consecutive_errors=5,
        repeat=False,
    )

    assert not config.skip_corrupted_samples, "skip_corrupted_samples 应为 False"
    assert config.max_consecutive_errors == 5, "max_consecutive_errors 应为 5"


@requires_torch
@requires_torchtitan
def test_shard_error_handling():
    """Verify shard error handling."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset


    mock_ds = MockStreamingDataset([
        {"text": "正常样本"},
        {"text": "另一个正常样本"},
    ])
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    error_stats = dataset.get_error_stats()
    total_errors = sum(error_stats.values())
    assert total_errors == 0, "不应有错误记录"






@pytest.mark.integration
@requires_torch
@requires_torchtitan
def test_real_streaming_with_error_handling():
    """Verify real streaming with error handling."""
    pytest.skip("需要准备真实 MDS 数据")






@requires_torch
@requires_torchtitan
def test_empty_dataset():
    """Verify empty dataset."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    mock_ds = MockStreamingDataset([])
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    assert valid_count == 0, "空数据集不应产生样本"


@requires_torch
@requires_torchtitan
def test_all_corrupted_samples():
    """Verify all corrupted samples."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    mock_ds = MockStreamingDataset([None, {}, {"text": ""}])
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=3,
        infinite=False,
    )

    with pytest.raises(RuntimeError) as exc_info:
        for _ in dataset:
            pass

    assert "数据损坏严重" in str(exc_info.value)


@requires_torch
@requires_torchtitan
def test_shard_corruption_skips_and_continues():
    """Verify shard corruption skips and continues."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "正常样本 A"},
        {"text": "（不会被读到，因为 __next__ 会先抛异常）"},
        {"text": "正常样本 B"},
        {"text": "正常样本 C"},
    ]
    mock_ds = ShardCorruptedStreamingDataset(samples, corrupt_indices={1})
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    assert valid_count > 0, "应成功处理部分样本"

    error_stats = dataset.get_error_stats()
    assert error_stats.get("shard_errors", 0) >= 1, "应记录至少 1 个 shard_errors"


@requires_torch
@requires_torchtitan
def test_shard_corruption_raises_when_skip_disabled():
    """Verify shard corruption raises when skip disabled."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [{"text": "正常"}, {"text": "损坏"}]
    mock_ds = ShardCorruptedStreamingDataset(samples, corrupt_indices={1})
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=False,
        max_consecutive_errors=10,
        infinite=False,
    )

    with pytest.raises(ValueError, match="not enough values to unpack"):
        for _ in dataset:
            pass






@requires_torch
@requires_torchtitan
def test_invalid_sample_types():
    """Verify invalid sample types."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    test_samples = [
        b"bytes data",
        12345,
        [1, 2, 3],
        {"text": "正常样本"},
    ]
    mock_ds = MockStreamingDataset(test_samples)
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    error_stats = dataset.get_error_stats()
    assert error_stats.get("invalid_type", 0) == 3, "应检测到 3 个 invalid_type 错误"


@requires_torch
@requires_torchtitan
def test_missing_text_key():
    """Verify missing text key."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    test_samples = [
        {"content": "字段名不对"},
        {"other": "也不对"},
        {"text": "正常"},
    ]
    mock_ds = MockStreamingDataset(test_samples)
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    for _ in dataset:
        pass

    error_stats = dataset.get_error_stats()
    assert error_stats.get("missing_key", 0) == 2, "应检测到 2 个 missing_key"

    corrupted = dataset.get_corrupted_samples()
    assert any("content" in str(c.get("error", "")) for c in corrupted), \
        "损坏样本详情应包含可用字段列表"


@requires_torch
@requires_torchtitan
def test_invalid_text_field_type():
    """Verify invalid text field type."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    test_samples = [
        {"text": 12345},
        {"text": [1, 2, 3]},
        {"text": {"nested": True}},
        {"text": "正常"},
    ]
    mock_ds = MockStreamingDataset(test_samples)
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    for _ in dataset:
        pass

    error_stats = dataset.get_error_stats()
    assert error_stats.get("invalid_text_type", 0) == 3, "应检测到 3 个 invalid_text_type"


@requires_torch
@requires_torchtitan
def test_tokenize_error_skipping():
    """Verify tokenize error skipping."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    test_samples = [
        {"text": "触发失败"},
        {"text": "正常样本 ABC"},
    ]
    mock_ds = MockStreamingDataset(test_samples)
    tokenizer = ErrorTokenizer(fail_on={"触发"})

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    error_stats = dataset.get_error_stats()
    assert error_stats.get("tokenize_error", 0) == 1, "应检测到 1 个 tokenize_error"


@requires_torch
@requires_torchtitan
def test_empty_tokens_skipping():
    """Verify empty tokens skipping."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    test_samples = [
        {"text": "空结果"},
        {"text": "正常样本 XYZ"},
    ]
    mock_ds = MockStreamingDataset(test_samples)
    tokenizer = EmptyTokenizer(empty_on={"空结果"})

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    error_stats = dataset.get_error_stats()
    assert error_stats.get("empty_tokens", 0) == 1, "应检测到 1 个 empty_tokens"


@requires_torch
@requires_torchtitan
def test_multiple_consecutive_shard_errors():
    """Verify multiple consecutive shard errors."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "正常 A"},
        {"text": "损坏 1"},
        {"text": "损坏 2"},
        {"text": "损坏 3"},
        {"text": "正常 B"},
    ]
    mock_ds = ShardCorruptedStreamingDataset(samples, corrupt_indices={1, 2, 3})
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=2,
        infinite=False,
    )

    with pytest.raises(RuntimeError, match="数据损坏严重"):
        for _ in dataset:
            pass


@requires_torch
@requires_torchtitan
def test_shard_errors_interleaved_with_success():
    """Verify shard errors interleaved with success."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "正常 A"},
        {"text": "损坏 1"},
        {"text": "正常 B"},
        {"text": "损坏 2"},
        {"text": "正常 C"},
    ]
    mock_ds = ShardCorruptedStreamingDataset(samples, corrupt_indices={1, 3})
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=2,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    assert valid_count > 0, "应成功处理正常样本"
    error_stats = dataset.get_error_stats()
    assert error_stats.get("shard_errors", 0) == 2, "应记录 2 个 shard_errors"


@requires_torch
@requires_torchtitan
def test_os_error_as_shard_error():
    """Verify os error as shard error."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "This is a normal sample A with enough tokens for packing"},
        {"text": "I/O 错误"},
        {"text": "文件缺失"},
        {"text": "This is a normal sample B with enough tokens for packing"},
    ]
    error_map = {
        1: OSError("I/O operation on closed file"),
        2: FileNotFoundError("/data/shard_003.mds"),
    }
    mock_ds = MultiErrorStreamingDataset(samples, error_map)
    tokenizer = MockTokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_ds,
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        skip_corrupted_samples=True,
        max_consecutive_errors=10,
        infinite=False,
    )

    valid_count = 0
    for _ in dataset:
        valid_count += 1

    assert valid_count > 0, "应成功处理正常样本"
    error_stats = dataset.get_error_stats()
    assert error_stats.get("shard_errors", 0) == 2, "应记录 2 个 shard_errors"
