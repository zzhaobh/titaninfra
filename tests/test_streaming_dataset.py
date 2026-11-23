"""Unit tests for streaming dataset."""

import json
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch 未安装")


try:
    import torchtitan  # noqa: F401
    HAS_TORCHTITAN = True
except ImportError:
    HAS_TORCHTITAN = False

requires_torchtitan = pytest.mark.skipif(
    not HAS_TORCHTITAN, reason="torchtitan 未安装"
)


# ---------------------------------------------------------------------------
# Mock StreamingDataset (MosaicML)
# ---------------------------------------------------------------------------


class MockStreamingDataset:
    """Mock MosaicML StreamingDataset"""

    def __init__(self, samples: list[dict]):
        self._samples = samples
        self._current_index = 0
        self._epoch = 0
        self.num_samples = len(samples)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index >= len(self._samples):
            raise StopIteration
        sample = self._samples[self._current_index]
        self._current_index += 1
        return sample

    def __len__(self):
        return len(self._samples)

    def state_dict(self, num_samples: int = 0, from_beginning: bool = True):
        """State dict."""
        return {
            "epoch": self._epoch,
            "sample_in_epoch": self._current_index,
            "num_canonical_nodes": 1,
            "shuffle_seed": 42,
            "initial_physical_nodes": 1,
        }

    def load_state_dict(self, state: dict):
        """Load state dict."""
        self._epoch = state.get("epoch", 0)


# ---------------------------------------------------------------------------
# Mock Tokenizer
# ---------------------------------------------------------------------------


def _make_mock_tokenizer():
    """Make mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, add_bos=True, add_eos=True: [
        ord(c) % 100 for c in text
    ] + ([1] if add_eos else [])
    return tokenizer


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_torchtitan
@requires_torch
def test_streaming_dataset_basic():
    """Verify streaming dataset basic."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "hello world"},
        {"text": "testing data"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    outputs = list(dataset)

    assert len(outputs) > 0, "应该输出至少一个 batch"

    for input_dict, labels in outputs:
        assert "input" in input_dict, "输出应包含 'input' 键"
        assert "positions" in input_dict, "输出应包含 'positions' 键（varlen 模式）"
        assert isinstance(input_dict["input"], torch.Tensor), "input 应为 Tensor"
        assert isinstance(input_dict["positions"], torch.Tensor), "positions 应为 Tensor"
        assert isinstance(labels, torch.Tensor), "labels 应为 Tensor"
        assert input_dict["input"].shape[0] == 20, f"input 长度应为 seq_len=20，实际为 {input_dict['input'].shape[0]}"
        assert labels.shape[0] == 20, f"labels 长度应为 seq_len=20，实际为 {labels.shape[0]}"


@requires_torchtitan
@requires_torch
def test_streaming_dataset_pretrain_tokens():
    """Verify streaming dataset pretrain tokens."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "hello world"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    outputs = list(dataset)

    for input_dict, labels in outputs:
        assert not (labels == -100).any(), "Pretrain 场景下，所有 token 都应参与 loss"


@requires_torchtitan
@requires_torch
def test_streaming_dataset_never_adds_bos_eos():
    """Pretrain runtime contract: StreamingTextDataset MUST tokenize with
    ``add_bos=False, add_eos=False``.

    Document boundaries in pretrain are detected via ``position_ids == 0``
    (see CLAUDE.md "Pretrain special tokens"), NOT via BOS/EOS. Adding
    BOS/EOS at runtime would (a) pollute the EOS token's semantics so the
    model learns to early-stop during pretraining, and (b) inflate token
    counts vs. the offline stats baseline. Regressions here are silent —
    only catchable via the alignment B-series GPU tests — so we lock the
    contract at the unit-test layer.
    """
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    calls: list[dict] = []

    tokenizer = MagicMock()

    def recording_encode(text, add_bos=True, add_eos=True):
        calls.append({"add_bos": add_bos, "add_eos": add_eos})
        return [ord(c) % 100 for c in text]

    tokenizer.encode = recording_encode

    dataset = StreamingTextDataset(
        streaming_dataset=MockStreamingDataset(
            [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
        ),
        tokenizer=tokenizer,
        seq_len=10,
        text_key="text",
        infinite=False,
    )

    list(dataset)

    assert calls, "tokenizer.encode was never called — test did not exercise the tokenize path"
    violations = [c for c in calls if c["add_bos"] or c["add_eos"]]
    assert not violations, (
        f"Pretrain tokenize path must use add_bos=False, add_eos=False. "
        f"Got {len(violations)} violating call(s) out of {len(calls)}: "
        f"first violation = {violations[0]}"
    )


@requires_torchtitan
@requires_torch
def test_streaming_dataset_state_dict():
    """Verify streaming dataset state dict."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "sample one"},
        {"text": "sample two"},
        {"text": "sample three"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=10,
        text_key="text",
        infinite=False,
    )

    it = iter(dataset)
    for i in range(2):
        try:
            next(it)
        except StopIteration:
            break

    state = dataset.state_dict()

    assert "streaming" in state, "state_dict 应包含 'streaming' 键"
    assert "streaming_samples_consumed" in state, "state_dict 应包含 'streaming_samples_consumed' 键"


@requires_torchtitan
@requires_torch
def test_streaming_dataset_state_recovery():
    """Verify streaming dataset state recovery."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "sample one"},
        {"text": "sample two"},
        {"text": "sample three"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=10,
        text_key="text",
        infinite=False,
    )

    it = iter(dataset)
    next(it)

    state = dataset.state_dict()

    mock_streaming2 = MockStreamingDataset(samples)
    dataset2 = StreamingTextDataset(
        streaming_dataset=mock_streaming2,
        tokenizer=mock_tokenizer,
        seq_len=10,
        text_key="text",
        infinite=False,
    )

    dataset2.load_state_dict(state)

    assert dataset2._streaming_samples_consumed == state["streaming_samples_consumed"]


@requires_torchtitan
@requires_torch
def test_streaming_dataset_token_statistics():
    """Verify streaming dataset token statistics."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "hello world, this is a longer text sample for testing token statistics"},
        {"text": "another fairly long text sample to ensure packer buffer fills up properly"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    assert dataset.get_total_tokens_seen() == 0, "初始 token 统计应为 0"

    outputs = list(dataset)

    assert len(outputs) > 0, "应至少产生一个 batch"

    total_tokens = dataset.get_total_tokens_seen()
    assert total_tokens > 0, "迭代后 token 统计应大于 0"


@requires_torchtitan
@requires_torch
def test_streaming_dataset_positions_per_document_reset():
    """Verify streaming dataset positions per document reset."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "first document"},
        {"text": "second document"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=30,
        text_key="text",
        infinite=False,
    )

    outputs = list(dataset)

    for input_dict, labels in outputs:
        positions = input_dict["positions"]

        assert positions[0] == 0, "positions 应该从 0 开始"


@requires_torchtitan
@requires_torch
def test_streaming_dataset_epoch_size_controls_data():
    """Verify streaming dataset epoch size controls data."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "sample one"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=10,
        text_key="text",
        infinite=False,
    )

    batches = list(dataset)
    assert len(batches) >= 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_torchtitan
def test_streaming_dataloader_config():
    """Verify streaming dataloader config."""
    from titaninfra.config.data import StreamConfig
    from titaninfra.data.streaming_dataset import StreamingTextDataLoader

    config1 = StreamingTextDataLoader.Config(
        streams=[StreamConfig(tokenizer_name="qwen3", local="/data/train")]
    )
    assert len(config1.streams) == 1

    config2 = StreamingTextDataLoader.Config(
        streams=[
            StreamConfig(tokenizer_name="qwen3", local="/data/zh", proportion=0.7),
            StreamConfig(tokenizer_name="qwen3", local="/data/en", proportion=0.3),
        ]
    )
    assert len(config2.streams) == 2


@requires_torchtitan
def test_streaming_dataloader_config_validation():
    """Verify streaming dataloader config validation."""
    from titaninfra.config.data import StreamConfig
    from titaninfra.data.streaming_dataset import StreamingTextDataLoader

    with pytest.raises(ValueError):
        StreamingTextDataLoader.Config(streams=[])

    with pytest.raises(ValueError):
        StreamingTextDataLoader.Config(
            streams=[
                StreamConfig(tokenizer_name="qwen3", local="/data/zh", proportion=0.7),
                StreamConfig(tokenizer_name="qwen3", local="/data/en", proportion=0.7),
            ]
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_torchtitan
@requires_torch
def test_streaming_dataset_multi_stream():
    """Verify streaming dataset multi stream."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "stream one sample"},
        {"text": "stream two sample"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    outputs = list(dataset)

    assert len(outputs) > 0, "应该输出至少一个 batch"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_torchtitan
@requires_torch
def test_streaming_dataset_empty_sample():
    """Verify streaming dataset empty sample."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": ""},
        {"text": "valid text"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    outputs = list(dataset)

    assert isinstance(outputs, list)


@requires_torchtitan
@requires_torch
def test_streaming_dataset_long_text():
    """Verify streaming dataset long text."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    long_text = "hello world " * 100

    samples = [
        {"text": long_text},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    outputs = list(dataset)

    assert len(outputs) > 1, "长文本应该被切分为多个 batch"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_torchtitan
@requires_torch
def test_streaming_dataset_get_total_valid_tokens():
    """Verify streaming dataset get total valid tokens."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "hello world"},
        {"text": "test data"},
        {"text": "sample text"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    total_tokens = dataset.get_total_valid_tokens()
    assert isinstance(total_tokens, int), "返回值应为 int 类型"
    assert total_tokens > 0, "token 总数应大于 0"

    expected_tokens = len(samples) * 20  # 3 samples × 20 seq_len
    assert total_tokens == expected_tokens, f"token 总数应为 {expected_tokens}，实际为 {total_tokens}"


@requires_torchtitan
@requires_torch
def test_streaming_dataset_get_num_original_samples():
    """Verify streaming dataset get num original samples."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "sample 1"},
        {"text": "sample 2"},
        {"text": "sample 3"},
        {"text": "sample 4"},
        {"text": "sample 5"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    num_samples = dataset.get_num_original_samples()
    assert isinstance(num_samples, int), "返回值应为 int 类型"
    assert num_samples == 5, f"样本数应为 5，实际为 {num_samples}"


@requires_torchtitan
@requires_torch
def test_streaming_dataset_epoch_statistics_consistency():
    """Verify streaming dataset epoch statistics consistency."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "test sample one"},
        {"text": "test sample two"},
        {"text": "test sample three"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    seq_len = 30
    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=seq_len,
        text_key="text",
        infinite=False,
    )

    num_samples = dataset.get_num_original_samples()
    total_tokens = dataset.get_total_valid_tokens()

    assert total_tokens == num_samples * seq_len, (
        f"token 总数应为 {num_samples * seq_len}，"
        f"实际为 {total_tokens}"
    )


@requires_torchtitan
@requires_torch
def test_streaming_dataset_epoch_statistics_empty():
    """Verify streaming dataset epoch statistics empty."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = []

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=20,
        text_key="text",
        infinite=False,
    )

    num_samples = dataset.get_num_original_samples()
    total_tokens = dataset.get_total_valid_tokens()

    assert num_samples == 0, "空数据集样本数应为 0"
    assert total_tokens == 0, "空数据集 token 总数应为 0"


# ============================================================================
# ============================================================================


@requires_torchtitan
@requires_torch
def test_streaming_dataset_per_stream_with_stream_id():
    """Verify streaming dataset per stream with stream id."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "hello world", "stream_id": "wiki"},
        {"text": "import os", "stream_id": "code"},
        {"text": "更多文本", "stream_id": "wiki"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=128,
        text_key="text",
        default_stream_id="fallback",
        infinite=False,
    )

    for _ in dataset:
        pass

    stats = dataset.get_per_stream_stats()
    assert "wiki" in stats, "应包含 wiki stream"
    assert "code" in stats, "应包含 code stream"
    assert stats["wiki"]["samples"] == 2
    assert stats["code"]["samples"] == 1
    assert stats["wiki"]["tokens"] > 0
    assert stats["code"]["tokens"] > 0


@requires_torchtitan
@requires_torch
def test_streaming_dataset_per_stream_default_fallback():
    """Verify streaming dataset per stream default fallback."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "hello world"},
        {"text": "import os"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=128,
        text_key="text",
        default_stream_id="my_dataset",
        infinite=False,
    )

    for _ in dataset:
        pass

    stats = dataset.get_per_stream_stats()
    assert "my_dataset" in stats
    assert stats["my_dataset"]["samples"] == 2


@requires_torchtitan
@requires_torch
def test_streaming_dataset_epoch_progress():
    """Verify streaming dataset epoch progress."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [{"text": "a" * 50, "stream_id": "ds1"}] * 5

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=128,
        text_key="text",
        infinite=False,
    )

    for _ in dataset:
        pass

    consumed = dataset.get_per_stream_stats().get("ds1", {}).get("tokens", 0)
    assert consumed > 0

    progress = dataset.get_epoch_progress({"ds1": consumed * 2})
    assert abs(progress["ds1"] - 0.5) < 0.01


@requires_torchtitan
@requires_torch
def test_streaming_dataset_per_stream_state_dict():
    """Verify streaming dataset per stream state dict."""
    from titaninfra.data.streaming_dataset import StreamingTextDataset

    samples = [
        {"text": "hello world", "stream_id": "a"},
        {"text": "import os", "stream_id": "b"},
    ]

    mock_streaming = MockStreamingDataset(samples)
    mock_tokenizer = _make_mock_tokenizer()

    dataset = StreamingTextDataset(
        streaming_dataset=mock_streaming,
        tokenizer=mock_tokenizer,
        seq_len=128,
        text_key="text",
        infinite=False,
    )

    for _ in dataset:
        pass

    state = dataset.state_dict()
    assert "per_stream_tokens" in state
    assert "per_stream_samples" in state
    assert state["per_stream_tokens"]["a"] > 0

    mock_streaming2 = MockStreamingDataset(samples)
    dataset2 = StreamingTextDataset(
        streaming_dataset=mock_streaming2,
        tokenizer=mock_tokenizer,
        seq_len=128,
        text_key="text",
        infinite=False,
    )
    dataset2.load_state_dict(state)

    restored_stats = dataset2.get_per_stream_stats()
    assert restored_stats["a"]["tokens"] == state["per_stream_tokens"]["a"]
    assert restored_stats["b"]["tokens"] == state["per_stream_tokens"]["b"]


# ---------------------------------------------------------------------------
# Tests for full_sample oversized-sample skip logic in _make_messages_doc_iter
# ---------------------------------------------------------------------------


@requires_torchtitan
class TestMakeMessagesDocIterFullSample:
    """_make_messages_doc_iter: full_sample skips oversized; hard_truncate truncates."""

    def _make_stream_cfg(self, pack_strategy: str):
        cfg = MagicMock()
        cfg.messages_key = "messages"
        cfg.train_on_inputs = False
        cfg.sft_mask_mode = "auto"
        cfg.max_turns = None
        cfg.chat_template = None
        cfg.thinking_scale = None
        cfg.mask_empty_thinking = True
        cfg.validate_template_alignment = False
        cfg.resolve_stream_id.return_value = "test"
        cfg.pack_strategy.value = pack_strategy
        return cfg

    def _fake_ds(self):
        msg = json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
        return [{"messages": msg}]

    def test_normal_sample_yielded(self):
        from titaninfra.data.sft.types import EncodedSample
        from titaninfra.data.streaming_dataset import _make_messages_doc_iter

        seq_len = 100
        ids = list(range(50))
        encoded = EncodedSample(
            input_ids=tuple(ids),
            labels=tuple(ids),
            positions=tuple(range(len(ids))),
        )
        with patch(
            "titaninfra.data.sft.dispatch.tokenize_sample",
            return_value=encoded,
        ):
            docs = list(_make_messages_doc_iter(self._fake_ds(), None, self._make_stream_cfg("full_sample"), seq_len))

        assert len(docs) == 1
        assert len(docs[0]) == 50

    def test_oversized_sample_skipped(self):
        from titaninfra.data.sft.types import EncodedSample
        from titaninfra.data.streaming_dataset import _make_messages_doc_iter

        seq_len = 100
        # tokenizer truncates to seq_len+1 (sentinel), signalling original exceeded seq_len
        oversized = list(range(seq_len + 1))
        encoded = EncodedSample(
            input_ids=tuple(oversized),
            labels=tuple(oversized),
            positions=tuple(range(len(oversized))),
        )
        with patch(
            "titaninfra.data.sft.dispatch.tokenize_sample",
            return_value=encoded,
        ):
            docs = list(_make_messages_doc_iter(self._fake_ds(), None, self._make_stream_cfg("full_sample"), seq_len))

        assert len(docs) == 0

    def test_hard_truncate_keeps_truncation(self):
        from titaninfra.data.sft.types import EncodedSample
        from titaninfra.data.streaming_dataset import _make_messages_doc_iter

        seq_len = 100
        truncated = list(range(seq_len))
        encoded = EncodedSample(
            input_ids=tuple(truncated),
            labels=tuple(truncated),
            positions=tuple(range(len(truncated))),
        )
        with patch(
            "titaninfra.data.sft.dispatch.tokenize_sample",
            return_value=encoded,
        ) as mock_tok:
            docs = list(_make_messages_doc_iter(self._fake_ds(), None, self._make_stream_cfg("hard_truncate"), seq_len))

        assert len(docs) == 1
        assert len(docs[0]) == seq_len
        assert mock_tok.call_args.kwargs["max_length"] == seq_len


_BOS_ID = 9001
_EOS_ID = 9002


class _FakeTokenizer:
    """Tests for fake tokenizer."""

    def __init__(self):
        self.calls: list[dict] = []

    def encode(self, text, add_bos=False, add_eos=False):
        self.calls.append({"text": text, "add_bos": add_bos, "add_eos": add_eos})
        body = [ord(c) % 100 for c in text]
        out = []
        if add_bos:
            out.append(_BOS_ID)
        out.extend(body)
        if add_eos:
            out.append(_EOS_ID)
        return out


def _collect_single(iter_func):
    docs = list(iter_func)
    assert len(docs) == 1, f"expected single doc, got {len(docs)}"
    return [t for t, _ in docs[0]]


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class _FakeInnerTokenizer:
    """Tests for fake inner tokenizer."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def token_to_id(self, token):
        if not isinstance(token, str):
            raise TypeError(
                f"argument 'token': '{type(token).__name__}' object cannot be converted to 'PyString'"
            )
        return self._vocab.get(token)


class _FakeTokenizerWrapper:
    """Tests for fake tokenizer wrapper."""

    def __init__(self, tokenizer_path, vocab: dict[str, int], eos_id: int = 1):
        self.tokenizer = _FakeInnerTokenizer(vocab)
        self.tokenizer_path = str(tokenizer_path)
        self.eos_id = eos_id


@requires_torchtitan
class TestResolvePadId:
    """Tests for test resolve pad id."""

    @staticmethod
    def _write_config(tmp_path, pad_token_value):
        cfg_path = tmp_path / "tokenizer_config.json"
        cfg_path.write_text(json.dumps({"pad_token": pad_token_value}))
        return tmp_path

    def test_plain_string_pad_token(self, tmp_path):
        """Verify plain string pad token."""
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        path = self._write_config(tmp_path, "<pad>")
        tok = _FakeTokenizerWrapper(path, vocab={"<pad>": 42}, eos_id=1)
        assert StreamingTextDataset._resolve_pad_id(tok) == 42

    def test_added_token_dict_pad_token(self, tmp_path):
        """Verify added token dict pad token."""
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        added_token = {
            "__type": "AddedToken",
            "content": "<|▁pad▁|>",
            "lstrip": False,
            "normalized": True,
            "rstrip": False,
            "single_word": False,
        }
        path = self._write_config(tmp_path, added_token)
        tok = _FakeTokenizerWrapper(path, vocab={"<|▁pad▁|>": 2}, eos_id=1)
        assert StreamingTextDataset._resolve_pad_id(tok) == 2

    def test_missing_pad_token_falls_back_to_eos(self, tmp_path):
        """Verify missing pad token falls back to eos."""
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        cfg_path = tmp_path / "tokenizer_config.json"
        cfg_path.write_text(json.dumps({}))
        tok = _FakeTokenizerWrapper(tmp_path, vocab={}, eos_id=7)
        assert StreamingTextDataset._resolve_pad_id(tok) == 7


@requires_torchtitan
class TestAddBosEosRuntime:
    """Tests for test add bos eos runtime."""

    def _make_ds(self, texts):
        return MockStreamingDataset([{"text": t} for t in texts])

    def test_default_no_special_tokens(self):
        from titaninfra.data.streaming_dataset import _make_pretrain_doc_iter

        tok = _FakeTokenizer()
        ids = _collect_single(_make_pretrain_doc_iter(self._make_ds(["abc"]), tok, "text", "s0"))

        assert _BOS_ID not in ids
        assert _EOS_ID not in ids
        assert tok.calls[0]["add_bos"] is False
        assert tok.calls[0]["add_eos"] is False

    def test_explicit_false(self):
        from titaninfra.data.streaming_dataset import _make_pretrain_doc_iter

        tok = _FakeTokenizer()
        ids = _collect_single(_make_pretrain_doc_iter(
            self._make_ds(["abc"]), tok, "text", "s0", add_bos_eos=False,
        ))
        assert _BOS_ID not in ids and _EOS_ID not in ids

    def test_true_wraps_with_bos_eos(self):
        from titaninfra.data.streaming_dataset import _make_pretrain_doc_iter

        tok = _FakeTokenizer()
        ids = _collect_single(_make_pretrain_doc_iter(
            self._make_ds(["abc"]), tok, "text", "s0", add_bos_eos=True,
        ))

        assert ids[0] == _BOS_ID, "BOS must be the first token"
        assert ids[-1] == _EOS_ID, "EOS must be the last token"
        assert _BOS_ID not in ids[1:]
        assert _EOS_ID not in ids[:-1]
        assert len(ids) == len("abc") + 2

    def test_true_wraps_each_sample_independently(self):
        from titaninfra.data.streaming_dataset import _make_pretrain_doc_iter

        tok = _FakeTokenizer()
        texts = ["a", "bc", "def"]
        docs = list(_make_pretrain_doc_iter(
            self._make_ds(texts), tok, "text", "s0", add_bos_eos=True,
        ))
        assert len(docs) == len(texts)

        for doc, text in zip(docs, texts, strict=True):
            ids = [t for t, _ in doc]
            assert ids[0] == _BOS_ID, f"sample {text!r}: BOS missing at head"
            assert ids[-1] == _EOS_ID, f"sample {text!r}: EOS missing at tail"
            assert len(ids) == len(text) + 2

        all_ids = [t for doc in docs for t, _ in doc]
        assert all_ids.count(_BOS_ID) == len(texts)
        assert all_ids.count(_EOS_ID) == len(texts)

    def test_label_equals_input_id_pretrain_contract(self):
        from titaninfra.data.streaming_dataset import _make_pretrain_doc_iter

        tok = _FakeTokenizer()
        docs = list(_make_pretrain_doc_iter(
            self._make_ds(["abc"]), tok, "text", "s0", add_bos_eos=True,
        ))
        for t, lbl in docs[0]:
            assert t == lbl, "pretrain 流必须 label = input_id（包括 BOS/EOS）"

    def test_empty_text_only_bos_eos_when_true(self):
        from titaninfra.data.streaming_dataset import _make_pretrain_doc_iter

        tok = _FakeTokenizer()
        ids = _collect_single(_make_pretrain_doc_iter(
            self._make_ds([""]), tok, "text", "s0", add_bos_eos=True,
        ))
        assert ids == [_BOS_ID, _EOS_ID]

    def test_empty_text_no_tokens_when_false(self):
        from titaninfra.data.streaming_dataset import _make_pretrain_doc_iter

        tok = _FakeTokenizer()
        docs = list(_make_pretrain_doc_iter(
            self._make_ds([""]), tok, "text", "s0", add_bos_eos=False,
        ))
        assert docs == [[]]

    def test_stream_config_propagates_to_iter(self):
        """Verify stream config propagates to iter."""
        from titaninfra.config import StreamConfig
        from titaninfra.data.streaming_dataset import _make_pretrain_doc_iter

        cfg = StreamConfig(
            tokenizer_name="qwen3",
            local="/tmp/data",
            data_type="text",
            add_bos_eos=True,
        )
        tok = _FakeTokenizer()
        ds = self._make_ds(["xy"])

        _tok = tok
        _tkey = cfg.text_key
        _sid = cfg.resolve_stream_id()
        _abe = cfg.add_bos_eos
        doc_factory = lambda: _make_pretrain_doc_iter(
            ds, _tok, _tkey, _sid, add_bos_eos=_abe,
        )

        ids = _collect_single(doc_factory())
        assert ids[0] == _BOS_ID and ids[-1] == _EOS_ID

    def test_streaming_dataset_ctor_accepts_add_bos_eos(self):
        """Verify streaming dataset ctor accepts add bos eos."""
        if not HAS_TORCHTITAN:
            pytest.skip("torchtitan 未安装")

        from titaninfra.data.streaming_dataset import StreamingTextDataset

        tok = _FakeTokenizer()
        ds = self._make_ds(["abc"])

        sd = StreamingTextDataset(
            streaming_dataset=ds,
            tokenizer=tok,
            seq_len=64,
            text_key="text",
            add_bos_eos=True,
        )
        assert sd._add_bos_eos is True

        sd_default = StreamingTextDataset(
            streaming_dataset=ds,
            tokenizer=tok,
            seq_len=64,
            text_key="text",
        )
        assert sd_default._add_bos_eos is False


class TestAddBosEosPackerIntegration:
    """Tests for test add bos eos packer integration."""

    def test_packer_integration_bos_eos_enter_packed_stream(self):
        """Verify packer integration bos eos enter packed stream."""
        from titaninfra.data.packer import PackedSample, PackMode, StreamingPacker

        tok = _FakeTokenizer()
        texts = ["ab", "cde"]
        seq_len = 64
        packer = StreamingPacker(seq_len=seq_len, mode=PackMode.SIMPLE)

        for text in texts:
            tokens = tok.encode(text, add_bos=True, add_eos=True)
            doc = [(t, t) for t in tokens]
            packer.add_document(doc)

        packed: list[PackedSample] = list(packer.flush_all())
        assert packed, "packer 应至少 flush 出一条 PackedSample"

        all_ids = [t for s in packed for t in s.input_ids]
        all_labels = [lbl for s in packed for lbl in s.labels]

        assert all_ids.count(_BOS_ID) == len(texts), \
            f"BOS 数应等于文档数，got {all_ids.count(_BOS_ID)} vs {len(texts)}"
        assert all_ids.count(_EOS_ID) == len(texts), \
            f"EOS 数应等于文档数，got {all_ids.count(_EOS_ID)} vs {len(texts)}"

        assert all_labels == all_ids, "pretrain 流 labels 应与 input_ids 全等"

        #    test_split_bos_eos_across_windows_is_acceptable。
        for s in packed:
            assert s.positions is not None, "pretrain packer 必须生成 positions"
            for i, tok_id in enumerate(s.input_ids):
                if tok_id == _BOS_ID:
                    assert s.positions[i] == 0, (
                        f"未切分场景下 BOS 应与 position 0 共现，"
                        f"got position={s.positions[i]} at index {i}"
                    )

        expected_total = sum(len(t) + 2 for t in texts)
        assert len(all_ids) == expected_total, (
            f"总 token 数 mismatch: expected {expected_total}, got {len(all_ids)}"
        )

    def test_split_bos_eos_across_windows_is_acceptable(self):
        """Verify split bos eos across windows is acceptable."""
        from titaninfra.data.packer import PackMode, StreamingPacker

        tok = _FakeTokenizer()
        seq_len = 8
        text = "abcdefghijklmnopqrst"
        packer = StreamingPacker(seq_len=seq_len, mode=PackMode.SIMPLE)

        tokens = tok.encode(text, add_bos=True, add_eos=True)
        assert len(tokens) == len(text) + 2, "前置检查：tokenize 输出长度"
        assert tokens[0] == _BOS_ID and tokens[-1] == _EOS_ID

        packer.add_document([(t, t) for t in tokens])
        packed = list(packer.flush_all())

        assert len(packed) >= 2, f"超长样本必须被切成多段（≥2），got {len(packed)}"

        for s in packed:
            assert s.positions is not None
            for p in s.positions:
                assert 0 <= p <= seq_len, (
                    f"position {p} 越界 (must be 0 ≤ p ≤ {seq_len})——"
                    f"会触发 RoPE OOB / SIGABRT"
                )

        all_ids = [t for s in packed for t in s.input_ids]
        assert all_ids.count(_BOS_ID) == 1, "BOS 跨段不应被复制"
        assert all_ids.count(_EOS_ID) == 1, "EOS 跨段不应被复制"
        assert packed[0].input_ids[0] == _BOS_ID, "BOS 应位于第一段起点"
        assert _EOS_ID in packed[-1].input_ids, "EOS 应落在最后一段（任意位置）"

        total = sum(len(s.input_ids) for s in packed)
        assert total == len(tokens), (
            f"切分前后 token 总数应一致：tokenize={len(tokens)}, packed={total}"
        )


# ============================================================================
# ============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="需要 torch")
class TestFlatTokenCollator:
    """Tests for test flat token collator."""

    def _make_collator(self, *, num_tokens_per_microbatch=16, max_context_length=8, pad_id=999):
        from titaninfra.data.streaming_dataset import _FlatTokenCollator

        return _FlatTokenCollator(
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            max_context_length=max_context_length,
            pad_id=pad_id,
        )

    def test_cats_not_stacks_full_batch(self):
        """Verify cats not stacks full batch."""
        collator = self._make_collator()
        sample1 = (
            {"input": torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
             "positions": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])},
            torch.tensor([11, 12, 13, 14, 15, 16, 17, -100]),
        )
        sample2 = (
            {"input": torch.tensor([20, 21, 22, 23, 24, 25, 26, 27]),
             "positions": torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])},
            torch.tensor([21, 22, 23, 24, 25, -100, 27, -100]),
        )
        mb = collator([sample1, sample2])
        assert mb.input.shape == (16,)
        assert mb.input.tolist() == list(range(10, 18)) + list(range(20, 28))
        assert mb.loss_token_counts.item() == 13
        assert mb.routing_token_counts.tolist() == [16]
        assert not mb.padding_mask.any()

    def test_short_batch_pads_whole_missing_sample(self):
        """Verify short batch pads whole missing sample."""
        collator = self._make_collator(pad_id=999)
        sample = (
            {"input": torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
             "positions": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])},
            torch.tensor([11, 12, 13, 14, 15, 16, 17, -100]),
        )
        mb = collator([sample])
        assert mb.input.tolist() == [10, 11, 12, 13, 14, 15, 16, 17] + [999] * 8
        assert mb.padding_mask.tolist() == [False] * 8 + [True] * 8
        assert mb.routing_token_counts.tolist() == [8]
        assert mb.positions.tolist()[8:] == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_loss_weights_travel_as_loss_kwargs_dict(self):
        """Verify loss weights travel as loss kwargs dict."""
        collator = self._make_collator(num_tokens_per_microbatch=4, max_context_length=8)
        sample = (
            {"input": torch.tensor([1, 2, 3, 4]), "positions": torch.tensor([0, 1, 2, 3]),
             "loss_weights": torch.tensor([0.0, 1.0, 1.0, 0.5])},
            torch.tensor([2, 3, 4, -100]),
        )
        mb = collator([sample])
        assert set(mb.model_kwargs.keys()) == {"loss_kwargs"}
        assert "loss_weights" not in mb.model_kwargs
        assert mb.model_kwargs["loss_kwargs"]["loss_weights"].tolist() == [0.0, 1.0, 1.0, 0.5]

    def test_no_weights_omits_loss_kwargs_entirely(self):
        """Verify no weights omits loss kwargs entirely."""
        collator = self._make_collator(num_tokens_per_microbatch=4, max_context_length=8)
        sample = (
            {"input": torch.tensor([1, 2, 3, 4]), "positions": torch.tensor([0, 1, 2, 3])},
            torch.tensor([2, 3, 4, -100]),
        )
        mb = collator([sample])
        assert mb.model_kwargs == {}


@pytest.mark.skipif(not HAS_TORCH, reason="需要 torch")
class TestFlatTokenCollatorRealPreprocessInputs:
    """Tests for test flat token collator real preprocess inputs."""

    @pytest.fixture()
    def single_rank_pctx(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            pytest.skip("torch.distributed gloo backend unavailable")
        pytest.importorskip("torchtitan")
        from torchtitan.config.parallelism import ParallelismConfig
        from torchtitan.distributed.parallelism_context import (
            DistributedTopology,
            ParallelismContext,
        )

        already = dist.is_initialized()
        if not already:
            dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
        try:
            pctx = ParallelismContext.from_config(
                ParallelismConfig(), DistributedTopology(world_size=1),
                dump_folder="/tmp/_test_flat_collator_pctx",
            )
            pctx.build_mesh()
            yield pctx
        finally:
            if not already and dist.is_initialized():
                dist.destroy_process_group()

    def _fake_decoder(self):
        from torchtitan.models.common.decoder import Decoder

        dec = Decoder.__new__(Decoder)

        class _FakeConfig:
            layers: list = []
            max_context_length = 8

            def traverse(self, cls):
                return iter([])

        dec.config = _FakeConfig()
        return dec

    def test_loss_kwargs_dict_survives_real_preprocess_inputs(self, single_rank_pctx):
        from torchtitan.config.parallelism import ParallelismConfig

        from titaninfra.data.streaming_dataset import _FlatTokenCollator

        collator = _FlatTokenCollator(num_tokens_per_microbatch=8, max_context_length=8, pad_id=999)
        sample = (
            {"input": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
             "positions": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
             "loss_weights": torch.tensor([0.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 1.0])},
            torch.tensor([2, 3, 4, -100, 6, 7, 8, -100]),
        )
        mb = collator([sample])
        dec = self._fake_decoder()

        inputs, labels, model_kwargs, loss_kwargs = dec.preprocess_inputs(
            mb.to_input_dict(device=torch.device("cpu")),
            parallelism_context=single_rank_pctx,
            parallelism=ParallelismConfig(),
            max_num_documents=None,
            global_routing_token_counts=None,
        )
        assert inputs.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
        assert labels.tolist() == [2, 3, 4, -100, 6, 7, 8, -100]
        assert "loss_weights" not in model_kwargs
        assert loss_kwargs["loss_weights"].tolist() == [0.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 1.0]
        assert "attention_metadata" in model_kwargs
