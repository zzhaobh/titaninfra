"""Unit tests for sft utils."""

import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import datasets  # noqa: F401
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

requires_pandas = pytest.mark.skipif(not HAS_PANDAS, reason="pandas 未安装")
requires_datasets = pytest.mark.skipif(not HAS_DATASETS, reason="datasets 未安装")


# ===========================================================================
# ===========================================================================


@requires_pandas
def test_load_sft_dataset_from_parquet(tmp_path):
    """Verify load sft dataset from parquet."""
    from titaninfra.data.sft_utils import load_sft_dataset

    df = pd.DataFrame({"messages": [
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}],
    ]})
    path = str(tmp_path / "data.parquet")
    df.to_parquet(path)

    ds = load_sft_dataset(train_files=path)
    assert len(ds) == 2


@requires_pandas
def test_load_sft_dataset_dp_sharding(tmp_path):
    """Verify load sft dataset dp sharding."""
    from titaninfra.data.sft_utils import load_sft_dataset

    df = pd.DataFrame({"messages": [[{"role": "user", "content": f"q{i}"}] for i in range(4)]})
    path = str(tmp_path / "data.parquet")
    df.to_parquet(path)

    ds0 = load_sft_dataset(train_files=path, dp_rank=0, dp_size=2)
    ds1 = load_sft_dataset(train_files=path, dp_rank=1, dp_size=2)
    assert len(ds0) + len(ds1) == 4
    assert len(ds0) == 2 and len(ds1) == 2


@requires_pandas
def test_load_sft_dataset_from_directory(tmp_path):
    """Verify load sft dataset from directory."""
    from titaninfra.data.sft_utils import load_sft_dataset

    for i in range(2):
        df = pd.DataFrame({"messages": [[{"role": "user", "content": f"x{i}"}]]})
        df.to_parquet(str(tmp_path / f"data_{i}.parquet"))

    ds = load_sft_dataset(train_files=str(tmp_path))
    assert len(ds) == 2


def test_load_sft_dataset_no_source_raises():
    """Verify load sft dataset no source raises."""
    from titaninfra.data.sft_utils import load_sft_dataset

    with pytest.raises(ValueError, match="dataset 或 train_files"):
        load_sft_dataset()


@requires_pandas
def test_load_sft_dataset_from_jsonl(tmp_path):
    """Verify load sft dataset from jsonl."""
    import json

    from titaninfra.data.sft_utils import load_sft_dataset

    jsonl_path = tmp_path / "data.jsonl"
    with open(str(jsonl_path), "w") as f:
        for i in range(3):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"q{i}"}]}) + "\n")

    ds = load_sft_dataset(train_files=str(jsonl_path))
    assert len(ds) == 3


@requires_pandas
def test_load_sft_dataset_from_json(tmp_path):
    """Verify load sft dataset from json."""
    import json

    from titaninfra.data.sft_utils import load_sft_dataset

    json_path = tmp_path / "data.json"
    data = [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(2)]
    with open(str(json_path), "w") as f:
        json.dump(data, f)

    ds = load_sft_dataset(train_files=str(json_path))
    assert len(ds) == 2


@requires_datasets
def test_load_sft_dataset_empty_dir_raises(tmp_path):
    """Verify load sft dataset empty dir raises."""
    from titaninfra.data.sft_utils import load_sft_dataset

    with pytest.raises(ValueError):
        load_sft_dataset(train_files=str(tmp_path))


# ===========================================================================
# ===========================================================================


def test_tokenize_conversation_masks_prompt():
    """Verify tokenize conversation masks prompt."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft.types import EncodedSample
    from titaninfra.data.sft_utils import tokenize_conversation
    from titaninfra.data.templates import IGNORE_INDEX

    tokenizer = MagicMock()
    messages = [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "xyz"},
    ]
    encoded = EncodedSample(
        input_ids=(1, 2, 3, 4, 5),
        labels=(IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5),
        positions=(0, 1, 2, 3, 4),
    )
    with patch("titaninfra.data.sft.dispatch.tokenize_sample", return_value=encoded):
        input_ids, labels = tokenize_conversation(messages, tokenizer, template=None)

    assert len(input_ids) == len(labels)
    assert all(lbl == IGNORE_INDEX for lbl in labels[:2])
    assert all(lbl != IGNORE_INDEX for lbl in labels[2:])


def test_tokenize_conversation_train_on_inputs():
    """Verify tokenize conversation train on inputs."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft.types import EncodedSample
    from titaninfra.data.sft_utils import tokenize_conversation
    from titaninfra.data.templates import IGNORE_INDEX

    tokenizer = MagicMock()
    messages = [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "xyz"},
    ]
    encoded = EncodedSample(
        input_ids=(1, 2, 3, 4, 5),
        labels=(1, 2, 3, 4, 5),
        positions=(0, 1, 2, 3, 4),
    )
    with patch("titaninfra.data.sft.dispatch.tokenize_sample", return_value=encoded) as mock_fn:
        _, labels = tokenize_conversation(messages, tokenizer, train_on_inputs=True)
    mock_fn.assert_called_once()
    assert IGNORE_INDEX not in labels


def test_tokenize_conversation_max_length():
    """Verify tokenize conversation max length."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft.types import EncodedSample
    from titaninfra.data.sft_utils import tokenize_conversation

    tokenizer = MagicMock()
    messages = [
        {"role": "user", "content": "longtext"},
        {"role": "assistant", "content": "longresponse"},
    ]
    encoded = EncodedSample(
        input_ids=(1, 2, 3),
        labels=(-100, 2, 3),
        positions=(0, 1, 2),
    )
    with patch(
        "titaninfra.data.sft.dispatch.tokenize_sample", return_value=encoded,
    ) as mock_fn:
        input_ids, labels = tokenize_conversation(messages, tokenizer, max_length=5)
        assert mock_fn.call_args.kwargs.get("max_length") == 5
    assert len(input_ids) <= 5
    assert len(labels) <= 5


def test_tokenize_conversation_with_template():
    """Verify tokenize conversation with template."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft_utils import tokenize_conversation
    from titaninfra.data.templates import get_template

    template = get_template("chatml")
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    with patch("titaninfra.data.sft_utils.encode_with_template") as mock_encode:
        mock_encode.return_value = ([1, 2, 3], [1, 2, 3])
        input_ids, labels = tokenize_conversation(messages, tokenizer, template=template)
        mock_encode.assert_called_once()


def test_tokenize_conversation_returns_equal_length():
    """Verify tokenize conversation returns equal length."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft.types import EncodedSample
    from titaninfra.data.sft_utils import tokenize_conversation

    tokenizer = MagicMock()
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    encoded = EncodedSample(
        input_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        labels=(-100, -100, -100, -100, -100, 6, 7, 8),
        positions=tuple(range(8)),
    )
    with patch("titaninfra.data.sft.dispatch.tokenize_sample", return_value=encoded):
        input_ids, labels = tokenize_conversation(messages, tokenizer)
    assert len(input_ids) == len(labels)


# ===========================================================================
# ===========================================================================


def test_load_hf_tokenizer_sets_pad_token():
    """Verify load hf tokenizer sets pad token."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft_utils import load_hf_tokenizer

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    with patch("titaninfra.data.sft_utils.AutoTokenizer") as MockAutoTokenizer:
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer
        result = load_hf_tokenizer("/fake/path")

    assert result.pad_token == "<eos>"


def test_load_hf_tokenizer_keeps_existing_pad_token():
    """Verify load hf tokenizer keeps existing pad token."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft_utils import load_hf_tokenizer

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "<eos>"

    with patch("titaninfra.data.sft_utils.AutoTokenizer") as MockAutoTokenizer:
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer
        result = load_hf_tokenizer("/fake/path")

    assert result.pad_token == "<pad>"


# ===========================================================================
# ===========================================================================


def test_tokenize_hf_template_path_calls_apply_chat_template():
    """Verify tokenize hf template path calls apply chat template."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft.types import EncodedSample
    from titaninfra.data.sft_utils import tokenize_conversation

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    encoded = EncodedSample(
        input_ids=(1, 2, 3, 4, 5),
        labels=(-100, -100, 3, 4, 5),
        positions=(0, 1, 2, 3, 4),
    )
    with patch(
        "titaninfra.data.sft.dispatch.tokenize_sample",
        return_value=encoded,
    ) as mock_fn:
        tokenizer = MagicMock()
        input_ids, labels = tokenize_conversation(messages, tokenizer, template=None)
        mock_fn.assert_called_once()
        assert input_ids == [1, 2, 3, 4, 5]
        assert labels == [-100, -100, 3, 4, 5]


def test_tokenize_hf_template_raises_on_template_error():
    """Verify tokenize hf template raises on template error."""
    from unittest.mock import MagicMock, patch

    from titaninfra.data.sft_utils import tokenize_conversation

    tokenizer = MagicMock()

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    with patch(
        "titaninfra.data.sft.dispatch.tokenize_sample",
        side_effect=RuntimeError("unexpected"),
    ), pytest.raises(RuntimeError, match="unexpected"):
        tokenize_conversation(messages, tokenizer, template=None)
