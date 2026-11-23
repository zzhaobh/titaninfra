"""Unit tests for basic."""

from tests.conftest import make_test_model_config, make_test_train_config
from titaninfra.config import PackStrategy, StreamConfig


def test_basic_config_creation():
    """Verify basic config creation."""
    model = make_test_model_config(name="qwen3/8B")
    assert model.name == "qwen3/8B"

    train = make_test_train_config()
    assert train.lr > 0

    train_sft = make_test_train_config()
    assert train_sft.lr > 0


def test_basic_data_config_creation():
    """Verify basic data config creation."""
    stream = StreamConfig(tokenizer_name="qwen3", local="/tmp/test")
    assert stream.data_type == "text"

    stream_sft = StreamConfig(tokenizer_name="qwen3", local="/tmp/sft", data_type="messages", add_bos_eos=False)
    assert stream_sft.data_type == "messages"


def test_stream_config_pack_strategy():
    """Verify stream config pack strategy."""
    stream = StreamConfig(
        tokenizer_name="qwen3",
        local="/tmp/test",
        pack_strategy=PackStrategy.SENTENCE_BOUNDARY,
        sentence_end_tokens=["\n"],
    )
    assert stream.pack_strategy == PackStrategy.SENTENCE_BOUNDARY
    assert stream.sentence_end_tokens == ["\n"]


def test_stream_config_resolve_stream_id():
    """Verify stream config resolve stream id."""
    stream = StreamConfig(tokenizer_name="qwen3", local="/data/zh/mds/train")
    assert stream.resolve_stream_id() == "train"

    stream_with_id = StreamConfig(tokenizer_name="qwen3", local="/data/test", stream_id="custom_id")
    assert stream_with_id.resolve_stream_id() == "custom_id"


def test_stream_config_to_mosaicml_kwargs():
    """Verify stream config to mosaicml kwargs."""
    stream = StreamConfig(tokenizer_name="qwen3", local="/data/test", remote="s3://bucket/test", split="train")
    kwargs = stream.to_mosaicml_kwargs()
    assert kwargs["local"] == "/data/test"
    assert kwargs["remote"] == "s3://bucket/test"
    assert kwargs["split"] == "train"
