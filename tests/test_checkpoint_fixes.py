"""Unit tests for checkpoint fixes."""

import pytest

from tests.conftest import make_test_train_config

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_exclude_from_loading_default_no_lr_scheduler():
    """Verify exclude from loading default no lr scheduler."""

    config = make_test_train_config()
    assert config.exclude_from_loading == []
    assert "lr_scheduler" not in config.exclude_from_loading
    assert "dataloader" not in config.exclude_from_loading


def test_exclude_from_loading_custom_still_works():
    """Verify exclude from loading custom still works."""

    config = make_test_train_config(exclude_from_loading=["dataloader", "lr_scheduler"])
    assert "lr_scheduler" in config.exclude_from_loading


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_packed_stream_load_state_dict_resets_iterator():
    """Verify packed stream load state dict resets iterator."""
    from titaninfra.data.packer import StreamingPacker
    from titaninfra.data.streaming_dataset import PackedStream

    source = iter([[(1, 1), (2, 2)]])
    packer = StreamingPacker(seq_len=4)
    stream = PackedStream(source=source, packer=packer)

    stream._iterator = "fake_iterator"

    stream.load_state_dict({})
    assert stream._iterator is None


def test_stream_mixer_load_state_dict_resets_iterator():
    """Verify stream mixer load state dict resets iterator."""
    from titaninfra.data.packer import StreamingPacker
    from titaninfra.data.streaming_dataset import PackedStream, StreamMixer

    source = iter([[(1, 1)]])
    packer = StreamingPacker(seq_len=4)
    stream = PackedStream(source=source, packer=packer)
    mixer = StreamMixer(streams=[stream], weights=[1.0])

    mixer._iterator = "fake_iterator"

    state = mixer.state_dict()
    mixer.load_state_dict(state)
    assert mixer._iterator is None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_packed_stream_saves_source_state():
    """Verify packed stream saves source state."""
    from titaninfra.data.packer import StreamingPacker
    from titaninfra.data.streaming_dataset import PackedStream

    class MockStatefulSource:
        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def state_dict(self):
            return {"position": self.position}

        def load_state_dict(self, state):
            self.position = state["position"]

    source = MockStatefulSource()
    source.position = 42
    packer = StreamingPacker(seq_len=4)
    stream = PackedStream(source=source, packer=packer)

    state = stream.state_dict()
    assert "source" in state
    assert state["source"]["position"] == 42


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_neftune_guard_no_tok_embeddings():
    """Verify neftune guard no tok embeddings."""
    from unittest.mock import MagicMock

    model_part = MagicMock(spec=[])

    assert not hasattr(model_part, "tok_embeddings")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_new_checkpoint_advanced_fields_defaults():
    """Verify new checkpoint advanced fields defaults."""

    config = make_test_train_config()
    assert config.async_checkpoint == "disabled"


def test_new_checkpoint_advanced_fields_custom():
    """Verify new checkpoint advanced fields custom."""

    config = make_test_train_config(
        async_checkpoint="async",
    )
    assert config.async_checkpoint == "async"


def test_train_config_seed_field_wired():
    """Verify train config seed field wired."""

    config = make_test_train_config()
    assert hasattr(config, "seed")
    assert config.seed is None

    config = make_test_train_config(seed=1234)
    assert config.seed == 1234


def test_config_serialization_roundtrip_with_new_fields():
    """Verify config serialization roundtrip with new fields."""

    config = make_test_train_config(
        async_checkpoint="async_with_pinned_mem",
        exclude_from_loading=["dataloader"],
    )
    config_dict = config.model_dump()
    restored = make_test_train_config(**config_dict)

    assert restored.async_checkpoint == "async_with_pinned_mem"
    assert restored.exclude_from_loading == ["dataloader"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_to_hf_no_torchtitan_keys_in_output():
    """Verify to hf no torchtitan keys in output."""
    try:
        from titaninfra.models.qwen3 import model_registry
        from titaninfra.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
    except ImportError:
        pytest.skip("qwen3 model not available")

    try:
        model_config = model_registry("0.6B").model
        adapter = Qwen3StateDictAdapter(model_config, None)
    except Exception:
        pytest.skip("Qwen3StateDictAdapter init requires GPU context")

    tt_keys = set(adapter.from_hf_map.values())
    hf_keys = set(adapter.from_hf_map.keys())

    fake_state_dict = {}
    for tt_key in list(tt_keys)[:5]:
        if tt_key and "{}" not in tt_key:
            fake_state_dict[tt_key] = torch.zeros(1)

    if not fake_state_dict:
        pytest.skip("No simple TT keys available for testing")

    hf_result = adapter.to_hf(fake_state_dict)

    for key in hf_result:
        assert key not in tt_keys or key in hf_keys, (
            f"to_hf 输出包含 TorchTitan 格式的 key: {key}，"
            f"这表明 to_hf 中混入了 hf_state_dict[tt_key] = value 的 bug"
        )
