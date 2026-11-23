"""Unit tests for sft dataset."""

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch 未安装")
requires_pandas = pytest.mark.skipif(
    not HAS_TORCH, reason="torch 未安装"
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@requires_torch
def test_pack_samples_basic():
    """Verify pack samples basic."""
    from titaninfra.data.packer import PackMode, SamplePacker
    _pack_samples = lambda samples, seq_len: SamplePacker(seq_len=seq_len, mode=PackMode.SIMPLE).pack(samples)

    samples = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
        {"input_ids": [6, 7, 8], "labels": [6, 7, 8]},
    ]

    packed = _pack_samples(samples, seq_len=10)

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert packed[0]["labels"] == [-100, 2, 3, 4, 5, 6, 7, 8]


@requires_torch
def test_pack_samples_overflow():
    """Verify pack samples overflow."""
    from titaninfra.data.packer import PackMode, SamplePacker
    _pack_samples = lambda samples, seq_len: SamplePacker(seq_len=seq_len, mode=PackMode.SIMPLE).pack(samples)

    samples = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5, 6, 7, 8], "labels": [4, 5, 6, 7, 8]},
        {"input_ids": [9, 10], "labels": [9, 10]},
    ]

    packed = _pack_samples(samples, seq_len=5)

    assert len(packed) == 3
    assert packed[0]["input_ids"] == [1, 2, 3]
    assert packed[1]["input_ids"] == [4, 5, 6, 7, 8]
    assert packed[2]["input_ids"] == [9, 10]


@requires_torch
def test_pack_samples_boundary():
    """Verify pack samples boundary."""
    from titaninfra.data.packer import PackMode, SamplePacker
    _pack_samples = lambda samples, seq_len: SamplePacker(seq_len=seq_len, mode=PackMode.SIMPLE).pack(samples)

    samples = [
        {"input_ids": [1, 2], "labels": [1, 2]},
        {"input_ids": [3, 4], "labels": [3, 4]},
        {"input_ids": [5, 6], "labels": [5, 6]},
    ]

    packed = _pack_samples(samples, seq_len=6)

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_pack_samples_varlen_basic():
    """Verify pack samples varlen basic."""
    from titaninfra.data.packer import PackMode, SamplePacker
    _pack_samples_varlen = lambda samples, seq_len: SamplePacker(seq_len=seq_len, mode=PackMode.SIMPLE).pack(samples)

    samples = [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, 4]},
        {"input_ids": [5, 6, 7], "labels": [5, 6, 7]},
    ]
    packed = _pack_samples_varlen(samples, seq_len=10)

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert packed[0]["labels"] == [-100, 2, 3, 4, 5, 6, 7]
    assert packed[0]["positions"] == [0, 1, 2, 3, 0, 1, 2]


def test_pack_samples_varlen_overflow():
    """Verify pack samples varlen overflow."""
    from titaninfra.data.packer import PackMode, SamplePacker
    _pack_samples_varlen = lambda samples, seq_len: SamplePacker(seq_len=seq_len, mode=PackMode.SIMPLE).pack(samples)

    samples = [
        {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [1, 2, 3, 4, 5, 6]},
    ]
    packed = _pack_samples_varlen(samples, seq_len=4)

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4]
    assert packed[0]["positions"] == [0, 1, 2, 3]


def test_pack_samples_varlen_boundary():
    """Verify pack samples varlen boundary."""
    from titaninfra.data.packer import PackMode, SamplePacker
    _pack_samples_varlen = lambda samples, seq_len: SamplePacker(seq_len=seq_len, mode=PackMode.SIMPLE).pack(samples)

    samples = [
        {"input_ids": [1, 2], "labels": [1, 2]},
        {"input_ids": [3, 4], "labels": [3, 4]},
    ]
    packed = _pack_samples_varlen(samples, seq_len=4)

    assert len(packed) == 1
    assert packed[0]["positions"] == [0, 1, 0, 1]


def test_pack_samples_varlen_splits_when_full():
    """Verify pack samples varlen splits when full."""
    from titaninfra.data.packer import PackMode, SamplePacker
    _pack_samples_varlen = lambda samples, seq_len: SamplePacker(seq_len=seq_len, mode=PackMode.SIMPLE).pack(samples)

    samples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
        {"input_ids": [7, 8], "labels": [7, 8]},
    ]
    packed = _pack_samples_varlen(samples, seq_len=5)

    assert len(packed) == 2
    assert packed[0]["input_ids"] == [1, 2, 3]
    assert packed[0]["positions"] == [0, 1, 2]
    assert packed[1]["input_ids"] == [4, 5, 6, 7, 8]
    assert packed[1]["positions"] == [0, 1, 2, 0, 1]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_steps_per_epoch_ceil():
    """Verify steps per epoch ceil."""
    import math

    total_valid_tokens = 1000
    effective_tokens_per_step = 100
    steps_per_epoch = max(1, math.ceil(total_valid_tokens / effective_tokens_per_step))
    assert steps_per_epoch == 10

    total_valid_tokens = 1070
    effective_tokens_per_step = 100
    steps_per_epoch = max(1, math.ceil(total_valid_tokens / effective_tokens_per_step))
    assert steps_per_epoch == 11

    total_valid_tokens = 50
    effective_tokens_per_step = 100
    steps_per_epoch = max(1, math.ceil(total_valid_tokens / effective_tokens_per_step))
    assert steps_per_epoch == 1


def test_steps_per_epoch_covers_all_tokens():
    """Verify steps per epoch covers all tokens."""
    import math

    total_valid_tokens = 1234
    effective_tokens_per_step = 100

    steps_per_epoch = max(1, math.ceil(total_valid_tokens / effective_tokens_per_step))

    max_tokens_covered = steps_per_epoch * effective_tokens_per_step

    assert max_tokens_covered >= total_valid_tokens

    old_steps = int(total_valid_tokens / effective_tokens_per_step)
    old_tokens_covered = old_steps * effective_tokens_per_step
    assert old_tokens_covered < total_valid_tokens


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_epoch_boundary_detection_logic():
    """Verify epoch boundary detection logic."""

    def check_epoch_end(current_tokens, tokens_per_epoch, tolerance):
        """Check epoch end."""
        if current_tokens == 0:
            return False, 0

        progress = current_tokens % tokens_per_epoch
        epoch_num = current_tokens // tokens_per_epoch

        if progress < tolerance and epoch_num > 0:
            return True, epoch_num

        return False, epoch_num

    tokens_per_epoch = 1000
    tolerance = 100
    current_tokens = 1050

    is_epoch_end, epoch_num = check_epoch_end(current_tokens, tokens_per_epoch, tolerance)
    assert is_epoch_end is True
    assert epoch_num == 1

    current_tokens = 500
    is_epoch_end, epoch_num = check_epoch_end(current_tokens, tokens_per_epoch, tolerance)
    assert is_epoch_end is False
    assert epoch_num == 0

    current_tokens = 950
    is_epoch_end, epoch_num = check_epoch_end(current_tokens, tokens_per_epoch, tolerance)
    assert is_epoch_end is False
    assert epoch_num == 0

    current_tokens = 2010
    is_epoch_end, epoch_num = check_epoch_end(current_tokens, tokens_per_epoch, tolerance)
    assert is_epoch_end is True
    assert epoch_num == 2


def test_epoch_boundary_detection_wrong_logic():
    """Verify epoch boundary detection wrong logic."""

    def check_epoch_end_old(current_tokens, tokens_per_epoch, tolerance):
        """Check epoch end old."""
        if current_tokens == 0:
            return False, 0

        progress = current_tokens % tokens_per_epoch
        epoch_num = current_tokens // tokens_per_epoch

        if progress > 0 and (tokens_per_epoch - progress) < tolerance:
            return True, epoch_num
        elif progress < tolerance and epoch_num > 0:
            return True, epoch_num

        return False, epoch_num

    tokens_per_epoch = 1000
    tolerance = 100

    current_tokens = 1050

    is_epoch_end_old, epoch_num_old = check_epoch_end_old(
        current_tokens, tokens_per_epoch, tolerance
    )

    is_epoch_end_new, epoch_num_new = check_epoch_end_old(
        current_tokens, tokens_per_epoch, tolerance
    )

    assert is_epoch_end_old is True
    assert is_epoch_end_new is True


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_packing_efficiency_calculation():
    """Verify packing efficiency calculation."""
    from titaninfra.data.templates import IGNORE_INDEX

    samples = [
        {
            "input_ids": list(range(100)),
            "labels": [IGNORE_INDEX] * 20 + list(range(20, 100)),  # 80 valid
        },
        {
            "input_ids": list(range(80)),
            "labels": [IGNORE_INDEX] * 20 + list(range(20, 80)),  # 60 valid
        },
        {
            "input_ids": list(range(50)),
            "labels": [IGNORE_INDEX] * 10 + list(range(10, 50)),  # 40 valid
        },
    ]

    # Total valid tokens = 80 + 60 + 40 = 180
    total_valid_tokens = 180

    total_packed_tokens = sum(
        min(len(sample["input_ids"]), 100)
        for sample in samples
    )
    assert total_packed_tokens == 230

    # Packing efficiency = 180 / 230 = 0.7826
    packing_efficiency = total_valid_tokens / total_packed_tokens
    assert abs(packing_efficiency - 0.7826) < 0.01

    old_packing_efficiency = total_valid_tokens / (len(samples) * 100)
    assert abs(old_packing_efficiency - 0.6) < 0.01  # 180 / 300 = 0.6

    assert packing_efficiency > old_packing_efficiency


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_stream_config_enable_thinking_removed():
    """Verify stream config enable thinking removed."""
    from titaninfra.config.data import StreamConfig

    assert "enable_thinking" not in StreamConfig.model_fields
    cfg = StreamConfig(
        tokenizer_name="qwen3",
        local="/data/train",
        data_type="messages",
        add_bos_eos=False,
        enable_thinking=False,
    )
    assert not hasattr(cfg, "enable_thinking")
