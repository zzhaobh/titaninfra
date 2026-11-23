"""Unit tests for integration varlen positions."""

import pytest

from titaninfra.data.packer import (
    PackMode,
    SamplePacker,
    StreamingPacker,
)


def test_sft_and_pretrain_produce_same_positions():
    """Verify sft and pretrain produce same positions."""
    sft_packer = SamplePacker(seq_len=20, mode=PackMode.SIMPLE)
    sft_samples = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},      # 3 tokens
        {"input_ids": [4, 5, 6, 7], "labels": [4, 5, 6, 7]},    # 4 tokens
    ]
    sft_packed = sft_packer.pack(sft_samples)

    assert len(sft_packed) == 1
    assert sft_packed[0]["positions"] == [0, 1, 2, 0, 1, 2, 3]

    pretrain_packer = StreamingPacker(seq_len=10)
    pretrain_packer.add_document([(1, 1), (2, 2), (3, 3)])  # doc 1: 3 tokens
    pretrain_packer.add_document([(4, 4), (5, 5), (6, 6), (7, 7)])  # doc 2: 4 tokens
    pretrain_packer.add_document([(i, i) for i in range(8, 20)])  # doc 3: 12 tokens
    pretrain_packed = list(pretrain_packer.flush_ready())

    assert len(pretrain_packed) == 1
    # doc1 (3) + doc2 (4) + doc3 (4) = 11 tokens
    assert pretrain_packed[0].positions == [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]

    assert sft_packed[0]["positions"][:3] == [0, 1, 2]
    assert pretrain_packed[0].positions[:3] == [0, 1, 2]

    assert sft_packed[0]["positions"][3:7] == [0, 1, 2, 3]
    assert pretrain_packed[0].positions[3:7] == [0, 1, 2, 3]


def test_simple_positions_multiple_docs():
    """Verify simple positions multiple docs."""
    packer = SamplePacker(seq_len=20, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": [1, 2], "labels": [1, 2]},
        {"input_ids": [3, 4, 5], "labels": [3, 4, 5]},
        {"input_ids": [6], "labels": [6]},
        {"input_ids": [7, 8, 9, 10], "labels": [7, 8, 9, 10]},
        {"input_ids": [11, 12], "labels": [11, 12]},
    ]
    packed = packer.pack(samples)

    expected_positions = [
        0, 1,           # doc 1: length 2
        0, 1, 2,        # doc 2: length 3
        0,              # doc 3: length 1
        0, 1, 2, 3,     # doc 4: length 4
        0, 1,           # doc 5: length 2
    ]

    assert len(packed) == 1
    assert packed[0]["positions"] == expected_positions


def test_simple_positions_overflow():
    """Verify simple positions overflow."""
    sft_packer = SamplePacker(seq_len=5, mode=PackMode.SIMPLE)
    sft_samples = [{"input_ids": list(range(10)), "labels": list(range(10))}]
    sft_packed = sft_packer.pack(sft_samples)

    pretrain_packer = StreamingPacker(seq_len=5)
    pretrain_packer.add_document([(i, i) for i in range(10)])
    pretrain_packed = list(pretrain_packer.flush_ready())

    assert len(sft_packed) == 1
    assert sft_packed[0]["positions"] == [0, 1, 2, 3, 4]

    assert len(pretrain_packed) == 1
    assert pretrain_packed[0].positions == [0, 1, 2, 3, 4, 5]


def test_simple_positions_split_across_chunks():
    """Verify simple positions split across chunks."""
    packer = StreamingPacker(seq_len=5)
    packer.add_document([(i, i) for i in range(15)])

    samples = list(packer.flush_ready())

    assert len(samples) == 2

    # First chunk: positions [0..5] from the start of the document
    assert samples[0].positions == [0, 1, 2, 3, 4, 5]

    # Second chunk: tail renumbered from 0 to prevent RoPE cache overflow
    assert samples[1].positions == [0, 1, 2, 3, 4, 5]



def test_simple_positions_empty_doc():
    """Verify simple positions empty doc."""
    sft_packer = SamplePacker(seq_len=10, mode=PackMode.SIMPLE)
    sft_samples = [
        {"input_ids": [1, 2], "labels": [1, 2]},
        {"input_ids": [], "labels": []},
        {"input_ids": [3, 4], "labels": [3, 4]},
    ]
    sft_packed = sft_packer.pack(sft_samples)

    pretrain_packer = StreamingPacker(seq_len=10)
    pretrain_packer.add_document([(1, 1), (2, 2)])
    pretrain_packer.add_document([])
    pretrain_packer.add_document(
        [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11)]
    )
    pretrain_packed = list(pretrain_packer.flush_ready())

    assert sft_packed[0]["positions"] == [0, 1, 0, 1]

    # doc1: [0, 1], doc3: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert pretrain_packed[0].positions == [0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_positions_wrap_around():
    """Verify positions wrap around."""
    packer = SamplePacker(seq_len=10, mode=PackMode.SIMPLE)

    sample = {"input_ids": list(range(15)), "labels": list(range(15))}
    packed = packer.pack([sample])

    assert len(packed) == 1
    assert packed[0]["positions"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_positions_for_block_causal_attention():
    """Verify positions for block causal attention."""
    packer = SamplePacker(seq_len=20, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": [1, 2, 3, 151645, 4, 5, 6], "labels": [-100, 2, 3, 151645, 4, 5, 6]},
        {"input_ids": [7, 8], "labels": [7, 8]},
    ]
    packed = packer.pack(samples)

    # Sample 1: [0, 1, 2, 3, 4, 5, 6] (7 tokens)
    # Sample 2: [0, 1] (2 tokens)
    assert len(packed) == 1
    assert packed[0]["positions"] == [0, 1, 2, 3, 4, 5, 6, 0, 1]



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
