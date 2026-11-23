"""Unit tests for packer."""

import pytest

from titaninfra.data.packer import (
    PackedSample,
    PackMode,
    SamplePacker,
    StreamingPacker,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_pack_mode_none():
    """Verify pack mode none."""
    packer = SamplePacker(seq_len=10, mode=PackMode.NONE)
    samples = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
    ]
    packed = packer.pack(samples)
    assert packed == samples


def test_pack_mode_simple_basic():
    """Verify pack mode simple basic."""
    packer = SamplePacker(seq_len=10, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert packed[0]["labels"] == [-100, 2, 3, 4, 5, 6]
    assert packed[0]["positions"] == [0, 1, 2, 0, 1, 2]


def test_pack_mode_simple_overflow():
    """Verify pack mode simple overflow."""
    packer = SamplePacker(seq_len=5, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8], "labels": [1, 2, 3, 4, 5, 6, 7, 8]},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert packed[0]["labels"] == [1, 2, 3, 4, 5]
    assert packed[0]["positions"] == [0, 1, 2, 3, 4]


def test_pack_mode_simple_cannot_fit():
    """Verify pack mode simple cannot fit."""
    packer = SamplePacker(seq_len=5, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5, 6, 7], "labels": [4, 5, 6, 7]},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 2
    assert packed[0]["input_ids"] == [1, 2, 3]
    assert packed[0]["positions"] == [0, 1, 2]
    assert packed[1]["input_ids"] == [4, 5, 6, 7]
    assert packed[1]["positions"] == [0, 1, 2, 3]


def test_pack_mode_simple_empty():
    """Verify pack mode simple empty."""
    packer = SamplePacker(seq_len=10, mode=PackMode.SIMPLE)
    packed = packer.pack([])
    assert packed == []


def test_pack_mode_simple_wrap_around():
    """Verify pack mode simple wrap around."""
    packer = SamplePacker(seq_len=5, mode=PackMode.SIMPLE)

    samples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
    ]
    packed = packer.pack(samples)
    assert packed[0]["positions"] == [0, 1, 2]

    samples_long = [
        {"input_ids": list(range(12)), "labels": list(range(12))},  # 12 > seq_len=5
    ]
    packed_long = packer.pack(samples_long)

    assert packed_long[0]["input_ids"] == [0, 1, 2, 3, 4]
    assert packed_long[0]["positions"] == [0, 1, 2, 3, 4]

    packer2 = SamplePacker(seq_len=10, mode=PackMode.SIMPLE)
    samples_multi = [
        {"input_ids": list(range(3)), "labels": list(range(3))},   # doc 1: 3 tokens
        {"input_ids": list(range(3, 7)), "labels": list(range(3, 7))},  # doc 2: 4 tokens
    ]
    packed_multi = packer2.pack(samples_multi)

    assert len(packed_multi) == 1

    assert packed_multi[0]["input_ids"] == list(range(7))
    # positions: doc1 [0,1,2] + doc2 [0,1,2,3] = [0,1,2,0,1,2,3]
    assert packed_multi[0]["positions"] == [0, 1, 2, 0, 1, 2, 3]


def test_pack_mode_simple_multiple_docs():
    """Verify pack mode simple multiple docs."""
    packer = SamplePacker(seq_len=20, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},      # doc 1: length 3
        {"input_ids": [4, 5, 6, 7], "labels": [4, 5, 6, 7]},  # doc 2: length 4
        {"input_ids": [8, 9], "labels": [8, 9]},            # doc 3: length 2
    ]
    packed = packer.pack(samples)

    assert len(packed) == 1
    assert packed[0]["positions"] == [0, 1, 2, 0, 1, 2, 3, 0, 1]


def test_packed_sample_to_dict():
    """Verify packed sample to dict."""
    sample = PackedSample(
        input_ids=[1, 2, 3],
        labels=[-100, 2, 3],
        positions=[0, 1, 2]
    )

    result = sample.to_dict()
    assert result == {
        "input_ids": [1, 2, 3],
        "labels": [-100, 2, 3],
        "positions": [0, 1, 2],
    }

    sample_no_pos = PackedSample(
        input_ids=[1, 2, 3],
        labels=[-100, 2, 3]
    )
    result_no_pos = sample_no_pos.to_dict()
    assert result_no_pos == {
        "input_ids": [1, 2, 3],
        "labels": [-100, 2, 3],
    }
    assert "positions" not in result_no_pos


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_streaming_packer_basic():
    """Verify streaming packer basic."""
    packer = StreamingPacker(seq_len=5)

    packer.add_document([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])

    samples = list(packer.flush_ready())

    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 3, 4, 5, 6]
    assert samples[0].labels == [1, 2, 3, 4, 5, 6]
    assert samples[0].positions is not None
    assert samples[0].positions == [0, 1, 2, 3, 4, 5]


def test_streaming_packer_multiple_docs():
    """Verify streaming packer multiple docs."""
    packer = StreamingPacker(seq_len=5)

    packer.add_document([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])  # doc 1: 6 tokens
    packer.add_document([(7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12)])  # doc 2: 6 tokens

    samples = list(packer.flush_ready())

    assert len(samples) == 2

    assert samples[0].positions == [0, 1, 2, 3, 4, 5]

    assert samples[1].positions == [0, 1, 2, 3, 4, 5]


def test_streaming_packer_long_doc_positions_renumbered():
    """Long doc positions are renumbered after cut to prevent RoPE cache overflow.

    When a document exceeds seq_len, _flush_simple cuts at seq_len+1.
    The remaining buffer must have positions renumbered from 0 so they
    stay within the RoPE cache (sized to seq_len).  Without renumbering,
    positions like [6,7,8,9,10,11] would exceed seq_len=5 → CUDA OOB.
    """
    packer = StreamingPacker(seq_len=5)

    packer.add_document([(i, i) for i in range(12)])

    samples = list(packer.flush_ready())

    assert len(samples) == 2

    # First chunk: original positions [0..5] (within seq_len)
    assert samples[0].positions == [0, 1, 2, 3, 4, 5]

    # Second chunk: tail renumbered from 0 (was [6..11], now [0..5])
    assert samples[1].positions == [0, 1, 2, 3, 4, 5]

    # Multi-doc: short docs concatenated, each doc starts from 0
    packer2 = StreamingPacker(seq_len=5)
    packer2.add_document([(i, i) for i in range(3)])   # doc 1: 3 tokens
    packer2.add_document([(i, i) for i in range(3)])   # doc 2: 3 tokens
    packer2.add_document([(i, i) for i in range(5)])   # doc 3: 5 tokens

    samples2 = list(packer2.flush_ready())

    assert len(samples2) == 1

    # doc1 [0,1,2] + doc2 [0,1,2] = [0,1,2,0,1,2]
    assert samples2[0].positions == [0, 1, 2, 0, 1, 2]


def test_streaming_packer_partial_flush():
    """Verify streaming packer partial flush."""
    packer = StreamingPacker(seq_len=10)

    packer.add_document([(1, 1), (2, 2)])

    samples = list(packer.flush_ready())

    assert len(samples) == 0


def test_streaming_packer_total_tokens():
    """Verify streaming packer total tokens."""
    packer = StreamingPacker(seq_len=5)

    packer.add_document([(i, i) for i in range(10)])

    # Flush
    list(packer.flush_ready())

    assert packer.get_total_tokens_seen() == 5  # seq_len


def test_streaming_packer_empty_document():
    """Verify streaming packer empty document."""
    packer = StreamingPacker(seq_len=10)

    packer.add_document([])

    packer.add_document([(1, 1), (2, 2)])

    samples = list(packer.flush_ready())
    assert len(samples) == 0



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_exact_seq_len():
    """Verify exact seq len."""
    packer = SamplePacker(seq_len=5, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 1
    assert len(packed[0]["input_ids"]) == 5
    assert packed[0]["positions"] == [0, 1, 2, 0, 1]


def test_single_long_sample():
    """Verify single long sample."""
    packer = SamplePacker(seq_len=5, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": list(range(100)), "labels": list(range(100))},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 1
    assert len(packed[0]["input_ids"]) == 5
    assert packed[0]["positions"] == [0, 1, 2, 3, 4]


def test_very_short_samples():
    """Verify very short samples."""
    packer = SamplePacker(seq_len=100, mode=PackMode.SIMPLE)
    samples = [
        {"input_ids": [i], "labels": [i]}
        for i in range(10)
    ]
    packed = packer.pack(samples)

    assert len(packed) == 1
    assert len(packed[0]["input_ids"]) == 10
    assert packed[0]["positions"] == [0] * 10


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_pack_mode_sft_basic():
    """Verify pack mode sft basic."""
    packer = SamplePacker(seq_len=10, mode=PackMode.SFT)
    samples = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert packed[0]["labels"] == [-100, 2, 3, 4, 5, 6]
    assert packed[0]["positions"] == [0, 1, 2, 0, 1, 2]


def test_pack_mode_sft_long_sample_cross_batch():
    """Verify pack mode sft long sample cross batch."""
    packer = SamplePacker(seq_len=5, mode=PackMode.SFT)

    samples = [
        {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8], "labels": [1, 2, 3, 4, 5, 6, 7, 8]},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 2

    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert packed[0]["labels"] == [1, 2, 3, 4, 5]
    assert packed[0]["positions"] == [0, 1, 2, 3, 4]

    assert packed[1]["input_ids"] == [6, 7, 8]
    assert packed[1]["labels"] == [6, 7, 8]
    assert packed[1]["positions"] == [0, 1, 2]


def test_pack_mode_sft_very_long_sample():
    """Verify pack mode sft very long sample."""
    packer = SamplePacker(seq_len=5, mode=PackMode.SFT)

    samples = [
        {"input_ids": list(range(15)), "labels": list(range(15))},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 3

    assert packed[0]["input_ids"] == [0, 1, 2, 3, 4]
    assert packed[1]["input_ids"] == [5, 6, 7, 8, 9]
    assert packed[2]["input_ids"] == [10, 11, 12, 13, 14]

    assert packed[0]["positions"] == [0, 1, 2, 3, 4]
    assert packed[1]["positions"] == [0, 1, 2, 3, 4]
    assert packed[2]["positions"] == [0, 1, 2, 3, 4]


def test_pack_mode_sft_mixed_lengths():
    """Verify pack mode sft mixed lengths."""
    packer = SamplePacker(seq_len=10, mode=PackMode.SFT)

    samples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": list(range(4, 20)), "labels": list(range(4, 20))},
        {"input_ids": [20, 21], "labels": [20, 21]},
    ]
    packed = packer.pack(samples)

    assert len(packed) == 4

    # Batch 1
    assert packed[0]["input_ids"] == [1, 2, 3]
    assert packed[0]["positions"] == [0, 1, 2]

    # Batch 2
    assert packed[1]["input_ids"] == list(range(4, 14))
    assert packed[1]["positions"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Batch 3
    assert packed[2]["input_ids"] == list(range(14, 20))
    assert packed[2]["positions"] == [0, 1, 2, 3, 4, 5]

    # Batch 4
    assert packed[3]["input_ids"] == [20, 21]
    assert packed[3]["positions"] == [0, 1]


def test_pack_mode_sft_no_truncation():
    """Verify pack mode sft no truncation."""
    packer = SamplePacker(seq_len=5, mode=PackMode.SFT)

    samples = [
        {"input_ids": list(range(100)), "labels": list(range(100))},
    ]
    packed = packer.pack(samples)

    total_tokens = sum(len(p["input_ids"]) for p in packed)
    assert total_tokens == 100

    all_input_ids = []
    for p in packed:
        all_input_ids.extend(p["input_ids"])
    assert all_input_ids == list(range(100))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_streaming_packer_sentence_boundary_disabled():
    """Verify streaming packer sentence boundary disabled."""
    packer = StreamingPacker(seq_len=5)

    packer.add_document([(1, 1), (2, 2), (46, 46), (4, 4), (5, 5), (6, 6)])

    samples = list(packer.flush_ready())

    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 46, 4, 5, 6]


def test_streaming_packer_sentence_boundary_basic():
    """Verify streaming packer sentence boundary basic."""
    packer = StreamingPacker(
        seq_len=5, mode=PackMode.PRETRAIN, sentence_end_tokens=[46],
    )

    packer.add_document([(1, 1), (2, 2), (46, 46), (4, 4), (5, 5), (6, 6)])

    samples = list(packer.flush_ready())
    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 46]
    assert samples[0].positions == [0, 1, 2]

    remaining = list(packer.flush_all())
    assert len(remaining) == 1
    assert remaining[0].input_ids == [4, 5, 6]


def test_streaming_packer_sentence_boundary_chinese():
    """Verify streaming packer sentence boundary chinese."""
    packer = StreamingPacker(
        seq_len=5, mode=PackMode.PRETRAIN, sentence_end_tokens=[12290],
    )

    packer.add_document([
        (1, 1), (2, 2), (12290, 12290),
        (4, 4), (5, 5), (12290, 12290),
    ])

    samples = list(packer.flush_ready())

    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 12290, 4, 5, 12290]


def test_streaming_packer_sentence_boundary_fallback():
    """Verify streaming packer sentence boundary fallback."""
    packer = StreamingPacker(
        seq_len=5, mode=PackMode.PRETRAIN, sentence_end_tokens=[46],
    )

    packer.add_document([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])

    samples = list(packer.flush_ready())

    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 3, 4, 5, 6]


def test_streaming_packer_sentence_boundary_multiple_types():
    """Verify streaming packer sentence boundary multiple types."""
    packer = StreamingPacker(
        seq_len=8, mode=PackMode.PRETRAIN,
        sentence_end_tokens=[46, 33, 63, 12290],
    )

    packer.add_document([
        (1, 1), (2, 2), (46, 46),
        (4, 4), (4, 4), (33, 33),
        (6, 6), (6, 6), (63, 63),
    ])

    samples = list(packer.flush_ready())

    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 46, 4, 4, 33, 6, 6, 63]


def test_streaming_packer_sentence_boundary_prefers_last():
    """Verify streaming packer sentence boundary prefers last."""
    packer = StreamingPacker(
        seq_len=8, mode=PackMode.PRETRAIN,
        sentence_end_tokens=[46],  # .
    )

    packer.add_document([
        (1, 1), (2, 2), (46, 46),   # index 0-2
        (4, 4), (5, 5), (46, 46),   # index 3-5
        (7, 7), (8, 8), (9, 9),     # index 6-8
        (10, 10), (11, 11), (12, 12),
    ])

    samples = list(packer.flush_ready())

    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 46, 4, 5, 46]

    remaining = list(packer.flush_all())
    assert len(remaining) == 1
    assert remaining[0].input_ids == [7, 8, 9, 10, 11, 12]


def test_streaming_packer_sentence_boundary_short_buffer():
    """Verify streaming packer sentence boundary short buffer."""
    packer = StreamingPacker(
        seq_len=10, mode=PackMode.PRETRAIN,
        sentence_end_tokens=[46, 33, 63, 12290, 65281, 65311, 10],
    )

    packer.add_document([(1, 1), (2, 2), (46, 46), (4, 4), (5, 5)])

    samples = list(packer.flush_ready())
    assert len(samples) == 0

    remaining = list(packer.flush_all())
    assert len(remaining) == 1
    assert remaining[0].input_ids == [1, 2, 46, 4, 5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_flush_pretrain_does_not_drain_short_buffer():
    """Verify flush pretrain does not drain short buffer."""
    packer = StreamingPacker(
        seq_len=16,
        mode=PackMode.PRETRAIN,
        sentence_end_tokens=[46],  # '.' = 46
    )

    packer.add_document([(i, i) for i in range(5)])
    samples = list(packer.flush_ready())
    assert len(samples) == 0, f"不足 seq_len 且无句子边界时不应输出，但得到 {len(samples)} 个样本"

    assert len(packer._buffer) == 5


def test_flush_all_drains_remaining():
    """Verify flush all drains remaining."""
    packer = StreamingPacker(
        seq_len=16,
        mode=PackMode.PRETRAIN,
        sentence_end_tokens=[46],  # '.' = 46
    )

    packer.add_document([(i, i) for i in range(5)])

    assert len(list(packer.flush_ready())) == 0

    samples = list(packer.flush_all())
    assert len(samples) == 1
    assert samples[0].input_ids == list(range(5))
    assert len(packer._buffer) == 0


def test_flush_all_outputs_complete_then_remainder():
    """Verify flush all outputs complete then remainder."""
    packer = StreamingPacker(
        seq_len=16,
        mode=PackMode.PRETRAIN,
        sentence_end_tokens=[46],  # '.' = 46
    )

    packer.add_document([(1, 1), (2, 2), (46, 46), (4, 4), (5, 5)])

    samples = list(packer.flush_all())
    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 46, 4, 5]


def test_flush_all_preserves_per_document_positions():
    """Verify flush all preserves per document positions."""
    packer = StreamingPacker(
        seq_len=16,
        mode=PackMode.PRETRAIN,
        sentence_end_tokens=[],
    )

    packer.add_document([(10, 10), (11, 11), (12, 12)])
    packer.add_document([(20, 20), (21, 21)])

    assert len(list(packer.flush_ready())) == 0

    samples = list(packer.flush_all())
    assert len(samples) == 1
    assert samples[0].input_ids == [10, 11, 12, 20, 21]
    assert samples[0].positions == [0, 1, 2, 0, 1]


def test_position_overflow_prevented_after_flush():
    """Positions must never exceed seq_len after buffer cut.

    Reproduces the SIGABRT root cause: a 20-token document in a seq_len=5
    packer produces 3 chunks.  Without renumbering, chunk 2 would have
    positions [6..11] and chunk 3 [12..17] — both exceed the RoPE cache.
    After the fix, every chunk's positions start from 0.
    """
    packer = StreamingPacker(seq_len=5)

    packer.add_document([(i, i) for i in range(20)])

    samples = list(packer.flush_ready())
    assert len(samples) == 3  # 20 tokens / (5+1) = 3 full chunks, 2 leftover

    for idx, sample in enumerate(samples):
        assert max(sample.positions) <= 5, (
            f"Chunk {idx}: max position {max(sample.positions)} exceeds seq_len=5"
        )
        assert sample.positions[0] == 0, (
            f"Chunk {idx}: positions must start from 0, got {sample.positions[0]}"
        )

    residual = list(packer.flush_all())
    assert len(residual) == 1
    assert residual[0].positions == [0, 1]


def test_position_renumber_preserves_multi_doc_boundaries():
    """After renumbering, document boundaries (position resets) are preserved.

    Buffer: doc1(8 tokens) + doc2(4 tokens) = 12 tokens, seq_len=5.
    Chunk 1 takes [0..5] from doc1, remaining buffer has doc1 tail(2) + doc2(4).
    Renumbering should keep the boundary between doc1-tail and doc2.
    """
    packer = StreamingPacker(seq_len=5)

    packer.add_document([(100 + i, 100 + i) for i in range(8)])
    packer.add_document([(200 + i, 200 + i) for i in range(4)])

    samples = list(packer.flush_ready())
    assert len(samples) == 2

    # Chunk 1: first 6 tokens from doc1 → positions [0,1,2,3,4,5]
    assert samples[0].positions == [0, 1, 2, 3, 4, 5]

    # Chunk 2: doc1 tail (2 tokens, renumbered [0,1]) + doc2 (4 tokens, [0,1,2,3])
    assert samples[1].positions == [0, 1, 0, 1, 2, 3]


def test_position_renumber_pretrain_mode():
    """PRETRAIN mode also renumbers positions after cuts."""
    packer = StreamingPacker(seq_len=5, mode=PackMode.PRETRAIN)

    packer.add_document([(i, i) for i in range(14)])

    samples = list(packer.flush_ready())
    assert len(samples) == 2

    assert samples[0].positions[0] == 0
    assert samples[1].positions[0] == 0
    assert max(samples[1].positions) <= 5


# ============================================================================
# ============================================================================


def test_no_pack_single_document():
    """Verify no pack single document."""
    packer = StreamingPacker(seq_len=8, mode=PackMode.NONE)

    packer.add_document([(1, 1), (2, 2), (3, 3)])
    samples = list(packer.flush_ready())
    assert len(samples) == 1
    assert samples[0].input_ids == [1, 2, 3]
    assert samples[0].labels == [1, 2, 3]
    assert samples[0].positions == [0, 1, 2]


def test_no_pack_multiple_documents_separate():
    """Verify no pack multiple documents separate."""
    packer = StreamingPacker(seq_len=8, mode=PackMode.NONE)

    packer.add_document([(10, 10), (11, 11)])
    packer.add_document([(20, 20), (21, 21), (22, 22)])

    samples = list(packer.flush_ready())
    assert len(samples) == 2
    assert samples[0].input_ids == [10, 11]
    assert samples[1].input_ids == [20, 21, 22]
    assert samples[0].positions == [0, 1]
    assert samples[1].positions == [0, 1, 2]


def test_no_pack_truncate_long_document():
    """Verify no pack truncate long document."""
    packer = StreamingPacker(seq_len=4, mode=PackMode.NONE)

    packer.add_document([(i, i) for i in range(10)])
    samples = list(packer.flush_ready())
    assert len(samples) == 1
    assert len(samples[0].input_ids) == 5  # seq_len + 1
    assert samples[0].input_ids == [0, 1, 2, 3, 4]


def test_no_pack_empty_document():
    """Verify no pack empty document."""
    packer = StreamingPacker(seq_len=8, mode=PackMode.NONE)

    packer.add_document([])
    samples = list(packer.flush_ready())
    assert len(samples) == 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestSftBinPacking:
    """Tests for test sft bin packing."""

    def _doc(self, length: int, base: int = 0):
        return [(base + i, base + i) for i in range(length)]

    def test_sft_packs_multiple_docs_into_one_sample(self):
        """Verify sft packs multiple docs into one sample."""
        packer = StreamingPacker(seq_len=4096, mode=PackMode.SFT)
        for i in range(5):
            packer.add_document(self._doc(200, base=i * 1000))

        ready = list(packer.flush_ready())
        assert ready == []

        all_samples = list(packer.flush_all())
        assert len(all_samples) == 1
        assert len(all_samples[0].input_ids) == 1000

    def test_sft_emits_when_next_doc_overflows(self):
        """Verify sft emits when next doc overflows."""
        packer = StreamingPacker(seq_len=4096, mode=PackMode.SFT)

        packer.add_document(self._doc(1500, base=0))
        assert list(packer.flush_ready()) == []

        packer.add_document(self._doc(1500, base=10000))
        assert list(packer.flush_ready()) == []

        packer.add_document(self._doc(1500, base=20000))
        emitted = list(packer.flush_ready())
        assert len(emitted) == 1
        assert len(emitted[0].input_ids) == 3000

        tail = list(packer.flush_all())
        assert len(tail) == 1
        assert len(tail[0].input_ids) == 1500

    def test_sft_positions_reset_per_doc(self):
        """Verify sft positions reset per doc."""
        packer = StreamingPacker(seq_len=4096, mode=PackMode.SFT)
        packer.add_document(self._doc(200, base=0))
        packer.add_document(self._doc(300, base=1000))

        samples = list(packer.flush_all())
        assert len(samples) == 1
        expected = list(range(200)) + list(range(300))
        assert samples[0].positions == expected

    def test_sft_skip_oversized_defensive(self):
        """Verify sft skip oversized defensive."""
        packer = StreamingPacker(seq_len=4096, mode=PackMode.SFT)
        packer.add_document(self._doc(10000, base=0))   # > seq_len+1=4097
        packer.add_document(self._doc(200, base=20000))

        emitted = list(packer.flush_ready())
        assert emitted == []

        tail = list(packer.flush_all())
        assert len(tail) == 1
        assert len(tail[0].input_ids) == 200

    def test_sft_no_overwrite_per_doc_reset(self):
        """Verify sft no overwrite per doc reset."""
        packer = StreamingPacker(seq_len=4096, mode=PackMode.SFT)
        packer.add_document(self._doc(100, base=0))
        packer.add_document(self._doc(200, base=1000))
        packer.add_document(self._doc(50, base=2000))

        samples = list(packer.flush_all())
        assert len(samples) == 1
        expected = list(range(100)) + list(range(200)) + list(range(50))
        assert samples[0].positions == expected
        assert samples[0].positions != list(range(350))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


IGNORE_INDEX = -100


class TestWeightBuffer:
    """Tests for test weight buffer."""

    def test_2tuple_after_3tuple_respects_ignore_index(self):
        """Verify 2tuple after 3tuple respects ignore index."""
        packer = StreamingPacker(seq_len=20, mode=PackMode.SIMPLE)
        doc1 = [(1, 1, 0.5), (2, 2, 1.0), (3, IGNORE_INDEX, 0.0)]
        packer.add_document(doc1)

        doc2 = [(4, IGNORE_INDEX), (5, 5), (6, IGNORE_INDEX)]
        packer.add_document(doc2)

        samples = list(packer.flush_all())
        assert len(samples) == 1
        w = samples[0].loss_weights
        assert w is not None
        assert w == [0.5, 1.0, 0.0, 0.0, 1.0, 0.0]

    def test_3tuple_after_2tuple_backfills_from_labels(self):
        """Verify 3tuple after 2tuple backfills from labels."""
        packer = StreamingPacker(seq_len=20, mode=PackMode.SIMPLE)
        doc1 = [(1, IGNORE_INDEX), (2, 2), (3, IGNORE_INDEX)]
        packer.add_document(doc1)

        doc2 = [(4, 4, 0.7), (5, IGNORE_INDEX, 0.0)]
        packer.add_document(doc2)

        samples = list(packer.flush_all())
        assert len(samples) == 1
        w = samples[0].loss_weights
        assert w is not None
        assert w == [0.0, 1.0, 0.0, 0.7, 0.0]

    def test_no_weights_stays_none(self):
        """Verify no weights stays none."""
        packer = StreamingPacker(seq_len=20, mode=PackMode.SIMPLE)
        packer.add_document([(1, 1), (2, IGNORE_INDEX)])
        packer.add_document([(3, 3), (4, IGNORE_INDEX)])

        samples = list(packer.flush_all())
        assert len(samples) == 1
        assert samples[0].loss_weights is None
