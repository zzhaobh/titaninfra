"""Unit tests for a6 packing."""


from titaninfra.data.packer import PackMode, StreamingPacker


class TestA6Packing:
    """Tests for test a6 packing."""

    # ── Position ID Reset ─────────────────────────────────────

    def test_single_doc_positions_start_at_zero(self):
        """Verify single doc positions start at zero."""
        packer = StreamingPacker(seq_len=10, mode=PackMode.SIMPLE)
        doc = [(i, i) for i in range(11)]  # 11 tokens = seq_len + 1
        packer.add_document(doc)

        samples = list(packer.flush_ready())
        assert len(samples) == 1
        sample = samples[0]
        expected = list(range(11))
        assert sample.positions == expected

    def test_multi_doc_positions_reset(self):
        """Verify multi doc positions reset."""
        packer = StreamingPacker(seq_len=10, mode=PackMode.SIMPLE)

        doc_a = [(100 + i, 100 + i) for i in range(5)]
        doc_b = [(200 + i, 200 + i) for i in range(6)]

        packer.add_document(doc_a)
        packer.add_document(doc_b)

        samples = list(packer.flush_ready())
        assert len(samples) >= 1
        sample = samples[0]

        expected_pos = list(range(5)) + list(range(6))
        assert sample.positions == expected_pos, (
            f"positions 不匹配:\n  got:      {sample.positions}\n  expected: {expected_pos}"
        )


    def test_input_ids_labels_aligned(self):
        """Verify input ids labels aligned."""
        packer = StreamingPacker(seq_len=10, mode=PackMode.SIMPLE)
        # Pretrain: label = token_id
        doc = [(i, i) for i in range(15)]
        packer.add_document(doc)

        samples = list(packer.flush_ready())
        assert len(samples) >= 1
        for sample in samples:
            assert len(sample.input_ids) == len(sample.labels)
            assert len(sample.input_ids) == len(sample.positions)

    def test_sft_labels_preserved(self):
        """Verify sft labels preserved."""
        packer = StreamingPacker(seq_len=20, mode=PackMode.SIMPLE)

        IGNORE = -100
        # SFT doc: prompt (masked) + response (has loss)
        doc = (
            [(10, IGNORE), (11, IGNORE), (12, IGNORE)]  # prompt
            + [(20, 20), (21, 21), (22, 22)]  # response
        )
        packer.add_document(doc)

        padding_doc = [(0, 0)] * 15
        packer.add_document(padding_doc)

        samples = list(packer.flush_ready())
        assert len(samples) >= 1

        sample = samples[0]
        # prompt labels should be -100
        assert sample.labels[0] == IGNORE
        assert sample.labels[1] == IGNORE
        assert sample.labels[2] == IGNORE
        # response labels should be token_id
        assert sample.labels[3] == 20
        assert sample.labels[4] == 21
        assert sample.labels[5] == 22


    def test_output_length_matches_seq_len_plus_one(self):
        """Verify output length matches seq len plus one."""
        seq_len = 16
        packer = StreamingPacker(seq_len=seq_len, mode=PackMode.SIMPLE)

        doc = [(i, i) for i in range(50)]
        packer.add_document(doc)

        samples = list(packer.flush_ready())
        for sample in samples:
            assert len(sample.input_ids) == seq_len + 1, (
                f"输出长度 {len(sample.input_ids)} != {seq_len + 1}"
            )


    def test_no_pack_mode_single_doc(self):
        """Verify no pack mode single doc."""
        packer = StreamingPacker(seq_len=10, mode=PackMode.NONE)

        doc = [(i, i) for i in range(20)]
        packer.add_document(doc)

        samples = list(packer.flush_ready())
        assert len(samples) == 1
        assert len(samples[0].input_ids) == 11  # seq_len + 1

    def test_no_pack_multiple_docs_separate(self):
        """Verify no pack multiple docs separate."""
        packer = StreamingPacker(seq_len=10, mode=PackMode.NONE)

        doc_a = [(100 + i, 100 + i) for i in range(15)]
        doc_b = [(200 + i, 200 + i) for i in range(15)]
        packer.add_document(doc_a)
        packer.add_document(doc_b)

        samples = list(packer.flush_ready())
        assert len(samples) == 2

        assert samples[0].input_ids[0] == 100
        assert samples[1].input_ids[0] == 200


    def test_sft_mode_preserves_complete_samples(self):
        """Verify sft mode preserves complete samples."""
        packer = StreamingPacker(seq_len=20, mode=PackMode.SFT)

        short_doc = [(i, i) for i in range(8)]
        packer.add_document(short_doc)

        samples = list(packer.flush_ready())

        another_doc = [(100 + i, 100 + i) for i in range(15)]
        packer.add_document(another_doc)

        samples = list(packer.flush_ready())
        assert len(samples) >= 1


    def test_empty_document_ignored(self):
        """Verify empty document ignored."""
        packer = StreamingPacker(seq_len=10, mode=PackMode.SIMPLE)

        packer.add_document([])
        doc = [(i, i) for i in range(11)]
        packer.add_document(doc)

        samples = list(packer.flush_ready())
        assert len(samples) == 1
        assert len(samples[0].input_ids) == 11

    def test_reset_clears_state(self):
        """Verify reset clears state."""
        packer = StreamingPacker(seq_len=10, mode=PackMode.SIMPLE)

        doc = [(i, i) for i in range(5)]
        packer.add_document(doc)
        packer.reset()

        assert len(packer._buffer) == 0
        assert len(packer._position_buffer) == 0
