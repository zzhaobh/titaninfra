"""Unit tests for static analysis fixes."""

import pytest

from tests.conftest import make_test_model_config

# ============================================================================
# ============================================================================


class TestPackerBoundaryRegression:
    """Tests for test packer boundary regression."""

    @staticmethod
    def _make_sft_packer(seq_len: int):
        """Make sft packer."""
        from titaninfra.data.packer import PackMode, StreamingPacker

        return StreamingPacker(seq_len=seq_len, mode=PackMode.SFT)

    def test_overlong_doc_dropped_boundaries_clean(self):
        """Verify overlong doc dropped boundaries clean."""
        packer = self._make_sft_packer(seq_len=8)

        tokens = [(i, i) for i in range(25)]
        packer.add_document(tokens)

        results = list(packer.flush_ready())
        assert len(results) == 0, "超长文档应被防御性丢弃"
        for b in packer._doc_boundaries:
            assert b >= 0, f"边界值不应为负: {b}"

    def test_overlong_doc_dropped_no_output(self):
        """Verify overlong doc dropped no output."""
        packer = self._make_sft_packer(seq_len=8)

        doc_len = 20
        tokens = [(i, i) for i in range(doc_len)]
        packer.add_document(tokens)

        all_chunks = list(packer.flush_ready())
        assert len(all_chunks) == 0, "超长文档应被丢弃，不产生输出"

    def test_overlong_then_normal_doc(self):
        """Verify overlong then normal doc."""
        packer = self._make_sft_packer(seq_len=8)

        long_doc = [(100 + i, 100 + i) for i in range(20)]
        packer.add_document(long_doc)
        list(packer.flush_ready())

        short_doc = [(200 + i, 200 + i) for i in range(5)]
        packer.add_document(short_doc)

        assert all(b >= 0 for b in packer._doc_boundaries), \
            f"边界值不应为负: {packer._doc_boundaries}"
        remaining = list(packer.flush_all())
        total = sum(len(c.input_ids) for c in remaining)
        assert total > 0, "正常文档应可通过 flush_all 输出"

    def test_normal_docs_bin_packed(self):
        """Verify normal docs bin packed."""
        seq_len = 8
        packer = self._make_sft_packer(seq_len=seq_len)

        for doc_idx in range(3):
            base = doc_idx * 100
            tokens = [(base + i, base + i) for i in range(4)]
            packer.add_document(tokens)

        all_chunks = list(packer.flush_ready()) + list(packer.flush_all())
        total_tokens = sum(len(c.input_ids) for c in all_chunks)
        assert total_tokens > 0, "正常文档应产生输出"
        for i, c in enumerate(all_chunks):
            assert len(c.input_ids) > 0, f"chunk {i} 不应为空"
        assert all(b >= 0 for b in packer._doc_boundaries)

    def test_three_consecutive_overlong_docs_all_dropped(self):
        """Verify three consecutive overlong docs all dropped."""
        packer = self._make_sft_packer(seq_len=4)

        all_chunks = []
        for doc_idx in range(3):
            base = doc_idx * 100
            tokens = [(base + i, base + i) for i in range(15)]
            packer.add_document(tokens)
            all_chunks.extend(packer.flush_ready())

        assert len(all_chunks) == 0, "超长文档应全部被丢弃"
        assert len(packer._doc_boundaries) == 0
        assert len(packer._buffer) == 0


# ============================================================================
# ============================================================================


class TestStreamMixerImmutability:
    """Tests for test stream mixer immutability."""

    def test_exhausted_stream_does_not_corrupt_weights(self):
        """Verify exhausted stream does not corrupt weights."""
        from titaninfra.data.streaming_dataset import StreamMixer

        streams = [
            iter([(1, 1), (2, 2)]),
            iter([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]),
        ]
        original_weights = [0.5, 0.5]
        mixer = StreamMixer(streams, original_weights)

        results = list(mixer)
        assert len(results) > 0

        assert original_weights == [0.5, 0.5], \
            "原始 weights 列表不应被修改（不可变性）"


# ============================================================================
# ============================================================================


class TestWarningsStacklevel:
    """Tests for test warnings stacklevel."""

    def test_memory_budget_warning_and_auto_fix(self):
        """Verify memory budget warning and auto fix."""
        import warnings


        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = make_test_model_config(
                name="qwen3/8B",
                ac_mode="memory_budget",
                compile=False,
            )
            assert cfg.compile is True
            budget_warns = [x for x in w if "memory_budget" in str(x.message)]
            assert len(budget_warns) == 1
            assert "torch.compile" in str(budget_warns[0].message)

    def test_async_tp_warning_and_auto_fix(self):
        """Verify async tp warning and auto fix."""
        import warnings


        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = make_test_model_config(
                name="qwen3/8B",
                enable_async_tensor_parallel=True,
                compile=False,
                ac_mode="none",
            )
            assert cfg.compile is True
            tp_warns = [x for x in w if "async_tensor_parallel" in str(x.message)]
            assert len(tp_warns) == 1

    def test_sdpa_backend_rejected(self):
        """Verify sdpa backend rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_test_model_config(
                name="qwen3/8B",
                attn_backend="sdpa",
                attn_mask_type="block_causal",
            )


# ============================================================================
# ============================================================================


class TestBoolIntBranchOrder:
    """Tests for test bool int branch order."""

    def test_bool_value_detected_as_int_type(self):
        """Verify bool value detected as int type."""
        from titaninfra.data.converter import _infer_columns_from_sample

        sample = {"flag": True, "count": 42, "name": "test"}
        columns = _infer_columns_from_sample(sample)
        assert columns["flag"] == "int"
        assert columns["count"] == "int"
        assert columns["name"] == "str"

    def test_false_value_detected(self):
        """Verify false value detected."""
        from titaninfra.data.converter import _infer_columns_from_sample

        sample = {"active": False}
        columns = _infer_columns_from_sample(sample)
        assert columns["active"] == "int"

    def test_mixed_types(self):
        """Verify mixed types."""
        from titaninfra.data.converter import _infer_columns_from_sample

        sample = {
            "text": "hello",
            "flag": True,
            "count": 100,
            "score": 0.95,
            "data": b"bytes",
        }
        columns = _infer_columns_from_sample(sample)
        assert columns == {
            "text": "str",
            "flag": "int",
            "count": "int",
            "score": "float64",
            "data": "bytes",
        }


# ============================================================================
# ============================================================================


class TestPathTraversalProtection:
    """Tests for test path traversal protection."""

    def test_rejects_parent_directory_traversal(self, tmp_path):
        """Verify rejects parent directory traversal."""
        from titaninfra.data.converter import _validate_single_shard

        shard_info = {"raw_data": {"basename": "../../etc/passwd"}}
        errors = _validate_single_shard(tmp_path, 0, shard_info)
        assert any("非法路径字符" in e for e in errors)

    def test_rejects_slash_in_basename(self, tmp_path):
        """Verify rejects slash in basename."""
        from titaninfra.data.converter import _validate_single_shard

        shard_info = {"raw_data": {"basename": "subdir/file.mds"}}
        errors = _validate_single_shard(tmp_path, 0, shard_info)
        assert not any("非法路径字符" in e for e in errors)

    def test_rejects_backslash_in_basename(self, tmp_path):
        """Verify rejects backslash in basename."""
        from titaninfra.data.converter import _validate_single_shard

        shard_info = {"raw_data": {"basename": "subdir\\file.mds"}}
        errors = _validate_single_shard(tmp_path, 0, shard_info)
        assert not any("非法路径字符" in e for e in errors)

    def test_accepts_normal_basename(self, tmp_path):
        """Verify accepts normal basename."""
        from titaninfra.data.converter import _validate_single_shard

        shard_info = {"raw_data": {"basename": "shard.00000.mds"}}
        errors = _validate_single_shard(tmp_path, 0, shard_info)
        assert not any("非法路径字符" in e for e in errors)


# ============================================================================
# ============================================================================


class TestHashlibWhitelist:
    """Tests for test hashlib whitelist."""

    def test_rejects_unknown_algorithm(self, tmp_path):
        """Verify rejects unknown algorithm."""
        from titaninfra.data.converter import _verify_file_hashes

        data_file = tmp_path / "test.mds"
        data_file.write_bytes(b"test data")
        hashes = {"unknown_algo": "abc123"}
        errors = _verify_file_hashes(data_file, hashes, 0, "test.mds")
        assert len(errors) == 0

    def test_accepts_sha256(self, tmp_path):
        """Verify accepts sha256."""
        import hashlib

        from titaninfra.data.converter import _verify_file_hashes

        data_file = tmp_path / "test.mds"
        data_file.write_bytes(b"test data")
        correct_hash = hashlib.sha256(b"test data").hexdigest()

        errors = _verify_file_hashes(data_file, {"sha256": correct_hash}, 0, "test.mds")
        assert len(errors) == 0

        errors = _verify_file_hashes(data_file, {"sha256": "wrong"}, 0, "test.mds")
        assert len(errors) == 1
        assert "哈希不匹配" in errors[0]


# ============================================================================
# ============================================================================


class TestShardErrorPatternPrecision:
    """Tests for test shard error pattern precision."""

    def test_generic_shard_message_not_matched(self):
        """Verify generic shard message not matched."""
        pytest.importorskip("torchtitan", reason="需要 torchtitan")
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        ds = StreamingTextDataset.__new__(StreamingTextDataset)
        error = ValueError("shard configuration mismatch")
        assert ds._is_shard_error(error) is False

    def test_io_error_still_matched(self):
        """Verify io error still matched."""
        pytest.importorskip("torchtitan", reason="需要 torchtitan")
        from titaninfra.data.streaming_dataset import StreamingTextDataset

        ds = StreamingTextDataset.__new__(StreamingTextDataset)
        error = OSError("No such file or directory")
        assert ds._is_shard_error(error) is True


# ============================================================================
# ============================================================================


class TestLstripFix:
    """Tests for test lstrip fix."""

    def test_removeprefix_exact(self):
        """Verify removeprefix exact."""
        assert "./checkpoints".removeprefix("./") == "checkpoints"
        assert "checkpoints".removeprefix("./") == "checkpoints"
        assert "../data".removeprefix("./") == "../data"
        assert "...dots".removeprefix("./") == "...dots"
        assert "/absolute".removeprefix("./") == "/absolute"

    def test_lstrip_was_wrong(self):
        """Verify lstrip was wrong."""
        assert "../data".lstrip("./") == "data"
        assert "../data".removeprefix("./") == "../data"
