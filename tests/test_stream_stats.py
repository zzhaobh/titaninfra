"""Unit tests for stream stats."""

from __future__ import annotations

import json

import pytest

from titaninfra.data.schema import StreamStats, TokenizerStats, embed_tokenizer

# ============================================================================
# v3 round-trip
# ============================================================================


class TestStreamStatsV3RoundTrip:
    """Tests for test stream stats v3 round trip."""

    def test_round_trip(self, tmp_path):
        original = StreamStats(
            stream_id="en_books",
            stream_type="pretrain",
            num_samples=200,
            default_tokenizer="qwen3",
            tokenizer_stats={
                "qwen3": TokenizerStats(
                    total_tokens=100000,
                    avg_tokens_per_sample=500.0,
                    created_at="2026-04-17T12:00:00",
                ),
                "llama3": TokenizerStats(
                    total_tokens=90000,
                    avg_tokens_per_sample=450.0,
                    created_at="2026-04-18T10:00:00",
                ),
            },
            created_at="2026-04-17T12:00:00",
        )
        original.save(str(tmp_path))

        loaded = StreamStats.load(str(tmp_path))
        assert loaded is not None
        assert loaded.version == 3
        assert loaded.num_samples == 200
        assert loaded.default_tokenizer == "qwen3"
        assert len(loaded.tokenizer_stats) == 2
        assert loaded.tokenizer_stats["llama3"].total_tokens == 90000

    def test_no_top_level_token_fields(self, tmp_path):
        """Verify no top level token fields."""
        stats = StreamStats(
            num_samples=100,
            default_tokenizer="qwen3",
            tokenizer_stats={"qwen3": TokenizerStats(total_tokens=50000, avg_tokens_per_sample=500.0)},
        )
        stats.save(str(tmp_path))
        raw = json.loads((tmp_path / "stream_stats.json").read_text())
        assert "total_tokens" not in raw
        assert "avg_tokens_per_sample" not in raw


# ============================================================================
# ============================================================================


class TestResolveTokens:
    """Tests for test resolve tokens."""

    @pytest.fixture()
    def stats(self):
        return StreamStats(
            num_samples=100,
            default_tokenizer="qwen3",
            tokenizer_stats={
                "qwen3": TokenizerStats(
                    total_tokens=50000, avg_tokens_per_sample=500.0,
                ),
                "llama3": TokenizerStats(
                    total_tokens=45000, avg_tokens_per_sample=450.0,
                ),
            },
        )

    def test_explicit_tokenizer(self, stats):
        total, avg = stats.resolve_tokens("llama3")
        assert total == 45000
        assert avg == 450.0

    def test_default_tokenizer(self, stats):
        total, avg = stats.resolve_tokens(None)
        assert total == 50000
        assert avg == 500.0

    def test_explicit_mismatch_raises(self, stats):
        """Verify explicit mismatch raises."""
        with pytest.raises(ValueError, match="unknown_tok"):
            stats.resolve_tokens("unknown_tok")

    def test_zero_when_no_tokenizer_stats_and_none(self):
        """Verify zero when no tokenizer stats and none."""
        stats = StreamStats(num_samples=10)
        total, avg = stats.resolve_tokens(None)
        assert total == 0
        assert avg == 0.0

    def test_explicit_mismatch_raises_empty_stats(self):
        """Verify explicit mismatch raises empty stats."""
        stats = StreamStats(num_samples=10)
        with pytest.raises(ValueError, match="anything"):
            stats.resolve_tokens("anything")


# ============================================================================
# add_tokenizer_stats
# ============================================================================


class TestAddTokenizerStats:
    def test_add_new(self):
        stats = StreamStats(num_samples=100)
        stats.add_tokenizer_stats("qwen3", 50000, "2026-04-17")
        assert "qwen3" in stats.tokenizer_stats
        assert stats.tokenizer_stats["qwen3"].total_tokens == 50000
        assert stats.tokenizer_stats["qwen3"].avg_tokens_per_sample == 500.0

    def test_add_second_tokenizer(self):
        stats = StreamStats(
            num_samples=100,
            tokenizer_stats={
                "qwen3": TokenizerStats(total_tokens=50000, avg_tokens_per_sample=500.0),
            },
        )
        stats.add_tokenizer_stats("llama3", 45000)
        assert len(stats.tokenizer_stats) == 2
        assert stats.tokenizer_stats["llama3"].total_tokens == 45000

    def test_overwrite_existing(self):
        stats = StreamStats(
            num_samples=100,
            tokenizer_stats={
                "qwen3": TokenizerStats(total_tokens=50000, avg_tokens_per_sample=500.0),
            },
        )
        stats.add_tokenizer_stats("qwen3", 60000)
        assert stats.tokenizer_stats["qwen3"].total_tokens == 60000
        assert stats.tokenizer_stats["qwen3"].avg_tokens_per_sample == 600.0

    def test_fine_token_length_dist_roundtrip(self, tmp_path):
        fine_dist = {"0-64": 5, "64-128": 10, "128-192": 8}
        stats = StreamStats(num_samples=23, stream_id="test")
        stats.add_tokenizer_stats(
            "qwen3", 5000, "2026-05-22",
            fine_token_length_dist=fine_dist,
        )
        stats.default_tokenizer = "qwen3"
        stats.save(str(tmp_path))

        loaded = StreamStats.load(str(tmp_path))
        assert loaded is not None
        ts = loaded.tokenizer_stats["qwen3"]
        assert ts.fine_token_length_dist == fine_dist

    def test_fine_token_length_dist_none_compat(self):
        stats = StreamStats(num_samples=10)
        stats.add_tokenizer_stats("qwen3", 1000)
        assert stats.tokenizer_stats["qwen3"].fine_token_length_dist is None


# ============================================================================
# embed_tokenizer
# ============================================================================


class TestEmbedTokenizer:
    def test_copies_files(self, tmp_path):
        tok_dir = tmp_path / "src_tok"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.json").write_text('{"key": "val"}')
        (tok_dir / "tokenizer_config.json").write_text('{"model_type": "qwen3"}')

        mds_dir = tmp_path / "mds"
        mds_dir.mkdir()

        embed_tokenizer(str(tok_dir), str(mds_dir), "qwen3")

        dest = mds_dir / "tokenizers" / "qwen3"
        assert dest.exists()
        assert (dest / "tokenizer.json").read_text() == '{"key": "val"}'
        assert (dest / "tokenizer_config.json").read_text() == '{"model_type": "qwen3"}'

    def test_skips_if_path_missing(self, tmp_path):
        mds_dir = tmp_path / "mds"
        mds_dir.mkdir()
        embed_tokenizer("/nonexistent/path", str(mds_dir), "qwen3")
        assert not (mds_dir / "tokenizers").exists()

    def test_overwrites_existing(self, tmp_path):
        tok_dir = tmp_path / "src_tok"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.json").write_text("v1")

        mds_dir = tmp_path / "mds"
        mds_dir.mkdir()

        embed_tokenizer(str(tok_dir), str(mds_dir), "qwen3")
        assert (mds_dir / "tokenizers" / "qwen3" / "tokenizer.json").read_text() == "v1"

        (tok_dir / "tokenizer.json").write_text("v2")
        embed_tokenizer(str(tok_dir), str(mds_dir), "qwen3")
        assert (mds_dir / "tokenizers" / "qwen3" / "tokenizer.json").read_text() == "v2"


# ============================================================================
# Edge cases
# ============================================================================


class TestStreamStatsEdgeCases:
    def test_load_missing_file(self, tmp_path):
        assert StreamStats.load(str(tmp_path)) is None

    def test_load_invalid_json(self, tmp_path):
        (tmp_path / "stream_stats.json").write_text("not json")
        assert StreamStats.load(str(tmp_path)) is None

    def test_load_zero_samples(self, tmp_path):
        stats = {"version": 3, "num_samples": 0}
        (tmp_path / "stream_stats.json").write_text(json.dumps(stats))
        assert StreamStats.load(str(tmp_path)) is None

    def test_load_wrong_version(self, tmp_path):
        stats = {"version": 2, "num_samples": 10}
        (tmp_path / "stream_stats.json").write_text(json.dumps(stats))
        assert StreamStats.load(str(tmp_path)) is None

    def test_load_version_0(self, tmp_path):
        stats = {"version": 0, "num_samples": 10}
        (tmp_path / "stream_stats.json").write_text(json.dumps(stats))
        assert StreamStats.load(str(tmp_path)) is None


# ============================================================================
# ============================================================================


class TestRealDataCompat:
    """Tests for test real data compat."""

    @pytest.mark.parametrize(
        "mds_dir,expected_stream_id,expected_type,has_tokenizer",
        [
            ("data/aime/mds/train", "aime", "pretrain", True),
            ("data/aime/mds_sft/train", "aime", "messages", True),
            ("data/sft_test/basic/mds/train", "train", "messages", True),
        ],
    )
    def test_load_real_data(self, mds_dir, expected_stream_id, expected_type, has_tokenizer):
        s = StreamStats.load(mds_dir)
        assert s is not None
        assert s.version == 3
        assert s.stream_id == expected_stream_id
        assert s.stream_type == expected_type
        assert s.num_samples > 0
        if has_tokenizer:
            assert s.default_tokenizer != ""
            assert s.default_tokenizer in s.tokenizer_stats
            tok = s.tokenizer_stats[s.default_tokenizer]
            assert tok.total_tokens > 0
        else:
            assert s.default_tokenizer == ""
            assert s.tokenizer_stats == {}

    def test_resolve_tokens_aime(self):
        s = StreamStats.load("data/aime/mds/train")
        assert s is not None
        total, avg = s.resolve_tokens()
        assert total == 206673
        assert abs(avg - 2152.84375) < 0.01

    def test_resolve_tokens_explicit_name(self):
        s = StreamStats.load("data/aime/mds/train")
        assert s is not None
        total, avg = s.resolve_tokens(s.default_tokenizer)
        assert total == 206673

    def test_read_single_stream_stats_real(self):
        """Verify read single stream stats real."""
        try:
            from titaninfra.data.streaming_dataset import _read_single_stream_stats
        except (ImportError, ModuleNotFoundError):
            pytest.skip("requires torchtitan")

        result = _read_single_stream_stats("data/aime/mds/train")
        assert result is not None
        assert result["num_samples"] > 0
        assert result["total_tokens"] > 0
        assert "avg_tokens_per_sample" in result
        assert "stream_id" in result

    def test_read_single_stream_stats_no_tokenizer(self):
        """sft_test has no tokenizer_stats → total_tokens=0 is acceptable."""
        try:
            from titaninfra.data.streaming_dataset import _read_single_stream_stats
        except (ImportError, ModuleNotFoundError):
            pytest.skip("requires torchtitan")

        result = _read_single_stream_stats("data/sft_test/basic/mds/train")
        # Returns None when total_tokens == 0 (no tokenizer stats)
        # OR returns a dict with total_tokens=0 — both are valid depending on implementation
        if result is not None:
            assert result["num_samples"] > 0


# ============================================================================
# ============================================================================


class TestReadSingleStreamStatsTokenizerSelection:
    """Tests for test read single stream stats tokenizer selection."""

    def _write_v3(self, tmp_path):
        stats = StreamStats(
            stream_id="test",
            num_samples=100,
            default_tokenizer="qwen3",
            tokenizer_stats={
                "qwen3": TokenizerStats(total_tokens=50000, avg_tokens_per_sample=500.0),
                "llama3": TokenizerStats(total_tokens=45000, avg_tokens_per_sample=450.0),
            },
        )
        stats.save(str(tmp_path))

    def test_explicit_tokenizer_name(self, tmp_path):
        try:
            from titaninfra.data.streaming_dataset import _read_single_stream_stats
        except (ImportError, ModuleNotFoundError):
            pytest.skip("requires torchtitan")

        self._write_v3(tmp_path)
        result = _read_single_stream_stats(str(tmp_path), tokenizer_name="llama3")
        assert result is not None
        assert result["total_tokens"] == 45000
        assert result["avg_tokens_per_sample"] == 450.0

    def test_none_uses_default(self, tmp_path):
        try:
            from titaninfra.data.streaming_dataset import _read_single_stream_stats
        except (ImportError, ModuleNotFoundError):
            pytest.skip("requires torchtitan")

        self._write_v3(tmp_path)
        result = _read_single_stream_stats(str(tmp_path))
        assert result is not None
        assert result["total_tokens"] == 50000
        assert result["avg_tokens_per_sample"] == 500.0

    def test_unknown_tokenizer_raises(self, tmp_path):
        try:
            from titaninfra.data.streaming_dataset import _read_single_stream_stats
        except (ImportError, ModuleNotFoundError):
            pytest.skip("requires torchtitan")

        self._write_v3(tmp_path)
        with pytest.raises(ValueError, match="nonexistent"):
            _read_single_stream_stats(str(tmp_path), tokenizer_name="nonexistent")


# ============================================================================
# ============================================================================


@pytest.mark.xdist_group("streaming_shm")
class TestRecomputeStatsE2E:
    """Tests for test recompute stats e2 e."""

    @pytest.fixture(autouse=True)
    def _acquire_streaming_lock(self):
        """Serialize StreamingDataset access across xdist workers."""
        from tests._mds_utils import _LOCK
        with _LOCK:
            yield

    def test_recompute_matches_original(self):
        pytest.importorskip("streaming")
        pytest.importorskip("transformers")
        from titaninfra.data.recompute_stats import recompute_token_stats

        num_samples, total_tokens, max_tc, min_tc, hist, fine_hist = recompute_token_stats(
            mds_path="data/aime/mds/train",
            tokenizer_path="tests/fixtures/tokenizers/qwen3",
            data_type="text",
        )
        assert num_samples == 96
        assert total_tokens > 200000
        assert max_tc > 0
        assert min_tc > 0
        assert max_tc >= min_tc
        assert sum(hist.values()) == num_samples

    def test_recompute_messages_basic(self):
        """Verify recompute messages basic."""
        pytest.importorskip("streaming")
        pytest.importorskip("transformers")
        from titaninfra.data.recompute_stats import recompute_token_stats

        num_samples, total_tokens, max_tc, min_tc, hist, fine_hist = recompute_token_stats(
            mds_path="data/sft_test/basic/mds/train",
            tokenizer_path="tests/fixtures/tokenizers/qwen3",
            data_type="messages",
            family="qwen3",
        )
        assert num_samples > 0
        assert total_tokens > 0
        assert max_tc > 0
        assert min_tc > 0
        assert sum(hist.values()) == num_samples

    def test_recompute_messages_aime_sft(self):
        """Verify recompute messages aime sft."""
        pytest.importorskip("streaming")
        pytest.importorskip("transformers")
        from titaninfra.data.recompute_stats import recompute_token_stats

        num_samples, total_tokens, max_tc, min_tc, hist, fine_hist = recompute_token_stats(
            mds_path="data/aime/mds_sft/train",
            tokenizer_path="tests/fixtures/tokenizers/qwen3",
            data_type="messages",
            family="qwen3",
        )
        assert num_samples > 0
        assert total_tokens > 0
        assert max_tc > 0
        assert min_tc > 0
        assert sum(hist.values()) == num_samples

    def test_add_tokenizer_to_mds_messages(self, tmp_path):
        """Verify add tokenizer to mds messages."""
        pytest.importorskip("streaming")
        pytest.importorskip("transformers")
        import shutil

        from titaninfra.data.recompute_stats import add_tokenizer_to_mds

        src = "data/sft_test/basic/mds/train"
        dst = str(tmp_path / "mds_copy")
        shutil.copytree(src, dst)

        add_tokenizer_to_mds(
            mds_path=dst,
            tokenizer_path="tests/fixtures/tokenizers/qwen3",
            tokenizer_name="qwen3",
            data_type="messages",
            family="qwen3",
            set_default=True,
        )

        s = StreamStats.load(dst)
        assert s is not None
        assert s.default_tokenizer == "qwen3"
        assert "qwen3" in s.tokenizer_stats
        assert s.tokenizer_stats["qwen3"].total_tokens > 0
        assert s.tokenizer_stats["qwen3"].avg_tokens_per_sample > 0
