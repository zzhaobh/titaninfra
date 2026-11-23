"""Unit tests for a5 data order."""

import os

import pytest

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
        reason="A5 需要 GPU 环境 + MDS 数据集",
    ),
    pytest.mark.xdist_group("a5_shared_mds_dataloader"),
]


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AIME_TRAIN = os.path.join(REPO_ROOT, "data/aime/mds/train")
TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH",
    os.path.join(REPO_ROOT, "tests/fixtures/tokenizers/qwen3"),
)


def _has_mds_data():
    return os.path.isdir(AIME_TRAIN) and os.path.exists(
        os.path.join(AIME_TRAIN, "index.json")
    )


@pytest.mark.skipif(not _has_mds_data(), reason="MDS 数据不存在")
class TestA5DataOrder:
    """Tests for test a5 data order."""

    def _build_dataloader(self, seed: int = 42, num_batches: int = 10):
        """Build dataloader."""
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        from titaninfra.config.data import StreamConfig
        from titaninfra.data.streaming_dataset import StreamingTextDataLoader

        stream = StreamConfig(
            tokenizer_name="qwen3",
            local=AIME_TRAIN,
            data_type="text",
            epoch_size_margin=2.0,
        )

        config = StreamingTextDataLoader.Config(
            streams=[stream],
            shuffle_seed=seed,
            shuffle=True,
            num_workers=0,
            repeat=False,
        )

        tokenizer = HuggingFaceTokenizer(tokenizer_path=TOKENIZER_PATH)

        loader = StreamingTextDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            max_context_length=512,
            num_tokens_per_microbatch=512 * 2,
        )

        batches = []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            # titaninfra/data/streaming_dataset.py::_FlatTokenCollator）。
            batches.append((batch.input.tolist(), batch.labels.tolist()))
        return batches

    def test_deterministic_with_same_seed(self):
        """Verify deterministic with same seed."""
        batches_1 = self._build_dataloader(seed=42, num_batches=10)
        batches_2 = self._build_dataloader(seed=42, num_batches=10)

        assert len(batches_1) == len(batches_2) == 10

        for i, (b1, b2) in enumerate(zip(batches_1, batches_2, strict=False)):
            assert b1[0] == b2[0], f"batch {i} input_ids 不一致"
            assert b1[1] == b2[1], f"batch {i} labels 不一致"

    def test_different_seed_different_order(self):
        """Verify different seed different order."""
        batches_42 = self._build_dataloader(seed=42, num_batches=5)
        batches_99 = self._build_dataloader(seed=99, num_batches=5)

        any_different = False
        for b42, b99 in zip(batches_42, batches_99, strict=False):
            if b42[0] != b99[0]:
                any_different = True
                break

        assert any_different, "不同 seed 应产生不同数据顺序"
