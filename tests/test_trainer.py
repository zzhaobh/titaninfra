"""Unit tests for trainer."""

from unittest.mock import patch

import pytest

from tests.conftest import make_test_model_config, make_test_train_config


class TestTrainResult:
    def test_creation(self):
        """Verify creation."""
        from titaninfra.trainer import TrainResult
        r = TrainResult(stage="pretrain", ckpt_dir="./ckpts", steps=1000, model_name="qwen3/8B")
        assert r.stage == "pretrain"
        assert r.ckpt_dir == "./ckpts"
        assert r.steps == 1000
        assert r.model_name == "qwen3/8B"

    def test_repr(self):
        """Verify repr."""
        from titaninfra.trainer import TrainResult
        r = TrainResult(stage="sft", ckpt_dir="./ckpts", steps=500, model_name="llama3/8B")
        repr_str = repr(r)
        assert "sft" in repr_str
        assert "llama3/8B" in repr_str



class TestTrainerRouting:
    """Tests for test trainer routing."""

    @patch("titaninfra.engine.pretrain.run_train")
    def test_train_config_routes_to_run_train(self, mock_run):
        """Verify train config routes to run train."""
        from titaninfra import StreamConfig, Trainer
        from titaninfra.trainer import TrainResult

        mock_run.return_value = TrainResult("train", "./ckpts", 1000, "qwen3/8B")

        result = Trainer(
            model=make_test_model_config(name="qwen3/8B"),
            train=make_test_train_config(steps=100),
            streams=[StreamConfig(tokenizer_name="qwen3", local="/tmp/data")],
        )

        mock_run.assert_called_once()
        assert result.stage == "train"

    @patch("titaninfra.engine.pretrain.run_train")
    def test_train_config_with_sft_fields(self, mock_run):
        """TrainConfig with SFT fields routes to run_train"""
        from titaninfra import StreamConfig, Trainer
        from titaninfra.trainer import TrainResult

        mock_run.return_value = TrainResult("train", "./ckpts", 500, "qwen3/8B")

        result = Trainer(
            model=make_test_model_config(name="qwen3/8B"),
            train=make_test_train_config(steps=100),
            streams=[StreamConfig(tokenizer_name="qwen3", family="qwen3", local="/tmp/sft", data_type="messages", add_bos_eos=False)],
        )

        mock_run.assert_called_once()
        assert result.stage == "train"

    def test_invalid_train_type_raises(self):
        """Verify invalid train type raises."""
        from titaninfra import StreamConfig, Trainer

        with pytest.raises(TypeError, match="无法从 train 类型"):
            Trainer(
                model=make_test_model_config(name="qwen3/8B"),
                train="not_a_config",
                streams=[StreamConfig(tokenizer_name="qwen3", local="/tmp/data")],
            )



class TestMultiStreamRouting:
    @patch("titaninfra.engine.pretrain.run_train")
    def test_multi_stream_cpt(self, mock_run):
        """Verify multi stream cpt."""
        from titaninfra import StreamConfig, Trainer
        from titaninfra.trainer import TrainResult

        mock_run.return_value = TrainResult("train", "./ckpts", 1000, "qwen3/8B")

        Trainer(
            model=make_test_model_config(name="qwen3/8B"),
            train=make_test_train_config(lr=5e-5, steps=100),
            streams=[
                StreamConfig(tokenizer_name="qwen3", local="/data/text", proportion=0.7),
                StreamConfig(tokenizer_name="qwen3", local="/data/sft", data_type="messages", add_bos_eos=False, proportion=0.3),
            ],
        )
        mock_run.assert_called_once()


def test_mixed_stream_stage_is_cpt():
    """Text+message streams produce CPT, not SFT checkpoint/result metadata."""
    from titaninfra import StreamConfig
    from titaninfra.engine.pretrain import _infer_stage

    text = StreamConfig(tokenizer_name="qwen3", local="/data/text", proportion=0.7)
    messages = StreamConfig(
        tokenizer_name="qwen3",
        local="/data/sft",
        data_type="messages",
        family="qwen3",
        add_bos_eos=False,
        proportion=0.3,
    )
    assert _infer_stage([text]) == "pretrain"
    assert _infer_stage([messages]) == "sft"
    assert _infer_stage([text, messages]) == "cpt"
