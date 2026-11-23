"""Tests for checkpoint-related configuration options."""

from tests.conftest import make_test_train_config


def test_config_with_checkpoint_options():
    """Checkpoint options are preserved on the configuration object."""
    config = make_test_train_config(
        ckpt_subdir="checkpoints/test",
        ckpt_interval=100,
        resume_from=None,
    )

    assert config.ckpt_subdir == "checkpoints/test"
    assert config.ckpt_interval == 100
    assert config.resume_from is None


def test_config_serialization():
    """Configuration values survive a dump-and-reload round trip."""
    config = make_test_train_config(ckpt_interval=500)

    config_dict = config.model_dump()
    assert config_dict["ckpt_interval"] == 500

    config2 = make_test_train_config(**config_dict)
    assert config2.ckpt_interval == 500
