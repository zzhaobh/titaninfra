"""TrainConfig must not carry an encoder_mode field."""
from tests.conftest import make_test_train_config
from titaninfra.config.train import TrainConfig


def test_field_absent():
    assert "encoder_mode" not in TrainConfig.model_fields


def test_extra_encoder_mode_ignored_or_rejected():
    """Pickled / JSON configs from past runs may carry encoder_mode.
    Whichever Pydantic policy is in force, instantiation must not raise
    unhandled errors that block resume. Use the factory so the A-class
    required fields are present and the only variable under test is
    encoder_mode handling."""
    import pydantic
    try:
        cfg = make_test_train_config(encoder_mode="jinja", lr=1e-5, steps=10)
        assert not hasattr(cfg, "encoder_mode")
    except pydantic.ValidationError:
        pass
