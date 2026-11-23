"""Unit tests for wandb utils."""

import importlib
import os
import sys
from unittest.mock import MagicMock

from tests.conftest import make_test_train_config


class TestSetupWandbEnv:
    """Tests for test setup wandb env."""

    def _make_config(self, **overrides):
        """Make config."""
        defaults = {
            "enable_wandb": True,
            "wandb_project": None,
            "wandb_entity": None,
            "wandb_run_name": None,
            "wandb_run_id": None,
            "wandb_tags": None,
            "wandb_notes": None,
            "wandb_mode": None,
        }
        defaults.update(overrides)
        config = MagicMock()
        for key, value in defaults.items():
            setattr(config, key, value)
        return config

    def test_disabled_wandb_sets_nothing(self, monkeypatch):
        """Verify disabled wandb sets nothing."""
        from titaninfra.utils.wandb import setup_wandb_env

        for key in ["WANDB_PROJECT", "WANDB_TEAM", "WANDB_RUN_NAME"]:
            monkeypatch.delenv(key, raising=False)

        config = self._make_config(enable_wandb=False, wandb_project="should-not-set")
        setup_wandb_env(config)

        assert "WANDB_PROJECT" not in os.environ
        assert "WANDB_TEAM" not in os.environ
        assert "WANDB_RUN_NAME" not in os.environ

    def test_enabled_wandb_sets_non_none_fields(self, monkeypatch):
        """Verify enabled wandb sets non none fields."""
        from titaninfra.utils.wandb import setup_wandb_env

        env_keys = [
            "WANDB_PROJECT", "WANDB_TEAM", "WANDB_RUN_NAME",
            "WANDB_RUN_ID", "WANDB_RUN_TAGS", "WANDB_RUN_NOTES",
            "WANDB_MODE",
        ]
        for key in env_keys:
            monkeypatch.delenv(key, raising=False)

        config = self._make_config(
            enable_wandb=True,
            wandb_project="my-project",
            wandb_entity="my-team",
            wandb_run_name="run-001",
            wandb_mode="offline",
        )
        setup_wandb_env(config)

        assert os.environ["WANDB_PROJECT"] == "my-project"
        assert os.environ["WANDB_TEAM"] == "my-team"
        assert os.environ["WANDB_RUN_NAME"] == "run-001"
        assert os.environ["WANDB_MODE"] == "offline"
        assert "WANDB_RUN_ID" not in os.environ
        assert "WANDB_RUN_TAGS" not in os.environ
        assert "WANDB_RUN_NOTES" not in os.environ

    def test_all_fields_set(self, monkeypatch):
        """Verify all fields set."""
        from titaninfra.utils.wandb import setup_wandb_env

        env_keys = [
            "WANDB_PROJECT", "WANDB_TEAM", "WANDB_RUN_NAME",
            "WANDB_RUN_ID", "WANDB_RUN_TAGS", "WANDB_RUN_NOTES",
            "WANDB_MODE",
        ]
        for key in env_keys:
            monkeypatch.delenv(key, raising=False)

        config = self._make_config(
            enable_wandb=True,
            wandb_project="proj",
            wandb_entity="team",
            wandb_run_name="run",
            wandb_run_id="abc123",
            wandb_tags="tag1,tag2",
            wandb_notes="test run notes",
            wandb_mode="online",
        )
        setup_wandb_env(config)

        assert os.environ["WANDB_PROJECT"] == "proj"
        assert os.environ["WANDB_TEAM"] == "team"
        assert os.environ["WANDB_RUN_NAME"] == "run"
        assert os.environ["WANDB_RUN_ID"] == "abc123"
        assert os.environ["WANDB_RUN_TAGS"] == "tag1,tag2"
        assert os.environ["WANDB_RUN_NOTES"] == "test run notes"
        assert os.environ["WANDB_MODE"] == "online"

    def test_none_fields_skipped(self, monkeypatch):
        """Verify none fields skipped."""
        from titaninfra.utils.wandb import setup_wandb_env

        env_keys = [
            "WANDB_PROJECT", "WANDB_TEAM", "WANDB_RUN_NAME",
            "WANDB_RUN_ID", "WANDB_RUN_TAGS", "WANDB_RUN_NOTES",
            "WANDB_MODE",
        ]
        for key in env_keys:
            monkeypatch.delenv(key, raising=False)

        config = self._make_config(enable_wandb=True)
        setup_wandb_env(config)

        for key in env_keys:
            assert key not in os.environ

    def test_uses_real_train_config(self, monkeypatch):
        """Verify uses real train config."""
        from titaninfra.utils.wandb import setup_wandb_env

        for key in ["WANDB_PROJECT", "WANDB_TEAM", "WANDB_MODE"]:
            monkeypatch.delenv(key, raising=False)

        config = make_test_train_config(
            enable_wandb=True,
            wandb_project="real-project",
            wandb_mode="dryrun",
        )
        setup_wandb_env(config)

        assert os.environ["WANDB_PROJECT"] == "real-project"
        assert os.environ["WANDB_MODE"] == "dryrun"




class TestBuildMetricsConfig:
    """Tests for test build metrics config."""

    def _call_build_metrics_config(self, train_config, mock_metrics_module):
        """Helper: call build_metrics_config with mocked torchtitan module."""
        # Ensure the mock is in sys.modules when the function's local import runs
        saved = sys.modules.get("torchtitan.observability.metrics")
        sys.modules["torchtitan.observability.metrics"] = mock_metrics_module
        try:
            # Reload to ensure the function picks up the mock on local import
            import titaninfra.utils.wandb as wandb_mod
            importlib.reload(wandb_mod)
            return wandb_mod.build_metrics_config(train_config)
        finally:
            if saved is not None:
                sys.modules["torchtitan.observability.metrics"] = saved
            else:
                sys.modules.pop("torchtitan.observability.metrics", None)

    def _mock_metrics_module(self):
        """A mock torchtitan.observability.metrics module with real-shaped
        (but identity-trackable) MetricsProcessor/TensorBoardLogger/WandBLogger."""
        mock_metrics_module = MagicMock()
        mock_metrics_module.MetricsProcessor.Config = MagicMock()

        class _TBConfig:
            pass

        class _WandBConfig:
            pass

        mock_metrics_module.TensorBoardLogger.Config = _TBConfig
        mock_metrics_module.WandBLogger.Config = _WandBConfig
        return mock_metrics_module, _TBConfig, _WandBConfig

    def test_returns_config_with_wandb_enabled(self):
        """Verify returns config with wandb enabled."""
        mock_metrics_module, tb_cls, wandb_cls = self._mock_metrics_module()
        mock_config_class = mock_metrics_module.MetricsProcessor.Config

        train_config = MagicMock()
        train_config.log_interval = 10
        train_config.enable_wandb = True

        self._call_build_metrics_config(train_config, mock_metrics_module)

        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args[1]
        assert call_kwargs["log_freq"] == 10
        assert call_kwargs["disable_color_printing"] is False
        loggers = call_kwargs["loggers"]
        assert len(loggers) == 2
        assert isinstance(loggers[0], tb_cls)
        assert isinstance(loggers[1], wandb_cls)

    def test_returns_config_with_wandb_disabled(self):
        """Verify returns config with wandb disabled."""
        mock_metrics_module, tb_cls, wandb_cls = self._mock_metrics_module()
        mock_config_class = mock_metrics_module.MetricsProcessor.Config

        train_config = MagicMock()
        train_config.log_interval = 50
        train_config.enable_wandb = False

        self._call_build_metrics_config(train_config, mock_metrics_module)

        call_kwargs = mock_config_class.call_args[1]
        assert call_kwargs["log_freq"] == 50
        assert call_kwargs["disable_color_printing"] is False
        loggers = call_kwargs["loggers"]
        assert len(loggers) == 1
        assert isinstance(loggers[0], tb_cls)

    def test_log_freq_matches_config(self):
        """Verify log freq matches config."""
        mock_metrics_module, _, _ = self._mock_metrics_module()
        mock_config_class = mock_metrics_module.MetricsProcessor.Config

        train_config = MagicMock()
        train_config.log_interval = 25
        train_config.enable_wandb = False

        self._call_build_metrics_config(train_config, mock_metrics_module)

        call_kwargs = mock_config_class.call_args[1]
        assert call_kwargs["log_freq"] == 25
