"""Unit tests for wandb resume."""

import json
import os

import pytest

from tests.conftest import make_test_train_config
from titaninfra.utils.wandb import WANDB_STATE_FILE, _load_wandb_run_id, save_wandb_run_id, setup_wandb_env


@pytest.fixture(autouse=True)
def _clean_wandb_env(monkeypatch):
    """Clean wandb env."""
    for key in [
        "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_PROJECT", "WANDB_TEAM",
        "WANDB_RUN_NAME", "WANDB_RUN_TAGS", "WANDB_RUN_NOTES", "WANDB_MODE",
    ]:
        monkeypatch.delenv(key, raising=False)


def _make_config(**overrides):
    """Make config."""
    defaults = {"enable_wandb": True}
    defaults.update(overrides)
    return make_test_train_config(**defaults)


# ============================================================================
# _load_wandb_run_id / save_wandb_run_id
# ============================================================================


class TestWandBRunIdPersistence:
    """Tests for test wand brun id persistence."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Verify save and load roundtrip."""
        path = tmp_path / WANDB_STATE_FILE
        with open(path, "w") as f:
            json.dump({"run_id": "abc123"}, f)

        result = _load_wandb_run_id(str(tmp_path))
        assert result == "abc123"

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Verify load nonexistent returns none."""
        result = _load_wandb_run_id(str(tmp_path))
        assert result is None

    def test_load_corrupted_json_returns_none(self, tmp_path):
        """Verify load corrupted json returns none."""
        path = tmp_path / WANDB_STATE_FILE
        path.write_text("{invalid json")

        result = _load_wandb_run_id(str(tmp_path))
        assert result is None

    def test_load_missing_run_id_key_returns_none(self, tmp_path):
        """Verify load missing run id key returns none."""
        path = tmp_path / WANDB_STATE_FILE
        with open(path, "w") as f:
            json.dump({"other_key": "value"}, f)

        result = _load_wandb_run_id(str(tmp_path))
        assert result is None

    def test_save_without_wandb_no_crash(self, tmp_path):
        """Verify save without wandb no crash."""
        save_wandb_run_id(str(tmp_path))
        assert not (tmp_path / WANDB_STATE_FILE).exists()


# ============================================================================
# ============================================================================


class TestSetupWandBEnvResume:
    """Tests for test setup wand benv resume."""

    def test_resume_auto_loads_run_id(self, tmp_path):
        """Verify resume auto loads run id."""
        path = tmp_path / WANDB_STATE_FILE
        with open(path, "w") as f:
            json.dump({"run_id": "prev_run_xyz"}, f)

        config = _make_config()
        setup_wandb_env(config, dump_folder=str(tmp_path), is_resume=True)

        assert os.environ.get("WANDB_RUN_ID") == "prev_run_xyz"
        assert os.environ.get("WANDB_RESUME") == "allow"

    def test_user_run_id_takes_precedence(self, tmp_path):
        """Verify user run id takes precedence."""
        path = tmp_path / WANDB_STATE_FILE
        with open(path, "w") as f:
            json.dump({"run_id": "old_run"}, f)

        config = _make_config(wandb_run_id="user_specified_run")
        setup_wandb_env(config, dump_folder=str(tmp_path), is_resume=True)

        assert os.environ.get("WANDB_RUN_ID") == "user_specified_run"
        assert os.environ.get("WANDB_RESUME") == "allow"

    def test_no_resume_no_wandb_resume_env(self, tmp_path):
        """Verify no resume no wandb resume env."""
        config = _make_config()
        setup_wandb_env(config, dump_folder=str(tmp_path), is_resume=False)

        assert os.environ.get("WANDB_RESUME") is None
        assert os.environ.get("WANDB_RUN_ID") is None

    def test_resume_no_state_file_no_crash(self, tmp_path):
        """Verify resume no state file no crash."""
        config = _make_config()
        setup_wandb_env(config, dump_folder=str(tmp_path), is_resume=True)

        assert os.environ.get("WANDB_RESUME") is None

    def test_wandb_disabled_no_env_set(self, tmp_path):
        """Verify wandb disabled no env set."""
        path = tmp_path / WANDB_STATE_FILE
        with open(path, "w") as f:
            json.dump({"run_id": "should_not_be_used"}, f)

        config = _make_config(enable_wandb=False)
        setup_wandb_env(config, dump_folder=str(tmp_path), is_resume=True)

        assert os.environ.get("WANDB_RUN_ID") is None
        assert os.environ.get("WANDB_RESUME") is None

    def test_first_run_with_explicit_id_sets_resume(self):
        """Verify first run with explicit id sets resume."""
        config = _make_config(wandb_run_id="my_run_001")
        setup_wandb_env(config, dump_folder="", is_resume=False)

        assert os.environ.get("WANDB_RUN_ID") == "my_run_001"
        assert os.environ.get("WANDB_RESUME") == "allow"
