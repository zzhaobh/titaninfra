"""Unit tests for resume."""

import os

import pytest

from titaninfra.utils.resume import (
    CheckpointSetup,
    _default_is_hf_path,
    _list_checkpoints,
    _resolve_resume_parent,
    resolve_checkpoint_config,
)


class TestCheckpointSetupImmutability:
    """Tests for test checkpoint setup immutability."""

    def test_frozen(self):
        setup = CheckpointSetup(
            ckpt_subdir="ckpt",
            dump_folder="output",
            initial_load_path=None,
            initial_load_in_hf=False,
        )
        with pytest.raises(AttributeError):
            setup.ckpt_subdir = "modified"


class TestResolveResumeParent:
    """Tests for test resolve resume parent."""

    def test_step_directory(self):
        assert _resolve_resume_parent("/mnt/ckpt/step-900") == "/mnt/ckpt"

    def test_step_underscore_directory(self):
        assert _resolve_resume_parent("/mnt/ckpt/step_900") == "/mnt/ckpt"

    def test_trailing_slash(self):
        assert _resolve_resume_parent("/mnt/ckpt/step-900/") == "/mnt/ckpt"

    def test_plain_directory(self):
        assert _resolve_resume_parent("/mnt/ckpt/checkpoints") == "/mnt/ckpt/checkpoints"


class TestListCheckpoints:
    """Tests for test list checkpoints."""

    def test_with_checkpoints(self, tmp_path):
        (tmp_path / "step-100").mkdir()
        (tmp_path / "step_200").mkdir()
        (tmp_path / "other_file.txt").touch()
        result = _list_checkpoints(str(tmp_path))
        assert sorted(result) == ["step-100", "step_200"]

    def test_empty_directory(self, tmp_path):
        assert _list_checkpoints(str(tmp_path)) == []

    def test_nonexistent_directory(self):
        assert _list_checkpoints("/nonexistent/path") == []


class TestDefaultIsHfPath:
    """Tests for test default is hf path."""

    def test_none_path(self):
        assert _default_is_hf_path(None) is False

    def test_empty_string(self):
        assert _default_is_hf_path("") is False

    def test_safetensors_dir(self, tmp_path):
        (tmp_path / "model.safetensors").touch()
        assert _default_is_hf_path(str(tmp_path)) is True

    def test_safetensors_index_dir(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").touch()
        assert _default_is_hf_path(str(tmp_path)) is True

    def test_config_json_dir(self, tmp_path):
        (tmp_path / "config.json").touch()
        assert _default_is_hf_path(str(tmp_path)) is True

    def test_distcp_dir(self, tmp_path):
        (tmp_path / "__0_0.distcp").touch()
        assert _default_is_hf_path(str(tmp_path)) is False

    def test_nonexistent_path_distcp(self):
        assert _default_is_hf_path("/nonexistent/path.distcp") is False

    def test_nonexistent_path_other(self):
        assert _default_is_hf_path("/nonexistent/hf_model") is True


class TestResolveCheckpointConfigNoResume:
    """Tests for test resolve checkpoint config no resume."""

    def test_default_paths_no_hf(self):
        """Verify default paths no hf."""
        setup = resolve_checkpoint_config(
            resume_from=None,
            ckpt_subdir="checkpoints",
            dump_folder="output",
            hf_ckpt_path=None,
            task_name=None,
        )
        assert setup.ckpt_subdir == "checkpoints"
        assert setup.dump_folder == "output"
        assert setup.initial_load_path is None
        assert setup.initial_load_in_hf is False

    def test_hf_ckpt_path_sets_initial_load(self, tmp_path):
        """Verify hf ckpt path sets initial load."""
        hf_dir = tmp_path / "hf_model"
        hf_dir.mkdir()
        (hf_dir / "model.safetensors").touch()

        setup = resolve_checkpoint_config(
            resume_from=None,
            ckpt_subdir="checkpoints",
            dump_folder="output",
            hf_ckpt_path=str(hf_dir),
            task_name=None,
        )
        assert setup.initial_load_path == str(hf_dir)
        assert setup.initial_load_in_hf is True

    def test_task_name_appended_to_dump_folder_only(self):
        """Verify task name appended to dump folder only."""
        setup = resolve_checkpoint_config(
            resume_from=None,
            ckpt_subdir="checkpoints",
            dump_folder="output",
            hf_ckpt_path=None,
            task_name="my_task",
        )
        assert setup.ckpt_subdir == "checkpoints"
        assert setup.dump_folder == "output_my_task"

    def test_task_name_none_no_append(self):
        """Verify task name none no append."""
        setup = resolve_checkpoint_config(
            resume_from=None,
            ckpt_subdir="checkpoints",
            dump_folder="output",
            hf_ckpt_path=None,
            task_name=None,
        )
        assert setup.ckpt_subdir == "checkpoints"
        assert setup.dump_folder == "output"

    def test_custom_is_hf_path_fn(self):
        """Verify custom is hf path fn."""
        setup = resolve_checkpoint_config(
            resume_from=None,
            ckpt_subdir="ckpt",
            dump_folder="out",
            hf_ckpt_path="/some/path",
            task_name=None,
            is_hf_path_fn=lambda p: False,
        )
        assert setup.initial_load_path == "/some/path"
        assert setup.initial_load_in_hf is False

    def test_normpath_cleans_paths(self):
        """Verify normpath cleans paths."""
        setup = resolve_checkpoint_config(
            resume_from=None,
            ckpt_subdir="./checkpoints",
            dump_folder="./output/",
            hf_ckpt_path=None,
            task_name=None,
        )
        assert setup.ckpt_subdir == "checkpoints"
        assert setup.dump_folder == "output"

    def test_normpath_cleans_dotdot(self):
        """Verify normpath cleans dotdot."""
        setup = resolve_checkpoint_config(
            resume_from=None,
            ckpt_subdir="./a/./b",
            dump_folder="output",
            hf_ckpt_path=None,
            task_name=None,
        )
        assert setup.ckpt_subdir == "a/b"


class TestResolveCheckpointConfigResume:
    """Tests for test resolve checkpoint config resume."""

    def test_resume_nonexistent_path_raises(self):
        """Verify resume nonexistent path raises."""
        with pytest.raises(ValueError, match="resume_from 路径不存在"):
            resolve_checkpoint_config(
                resume_from="/nonexistent/checkpoint/path",
                ckpt_subdir="checkpoints",
                dump_folder="output",
                hf_ckpt_path=None,
                task_name=None,
            )

    def test_resume_empty_dir_raises(self, tmp_path):
        """Verify resume empty dir raises."""
        empty_dir = tmp_path / "empty_ckpt"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="没有找到 checkpoint"):
            resolve_checkpoint_config(
                resume_from=str(empty_dir),
                ckpt_subdir="checkpoints",
                dump_folder="output",
                hf_ckpt_path=None,
                task_name=None,
            )

    def test_resume_same_as_ckpt_subdir(self, tmp_path):
        """Verify resume same as ckpt subdir."""
        dump_folder = str(tmp_path / "output")
        ckpt_subdir = "checkpoints"
        full_ckpt_dir = os.path.join(dump_folder, ckpt_subdir)
        os.makedirs(full_ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(full_ckpt_dir, "step-100"))

        setup = resolve_checkpoint_config(
            resume_from=full_ckpt_dir,
            ckpt_subdir=ckpt_subdir,
            dump_folder=dump_folder,
            hf_ckpt_path=None,
            task_name=None,
        )
        assert setup.ckpt_subdir == ckpt_subdir
        assert setup.dump_folder == dump_folder

    def test_resume_different_dir_copies_checkpoint(self, tmp_path):
        """Verify resume different dir copies checkpoint."""
        source_dir = tmp_path / "source_ckpt"
        source_dir.mkdir()
        step_dir = source_dir / "step-500"
        step_dir.mkdir()
        (step_dir / "model.pt").touch()

        dump_folder = str(tmp_path / "output")
        ckpt_subdir = "new_checkpoints"

        resolve_checkpoint_config(
            resume_from=str(source_dir),
            ckpt_subdir=ckpt_subdir,
            dump_folder=dump_folder,
            hf_ckpt_path=None,
            task_name=None,
        )

        full_ckpt_dir = os.path.join(dump_folder, ckpt_subdir)
        assert os.path.isdir(full_ckpt_dir)
        assert not os.path.islink(full_ckpt_dir)
        assert os.path.isdir(os.path.join(full_ckpt_dir, "step-500"))
        assert os.path.isfile(os.path.join(full_ckpt_dir, "step-500", "model.pt"))

    def test_resume_copies_latest_step_only(self, tmp_path):
        """Verify resume copies latest step only."""
        source_dir = tmp_path / "source_ckpt"
        source_dir.mkdir()
        (source_dir / "step-100").mkdir()
        (source_dir / "step-500").mkdir()
        (source_dir / "step-500" / "model.pt").touch()

        dump_folder = str(tmp_path / "output")
        ckpt_subdir = "new_ckpt"

        resolve_checkpoint_config(
            resume_from=str(source_dir),
            ckpt_subdir=ckpt_subdir,
            dump_folder=dump_folder,
            hf_ckpt_path=None,
            task_name=None,
        )

        full_ckpt_dir = os.path.join(dump_folder, ckpt_subdir)
        assert os.path.isdir(os.path.join(full_ckpt_dir, "step-500"))
        assert not os.path.exists(os.path.join(full_ckpt_dir, "step-100"))

    def test_resume_step_dir_resolves_parent(self, tmp_path):
        """Verify resume step dir resolves parent."""
        ckpt_root = tmp_path / "ckpt_root"
        ckpt_root.mkdir()
        step_dir = ckpt_root / "step-900"
        step_dir.mkdir()
        (step_dir / "model.pt").touch()

        dump_folder = str(tmp_path / "output")
        ckpt_subdir = "new_ckpt"

        resolve_checkpoint_config(
            resume_from=str(step_dir),
            ckpt_subdir=ckpt_subdir,
            dump_folder=dump_folder,
            hf_ckpt_path=None,
            task_name=None,
        )

        full_ckpt_dir = os.path.join(dump_folder, ckpt_subdir)
        assert os.path.isdir(full_ckpt_dir)
        assert not os.path.islink(full_ckpt_dir)
        assert os.path.isdir(os.path.join(full_ckpt_dir, "step-900"))

    def test_resume_target_has_existing_checkpoints(self, tmp_path):
        """Verify resume target has existing checkpoints."""
        source_dir = tmp_path / "source_ckpt"
        source_dir.mkdir()
        (source_dir / "step-100").mkdir()

        dump_folder = str(tmp_path / "output")
        ckpt_subdir = "existing_ckpt"
        full_ckpt_dir = os.path.join(dump_folder, ckpt_subdir)
        os.makedirs(full_ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(full_ckpt_dir, "step-200"))

        resolve_checkpoint_config(
            resume_from=str(source_dir),
            ckpt_subdir=ckpt_subdir,
            dump_folder=dump_folder,
            hf_ckpt_path=None,
            task_name=None,
        )

        assert not os.path.islink(full_ckpt_dir)
        assert os.path.isdir(full_ckpt_dir)
        assert os.path.isdir(os.path.join(full_ckpt_dir, "step-200"))
        assert not os.path.exists(os.path.join(full_ckpt_dir, "step-100"))

    def test_resume_target_empty_dir_gets_copy(self, tmp_path):
        """Verify resume target empty dir gets copy."""
        source_dir = tmp_path / "source_ckpt"
        source_dir.mkdir()
        (source_dir / "step-100").mkdir()
        (source_dir / "step-100" / "model.pt").touch()

        dump_folder = str(tmp_path / "output")
        ckpt_subdir = "empty_target"
        full_ckpt_dir = os.path.join(dump_folder, ckpt_subdir)
        os.makedirs(full_ckpt_dir, exist_ok=True)

        resolve_checkpoint_config(
            resume_from=str(source_dir),
            ckpt_subdir=ckpt_subdir,
            dump_folder=dump_folder,
            hf_ckpt_path=None,
            task_name=None,
        )

        assert os.path.isdir(full_ckpt_dir)
        assert not os.path.islink(full_ckpt_dir)
        assert os.path.isdir(os.path.join(full_ckpt_dir, "step-100"))

    def test_resume_with_task_name(self, tmp_path):
        """Verify resume with task name."""
        source_dir = tmp_path / "source_ckpt"
        source_dir.mkdir()
        (source_dir / "step-100").mkdir()
        (source_dir / "step-100" / "model.pt").touch()

        dump_folder = str(tmp_path / "output")
        ckpt_subdir = "checkpoints"

        setup = resolve_checkpoint_config(
            resume_from=str(source_dir),
            ckpt_subdir=ckpt_subdir,
            dump_folder=dump_folder,
            hf_ckpt_path=None,
            task_name="finetune_v2",
        )

        assert setup.ckpt_subdir == "checkpoints"
        assert setup.dump_folder == f"{dump_folder}_finetune_v2"


class TestResolveCheckpointConfigStage:
    """Tests for test resolve checkpoint config stage."""

    def test_sft_no_weights_logs_warning(self, tmp_path, caplog):
        """Verify sft no weights logs warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            resolve_checkpoint_config(
                resume_from=None,
                ckpt_subdir="ckpt",
                dump_folder=str(tmp_path / "output"),
                hf_ckpt_path=None,
                task_name=None,
                stage="sft",
            )
        assert "SFT 通常需要从预训练权重初始化" in caplog.text

    def test_pretrain_no_weights_logs_info(self, tmp_path, caplog):
        """Verify pretrain no weights logs info."""
        import logging
        with caplog.at_level(logging.INFO):
            resolve_checkpoint_config(
                resume_from=None,
                ckpt_subdir="ckpt",
                dump_folder=str(tmp_path / "output"),
                hf_ckpt_path=None,
                task_name=None,
                stage="pretrain",
            )
        assert "将从随机权重开始预训练" in caplog.text
