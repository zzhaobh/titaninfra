"""Unit tests for diagnose."""

import os

import pytest

from titaninfra.tools.diagnose import DiagnosisResult, diagnose_hf_checkpoint


@pytest.fixture()
def hf_dir(tmp_path):
    """Hf dir."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "qwen3", "vocab_size": 151936}')
    (d / "model.safetensors.index.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\x00")
    (d / "model-00002-of-00002.safetensors").write_bytes(b"\x00")
    (d / "tokenizer.json").write_text("{}")
    (d / "tokenizer_config.json").write_text("{}")
    return str(d)


def test_all_pass(hf_dir):
    result = diagnose_hf_checkpoint(hf_dir)
    assert isinstance(result, DiagnosisResult)
    assert result.path_exists is True
    assert result.is_hf_format is True
    assert result.missing_files == []
    assert result.all_pass is True
    assert result.has_shards is True


def test_nonexistent_path(tmp_path):
    result = diagnose_hf_checkpoint(str(tmp_path / "nope"))
    assert result.path_exists is False
    assert result.all_pass is False


def test_missing_weights(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    result = diagnose_hf_checkpoint(str(d))
    assert result.path_exists is True
    assert len(result.missing_files) > 0
    assert result.all_pass is False


def test_missing_tokenizer(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    result = diagnose_hf_checkpoint(str(d))
    assert any("tokenizer" in m for m in result.missing_files)
    assert result.all_pass is False


def test_model_config_parsed(hf_dir):
    result = diagnose_hf_checkpoint(hf_dir)
    assert result.model_config is not None
    assert result.model_config["model_type"] == "qwen3"
    assert result.model_config["vocab_size"] == 151936


def test_verbose_prints(hf_dir, capsys):
    diagnose_hf_checkpoint(hf_dir, verbose=True)
    captured = capsys.readouterr()
    assert "ALL CHECKS PASSED" in captured.out


def test_special_chars_in_path(tmp_path):
    d = tmp_path / "model (copy)"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    (d / "tokenizer.json").write_text("{}")
    result = diagnose_hf_checkpoint(str(d))
    assert "special characters" in result.issues[0]


def test_single_file_model(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    (d / "tokenizer_config.json").write_text("{}")
    result = diagnose_hf_checkpoint(str(d))
    assert result.has_shards is False
    assert result.all_pass is True


def test_symlink_detection(tmp_path):
    real = tmp_path / "real_model"
    real.mkdir()
    (real / "config.json").write_text("{}")
    (real / "model.safetensors").write_bytes(b"\x00")
    (real / "tokenizer.json").write_text("{}")

    link = tmp_path / "link_model"
    os.symlink(str(real), str(link))

    result = diagnose_hf_checkpoint(str(link))
    assert any("symlink" in i for i in result.issues)
