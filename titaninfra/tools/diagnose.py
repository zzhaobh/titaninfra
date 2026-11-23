"""Validate Hugging Face checkpoint structure and common path issues."""

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_REQUIRED_FILES = ("config.json", "model.safetensors", "model.safetensors.index.json",
                    "tokenizer.json", "tokenizer_config.json")
_OPTIONAL_FILES = ("vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json")


@dataclass
class DiagnosisResult:
    """Structured result returned by diagnose_hf_checkpoint."""
    path_exists: bool
    is_directory: bool
    file_status: dict[str, bool] = field(default_factory=dict)
    has_shards: bool = False
    missing_files: list[str] = field(default_factory=list)
    is_hf_format: bool = False
    model_config: dict | None = None
    issues: list[str] = field(default_factory=list)
    all_pass: bool = False


def _check_hf_files(hf_path: str) -> tuple[dict[str, bool], bool, list[str]]:
    """Check required files and return file status, shards, and missing files."""
    file_status: dict[str, bool] = {}
    has_shards = False

    try:
        files = set(os.listdir(hf_path))
    except Exception as e:
        logger.error(f"无法读取目录内容: {e}")
        return {f: False for f in _REQUIRED_FILES + _OPTIONAL_FILES}, False, list(_REQUIRED_FILES)

    for f in _REQUIRED_FILES + _OPTIONAL_FILES:
        file_status[f] = f in files

    has_shards = any(f.startswith("model-") and f.endswith(".safetensors") for f in files)

    missing = []
    if not file_status["config.json"]:
        missing.append("config.json")

    has_weights = file_status["model.safetensors"] or file_status["model.safetensors.index.json"] or has_shards
    if not has_weights:
        missing.append("model weights (model.safetensors / index.json / shards)")

    if not (file_status["tokenizer.json"] or file_status["tokenizer_config.json"]):
        missing.append("tokenizer (tokenizer.json / tokenizer_config.json)")

    return file_status, has_shards, missing


def _read_model_config(hf_path: str) -> dict | None:
    """Read common architecture fields from config.json, or return None."""
    config_path = os.path.join(hf_path, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        return {
            "model_type": config.get("model_type"),
            "vocab_size": config.get("vocab_size"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
        }
    except Exception:
        return None


def _check_common_issues(hf_path: str) -> list[str]:
    """Return common symlink, permission, and path-format issues."""
    issues = []
    if os.path.islink(hf_path):
        issues.append(f"path is a symlink -> {os.readlink(hf_path)}")
    try:
        with open(os.path.join(hf_path, "config.json")):
            pass
    except PermissionError:
        issues.append("permission denied on config.json")
    except FileNotFoundError:
        pass
    except Exception as e:
        issues.append(f"cannot read config.json: {e}")
    if " " in hf_path or "(" in hf_path or ")" in hf_path:
        issues.append("path contains special characters")
    return issues


def diagnose_hf_checkpoint(
    hf_path: str,
    model_name: str | None = None,
    verbose: bool = False,
) -> DiagnosisResult:
    """Diagnose HF checkpoint existence, files, format, and path issues.

    Set verbose=True to print a formatted report. The returned DiagnosisResult
    has all_pass=True only when all checks succeed.
    """
    from titaninfra.utils.resume import _default_is_hf_path

    result = DiagnosisResult(
        path_exists=os.path.exists(hf_path),
        is_directory=os.path.isdir(hf_path) if os.path.exists(hf_path) else False,
    )

    if not result.path_exists or not result.is_directory:
        if verbose:
            print(f"path does not exist or is not a directory: {hf_path}")
        return result

    result.file_status, result.has_shards, result.missing_files = _check_hf_files(hf_path)
    result.is_hf_format = _default_is_hf_path(hf_path)
    result.model_config = _read_model_config(hf_path)
    result.issues = _check_common_issues(hf_path)
    result.all_pass = (
        result.path_exists
        and not result.missing_files
        and result.is_hf_format
        and not result.issues
    )

    if verbose:
        _print_report(result, hf_path, model_name)

    return result


def _print_report(result: DiagnosisResult, hf_path: str, model_name: str | None) -> None:
    """Print a formatted diagnosis report."""
    print("=" * 80)
    print(f"  HF Checkpoint Diagnosis: {hf_path}")
    if model_name:
        print(f"  Model: {model_name}")
    print("=" * 80)

    _ok = "✅"
    _fail = "❌"

    print(f"\n  Path exists: {_ok if result.path_exists else _fail}")

    if result.file_status:
        print("\n  Required files:")
        for f in _REQUIRED_FILES:
            status = _ok if result.file_status.get(f) else _fail
            print(f"    {status} {f}")
        if result.has_shards:
            print(f"    {_ok} sharded model files detected")

    if result.missing_files:
        print(f"\n  {_fail} Missing: {', '.join(result.missing_files)}")

    print(f"\n  HF format detection: {_ok if result.is_hf_format else _fail}")

    if result.model_config:
        print("\n  Model config:")
        for k, v in result.model_config.items():
            if v is not None:
                print(f"    {k}: {v}")

    if result.issues:
        print("\n  Issues found:")
        for issue in result.issues:
            print(f"    - {issue}")

    print(f"\n  {'=' * 78}")
    print(f"  Result: {_ok + ' ALL CHECKS PASSED' if result.all_pass else _fail + ' ISSUES FOUND'}")
    print(f"  {'=' * 78}")
