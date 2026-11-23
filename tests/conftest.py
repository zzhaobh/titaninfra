"""Unit tests for conftest."""

import gc
import glob
import importlib.util
import os
import sys
from multiprocessing.shared_memory import SharedMemory

import pytest

# ============================================================
# ============================================================

_MODEL_DEFAULTS = {
    "dtype": "bf16",
    "param_dtype": "fp32",
    "attn_backend": "flex",
    "attn_mask_type": "causal",
    "ac_mode": "none",
    "compile": False,
}

_TRAIN_DEFAULTS = {
    "local_batch_size": 1,
    "global_batch_size": -1,
    "seq_len": 4096,
    "optimizer_name": "AdamW",
    "lr": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.1,
    "optimizer_state_dtype": "fp32",
    "min_lr_ratio": 0.1,
    "lr_decay_type": "cosine",
    "max_grad_norm": 1.0,
}

_MUON_DEFAULTS = {
    "muon_lr": 0.02,
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_weight_decay": 0.0,
    "muon_min_ndim_size": 4096,
    "muon_scale_factor": 1.0,
    "muon_ns_dtype": "bf16",
    "muon_gather_mode": False,
    "muon_include_router": False,
}


def make_test_model_config(**overrides):
    """Make test model config."""
    from titaninfra.config.model import ModelConfig

    return ModelConfig(**{**_MODEL_DEFAULTS, **overrides})


def make_test_train_config(**overrides):
    """Make test train config."""
    from titaninfra.config.train import TrainConfig

    merged = {**_TRAIN_DEFAULTS, **overrides}
    if "warmup_steps" not in overrides and "warmup_ratio" not in overrides:
        merged["warmup_steps"] = 200
    if merged.get("optimizer_name") == "MuonAdamW":
        merged = {**_MUON_DEFAULTS, **merged}
    return TrainConfig(**merged)


def pytest_configure(config):
    """Auto-download missing tokenizer fixtures at session startup.

    Uses tests/fixtures/tokenizers/download_all.py registry.
    Downloads from HF_ENDPOINT (default hf-mirror.com) if any fixture is missing.
    Runs once per session (xdist workers skip via workerinput check).
    """
    if hasattr(config, "workerinput"):
        return

    # Lightweight release/lint environments should never access the network.
    # Runtime suites with data dependencies opt into fixture materialization.
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("streaming") is None:
        return

    from tests.fixtures.tokenizers.download_all import check_missing, download_all

    missing = check_missing()
    if missing:
        print(f"\n[conftest] Downloading missing tokenizer fixtures: {missing}", file=sys.stderr)
        ok, fail = download_all()
        if fail:
            print(f"[conftest] WARNING: {fail} tokenizer(s) failed to download", file=sys.stderr)
        else:
            print(f"[conftest] All {ok} tokenizer fixtures ready", file=sys.stderr)

    # Build small MDS inputs from tracked raw fixtures when runtime dependencies
    # are available. Static release checks on lightweight hosts skip this path.
    from tests._fixture_mds import ensure_test_mds

    ensure_test_mds()


def _purge_mosaicml_shm() -> int:
    """Purge mosaicml shm."""
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return 0

    cleaned = 0
    for path in glob.glob(os.path.join(shm_dir, "[0-9][0-9][0-9][0-9][0-9][0-9]_*")):
        name = os.path.basename(path)
        try:
            shm = SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
            cleaned += 1
        except (FileNotFoundError, OSError):
            pass
    return cleaned


@pytest.fixture(autouse=True)
def _clean_stale_streaming_shm_fixture(request):
    """Clean stale streaming shm fixture."""
    marker = request.node.get_closest_marker("xdist_group")
    if marker is None or not marker.args or marker.args[0] != "streaming_shm":
        yield
        return

    _purge_mosaicml_shm()
    yield
    gc.collect()
    _purge_mosaicml_shm()
