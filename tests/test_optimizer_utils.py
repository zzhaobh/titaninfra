"""Unit tests for optimizer utils."""

import torch.nn as nn

from tests.conftest import make_test_train_config
from titaninfra.utils.optimizer_utils import _build_param_lr_map


class _TinyModel(nn.Module):
    """Tests for tiny model."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.router = nn.Linear(4, 2, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.embed(x).mean(dim=1))



def test_build_param_lr_map_all_base():
    """Verify build param lr map all base."""
    model = _TinyModel()
    lr_map = _build_param_lr_map(model, {}, base_lr=1e-4)
    assert all(v == 1e-4 for v in lr_map.values())


def test_build_param_lr_map_pattern_match():
    """Verify build param lr map pattern match."""
    model = _TinyModel()
    overrides = {"router": 1e-3, "embed": 1e-5}
    lr_map = _build_param_lr_map(model, overrides, base_lr=1e-4)

    for param_name, param in model.named_parameters():
        lr = lr_map[param]
        if "router" in param_name:
            assert lr == 1e-3, f"{param_name} should use 1e-3"
        elif "embed" in param_name:
            assert lr == 1e-5, f"{param_name} should use 1e-5"
        else:
            assert lr == 1e-4, f"{param_name} should use base 1e-4"


def test_build_param_lr_map_first_match_wins():
    """Verify build param lr map first match wins."""
    model = _TinyModel()
    overrides = {"router": 1e-3, "r": 1e-6}
    lr_map = _build_param_lr_map(model, overrides, base_lr=1e-4)
    assert lr_map[model.router.weight] == 1e-3


def test_build_param_lr_map_skips_frozen():
    """Verify build param lr map skips frozen."""
    model = _TinyModel()
    model.embed.weight.requires_grad = False
    lr_map = _build_param_lr_map(model, {}, base_lr=1e-4)
    assert model.embed.weight not in lr_map
    assert model.head.weight in lr_map



class TestLayerLrOverrides:
    def _cfg(self, **kw):
        return make_test_train_config(lr=2e-6, seq_len=4096, global_batch_size=8, **kw)

    def test_multiplier_stored_as_given(self):
        """Verify multiplier stored as given."""
        cfg = self._cfg(layer_lr_overrides={"router": 0.1, "embed": 0.01})
        assert cfg.layer_lr_overrides["router"] == 0.1
        assert cfg.layer_lr_overrides["embed"] == 0.01

    def test_none_by_default(self):
        """Verify none by default."""
        cfg = self._cfg()
        assert cfg.layer_lr_overrides is None

    def test_zero_multiplier_allowed(self):
        """Verify zero multiplier allowed."""
        cfg = self._cfg(layer_lr_overrides={"router": 0.0})
        assert cfg.layer_lr_overrides["router"] == 0.0


# ── TrainConfig: freeze_params ───────────────────────────────────────────────

class TestFreezeParams:
    def _cfg(self, **kw):
        return make_test_train_config(lr=2e-6, seq_len=4096, global_batch_size=8, **kw)

    def test_single_pattern(self):
        """Verify single pattern."""
        cfg = self._cfg(freeze_params=["router"])
        assert cfg.freeze_params == ["router"]

    def test_multiple_patterns(self):
        """Verify multiple patterns."""
        cfg = self._cfg(freeze_params=["router", "embed"])
        assert "router" in cfg.freeze_params
        assert "embed" in cfg.freeze_params

    def test_none_by_default(self):
        """Verify none by default."""
        cfg = self._cfg()
        assert cfg.freeze_params is None

    def test_independent_of_layer_lr_overrides(self):
        """Verify independent of layer lr overrides."""
        cfg = self._cfg(
            freeze_params=["router"],
            layer_lr_overrides={"embed": 0.01},
        )
        assert cfg.freeze_params == ["router"]
        assert cfg.layer_lr_overrides["embed"] == 0.01
