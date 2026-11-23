"""Unit tests for moe balance."""

import dataclasses
import logging

import pytest

from titaninfra.models.moe_balance import (
    _REGISTRY,
    MoEBalanceSpec,
    get_moe_balance,
    register_moe_balance,
    resolve_balance,
    resolve_gamma_at,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


class _FakeModelConfig:
    def __init__(self, moe_load_balance_coeff=None, moe_aux_loss_coeff=None):
        self.moe_load_balance_coeff = moe_load_balance_coeff
        self.moe_aux_loss_coeff = moe_aux_loss_coeff


def test_unregistered_family_returns_all_none():
    spec = get_moe_balance("nonexistent_family")
    assert spec.bias_gamma is None and spec.aux_coeff is None
    assert spec.aux_scope == "sequence"


def test_register_and_get():
    register_moe_balance("fam_x", MoEBalanceSpec(bias_gamma=1e-3, aux_coeff=1e-4))
    spec = get_moe_balance("fam_x")
    assert spec.bias_gamma == 1e-3 and spec.aux_coeff == 1e-4


def test_duplicate_register_raises():
    register_moe_balance("fam_dup", MoEBalanceSpec(bias_gamma=1e-3))
    with pytest.raises(ValueError, match="already registered"):
        register_moe_balance("fam_dup", MoEBalanceSpec(bias_gamma=2e-3))


def test_spec_is_frozen():
    spec = MoEBalanceSpec(bias_gamma=1e-3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.bias_gamma = 2e-3  # type: ignore[misc]


def test_resolve_override_numeric_when_family_declares():
    register_moe_balance("fam_r1", MoEBalanceSpec(bias_gamma=1e-3, aux_coeff=1e-4))
    cfg = _FakeModelConfig(moe_load_balance_coeff=5e-3, moe_aux_loss_coeff=2e-4)
    spec = resolve_balance("fam_r1", cfg)
    assert spec.bias_gamma == 5e-3 and spec.aux_coeff == 2e-4
    assert spec.aux_scope == "sequence"


def test_resolve_warns_and_ignores_bias_when_family_silent(caplog):
    register_moe_balance("fam_r2", MoEBalanceSpec())
    cfg = _FakeModelConfig(moe_load_balance_coeff=1e-3)
    with caplog.at_level(logging.WARNING):
        spec = resolve_balance("fam_r2", cfg)
    assert spec.bias_gamma is None
    assert "不支持 expert_bias" in caplog.text


def test_resolve_warns_and_ignores_aux_when_family_silent(caplog):
    register_moe_balance("fam_r2b", MoEBalanceSpec(bias_gamma=1e-3))
    cfg = _FakeModelConfig(moe_aux_loss_coeff=1e-4)
    with caplog.at_level(logging.WARNING):
        spec = resolve_balance("fam_r2b", cfg)
    assert spec.aux_coeff is None
    assert "未声明 aux-loss" in caplog.text


def test_resolve_coupled_bias_and_aux():
    register_moe_balance("fam_r3", MoEBalanceSpec(bias_gamma=1e-3, aux_coeff=1e-4))
    spec = resolve_balance("fam_r3", _FakeModelConfig())
    assert spec.bias_gamma == 1e-3 and spec.aux_coeff == 1e-4


def test_resolve_no_override_returns_base_identity():
    base = MoEBalanceSpec(bias_gamma=1e-3, aux_coeff=1e-4)
    register_moe_balance("fam_r4", base)
    spec = resolve_balance("fam_r4", _FakeModelConfig())
    assert spec is base


class TestResolveGammaAt:
    BASE = 1e-3

    def test_base_gamma_none_returns_none(self):
        assert resolve_gamma_at(5, None, calibration_steps=10,
                                calibration_multiplier=50.0, freeze_after=None) is None

    def test_calibration_boost_first_steps(self):
        g = resolve_gamma_at(1, self.BASE, calibration_steps=3,
                             calibration_multiplier=50.0, freeze_after=None)
        assert abs(g - 5e-2) < 1e-12

    def test_calibration_boundary_inclusive(self):
        g = resolve_gamma_at(3, self.BASE, calibration_steps=3,
                             calibration_multiplier=10.0, freeze_after=None)
        assert abs(g - 1e-2) < 1e-12

    def test_base_after_calibration(self):
        g = resolve_gamma_at(4, self.BASE, calibration_steps=3,
                             calibration_multiplier=10.0, freeze_after=None)
        assert abs(g - self.BASE) < 1e-12

    def test_freeze_after_returns_zero(self):
        assert resolve_gamma_at(101, self.BASE, calibration_steps=0,
                                calibration_multiplier=50.0, freeze_after=100) == 0.0

    def test_freeze_boundary_still_updates(self):
        g = resolve_gamma_at(100, self.BASE, calibration_steps=0,
                             calibration_multiplier=50.0, freeze_after=100)
        assert abs(g - self.BASE) < 1e-12

    def test_freeze_takes_priority_over_calibration(self):
        assert resolve_gamma_at(5, self.BASE, calibration_steps=10,
                                calibration_multiplier=50.0, freeze_after=3) == 0.0

    def test_no_schedule_returns_base(self):
        g = resolve_gamma_at(999, self.BASE, calibration_steps=0,
                             calibration_multiplier=50.0, freeze_after=None)
        assert abs(g - self.BASE) < 1e-12
