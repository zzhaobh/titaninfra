"""Unit tests for moe bias schedule."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from tests.conftest import make_test_train_config
from titaninfra.config.train import MoEBiasSchedule
from titaninfra.engine._pretrain_helpers import _setup_moe_bias_schedule, apply_calibration_schedule


class TestMoEBiasScheduleConfig:
    def test_defaults(self):
        cfg = MoEBiasSchedule()
        assert cfg.calibration_steps == 0
        assert cfg.calibration_multiplier == 50.0
        assert cfg.freeze_after is None

    def test_custom(self):
        cfg = MoEBiasSchedule(calibration_steps=200, calibration_multiplier=10.0, freeze_after=5000)
        assert cfg.calibration_steps == 200
        assert cfg.calibration_multiplier == 10.0
        assert cfg.freeze_after == 5000

    def test_train_config_none_by_default(self):
        t = make_test_train_config(lr=1e-4, seq_len=4096, global_batch_size=8)
        assert t.moe_bias_schedule is None

    def test_train_config_with_schedule(self):
        t = make_test_train_config(lr=1e-4, seq_len=4096, global_batch_size=8,
                        moe_bias_schedule=MoEBiasSchedule(calibration_steps=100, freeze_after=9000))
        assert t.moe_bias_schedule.calibration_steps == 100
        assert t.moe_bias_schedule.freeze_after == 9000

    def test_old_name_gone(self):
        with pytest.raises(ImportError):
            from titaninfra.config.train import MoECalibrationConfig  # noqa: F401

    def test_calibration_steps_ge_zero(self):
        with pytest.raises(ValueError):
            MoEBiasSchedule(calibration_steps=-1)

    def test_freeze_after_must_be_positive(self):
        with pytest.raises(ValueError):
            MoEBiasSchedule(freeze_after=0)
        with pytest.raises(ValueError):
            MoEBiasSchedule(freeze_after=-5)
        # None and positive remain valid
        assert MoEBiasSchedule(freeze_after=None).freeze_after is None
        assert MoEBiasSchedule(freeze_after=1).freeze_after == 1


class _MockMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.load_balance_coeff = 1e-3


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.moe = _MockMoE()


def _make_trainer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    class MockOptimizers:
        def register_step_pre_hook(self, fn):
            optimizer.register_step_pre_hook(fn)

    trainer = MagicMock()
    # upstream Trainer/TrainingEngine split: built runtime objects
    # (model_parts/.../lr_schedulers) now live on trainer.engine, and the
    # optimizer container itself moved one level deeper to
    # trainer.engine.optim.optimizers (Optim wraps OptimizersContainer) —
    # see _pretrain_helpers.py's matching trainer.engine.optim.optimizers
    # reads and engine/train_step.py's TitanInfraTrainingEngine for the
    # real optim_step()/num_completed_steps semantics this mock mirrors.
    trainer.engine.model_parts = [model]
    trainer.engine.optim.optimizers = MockOptimizers()
    trainer.engine.num_completed_steps = 0
    return trainer, optimizer


def _tc(calibration_steps=0, calibration_multiplier=50.0, freeze_after=None):
    return make_test_train_config(lr=0.1, seq_len=4096, global_batch_size=8,
                       moe_bias_schedule=MoEBiasSchedule(
                           calibration_steps=calibration_steps,
                           calibration_multiplier=calibration_multiplier,
                           freeze_after=freeze_after))


def _mc(coeff=1e-3):
    cfg = MagicMock()
    cfg.moe_load_balance_coeff = coeff
    return cfg


def _step(trainer, optimizer, model):
    model.weight.grad = torch.randn_like(model.weight)
    optimizer.step()  # fires the pre-hook with num_completed_steps(old)+1
    optimizer.zero_grad()
    trainer.engine.num_completed_steps += 1  # mirrors TrainingEngine.optim_step() post-assign


class TestSetupMoeBiasSchedule:
    def test_noop_when_schedule_none(self):
        model = _MockModel()
        trainer, optimizer = _make_trainer(model)
        tc = make_test_train_config(lr=0.1, seq_len=4096, global_batch_size=8)
        _setup_moe_bias_schedule(trainer, tc, _mc())
        _step(trainer, optimizer, model)
        assert model.moe.load_balance_coeff == 1e-3

    def test_calibration_boost_then_base(self):
        model = _MockModel()
        trainer, optimizer = _make_trainer(model)
        _setup_moe_bias_schedule(trainer, _tc(calibration_steps=2, calibration_multiplier=10.0), _mc(1e-3))
        _step(trainer, optimizer, model)  # step 1
        assert abs(model.moe.load_balance_coeff - 1e-2) < 1e-9
        _step(trainer, optimizer, model)  # step 2
        assert abs(model.moe.load_balance_coeff - 1e-2) < 1e-9
        _step(trainer, optimizer, model)  # step 3 → base
        assert abs(model.moe.load_balance_coeff - 1e-3) < 1e-9

    def test_freeze_sets_gamma_zero(self):
        model = _MockModel()
        trainer, optimizer = _make_trainer(model)
        _setup_moe_bias_schedule(trainer, _tc(freeze_after=2), _mc(1e-3))
        _step(trainer, optimizer, model)  # step 1 → base
        assert abs(model.moe.load_balance_coeff - 1e-3) < 1e-9
        _step(trainer, optimizer, model)
        assert abs(model.moe.load_balance_coeff - 1e-3) < 1e-9
        _step(trainer, optimizer, model)  # step 3 > freeze_after → 0
        assert model.moe.load_balance_coeff == 0.0

    def test_no_crash_when_coeff_none(self):
        model = _MockModel()
        model.moe.load_balance_coeff = None
        trainer, optimizer = _make_trainer(model)
        _setup_moe_bias_schedule(trainer, _tc(calibration_steps=2), _mc(None))
        _step(trainer, optimizer, model)
        assert model.moe.load_balance_coeff is None

    def test_calibration_then_base_then_freeze_full_curve(self):
        """Verify calibration then base then freeze full curve."""
        model = _MockModel()
        trainer, optimizer = _make_trainer(model)
        _setup_moe_bias_schedule(
            trainer, _tc(calibration_steps=2, calibration_multiplier=10.0, freeze_after=4), _mc(1e-3)
        )
        _step(trainer, optimizer, model)  # step 1 ≤ calib → boost
        assert abs(model.moe.load_balance_coeff - 1e-2) < 1e-9
        _step(trainer, optimizer, model)  # step 2 == calib → boost
        assert abs(model.moe.load_balance_coeff - 1e-2) < 1e-9
        _step(trainer, optimizer, model)  # step 3 → base
        assert abs(model.moe.load_balance_coeff - 1e-3) < 1e-9
        _step(trainer, optimizer, model)
        assert abs(model.moe.load_balance_coeff - 1e-3) < 1e-9
        _step(trainer, optimizer, model)  # step 5 > freeze_after → 0
        assert model.moe.load_balance_coeff == 0.0


class TestApplyCalibrationSchedule:
    """Tests for test apply calibration schedule."""

    def _make_trainer_with_scheduler(self, calibration_steps, total_steps=100, warmup_steps=10):
        model = _MockModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        def warmup_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

        class MockSchedulers:
            def __init__(self):
                self.schedulers = [scheduler]

        class MockConfig:
            class training:
                steps = total_steps

        trainer = MagicMock()
        trainer.engine.model_parts = [model]
        trainer.engine.optim.lr_schedulers = MockSchedulers()
        trainer.config = MockConfig()
        return trainer, optimizer, scheduler, _tc(calibration_steps=calibration_steps), model

    def test_total_steps_extended(self):
        trainer, _, _, train_config, _ = self._make_trainer_with_scheduler(
            calibration_steps=20, total_steps=100)
        apply_calibration_schedule(trainer, train_config)
        assert trainer.config.training.steps == 120

    def test_lr_zero_during_calibration(self):
        trainer, optimizer, scheduler, train_config, _ = self._make_trainer_with_scheduler(
            calibration_steps=5, total_steps=100, warmup_steps=10)
        apply_calibration_schedule(trainer, train_config)
        assert optimizer.param_groups[0]["lr"] == 0.0
        for step in range(4):
            scheduler.step()
            assert optimizer.param_groups[0]["lr"] == 0.0

    def test_warmup_starts_after_calibration(self):
        trainer, optimizer, scheduler, train_config, _ = self._make_trainer_with_scheduler(
            calibration_steps=5, total_steps=100, warmup_steps=10)
        apply_calibration_schedule(trainer, train_config)
        for _ in range(5):
            scheduler.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert 0 < lr < 0.1

    def test_warmup_completes_fully(self):
        trainer, optimizer, scheduler, train_config, _ = self._make_trainer_with_scheduler(
            calibration_steps=5, total_steps=100, warmup_steps=10)
        apply_calibration_schedule(trainer, train_config)
        for _ in range(15):
            scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - 0.1) < 1e-6

    def test_noop_when_no_schedule(self):
        train_config = make_test_train_config(lr=0.1, seq_len=4096, global_batch_size=8)
        trainer = MagicMock()
        trainer.config.training.steps = 100
        apply_calibration_schedule(trainer, train_config)
        assert trainer.config.training.steps == 100
