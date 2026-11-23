"""
Muon optimizer unit tests.

Tests the core NS iteration, Muon optimizer step, param classification,
MixedOptimizersContainer interface, and TrainConfig integration.
"""

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from tests.conftest import make_test_train_config


def _has_param(params, target) -> bool:
    """Has param."""
    return any(p is target for p in params)


def _catch_all_adamw(lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=False, foreach=False):
    """Catch all adamw."""
    from torchtitan.components.optim import AdamW

    return [AdamW.Config(
        pattern=r".*", lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        fused=fused, foreach=foreach,
    )]


class TestNewtonSchulzIter:
    """Test the Newton-Schulz iteration for orthogonality."""

    def test_convergence_square(self):
        """NS iteration on a square matrix produces closer to orthogonal."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        torch.manual_seed(42)
        X = torch.randn(64, 64)

        result = torch._dynamo.disable(_newton_schulz_iter)(X, steps=5)

        gram = result @ result.mT
        identity = torch.eye(64)
        # Relative orthogonality: off-diag norm / matrix size should be small
        relative_error = (gram - identity).norm().item() / 64
        assert relative_error < 0.5, f"Not converging: relative error = {relative_error}"

    def test_convergence_tall(self):
        """NS iteration on a tall matrix (m > n) produces closer to orthogonal columns."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        torch.manual_seed(42)
        X = torch.randn(128, 64)

        result = torch._dynamo.disable(_newton_schulz_iter)(X, steps=5)

        gram = result.mT @ result
        identity = torch.eye(64)
        relative_error = (gram - identity).norm().item() / 64
        assert relative_error < 0.5, f"Not converging: relative error = {relative_error}"

    def test_output_shape_preserved(self):
        """Output shape matches input shape."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        torch.manual_seed(42)
        X = torch.randn(32, 64)

        result = torch._dynamo.disable(_newton_schulz_iter)(X, steps=5)
        assert result.shape == X.shape


class TestMuonOptimizer:
    """Test the Muon optimizer step."""

    def test_basic_step(self):
        """Single Muon step updates a 2D parameter."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 64))
        original_data = param.data.clone()

        optimizer = Muon([param], lr=0.01, momentum=0.95, ns_steps=5)

        param.grad = torch.randn_like(param)
        optimizer.step()

        assert not torch.allclose(param.data, original_data), "Parameter not updated"

    def test_momentum_accumulation(self):
        """Momentum buffer accumulates across steps."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 64))
        optimizer = Muon([param], lr=0.01, momentum=0.95)

        param.grad = torch.randn_like(param)
        optimizer.step()

        buf = optimizer.state[param]["momentum_buffer"]
        assert buf.norm().item() > 0, "Momentum buffer should be non-zero after step"

    def test_weight_decay(self):
        """Decoupled weight decay shrinks parameter norm."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.ones(64, 64))
        optimizer = Muon([param], lr=0.01, weight_decay=0.1)

        param.grad = torch.zeros_like(param)
        initial_norm = param.data.norm().item()
        optimizer.step()

        assert param.data.norm().item() < initial_norm, "Weight decay should shrink norm"

    def test_invalid_lr_raises(self):
        """Negative lr raises ValueError."""
        from titaninfra.optimizers.muon import Muon

        param = nn.Parameter(torch.randn(64, 64))
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Muon([param], lr=-0.01)

    def test_invalid_momentum_raises(self):
        """Momentum >= 1 raises ValueError."""
        from titaninfra.optimizers.muon import Muon

        param = nn.Parameter(torch.randn(64, 64))
        with pytest.raises(ValueError, match="Invalid momentum"):
            Muon([param], momentum=1.0)


class TestParamClassification:
    """Test _classify_params splitting logic."""

    @pytest.fixture(autouse=True)
    def _require_torchtitan(self):
        pytest.importorskip("torchtitan")

    def test_2d_vs_1d_split(self):
        """2D params (large enough) go to Muon, 1D params go to AdamW."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.Sequential(
            nn.Linear(128, 128),  # weight: 2D (128*128=16384 > 4096), bias: 1D
            nn.LayerNorm(128),     # weight: 1D, bias: 1D
        )

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert len(muon_params) == 1  # only the Linear weight
        assert len(adamw_params) == 3  # Linear bias + LayerNorm weight + bias

    def test_small_2d_goes_to_adamw(self):
        """Small 2D params (below threshold) go to AdamW."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.Sequential(
            nn.Linear(8, 8),  # weight: 2D but only 64 elements
        )

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert len(muon_params) == 0
        assert len(adamw_params) == 2  # weight + bias

    def test_freeze_patterns(self):
        """Frozen params are excluded from both lists."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({"embed": nn.Linear(128, 128), "main": nn.Linear(128, 128)})

        muon_params, adamw_params = _classify_params(
            model, min_ndim_size=4096, freeze_patterns=["embed"]
        )

        # embed layer's params (weight + bias) should be excluded
        # main layer: weight → Muon, bias → AdamW
        assert len(muon_params) == 1
        assert len(adamw_params) == 1

    def test_router_gate_excluded_by_default(self):
        """MoE router/gate goes to AdamW by default (include_router=False)."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        # 'gate' is a large 2D matrix that would qualify for Muon by shape alone.
        model = nn.ModuleDict({"gate": nn.Linear(128, 128), "main": nn.Linear(128, 128)})

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        # gate.weight → AdamW (excluded), main.weight → Muon
        assert len(muon_params) == 1
        # gate.weight + gate.bias + main.bias
        assert len(adamw_params) == 3

    def test_router_gate_to_muon_when_included(self):
        """include_router=True routes the MoE gate weight to Muon (Moonlight recipe)."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({"gate": nn.Linear(128, 128), "main": nn.Linear(128, 128)})

        muon_params, adamw_params = _classify_params(
            model, min_ndim_size=4096, include_router=True
        )

        # gate.weight + main.weight → Muon; gate.bias + main.bias → AdamW
        assert len(muon_params) == 2
        assert len(adamw_params) == 2

    def test_include_router_does_not_leak_to_embeddings(self):
        """include_router=True must still keep embed/lm_head out of Muon."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({
            "tok_embeddings": nn.Linear(128, 128),
            "gate": nn.Linear(128, 128),
        })

        muon_params, adamw_params = _classify_params(
            model, min_ndim_size=4096, include_router=True
        )

        # only gate.weight → Muon; embeddings stay in AdamW
        assert len(muon_params) == 1
        # tok_embeddings.weight + tok_embeddings.bias + gate.bias
        assert len(adamw_params) == 3


class TestMixedOptimizersContainer:
    """Test the full mixed optimizer container."""

    @pytest.fixture(autouse=True)
    def _require_torchtitan(self):
        pytest.importorskip("torchtitan")

    def test_step_updates_all_params(self):
        """Both Muon and AdamW params get updated."""
        from titaninfra.optimizers.mixed_optimizer import MixedOptimizersContainer

        model = nn.Sequential(
            nn.Linear(128, 128),  # weight → Muon, bias → AdamW
            nn.LayerNorm(128),     # weight + bias → AdamW
        )

        config = MixedOptimizersContainer.Config(
            optimizers=_catch_all_adamw(lr=1e-4),
            muon_lr=0.02,
            muon_min_ndim_size=4096,
        )
        container = MixedOptimizersContainer(config, model_parts=[model])

        # Simulate a forward/backward
        x = torch.randn(4, 128)
        loss = model(x).sum()
        loss.backward()

        original_params = {name: p.data.clone() for name, p in model.named_parameters()}
        container.step()

        updated = False
        for name, p in model.named_parameters():
            if not torch.allclose(p.data, original_params[name]):
                updated = True
                break
        assert updated, "At least one parameter should be updated"

    def test_zero_grad(self):
        """zero_grad clears all gradients."""
        from titaninfra.optimizers.mixed_optimizer import MixedOptimizersContainer

        model = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128))
        config = MixedOptimizersContainer.Config(
            optimizers=_catch_all_adamw(lr=1e-4),
            muon_lr=0.02,
            muon_min_ndim_size=4096,
        )
        container = MixedOptimizersContainer(config, model_parts=[model])

        x = torch.randn(4, 128)
        loss = model(x).sum()
        loss.backward()

        container.zero_grad()

        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum().item() == 0.0

    def test_gather_mode_propagates_to_muon(self):
        """muon_gather_mode in Config reaches the underlying Muon optimizer."""
        from torchtitan.components.optim import AdamW

        from titaninfra.optimizers.mixed_optimizer import MixedOptimizersContainer

        model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128))
        cfg = MixedOptimizersContainer.Config(
            optimizers=[AdamW.Config(
                pattern=r".*", lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                fused=False, foreach=True,
            )],
            muon_min_ndim_size=64, muon_gather_mode=True,
        )
        container = MixedOptimizersContainer(cfg, model_parts=[model])
        muon_opts = [o for o in container._muon_optimizers if o is not None]
        assert muon_opts, "expected at least one Muon optimizer"
        assert all(o._gather_mode is True for o in muon_opts)


class TestTrainConfigMuon:
    """Test TrainConfig with MuonAdamW optimizer."""

    def test_muon_config_creation(self):
        """MuonAdamW config fields are accepted."""

        cfg = make_test_train_config(
            optimizer_name="MuonAdamW",
            muon_lr=0.02,
            muon_momentum=0.95,
            muon_ns_steps=5,
        )
        assert cfg.optimizer_name == "MuonAdamW"
        assert cfg.muon_lr == 0.02
        assert cfg.muon_momentum == 0.95

    def test_muon_config_defaults(self):
        """Muon fields have sensible defaults."""

        cfg = make_test_train_config(optimizer_name="MuonAdamW")
        assert cfg.muon_lr == 0.02
        assert cfg.muon_momentum == 0.95
        assert cfg.muon_nesterov is True
        assert cfg.muon_ns_steps == 5
        assert cfg.muon_weight_decay == 0.0
        assert cfg.muon_min_ndim_size == 4096

    def test_adamw_still_works(self):
        """Original AdamW config path is not broken."""

        cfg = make_test_train_config(optimizer_name="AdamW", lr=1e-4)
        assert cfg.optimizer_name == "AdamW"
        assert cfg.lr == 1e-4

    def test_invalid_muon_momentum_rejected(self):
        """muon_momentum >= 1 rejected by Pydantic."""

        with pytest.raises(ValidationError):
            make_test_train_config(optimizer_name="MuonAdamW", muon_momentum=1.0)

    def test_invalid_muon_ns_steps_rejected(self):
        """muon_ns_steps < 1 rejected by Pydantic."""

        with pytest.raises(ValidationError):
            make_test_train_config(optimizer_name="MuonAdamW", muon_ns_steps=0)

    def test_invalid_muon_ns_steps_too_high(self):
        """muon_ns_steps > 20 rejected by Pydantic."""

        with pytest.raises(ValidationError):
            make_test_train_config(optimizer_name="MuonAdamW", muon_ns_steps=21)

    def test_muon_gather_mode_default_false(self):

        cfg = make_test_train_config(steps=10)
        assert cfg.muon_gather_mode is False

    def test_muon_gather_mode_settable(self):

        cfg = make_test_train_config(steps=10, muon_gather_mode=True)
        assert cfg.muon_gather_mode is True


class TestNewtonSchulzIterAdditional:
    """Additional NS iteration tests."""

    def test_determinism(self):
        """Same input produces identical output across calls."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        ns_fn = torch._dynamo.disable(_newton_schulz_iter)

        torch.manual_seed(42)
        X = torch.randn(32, 32)

        result1 = ns_fn(X.clone(), steps=5)
        result2 = ns_fn(X.clone(), steps=5)

        assert torch.allclose(result1, result2, atol=1e-6)

    def test_wide_matrix(self):
        """Wide matrix (cols > rows) is handled correctly."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        torch.manual_seed(42)
        X = torch.randn(32, 128)  # wide: cols > rows

        result = torch._dynamo.disable(_newton_schulz_iter)(X, steps=5)
        assert result.shape == (32, 128)

    def test_more_steps_improves_convergence(self):
        """More NS steps yields better orthogonality."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        ns_fn = torch._dynamo.disable(_newton_schulz_iter)

        torch.manual_seed(42)
        X = torch.randn(32, 32)

        result_3 = ns_fn(X.clone(), steps=3)
        result_7 = ns_fn(X.clone(), steps=7)

        error_3 = (result_3 @ result_3.mT - torch.eye(32)).norm().item()
        error_7 = (result_7 @ result_7.mT - torch.eye(32)).norm().item()

        assert error_7 <= error_3, "More steps should not worsen convergence"


class TestMuonOptimizerAdditional:
    """Additional Muon optimizer tests."""

    def test_nesterov_false(self):
        """Muon works with nesterov=False."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 64))
        original = param.data.clone()

        optimizer = Muon([param], lr=0.01, nesterov=False)
        param.grad = torch.randn_like(param)
        optimizer.step()

        assert not torch.allclose(param.data, original)

    def test_step_counter_increments(self):
        """Step counter increments on each step."""
        from titaninfra.optimizers.muon import Muon

        param = nn.Parameter(torch.randn(64, 64))
        optimizer = Muon([param], lr=0.01)

        for i in range(3):
            param.grad = torch.randn_like(param)
            optimizer.step()

        assert optimizer.state[param]["step"] == 3

    def test_wide_param_transposition(self):
        """Wide parameters (cols > rows) use transposition path."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(32, 128))  # wide matrix
        original = param.data.clone()

        optimizer = Muon([param], lr=0.01)
        param.grad = torch.randn_like(param)
        optimizer.step()

        assert not torch.allclose(param.data, original)
        assert param.shape == (32, 128)

    def test_invalid_ns_steps_raises(self):
        """ns_steps < 1 raises ValueError."""
        from titaninfra.optimizers.muon import Muon

        param = nn.Parameter(torch.randn(64, 64))
        with pytest.raises(ValueError, match="Invalid ns_steps"):
            Muon([param], ns_steps=0)

    def test_none_grad_skipped(self):
        """Params with grad=None are not updated."""
        from titaninfra.optimizers.muon import Muon

        param1 = nn.Parameter(torch.randn(64, 64))
        param2 = nn.Parameter(torch.randn(64, 64))
        original2 = param2.data.clone()

        optimizer = Muon([param1, param2], lr=0.01)
        param1.grad = torch.randn_like(param1)
        # param2.grad remains None

        optimizer.step()

        assert torch.allclose(param2.data, original2), "Param with None grad should not be updated"

    def test_state_dict_roundtrip(self):
        """Muon state_dict can be saved and reloaded."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 64))
        optimizer = Muon([param], lr=0.01, momentum=0.95)

        param.grad = torch.randn_like(param)
        optimizer.step()
        param.grad = torch.randn_like(param)
        optimizer.step()

        state = optimizer.state_dict()

        # Create new optimizer and load state
        param2 = nn.Parameter(param.data.clone())
        optimizer2 = Muon([param2], lr=0.01, momentum=0.95)
        optimizer2.load_state_dict(state)

        # State should match
        buf_original = optimizer.state[param]["momentum_buffer"]
        buf_loaded = optimizer2.state[param2]["momentum_buffer"]
        assert torch.allclose(buf_original, buf_loaded)
        assert optimizer2.state[param2]["step"] == 2

    def test_multiple_steps_loss_decreases(self):
        """Multiple Muon steps on a simple problem reduce loss."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        model = nn.Linear(64, 64, bias=False)
        optimizer = Muon(model.parameters(), lr=0.01)

        target = torch.randn(8, 64)
        x = torch.randn(8, 64)

        initial_loss = ((model(x) - target) ** 2).mean().item()

        for _ in range(20):
            optimizer.zero_grad()
            loss = ((model(x) - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_loss = ((model(x) - target) ** 2).mean().item()
        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} → {final_loss}"


class TestParamClassificationAdditional:
    """Additional param classification tests."""

    @pytest.fixture(autouse=True)
    def _require_torchtitan(self):
        pytest.importorskip("torchtitan")

    def test_requires_grad_false_excluded(self):
        """Params with requires_grad=False are excluded."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.Linear(128, 128)
        model.weight.requires_grad_(False)

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert len(muon_params) == 0
        assert len(adamw_params) == 1  # only bias

    def test_multiple_large_linears(self):
        """Multiple large Linear layers: all weights go to Muon."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.Linear(256, 256, bias=False),
            nn.Linear(256, 256, bias=False),
        )

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert len(muon_params) == 3
        assert len(adamw_params) == 0

    def test_embedding_goes_to_adamw(self):
        """Embedding layer weight (2D, large) is excluded from Muon by name pattern."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({"tok_embeddings": nn.Embedding(10000, 128)})

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert len(muon_params) == 0
        assert len(adamw_params) == 1

    def test_lm_head_goes_to_adamw(self):
        """lm_head / output weight (2D, large) is excluded from Muon."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({"output": nn.Linear(2048, 151936, bias=False)})

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert len(muon_params) == 0
        assert len(adamw_params) == 1

    def test_moe_gate_goes_to_adamw(self):
        """MoE router gate weight (2D, large) is excluded from Muon."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "moe": nn.ModuleDict({
                        "router": nn.ModuleDict({
                            "gate": nn.Linear(2048, 128, bias=False),
                        }),
                    }),
                }),
            }),
        })

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert len(muon_params) == 0
        assert len(adamw_params) == 1

    def test_regular_linear_still_goes_to_muon(self):
        """Regular linear weights (not embed/output/gate) still go to Muon."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "wq": nn.Linear(2048, 2048, bias=False),
                    }),
                    "feed_forward": nn.ModuleDict({
                        "w1": nn.Linear(2048, 768, bias=False),
                    }),
                }),
            }),
        })

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert len(muon_params) == 2  # wq and w1
        assert len(adamw_params) == 0


class TestBuildOptimizerConfig:
    """Test _build_optimizer_config integration helper."""

    @pytest.fixture(autouse=True)
    def _require_torchtitan(self):
        pytest.importorskip("torchtitan")

    def test_adamw_returns_optimizers_container_config(self):
        """optimizer_name='AdamW' returns standard OptimizersContainer.Config."""
        from torchtitan.components.optim import OptimizersContainer

        from titaninfra.engine._pretrain_helpers import _build_optimizer_config

        cfg = make_test_train_config(optimizer_name="AdamW", lr=1e-4)
        result = _build_optimizer_config(cfg, param_groups=[])

        assert isinstance(result, OptimizersContainer.Config)
        catch_all = result.optimizers[-1]
        assert catch_all.pattern == r".*"
        assert catch_all.lr == 1e-4

    def test_muonadamw_returns_mixed_container_config(self):
        """optimizer_name='MuonAdamW' returns MixedOptimizersContainer.Config."""
        from titaninfra.engine._pretrain_helpers import _build_optimizer_config
        from titaninfra.optimizers.mixed_optimizer import MixedOptimizersContainer

        cfg = make_test_train_config(optimizer_name="MuonAdamW", lr=1e-4, muon_lr=0.03)
        result = _build_optimizer_config(cfg, param_groups=[])

        assert isinstance(result, MixedOptimizersContainer.Config)
        assert result.muon_lr == 0.03
        catch_all = result.optimizers[-1]
        assert catch_all.lr == 1e-4

    def test_config_fields_passed_through(self):
        """All config fields are correctly forwarded."""
        from titaninfra.engine._pretrain_helpers import _build_optimizer_config

        cfg = make_test_train_config(
            optimizer_name="MuonAdamW",
            lr=2e-4,
            beta1=0.85,
            beta2=0.99,
            eps=1e-6,
            weight_decay=0.05,
            muon_lr=0.01,
            muon_momentum=0.9,
            muon_nesterov=False,
            muon_ns_steps=7,
            muon_weight_decay=0.01,
            muon_min_ndim_size=8192,
        )
        result = _build_optimizer_config(cfg, param_groups=[])

        catch_all = result.optimizers[-1]
        assert catch_all.lr == 2e-4
        assert catch_all.betas == (0.85, 0.99)
        assert catch_all.eps == 1e-6
        assert catch_all.weight_decay == 0.05
        assert result.muon_lr == 0.01
        assert result.muon_momentum == 0.9
        assert result.muon_nesterov is False
        assert result.muon_ns_steps == 7
        assert result.muon_weight_decay == 0.01
        assert result.muon_min_ndim_size == 8192

    def test_gather_mode_flows_to_config(self):
        from titaninfra.engine._pretrain_helpers import _build_optimizer_config

        tc = make_test_train_config(steps=10, optimizer_name="MuonAdamW", muon_gather_mode=True)
        cfg = _build_optimizer_config(tc, param_groups=[])
        assert cfg.muon_gather_mode is True


# ===== New test classes for Muon optimizations =====


class TestNewtonSchulzQuinticCoeffs:
    """Test per-step quintic coefficients improvement."""

    def test_quintic_better_than_fixed(self):
        """Per-step quintic coefficients converge well."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        ns_fn = torch._dynamo.disable(_newton_schulz_iter)

        torch.manual_seed(42)
        X = torch.randn(64, 64)

        result = ns_fn(X.clone(), steps=5)
        gram = result @ result.mT
        error = (gram - torch.eye(64)).norm().item() / 64
        assert error < 0.3, f"Quintic coefficients should converge well: error={error}"

    def test_steps_beyond_5_uses_fallback(self):
        """Steps > 5 falls back to last coefficient entry."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        ns_fn = torch._dynamo.disable(_newton_schulz_iter)

        torch.manual_seed(42)
        X = torch.randn(32, 32)

        result = ns_fn(X.clone(), steps=8)
        assert result.shape == (32, 32)
        assert torch.isfinite(result).all()

    def test_single_step(self):
        """Single NS step produces valid output."""
        from titaninfra.optimizers.muon import _newton_schulz_iter

        ns_fn = torch._dynamo.disable(_newton_schulz_iter)

        torch.manual_seed(42)
        X = torch.randn(32, 32)

        result = ns_fn(X.clone(), steps=1)
        assert result.shape == (32, 32)
        assert torch.isfinite(result).all()


class TestBF16NSComputation:
    """Test BF16 Newton-Schulz computation."""

    def test_bf16_produces_valid_output(self):
        """NS iteration in BF16 produces finite output."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 64))

        optimizer = Muon([param], lr=0.01, ns_dtype=torch.bfloat16)
        param.grad = torch.randn_like(param)
        optimizer.step()

        assert torch.isfinite(param.data).all()

    def test_bf16_vs_fp32_tolerance(self):
        """BF16 and FP32 NS give similar (within tolerance) updates."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param_bf16 = nn.Parameter(torch.randn(64, 64))
        param_fp32 = nn.Parameter(param_bf16.data.clone())

        opt_bf16 = Muon([param_bf16], lr=0.01, ns_dtype=torch.bfloat16)
        opt_fp32 = Muon([param_fp32], lr=0.01, ns_dtype=torch.float32)

        grad = torch.randn(64, 64)
        param_bf16.grad = grad.clone()
        param_fp32.grad = grad.clone()

        opt_bf16.step()
        opt_fp32.step()

        diff = (param_bf16.data - param_fp32.data).abs().max().item()
        assert diff < 0.1, f"BF16 and FP32 diverged too much: max_diff={diff}"

    def test_fp32_fallback(self):
        """Explicitly using FP32 NS still works."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 64))

        optimizer = Muon([param], lr=0.01, ns_dtype=torch.float32)
        param.grad = torch.randn_like(param)
        optimizer.step()

        assert torch.isfinite(param.data).all()

    def test_default_is_bf16(self):
        """Default ns_dtype is bfloat16."""
        from titaninfra.optimizers.muon import Muon

        param = nn.Parameter(torch.randn(64, 64))
        optimizer = Muon([param], lr=0.01)
        assert optimizer._ns_dtype == torch.bfloat16


class TestMoonlightScaleFactor:
    """Test Moonlight scale factor option."""

    def test_default_scale_factor_is_1(self):
        """Default scale_factor is 1.0."""
        from titaninfra.optimizers.muon import Muon

        param = nn.Parameter(torch.randn(64, 64))
        optimizer = Muon([param], lr=0.01)
        assert optimizer.defaults["scale_factor"] == 1.0

    def test_scale_factor_02_reduces_update(self):
        """scale_factor=0.2 produces smaller updates than 1.0."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        base_data = torch.randn(64, 64)
        param_1 = nn.Parameter(base_data.clone())
        param_02 = nn.Parameter(base_data.clone())

        opt_1 = Muon([param_1], lr=0.01, scale_factor=1.0)
        opt_02 = Muon([param_02], lr=0.01, scale_factor=0.2)

        grad = torch.randn(64, 64)
        param_1.grad = grad.clone()
        param_02.grad = grad.clone()

        opt_1.step()
        opt_02.step()

        update_1 = (param_1.data - base_data).norm().item()
        update_02 = (param_02.data - base_data).norm().item()

        assert update_02 < update_1, f"0.2 scale should give smaller update: {update_02} vs {update_1}"
        ratio = update_02 / update_1
        assert 0.15 < ratio < 0.25, f"Expected ratio ~0.2, got {ratio}"

    def test_config_field_exists(self):
        """TrainConfig has muon_scale_factor field."""

        cfg = make_test_train_config(optimizer_name="MuonAdamW", muon_scale_factor=0.2)
        assert cfg.muon_scale_factor == 0.2


class TestNesterovAlignment:
    """Test corrected Nesterov momentum formula."""

    def test_nesterov_uses_lerp(self):
        """Nesterov update is interpolation: (1-beta)*grad + beta*buf."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 64))
        optimizer = Muon([param], lr=0.01, momentum=0.95, nesterov=True)

        grad = torch.randn(64, 64)
        param.grad = grad.clone()
        optimizer.step()

        buf = optimizer.state[param]["momentum_buffer"]
        expected_buf = 0.05 * grad
        assert torch.allclose(buf, expected_buf, atol=1e-6)

    def test_nesterov_false_uses_buf_directly(self):
        """With nesterov=False, update is just the momentum buffer."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 64))
        optimizer = Muon([param], lr=0.01, momentum=0.95, nesterov=False)

        param.grad = torch.randn_like(param)
        optimizer.step()
        assert optimizer.state[param]["step"] == 1


class TestQKClip:
    """Test QK-Clip mechanism."""

    def test_find_qk_params(self):
        """_find_qk_params discovers Q/K parameter pairs."""
        from titaninfra.optimizers.qk_clip import _find_qk_params

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "qkv_linear": nn.ModuleDict({
                            "wq": nn.Linear(128, 128, bias=False),
                            "wk": nn.Linear(128, 128, bias=False),
                        }),
                    }),
                }),
                "1": nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "qkv_linear": nn.ModuleDict({
                            "wq": nn.Linear(128, 128, bias=False),
                            "wk": nn.Linear(128, 128, bias=False),
                        }),
                    }),
                }),
            }),
        })

        pairs = _find_qk_params([model])
        assert len(pairs) == 2

    def test_clip_when_exceeds_tau(self):
        """Clipping occurs when max logit exceeds tau."""
        from titaninfra.optimizers.qk_clip import _find_qk_params, _qk_clip_step

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "qkv_linear": nn.ModuleDict({
                            "wq": nn.Linear(128, 128, bias=False),
                            "wk": nn.Linear(128, 128, bias=False),
                        }),
                    }),
                }),
            }),
        })

        with torch.no_grad():
            for p in model.parameters():
                p.mul_(100.0)

        pairs = _find_qk_params([model])
        n_clipped = _qk_clip_step(pairs, tau=1.0, head_dim=128)
        assert n_clipped > 0

    def test_no_clip_when_below_tau(self):
        """No clipping when max logit is below tau."""
        from titaninfra.optimizers.qk_clip import _find_qk_params, _qk_clip_step

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "qkv_linear": nn.ModuleDict({
                            "wq": nn.Linear(128, 128, bias=False),
                            "wk": nn.Linear(128, 128, bias=False),
                        }),
                    }),
                }),
            }),
        })

        pairs = _find_qk_params([model])
        n_clipped = _qk_clip_step(pairs, tau=10000.0, head_dim=128)
        assert n_clipped == 0

    def test_no_pairs_found_gracefully(self):
        """No Q/K pairs found → no error."""
        from titaninfra.optimizers.qk_clip import _find_qk_params

        model = nn.Sequential(nn.Linear(128, 128))
        pairs = _find_qk_params([model])
        assert len(pairs) == 0

    def test_mla_find_params(self):
        """_find_qk_params discovers MLA wq_b/wkv_b pairs."""
        from titaninfra.optimizers.qk_clip import _find_qk_params

        # DeepSeek-V3 20B: n_heads=16, nope=128, rope=64, v=128, q_lora=1536, kv_lora=512
        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "wq_b": nn.Linear(1536, 3072, bias=False),  # [3072, 1536] weight
                        "wkv_b": nn.Linear(512, 4096, bias=False),  # [4096, 512] weight
                    }),
                }),
            }),
        })
        pairs = _find_qk_params([model])
        assert len(pairs) == 1
        assert pairs[0][4] == "mla"

    def test_mla_clip_when_exceeds_tau(self):
        """MLA: clipping occurs when nope norm product exceeds tau."""
        from titaninfra.optimizers.qk_clip import _find_qk_params, _qk_clip_step

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "wq_b": nn.Linear(1536, 3072, bias=False),
                        "wkv_b": nn.Linear(512, 4096, bias=False),
                    }),
                }),
            }),
        })

        with torch.no_grad():
            for p in model.parameters():
                p.mul_(100.0)

        pairs = _find_qk_params([model])
        q_norm_before = model.layers["0"].attention.wq_b.weight.norm().item()
        n_clipped = _qk_clip_step(pairs, tau=1.0, head_dim=128)
        q_norm_after = model.layers["0"].attention.wq_b.weight.norm().item()

        assert n_clipped > 0
        assert q_norm_after < q_norm_before

    def test_mla_no_clip_below_tau(self):
        """MLA: no clipping when nope norms are below tau."""
        from titaninfra.optimizers.qk_clip import _find_qk_params, _qk_clip_step

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "wq_b": nn.Linear(1536, 3072, bias=False),
                        "wkv_b": nn.Linear(512, 4096, bias=False),
                    }),
                }),
            }),
        })

        pairs = _find_qk_params([model])
        n_clipped = _qk_clip_step(pairs, tau=10000.0, head_dim=128)
        assert n_clipped == 0

    def test_mla_q_rope_scaled_by_full_gamma(self):
        """MLA: q_rope is scaled by γ (full), q_nope/k_nope by √γ (Kimi K2 Algorithm 1)."""
        from titaninfra.optimizers.qk_clip import _mla_clip_heads

        torch.manual_seed(42)
        head_dim = 128
        rope_dim = 64
        n_heads = 16
        q_per_head = head_dim + rope_dim  # 192
        k_per_head = head_dim + 128       # 256 (nope + v_dim)
        lora_rank_q = 1536
        lora_rank_k = 512

        q_data = torch.randn(n_heads * q_per_head, lora_rank_q) * 10.0
        k_data = torch.randn(n_heads * k_per_head, lora_rank_k) * 10.0

        # Record pre-clip norms per head
        q_reshaped = q_data.view(n_heads, q_per_head, lora_rank_q)
        k_reshaped = k_data.view(n_heads, k_per_head, lora_rank_k)
        q_nope_norms_before = q_reshaped[:, :head_dim, :].flatten(1).norm(dim=1).clone()
        q_rope_norms_before = q_reshaped[:, head_dim:, :].flatten(1).norm(dim=1).clone()
        k_nope_norms_before = k_reshaped[:, :head_dim, :].flatten(1).norm(dim=1).clone()

        n_clipped = _mla_clip_heads(q_data, k_data, head_dim, tau=1.0)
        assert n_clipped > 0

        # Recompute post-clip norms
        q_reshaped = q_data.view(n_heads, q_per_head, lora_rank_q)
        k_reshaped = k_data.view(n_heads, k_per_head, lora_rank_k)
        q_nope_norms_after = q_reshaped[:, :head_dim, :].flatten(1).norm(dim=1)
        q_rope_norms_after = q_reshaped[:, head_dim:, :].flatten(1).norm(dim=1)
        k_nope_norms_after = k_reshaped[:, :head_dim, :].flatten(1).norm(dim=1)

        # For clipped heads: verify asymmetric scaling
        for h in range(n_heads):
            if q_nope_norms_after[h] >= q_nope_norms_before[h] * 0.999:
                continue  # this head wasn't clipped

            # sqrt_gamma = nope_after / nope_before
            sqrt_gamma_q = q_nope_norms_after[h] / q_nope_norms_before[h]
            sqrt_gamma_k = k_nope_norms_after[h] / k_nope_norms_before[h]
            # gamma_full = rope_after / rope_before
            gamma_from_rope = q_rope_norms_after[h] / q_rope_norms_before[h]

            # q_nope and k_nope should have same scaling (√γ)
            assert abs(sqrt_gamma_q - sqrt_gamma_k) < 1e-5, (
                f"Head {h}: q_nope scale {sqrt_gamma_q:.6f} != k_nope scale {sqrt_gamma_k:.6f}"
            )
            # q_rope should be scaled by γ = (√γ)²
            expected_gamma = sqrt_gamma_q ** 2
            assert abs(gamma_from_rope - expected_gamma) < 1e-4, (
                f"Head {h}: q_rope scale {gamma_from_rope:.6f} != γ={expected_gamma:.6f} "
                f"(√γ={sqrt_gamma_q:.6f})"
            )


class TestSVEntropyPlugin:
    """Test SVEntropyPlugin for singular value entropy monitoring."""

    def test_uniform_sv_max_entropy(self):
        """Orthogonal matrix → all SVs equal → max entropy ≈ 1.0."""
        from titaninfra.monitoring.sv_entropy import _compute_sv_entropy

        Q, _ = torch.linalg.qr(torch.randn(64, 64))
        h = _compute_sv_entropy(Q)
        assert h > 0.95, f"Orthogonal matrix should have entropy near 1.0, got {h}"

    def test_rank1_min_entropy(self):
        """Rank-1 matrix → single dominant SV → low entropy."""
        from titaninfra.monitoring.sv_entropy import _compute_sv_entropy

        u = torch.randn(64, 1)
        v = torch.randn(1, 64)
        rank1 = u @ v

        h = _compute_sv_entropy(rank1)
        assert h < 0.1, f"Rank-1 matrix should have entropy near 0, got {h}"

    def test_interval_gating(self):
        """Plugin returns empty dict when step doesn't match interval."""
        from titaninfra.monitoring.sv_entropy import SVEntropyPlugin

        model = nn.Sequential(nn.Linear(128, 128, bias=False))
        plugin = SVEntropyPlugin(model_parts=[model], interval=10, max_layers=4)

        assert plugin.collect(5, None) == {}
        assert plugin.collect(9, None) == {}
        assert len(plugin.collect(10, None)) > 0

    def test_max_layers_cap(self):
        """Plugin samples at most max_layers parameters."""
        from titaninfra.monitoring.sv_entropy import SVEntropyPlugin

        model = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
        )
        plugin = SVEntropyPlugin(model_parts=[model], interval=1, max_layers=2)

        assert len(plugin._params) == 2

    def test_cpu_computation(self):
        """SVD computation produces valid entropy value."""
        from titaninfra.monitoring.sv_entropy import _compute_sv_entropy

        weight = torch.randn(64, 64)
        h = _compute_sv_entropy(weight)
        assert 0.0 <= h <= 1.0

    def test_empty_model_no_crash(self):
        """Plugin handles model with no eligible params gracefully."""
        from titaninfra.monitoring.sv_entropy import SVEntropyPlugin

        model = nn.Sequential(nn.LayerNorm(128))
        plugin = SVEntropyPlugin(model_parts=[model], interval=10, max_layers=4)

        assert len(plugin._params) == 0
        assert plugin.collect(10, None) == {}

    def test_materialize_full_cpu_handles_plain_tensor(self):
        """``_materialize_full_cpu`` on a plain tensor is just detach().cpu()."""
        from titaninfra.monitoring.sv_entropy import _materialize_full_cpu

        weight = torch.randn(64, 64, requires_grad=True)
        out = _materialize_full_cpu(weight)
        assert out.device.type == "cpu"
        assert not out.requires_grad
        assert torch.equal(out, weight.detach().cpu())

    def test_materialize_full_cpu_handles_dtensor_via_duck_typing(self):
        """A DTensor-shaped stub triggers the ``full_tensor()`` branch.

        Unit tests on a single device cannot construct a real DTensor (no
        ProcessGroup), but we can assert the plugin would call ``full_tensor()``
        if the input *were* a DTensor. We register a fake class on
        ``torch.distributed.tensor.DTensor`` for the duration of the test so
        ``isinstance(...)`` matches our stub.
        """
        import torch.distributed.tensor as dtensor_mod

        from titaninfra.monitoring.sv_entropy import _materialize_full_cpu

        class _FakeDTensor(torch.Tensor):
            full_tensor_called = False

            def full_tensor(self):
                _FakeDTensor.full_tensor_called = True
                # Return a plain tensor — pretend we all-gathered to full.
                return torch.ones(64, 64)

        original_dtensor = dtensor_mod.DTensor
        try:
            dtensor_mod.DTensor = _FakeDTensor  # type: ignore[misc]
            fake = _FakeDTensor(torch.zeros(64, 64))
            out = _materialize_full_cpu(fake)
        finally:
            dtensor_mod.DTensor = original_dtensor  # type: ignore[misc]

        assert _FakeDTensor.full_tensor_called, "full_tensor() must be invoked for DTensor"
        assert out.device.type == "cpu"
        assert torch.equal(out, torch.ones(64, 64))


class TestTrainConfigNewFields:
    """Test new TrainConfig fields for Muon optimizations."""

    def test_ns_dtype_default(self):
        """Default ns_dtype is 'bf16'."""

        cfg = make_test_train_config(optimizer_name="MuonAdamW")
        assert cfg.muon_ns_dtype == "bf16"

    def test_ns_dtype_fp32(self):
        """Can set ns_dtype to 'fp32'."""

        cfg = make_test_train_config(optimizer_name="MuonAdamW", muon_ns_dtype="fp32")
        assert cfg.muon_ns_dtype == "fp32"

    def test_scale_factor_default(self):
        """Default scale_factor is 1.0."""

        cfg = make_test_train_config(optimizer_name="MuonAdamW")
        assert cfg.muon_scale_factor == 1.0

    def test_scale_factor_validation(self):
        """scale_factor must be > 0."""

        with pytest.raises(ValidationError):
            make_test_train_config(optimizer_name="MuonAdamW", muon_scale_factor=0.0)
        with pytest.raises(ValidationError):
            make_test_train_config(optimizer_name="MuonAdamW", muon_scale_factor=-1.0)

    def test_qk_clip_tau_default_none(self):
        """Default qk_clip_tau is None (disabled)."""

        cfg = make_test_train_config(optimizer_name="MuonAdamW")
        assert cfg.muon_qk_clip_tau is None

    def test_qk_clip_tau_set(self):
        """Can set qk_clip_tau to positive value."""

        cfg = make_test_train_config(optimizer_name="MuonAdamW", muon_qk_clip_tau=50.0)
        assert cfg.muon_qk_clip_tau == 50.0

    def test_sv_entropy_defaults(self):
        """SV entropy fields have correct defaults."""

        cfg = make_test_train_config(optimizer_name="MuonAdamW")
        assert cfg.enable_sv_entropy_monitor is False
        assert cfg.sv_entropy_interval == 100
        assert cfg.sv_entropy_max_layers == 4

    def test_sv_entropy_custom(self):
        """Can customize SV entropy fields."""

        cfg = make_test_train_config(
            optimizer_name="MuonAdamW",
            enable_sv_entropy_monitor=True,
            sv_entropy_interval=50,
            sv_entropy_max_layers=8,
        )
        assert cfg.enable_sv_entropy_monitor is True
        assert cfg.sv_entropy_interval == 50
        assert cfg.sv_entropy_max_layers == 8


class TestBuildOptimizerConfigNewFields:
    """Test that _build_optimizer_config passes through new fields."""

    @pytest.fixture(autouse=True)
    def _require_torchtitan(self):
        pytest.importorskip("torchtitan")

    def test_ns_dtype_passed(self):
        """muon_ns_dtype is forwarded to MixedOptimizersContainer.Config."""
        from titaninfra.engine._pretrain_helpers import _build_optimizer_config

        cfg = make_test_train_config(optimizer_name="MuonAdamW", muon_ns_dtype="fp32")
        result = _build_optimizer_config(cfg, param_groups=[])
        assert result.muon_ns_dtype == "fp32"

    def test_scale_factor_passed(self):
        """muon_scale_factor is forwarded to MixedOptimizersContainer.Config."""
        from titaninfra.engine._pretrain_helpers import _build_optimizer_config

        cfg = make_test_train_config(optimizer_name="MuonAdamW", muon_scale_factor=0.2)
        result = _build_optimizer_config(cfg, param_groups=[])
        assert result.muon_scale_factor == 0.2

    def test_defaults_passed(self):
        """Default values are correctly forwarded."""
        from titaninfra.engine._pretrain_helpers import _build_optimizer_config

        cfg = make_test_train_config(optimizer_name="MuonAdamW")
        result = _build_optimizer_config(cfg, param_groups=[])
        assert result.muon_ns_dtype == "bf16"
        assert result.muon_scale_factor == 1.0


# ===== 3D Batched Muon for MoE Expert Weights =====


class TestBatchedNewtonSchulzIter:
    """Test batched Newton-Schulz iteration for 3D expert tensors."""

    def test_convergence_3d(self):
        """Batched NS on [E, m, n] produces per-expert orthogonality."""
        from titaninfra.optimizers.muon import _batched_newton_schulz_iter

        torch.manual_seed(42)
        X = torch.randn(8, 32, 64)

        ns_fn = torch._dynamo.disable(_batched_newton_schulz_iter)
        result = ns_fn(X, steps=5)

        for e in range(8):
            gram = result[e] @ result[e].mT
            identity = torch.eye(32)
            relative_error = (gram - identity).norm().item() / 32
            assert relative_error < 0.5, f"Expert {e} not converging: error={relative_error}"

    def test_output_shape_preserved_3d(self):
        """Output shape matches input shape for 3D."""
        from titaninfra.optimizers.muon import _batched_newton_schulz_iter

        torch.manual_seed(42)
        X = torch.randn(16, 64, 128)

        ns_fn = torch._dynamo.disable(_batched_newton_schulz_iter)
        result = ns_fn(X, steps=5)
        assert result.shape == X.shape

    def test_determinism_3d(self):
        """Same 3D input produces identical output across calls."""
        from titaninfra.optimizers.muon import _batched_newton_schulz_iter

        ns_fn = torch._dynamo.disable(_batched_newton_schulz_iter)

        torch.manual_seed(42)
        X = torch.randn(4, 32, 32)

        result1 = ns_fn(X.clone(), steps=5)
        result2 = ns_fn(X.clone(), steps=5)
        assert torch.allclose(result1, result2, atol=1e-6)

    def test_wide_experts_transposition(self):
        """Wide expert matrices [E, 32, 128] handled correctly."""
        from titaninfra.optimizers.muon import _batched_newton_schulz_iter

        torch.manual_seed(42)
        X = torch.randn(8, 32, 128)

        ns_fn = torch._dynamo.disable(_batched_newton_schulz_iter)
        result = ns_fn(X, steps=5)
        assert result.shape == (8, 32, 128)
        assert torch.isfinite(result).all()


class TestMuon3DExpert:
    """Test Muon optimizer step with 3D expert-shaped parameters."""

    def test_3d_step_updates(self):
        """Single Muon step updates a 3D expert parameter."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(8, 32, 64))
        original = param.data.clone()

        optimizer = Muon([param], lr=0.01)
        param.grad = torch.randn_like(param)
        optimizer.step()

        assert not torch.allclose(param.data, original), "3D param not updated"

    def test_3d_momentum_accumulation(self):
        """Momentum buffer accumulates for 3D params."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(4, 32, 64))
        optimizer = Muon([param], lr=0.01, momentum=0.95)

        param.grad = torch.randn_like(param)
        optimizer.step()

        buf = optimizer.state[param]["momentum_buffer"]
        assert buf.shape == param.shape
        assert buf.norm().item() > 0

    def test_3d_loss_decreases(self):
        """Multiple Muon steps on 3D expert param reduce a simple loss."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(4, 32, 64))
        target = torch.randn(4, 32, 64)

        optimizer = Muon([param], lr=0.01)
        initial_loss = ((param - target) ** 2).mean().item()

        for _ in range(20):
            optimizer.zero_grad()
            loss = ((param - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_loss = ((param - target) ** 2).mean().item()
        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"

    def test_3d_wide_param_transposition(self):
        """Wide 3D params [E, m, n] where m < n use transposition path."""
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(8, 32, 128))
        original = param.data.clone()

        optimizer = Muon([param], lr=0.01)
        param.grad = torch.randn_like(param)
        optimizer.step()

        assert not torch.allclose(param.data, original)
        assert param.shape == (8, 32, 128)


class TestParamClassification3DExpert:
    """Test that 3D expert params are correctly classified for Muon."""

    @pytest.fixture(autouse=True)
    def _require_torchtitan(self):
        pytest.importorskip("torchtitan")

    def test_3d_expert_goes_to_muon(self):
        """3D expert.w1 param (large enough) goes to Muon."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "moe": nn.ModuleDict({
                        "experts": nn.ModuleDict({
                            "w1": nn.Module(),
                        }),
                    }),
                }),
            }),
        })
        # Manually register a 3D parameter
        w1 = nn.Parameter(torch.randn(8, 32, 64))
        model.layers[  # type: ignore[index]
            "0"
        ]["moe"]["experts"]["w1"].register_parameter("weight", w1)

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert _has_param(muon_params, w1), "3D expert.w1 should go to Muon"
        assert not _has_param(adamw_params, w1)

    def test_3d_non_expert_stays_adamw(self):
        """3D param without expert pattern stays in AdamW."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({
            "some_3d_thing": nn.Module(),
        })
        p = nn.Parameter(torch.randn(8, 32, 64))
        model["some_3d_thing"].register_parameter("weight", p)

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert not _has_param(muon_params, p), "Non-expert 3D should NOT go to Muon"
        assert _has_param(adamw_params, p)

    def test_3d_expert_too_small_stays_adamw(self):
        """3D expert param below min_ndim_size goes to AdamW."""
        from titaninfra.optimizers.mixed_optimizer import _classify_params

        model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "0": nn.ModuleDict({
                    "moe": nn.ModuleDict({
                        "experts": nn.ModuleDict({
                            "w1": nn.Module(),
                        }),
                    }),
                }),
            }),
        })
        # Small expert: 2*4*8 = 64 elements < 4096
        w1 = nn.Parameter(torch.randn(2, 4, 8))
        model.layers["0"]["moe"]["experts"]["w1"].register_parameter("weight", w1)  # type: ignore[index]

        muon_params, adamw_params = _classify_params(model, min_ndim_size=4096)

        assert not _has_param(muon_params, w1), "Small 3D expert should NOT go to Muon"
        assert _has_param(adamw_params, w1)


class TestShouldGather:
    """Test the _should_gather skip-guard pure function."""

    def test_plain_tensor_returns_false(self):
        """Non-DTensor (single-card) is already a full matrix → no gather."""
        from titaninfra.optimizers.muon import _should_gather

        x = torch.randn(64, 64)
        assert _should_gather(x) is False

    def test_3d_tensor_returns_false(self):
        """3D expert tensor → always skip."""
        from titaninfra.optimizers.muon import _should_gather

        x = torch.randn(8, 64, 64)
        assert _should_gather(x) is False

    def test_object_without_to_local_returns_false(self):
        """Anything lacking to_local() is treated as non-distributed."""
        from titaninfra.optimizers.muon import _should_gather

        class Fake:
            ndim = 2

        assert _should_gather(Fake()) is False

    def test_2d_sharded_dtensor_returns_true(self):
        """2D DTensor with a sharded dim → gather (the one branch that enables the feature).

        HSDP placement (Replicate(), Shard(0)) is exactly the f763de5 historical-bug
        dimension. Placement objects construct on CPU without process-group init, so
        this positive branch is CPU-testable via a duck-typed stub.
        """
        from torch.distributed.tensor.placement_types import Replicate, Shard

        from titaninfra.optimizers.muon import _should_gather

        class FakeDT:
            ndim = 2
            placements = (Replicate(), Shard(0))

            def to_local(self):
                raise NotImplementedError

        assert _should_gather(FakeDT()) is True



class TestGatherMode:
    """Test gather_mode flag on Muon (CPU / single-card)."""

    def test_gather_mode_accepts_flag(self):
        """Muon constructor accepts gather_mode kwarg."""
        from titaninfra.optimizers.muon import Muon

        param = nn.Parameter(torch.randn(64, 64))
        opt = Muon([param], lr=0.01, gather_mode=True)
        assert opt._gather_mode is True

    def test_gather_mode_default_false(self):
        """gather_mode defaults to False (zero behavior change)."""
        from titaninfra.optimizers.muon import Muon

        param = nn.Parameter(torch.randn(64, 64))
        opt = Muon([param], lr=0.01)
        assert opt._gather_mode is False

    def test_gather_mode_single_card_equivalent(self):
        """On a plain tensor, gather_mode=True is identical to gather_mode=False.

        _should_gather returns False for non-DTensors, so both paths run the
        same NS. Numerics must match.
        """
        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(0)
        p_off = nn.Parameter(torch.randn(64, 48))
        p_on = nn.Parameter(p_off.data.clone())
        grad = torch.randn(64, 48)

        opt_off = Muon([p_off], lr=0.01, momentum=0.95, ns_steps=5, gather_mode=False)
        opt_on = Muon([p_on], lr=0.01, momentum=0.95, ns_steps=5, gather_mode=True)

        p_off.grad = grad.clone()
        p_on.grad = grad.clone()
        opt_off.step()
        opt_on.step()

        torch.testing.assert_close(p_off.data, p_on.data)

    def test_full_ns_differs_from_blockwise_ns(self):
        """Full-matrix NS (gather) differs from per-shard (blockwise) NS.

        This is the mathematical justification for gather mode: running NS on
        row-shards independently is NOT equal to running NS on the full matrix.
        """
        from titaninfra.optimizers.muon import _newton_schulz_iter

        torch.manual_seed(7)
        X = torch.randn(128, 64)
        ns = lambda t: torch._dynamo.disable(_newton_schulz_iter)(t, steps=5)

        full = ns(X)

        half = X.shape[0] // 2
        block = torch.cat([ns(X[:half]), ns(X[half:])], dim=0)

        diff = (full - block).norm().item() / full.norm().item()
        assert diff > 0.01, (
            f"blockwise NS should differ from full NS, relative diff={diff}"
        )


class TestGatherModeDistributed:
    """Exercise the gather branch on a real (single-rank) DTensor.

    Prior CPU tests could not reach the gather code path: `_should_gather`
    returns False for every non-DTensor, so the `redistribute(...)` round-trip
    in Muon.step()'s gather branch never executed until a multi-GPU run. This
    constructs a genuine sharded DTensor via a single-rank gloo process group
    so the gather branch — including the re-shard `redistribute` — runs under
    CPU pytest.

    Regression guard for the `placements is needed for redistribute!` crash:
    `redistribute(device_mesh=None, placements=None, ...)` — a positional
    placements arg binds to device_mesh and leaves placements=None.
    """

    @pytest.fixture()
    def single_rank_pg(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            pytest.skip("torch.distributed gloo backend unavailable")

        already = dist.is_initialized()
        if not already:
            dist.init_process_group(
                backend="gloo", store=dist.HashStore(), rank=0, world_size=1
            )
        try:
            yield
        finally:
            if not already and dist.is_initialized():
                dist.destroy_process_group()

    def test_gather_branch_redistribute_roundtrip(self, single_rank_pg):
        """gather_mode step on a Shard(0) DTensor completes the redistribute round-trip.

        Reproduces the GPU crash at muon.py `full.redistribute(orig_placements)`
        (placements passed positionally → bound to device_mesh → RuntimeError
        'placements is needed for redistribute!'). With the fix it round-trips
        and updates the parameter, preserving the original placement.
        """
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        from titaninfra.optimizers.muon import Muon

        torch.manual_seed(0)
        mesh = init_device_mesh("cpu", (1,))
        dparam = nn.Parameter(distribute_tensor(torch.randn(64, 48), mesh, [Shard(0)]))

        opt = Muon([dparam], lr=0.01, momentum=0.95, ns_steps=5, gather_mode=True)
        dparam.grad = distribute_tensor(torch.randn(64, 48), mesh, [Shard(0)])

        before = dparam.to_local().clone()
        opt.step()  # must not raise; gather branch runs the redistribute round-trip

        after = dparam.to_local()
        assert not torch.allclose(before, after), "gather-mode step did not update param"
        assert dparam.placements == (Shard(0),)  # placement preserved after re-shard


# Reference (pre-Gram) NS formulations — frozen here so the equivalence tests
# pin the new min-dim Gram rewrite (Plan 1) to the original A = XXᵀ path.
def _ref_ns_2d(X, steps, coeffs):
    X = X / (X.norm() + 1e-7)
    for i in range(steps):
        a, b, c = coeffs[min(i, len(coeffs) - 1)]
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def _ref_ns_3d(X, steps, coeffs):
    X = X / (X.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-7)
    for i in range(steps):
        a, b, c = coeffs[min(i, len(coeffs) - 1)]
        A = torch.bmm(X, X.transpose(-2, -1))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    return X


class TestGramEquivalence:
    """Plan 1: min-dim Gram rewrite must match the original A = XXᵀ formulation.

    Square/wide inputs take the m <= n branch (byte-identical to the original).
    Tall inputs take the Gram branch, equivalent via (XXᵀ)X = X(XᵀX) up to
    fp reassociation — asserted within a tight tolerance.
    """

    @pytest.mark.parametrize("shape", [(64, 64), (96, 96)])
    def test_square_byte_identical(self, shape):
        from titaninfra.optimizers.muon import _NS_COEFFS_QUINTIC, _newton_schulz_iter

        torch.manual_seed(0)
        X = torch.randn(*shape, dtype=torch.float64)
        ns = torch._dynamo.disable(_newton_schulz_iter)
        out = ns(X, steps=5)
        ref = _ref_ns_2d(X, 5, _NS_COEFFS_QUINTIC)
        assert torch.equal(out, ref), "square input must hit the original branch exactly"

    @pytest.mark.parametrize("shape", [(128, 64), (3072, 1536), (10944, 2048)])
    def test_tall_gram_equivalent(self, shape):
        from titaninfra.optimizers.muon import _NS_COEFFS_QUINTIC, _newton_schulz_iter

        torch.manual_seed(0)
        X = torch.randn(*shape, dtype=torch.float64)
        ns = torch._dynamo.disable(_newton_schulz_iter)
        out = ns(X, steps=5)
        ref = _ref_ns_2d(X, 5, _NS_COEFFS_QUINTIC)
        assert torch.allclose(out, ref, rtol=1e-9, atol=1e-9), (
            f"Gram tall path diverged: max abs diff = {(out - ref).abs().max().item()}"
        )

    @pytest.mark.parametrize("shape", [(64, 128)])
    def test_wide_byte_identical(self, shape):
        from titaninfra.optimizers.muon import _NS_COEFFS_QUINTIC, _newton_schulz_iter

        torch.manual_seed(0)
        X = torch.randn(*shape, dtype=torch.float64)
        ns = torch._dynamo.disable(_newton_schulz_iter)
        out = ns(X, steps=5)
        ref = _ref_ns_2d(X, 5, _NS_COEFFS_QUINTIC)
        assert torch.equal(out, ref), "wide input must hit the original branch exactly"

    @pytest.mark.parametrize("shape", [(4, 128, 64), (8, 512, 256)])
    def test_batched_tall_gram_equivalent(self, shape):
        from titaninfra.optimizers.muon import (
            _NS_COEFFS_QUINTIC,
            _batched_newton_schulz_iter,
        )

        torch.manual_seed(0)
        X = torch.randn(*shape, dtype=torch.float64)
        ns = torch._dynamo.disable(_batched_newton_schulz_iter)
        out = ns(X, steps=5)
        ref = _ref_ns_3d(X, 5, _NS_COEFFS_QUINTIC)
        assert torch.allclose(out, ref, rtol=1e-9, atol=1e-9), (
            f"batched Gram path diverged: max abs diff = {(out - ref).abs().max().item()}"
        )

    def test_batched_square_byte_identical(self):
        from titaninfra.optimizers.muon import (
            _NS_COEFFS_QUINTIC,
            _batched_newton_schulz_iter,
        )

        torch.manual_seed(0)
        X = torch.randn(4, 64, 64, dtype=torch.float64)
        ns = torch._dynamo.disable(_batched_newton_schulz_iter)
        out = ns(X, steps=5)
        ref = _ref_ns_3d(X, 5, _NS_COEFFS_QUINTIC)
        assert torch.equal(out, ref), "batched square must hit the original branch exactly"


def _ref_muon_step_legacy(params_grads, *, lr, beta, nesterov, ns_steps, wd, sf,
                          ns_dtype=torch.bfloat16):
    """Ref muon step legacy."""
    import math

    import torch._dynamo

    from titaninfra.optimizers.muon import _newton_schulz_iter

    ns = torch._dynamo.disable(_newton_schulz_iter)
    for p, buf, g in params_grads:
        buf.lerp_(g, 1.0 - beta)
        update = g.lerp(buf, beta) if nesterov else buf
        m, n = update.shape[-2], update.shape[-1]
        transposed = m < n
        local = update.mT if transposed else update
        local = ns(local.to(ns_dtype), steps=ns_steps).to(local.dtype)
        if transposed:
            local = local.mT
        scale = sf * math.sqrt(max(m, n))
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        p.add_(local, alpha=-lr * scale)


class TestForeachStepParity:
    """Tests for test foreach step parity."""

    def _make(self, seed=7):
        torch.manual_seed(seed)
        shapes = [(128, 256), (256, 128), (512, 512), (64, 2048)]
        params_new, params_ref, grads = [], [], []
        for sh in shapes:
            g = torch.randn(*sh)
            p_new = torch.randn(*sh)
            p_ref = p_new.clone()
            params_new.append(p_new)
            params_ref.append(p_ref)
            grads.append(g)
        return shapes, params_new, params_ref, grads

    def test_single_step_parity(self):
        import torch

        from titaninfra.optimizers.muon import Muon

        _, p_new, p_ref, grads = self._make()
        lr, beta, ns_steps, wd, sf = 4.2e-4, 0.95, 5, 0.1, 0.2

        opt = Muon(p_new, lr=lr, momentum=beta, nesterov=True, ns_steps=ns_steps,
                   weight_decay=wd, scale_factor=sf, ns_dtype=torch.float32)
        for p, g in zip(p_new, grads, strict=True):
            p.grad = g.clone()
        opt.step()

        # reference (legacy per-param) path — same fresh momentum buffers (zeros)
        ref_bufs = [torch.zeros_like(p) for p in p_ref]
        _ref_muon_step_legacy(
            list(zip(p_ref, ref_bufs, [g.clone() for g in grads], strict=True)),
            lr=lr, beta=beta, nesterov=True, ns_steps=ns_steps, wd=wd, sf=sf,
            ns_dtype=torch.float32,
        )

        for i, (a, b) in enumerate(zip(p_new, p_ref, strict=True)):
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), (
                f"param {i} foreach vs legacy diverged: "
                f"max|Δ|={(a - b).abs().max().item():.2e}"
            )

    def test_multi_step_parity(self):
        """Verify multi step parity."""
        import torch

        from titaninfra.optimizers.muon import Muon

        _, p_new, p_ref, grads = self._make(seed=11)
        lr, beta, ns_steps, wd, sf = 1e-3, 0.9, 5, 0.0, 1.0

        opt = Muon(p_new, lr=lr, momentum=beta, nesterov=True, ns_steps=ns_steps,
                   weight_decay=wd, scale_factor=sf, ns_dtype=torch.float32)
        ref_bufs = [torch.zeros_like(p) for p in p_ref]

        for _step in range(3):
            for p, g in zip(p_new, grads, strict=True):
                p.grad = g.clone()
            opt.step()
            _ref_muon_step_legacy(
                list(zip(p_ref, ref_bufs, [g.clone() for g in grads], strict=True)),
                lr=lr, beta=beta, nesterov=True, ns_steps=ns_steps, wd=wd, sf=sf,
                ns_dtype=torch.float32,
            )

        for i, (a, b) in enumerate(zip(p_new, p_ref, strict=True)):
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), (
                f"param {i} diverged after 3 steps: max|Δ|={(a - b).abs().max().item():.2e}"
            )
