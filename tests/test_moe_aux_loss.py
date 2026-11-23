"""Tests for MoE auxiliary-loss load balancing (spec 2026-06-09)."""
from __future__ import annotations

from tests.conftest import make_test_model_config


class TestConfigCoupling:
    def test_coeff_coupling_allowed(self):
        """Verify coeff coupling allowed."""

        cfg = make_test_model_config(
            name="deepseek_v3/16B",
            moe_load_balance_coeff=1e-3,
            moe_aux_loss_coeff=1e-4,
        )
        assert cfg.moe_load_balance_coeff == 1e-3
        assert cfg.moe_aux_loss_coeff == 1e-4

    def test_aux_loss_coeff_alone_ok(self):

        cfg = make_test_model_config(name="qwen3/30B-A3B", moe_aux_loss_coeff=1e-2)
        assert cfg.moe_aux_loss_coeff == 1e-2
        assert cfg.moe_load_balance_coeff is None

    def test_neither_set_ok(self):

        cfg = make_test_model_config(name="qwen3/30B-A3B")
        assert cfg.moe_aux_loss_coeff is None


class TestPureCompute:
    def test_normalize_counts_sums_to_one(self):
        import torch

        from titaninfra.monitoring._moe_aux_compute import normalize_counts

        counts = torch.tensor([10.0, 30.0, 60.0, 0.0])
        f = normalize_counts(counts)
        assert torch.isclose(f.sum(), torch.tensor(1.0))
        assert torch.allclose(f, torch.tensor([0.1, 0.3, 0.6, 0.0]))

    def test_normalize_counts_uniform(self):
        import torch

        from titaninfra.monitoring._moe_aux_compute import normalize_counts

        counts = torch.full((8,), 5.0)
        f = normalize_counts(counts)
        assert torch.allclose(f, torch.full((8,), 1.0 / 8))

    def test_normalize_counts_all_zero_is_uniform(self):
        import torch

        from titaninfra.monitoring._moe_aux_compute import normalize_counts

        f = normalize_counts(torch.zeros(4))
        assert torch.allclose(f, torch.full((4,), 0.25))  # no NaN

    def test_balanced_routing_loss_is_one(self):
        import torch

        from titaninfra.monitoring._moe_aux_compute import compute_load_balancing_loss

        n = 8
        f_global = torch.full((n,), 1.0 / n)
        p_sum = torch.full((n,), 1.0 / n) * 100  # 100 tokens, uniform prob
        loss = compute_load_balancing_loss(p_sum, token_count=100, f_global=f_global, n_experts=n)
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)

    def test_collapsed_routing_loss_is_n(self):
        import torch

        from titaninfra.monitoring._moe_aux_compute import compute_load_balancing_loss

        n = 8
        f_global = torch.zeros(n)
        f_global[0] = 1.0
        p_sum = torch.zeros(n)
        p_sum[0] = 100.0  # all prob mass on expert 0
        loss = compute_load_balancing_loss(p_sum, token_count=100, f_global=f_global, n_experts=n)
        assert torch.isclose(loss, torch.tensor(float(n)), atol=1e-5)

    def test_matches_reference_implementation(self):
        import torch

        from titaninfra.monitoring._moe_aux_compute import compute_load_balancing_loss, normalize_counts

        torch.manual_seed(0)
        n, top_k, T = 8, 2, 64
        logits = torch.randn(T, n)
        probs = torch.softmax(logits, dim=-1)
        _, sel = torch.topk(probs, top_k, dim=-1)

        # Reference (sum-to-1 convention): f_i = count_i / (top_k * T), P_i = mean prob
        counts = torch.histc(sel.float().view(-1), bins=n, min=0, max=n)
        f_ref = counts / (top_k * T)
        p_ref = probs.mean(dim=0)
        ref_loss = n * (f_ref * p_ref).sum()

        f_global = normalize_counts(counts)  # == f_ref since counts sum to top_k*T
        p_sum = probs.sum(dim=0)
        got = compute_load_balancing_loss(p_sum, token_count=T, f_global=f_global, n_experts=n)
        assert torch.isclose(got, ref_loss, atol=1e-6)

    def test_gradient_flows_to_logits(self):
        import torch

        from titaninfra.monitoring._moe_aux_compute import compute_load_balancing_loss

        n, T = 8, 32
        logits = torch.randn(T, n, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        p_sum = probs.sum(dim=0)  # grad-carrying
        f_global = torch.full((n,), 1.0 / n)  # detached weighting
        loss = compute_load_balancing_loss(p_sum, token_count=T, f_global=f_global, n_experts=n)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_f_global_detached(self):
        import torch

        from titaninfra.monitoring._moe_aux_compute import compute_load_balancing_loss

        n, T = 4, 16
        f_global = torch.full((n,), 1.0 / n, requires_grad=True)
        # p_sum carries grad so the loss is connected to the autograd graph and
        # backward() can run; f_global is the input under test.
        p_sum = torch.full((n,), float(T) / n, requires_grad=True)
        loss = compute_load_balancing_loss(p_sum, token_count=T, f_global=f_global, n_experts=n)
        loss.backward()
        # f_global must be treated as a constant: compute detaches it internally,
        # so no grad reaches it (stays None), while p_sum does receive grad.
        assert f_global.grad is None
        assert p_sum.grad is not None


class _FakeAuxSource:
    """Minimal stand-in for MoEAuxLossPlugin: returns a fixed grad-carrying scalar."""

    def __init__(self, value_tensor):
        self._v = value_tensor

    def compute_microbatch_aux_loss(self):
        return self._v


class TestLossFnAux:
    def test_weighted_ce_adds_aux(self):
        import torch

        from titaninfra.data.loss import WeightedCELoss

        # inner CE (3-arg upstream contract) that returns a known sum tuple
        def inner(pred, labels, gvt=None):
            return torch.tensor(5.0), {}

        loss_fn = WeightedCELoss(inner)
        aux_val = torch.tensor(2.0, requires_grad=True)
        loss_fn.set_aux_loss_source(_FakeAuxSource(aux_val))

        pred = torch.zeros(1, 2, 4)
        labels = torch.full((1, 2), -100)
        out, _ = loss_fn(pred, labels)
        assert torch.isclose(out, torch.tensor(7.0))

    def test_weighted_ce_no_aux_source_unchanged(self):
        import torch

        from titaninfra.data.loss import WeightedCELoss

        def inner(pred, labels, gvt=None):
            return torch.tensor(5.0), {}

        loss_fn = WeightedCELoss(inner)
        pred = torch.zeros(1, 2, 4)
        labels = torch.full((1, 2), -100)
        out, _ = loss_fn(pred, labels)
        assert torch.isclose(out, torch.tensor(5.0))

class TestPlugin:
    def _make_fake_model(self, n_layers, n_experts, dim, top_k, device="cpu"):
        import torch
        from torch import nn

        class FakeRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(dim, n_experts, bias=False)
                self.top_k = top_k
                self.num_experts = n_experts
                self.score_func = "softmax"

            def forward(self, x):
                # Mirror torchtitan TokenChoiceTopKRouter (post-P4):
                #   returns (topk_scores, topk_expert_ids, scores_full)
                # scores_full is the already-normalized per-expert prob.
                scores = torch.softmax(self.gate(x), dim=-1)
                _, sel = torch.topk(scores, top_k, dim=-1)
                return scores.gather(-1, sel), sel, scores

        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = FakeRouter()
                self.gate = self.router.gate
                self.register_buffer("tokens_per_expert", torch.zeros(n_experts))

            def forward(self, x):
                return self.router(x)[0]

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_enabled = True
                self.moe = FakeMoE()

            def forward(self, x):
                self.moe(x)
                return x

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict({str(i): FakeBlock() for i in range(n_layers)})

            def get_submodule(self, target):
                return self.layers

            def forward(self, x):
                for b in self.layers.values():
                    b(x)
                return x

        return FakeModel().to(device)

    def test_plugin_no_moe_noop(self):
        from torch import nn

        from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

        class Dense(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict({"0": nn.Linear(4, 4)})

            def get_submodule(self, target):
                return self.layers

        plugin = MoEAuxLossPlugin(model_parts=[Dense()], coeff=1e-2)
        assert plugin.has_moe is False
        out = plugin.compute_microbatch_aux_loss()
        assert float(out) == 0.0
        plugin.reduce_and_cache_counts(loss_pg=None)

    def test_first_step_uniform_f_global(self):
        import torch

        from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

        model = self._make_fake_model(n_layers=2, n_experts=8, dim=16, top_k=2)
        plugin = MoEAuxLossPlugin(model_parts=[model], coeff=1.0)
        assert plugin.has_moe is True
        for f in plugin._f_global:
            assert torch.allclose(f, torch.full((8,), 1.0 / 8))

    def test_aux_loss_grad_flows_to_gate(self):
        import torch

        from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

        model = self._make_fake_model(n_layers=1, n_experts=8, dim=16, top_k=2)
        plugin = MoEAuxLossPlugin(model_parts=[model], coeff=1.0)
        plugin.set_step_scale(gvt=64.0, gas=1)

        x = torch.randn(64, 16)
        model(x)
        aux = plugin.compute_microbatch_aux_loss()
        assert aux.requires_grad
        aux.backward()
        gate_w = model.layers["0"].moe.router.gate.weight
        assert gate_w.grad is not None and gate_w.grad.abs().sum() > 0

    def test_compute_clears_microbatch_buffers(self):
        import torch

        from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

        model = self._make_fake_model(n_layers=1, n_experts=8, dim=16, top_k=2)
        plugin = MoEAuxLossPlugin(model_parts=[model], coeff=1.0)
        plugin.set_step_scale(gvt=64.0, gas=1)

        x = torch.randn(64, 16)
        model(x)
        _ = plugin.compute_microbatch_aux_loss()
        again = plugin.compute_microbatch_aux_loss()
        assert float(again) == 0.0

    def test_reduce_single_rank_updates_f_global(self):
        import torch

        from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

        model = self._make_fake_model(n_layers=1, n_experts=8, dim=16, top_k=2)
        plugin = MoEAuxLossPlugin(model_parts=[model], coeff=1.0)
        plugin.set_step_scale(gvt=64.0, gas=1)

        x = torch.randn(64, 16)
        model(x)
        _ = plugin.compute_microbatch_aux_loss()
        plugin.reduce_and_cache_counts(loss_pg=None)
        f = plugin._f_global[0]
        assert torch.isclose(f.sum(), torch.tensor(1.0), atol=1e-5)

    def test_collect_returns_aux_scalar(self):
        import torch

        from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

        # Seed for determinism — value must not depend on sibling-test RNG state.
        torch.manual_seed(0)
        n_experts = 8
        model = self._make_fake_model(n_layers=1, n_experts=n_experts, dim=16, top_k=2)
        plugin = MoEAuxLossPlugin(model_parts=[model], coeff=1.0)
        plugin.set_step_scale(gvt=64.0, gas=1)
        x = torch.randn(64, 16)
        model(x)
        _ = plugin.compute_microbatch_aux_loss()
        metrics = plugin.collect(step=10, trainer=None)
        assert "moe/aux_loss" in metrics
        # Unscaled L_aux ∈ [1, N_experts]. On step 1, f_global is uniform 1/N so
        # L_aux = Σ P_i = 1.0 in exact arithmetic; float32 softmax summation may
        # land a hair below 1.0, so allow a small tolerance on the lower bound.
        assert 1.0 - 1e-4 <= metrics["moe/aux_loss"] <= n_experts + 1e-4


class TestEngineWiring:
    def test_loss_fn_wrapped_when_aux_enabled(self):
        """Simulates the wiring build_and_inject_data_metrics performs: a plain
        loss_fn gets WeightedCELoss-wrapped and the aux source is attached."""
        import torch
        from torch import nn

        from titaninfra.data.loss import WeightedCELoss
        from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

        class Dense(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict({"0": nn.Linear(4, 4)})

            def get_submodule(self, target):
                return self.layers

        def plain_ce(pred, labels):
            return torch.tensor(1.0)

        loss_fn = plain_ce
        plugin = MoEAuxLossPlugin(model_parts=[Dense()], coeff=1e-2)
        if not hasattr(loss_fn, "set_aux_loss_source"):
            loss_fn = WeightedCELoss(loss_fn)
        loss_fn.set_aux_loss_source(plugin)

        assert hasattr(loss_fn, "set_aux_loss_source")
        assert loss_fn._aux_source is plugin
