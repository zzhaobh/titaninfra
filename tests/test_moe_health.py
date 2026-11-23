"""Unit tests for moe health."""

import torch
from torch import nn

from titaninfra.monitoring.moe_health import (
    LayerLoadMetrics,
    MoEHealthPlugin,
    _compute_layer_load_metrics,
    _find_moe_layers,
)


class MockGate(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockRouter(nn.Module):
    def __init__(self, dim: int = 16, num_experts: int = 8):
        super().__init__()
        self.gate = MockGate(dim, num_experts)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, expert_bias=None):
        # Mirror torchtitan P4 router.forward return:
        # (topk_scores_BLK, topk_expert_ids_BLK, scores_BLE). No count histogram.
        scores = self.gate(x)
        if expert_bias is not None:
            scores = scores + expert_bias
        top_scores, indices = torch.topk(scores, k=2, dim=-1)
        return top_scores, indices, scores


def _num_tokens_per_expert(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.histc(
        indices.view(-1).float(), bins=num_experts, min=0, max=num_experts
    )


class MockMoE(nn.Module):
    def __init__(self, num_experts: int = 8, dim: int = 16):
        super().__init__()
        self.router = MockRouter(dim=dim, num_experts=num_experts)
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_scores, indices, scores = self.router(x)
        num_tokens_per_expert = _num_tokens_per_expert(indices, self.router.num_experts)
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
        return x


class MockTransformerBlock(nn.Module):
    def __init__(self, moe_enabled: bool = True, num_experts: int = 8, dim: int = 16):
        super().__init__()
        self.moe_enabled = moe_enabled
        if moe_enabled:
            self.moe = MockMoE(num_experts, dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.moe_enabled:
            return self.moe(x)
        return x


class MockModelPart(nn.Module):
    def __init__(self, n_moe_layers: int = 4, n_dense_layers: int = 2, num_experts: int = 8, dim: int = 16):
        super().__init__()
        layers = {}
        for i in range(n_moe_layers):
            layers[str(i)] = MockTransformerBlock(moe_enabled=True, num_experts=num_experts, dim=dim)
        for i in range(n_moe_layers, n_moe_layers + n_dense_layers):
            layers[str(i)] = MockTransformerBlock(moe_enabled=False, dim=dim)
        self.layers = nn.ModuleDict(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            x = layer(x)
        return x


def _get_moes(model: MockModelPart) -> list[MockMoE]:
    return [block.moe for block in model.layers.values() if block.moe_enabled]


def _run_router_with_counts(moe: MockMoE, counts: torch.Tensor):
    """Directly fire the router hook with predetermined expert counts.

    Instead of relying on actual routing (which depends on random weights),
    we monkey-patch the router to emit expert ids whose one-hot histogram
    reproduces ``counts`` — matching torchtitan P4's (scores, ids, scores_BLE)
    return where the hook rebuilds counts from ids, not from a count tensor.
    """
    num_experts = moe.router.num_experts
    # One token-row per routed slot: expert e repeated counts[e] times, k=1.
    rows = torch.repeat_interleave(
        torch.arange(num_experts), counts.to(torch.long)
    )
    n = max(int(rows.numel()), 1)
    if rows.numel() == 0:
        rows = torch.zeros(1, dtype=torch.long)
    indices = rows.view(-1, 1)  # (N, 1) selected expert ids
    scores = torch.zeros(n, num_experts)

    original_forward = moe.router.forward

    def patched_forward(*args, **kwargs):
        top_scores = torch.zeros(n, 1)
        return top_scores, indices, scores

    moe.router.forward = patched_forward
    x = torch.randn(n, moe.router.gate.linear.in_features)
    moe(x)
    moe.router.forward = original_forward


# ── Tests ──


class TestFindMoeLayers:
    def test_finds_moe_layers(self):
        model = MockModelPart(n_moe_layers=4, n_dense_layers=2)
        moe_layers = _find_moe_layers([model])
        assert len(moe_layers) == 4

    def test_no_moe_layers(self):
        model = MockModelPart(n_moe_layers=0, n_dense_layers=4)
        moe_layers = _find_moe_layers([model])
        assert len(moe_layers) == 0

    def test_multiple_model_parts(self):
        m1 = MockModelPart(n_moe_layers=2, n_dense_layers=1)
        m2 = MockModelPart(n_moe_layers=3, n_dense_layers=0)
        moe_layers = _find_moe_layers([m1, m2])
        assert len(moe_layers) == 5


class TestMoEHealthPlugin:
    def _make_plugin(self, n_moe_layers=4, num_experts=8, has_ac=False, dead_window_size=100):
        model = MockModelPart(n_moe_layers=n_moe_layers, num_experts=num_experts)
        plugin = MoEHealthPlugin(model_parts=[model], has_ac=has_ac, dead_window_size=dead_window_size)
        return plugin, model

    def test_has_moe_true(self):
        plugin, _ = self._make_plugin(n_moe_layers=2)
        assert plugin.has_moe is True

    def test_has_moe_false(self):
        model = MockModelPart(n_moe_layers=0, n_dense_layers=4)
        plugin = MoEHealthPlugin(model_parts=[model])
        assert plugin.has_moe is False

    def test_collect_empty_when_no_moe(self):
        model = MockModelPart(n_moe_layers=0, n_dense_layers=4)
        plugin = MoEHealthPlugin(model_parts=[model])
        assert plugin.collect(step=1, trainer=None) == {}

    def test_collect_after_forward_uniform(self):
        """Verify collect after forward uniform."""
        plugin, model = self._make_plugin(n_moe_layers=2, num_experts=4)
        for moe in _get_moes(model):
            _run_router_with_counts(moe, torch.tensor([100.0, 100.0, 100.0, 100.0]))

        metrics = plugin.collect(step=1, trainer=None)
        assert "moe/expert_load_cv" in metrics
        assert metrics["moe/expert_load_cv"] < 0.01

    def test_collect_after_forward_skewed(self):
        """Verify collect after forward skewed."""
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4)
        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([0.0, 0.0, 100.0, 200.0]))

        metrics = plugin.collect(step=1, trainer=None)
        assert metrics["moe/expert_load_cv"] > 0.5
        assert metrics["moe/expert_util_min"] == 0.0

    def test_imbalance_ratio(self):
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4)
        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([10.0, 10.0, 10.0, 100.0]))

        metrics = plugin.collect(step=1, trainer=None)
        assert "moe/imbalance_ratio" in metrics
        assert metrics["moe/imbalance_ratio"] == 10.0

    def test_imbalance_ratio_with_zero_is_capped(self):
        """Verify imbalance ratio with zero is capped."""
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4)
        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([0.0, 10.0, 20.0, 30.0]))

        metrics = plugin.collect(step=1, trainer=None)
        assert "moe/imbalance_ratio" in metrics
        assert metrics["moe/imbalance_ratio"] <= 1000.0

    def test_imbalance_ratio_cap_at_1000(self):
        """Verify imbalance ratio cap at 1000."""
        counts = [torch.tensor([0.0001, 0.0, 0.0, 10000.0])]
        m = _compute_layer_load_metrics(counts, num_experts=4)
        assert m is not None
        assert m.imbalance_ratio <= 1000.0

    def test_dead_expert_sliding_window(self):
        """Verify dead expert sliding window."""
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4, dead_window_size=3)
        moe = _get_moes(model)[0]

        _run_router_with_counts(moe, torch.tensor([0.0, 100.0, 100.0, 100.0]))
        plugin.collect(step=1, trainer=None)

        _run_router_with_counts(moe, torch.tensor([50.0, 100.0, 100.0, 100.0]))
        m2 = plugin.collect(step=2, trainer=None)
        assert m2["moe/dead_expert_ratio"] == 0.0

    def test_dead_expert_sustained(self):
        """Verify dead expert sustained."""
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4, dead_window_size=3)
        moe = _get_moes(model)[0]

        for step in range(4):
            _run_router_with_counts(moe, torch.tensor([0.0, 100.0, 100.0, 100.0]))
            plugin.collect(step=step, trainer=None)

        _run_router_with_counts(moe, torch.tensor([0.0, 100.0, 100.0, 100.0]))
        metrics = plugin.collect(step=4, trainer=None)
        assert metrics["moe/dead_expert_ratio"] > 0

    def test_collect_resets_counters(self):
        """Verify collect resets counters."""
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4)
        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([50.0, 50.0, 50.0, 50.0]))
        plugin.collect(step=1, trainer=None)

        metrics2 = plugin.collect(step=2, trainer=None)
        assert "moe/expert_load_cv" not in metrics2 or metrics2["moe/expert_load_cv"] < 0.01

    def test_ac_halves_counts(self):
        """Verify ac halves counts."""
        plugin_no_ac, model_no_ac = self._make_plugin(n_moe_layers=1, num_experts=4, has_ac=False)
        plugin_ac, model_ac = self._make_plugin(n_moe_layers=1, num_experts=4, has_ac=True)

        counts = torch.tensor([100.0, 200.0, 300.0, 400.0])
        _run_router_with_counts(_get_moes(model_no_ac)[0], counts)
        _run_router_with_counts(_get_moes(model_ac)[0], counts)

        m_no_ac = plugin_no_ac.collect(step=1, trainer=None)
        m_ac = plugin_ac.collect(step=1, trainer=None)

        assert abs(m_ac["moe/expert_load_cv"] - m_no_ac["moe/expert_load_cv"]) < 0.01

    def test_per_layer_metrics(self):
        """Verify per layer metrics."""
        plugin, model = self._make_plugin(n_moe_layers=3, num_experts=4)
        for i, moe in enumerate(_get_moes(model)):
            _run_router_with_counts(moe, torch.tensor([100.0, 100.0, 100.0, 100.0]) + i * 50)
        metrics = plugin.collect(step=1, trainer=None)
        for i in range(3):
            assert f"moe/layer_{i}/expert_load_cv" in metrics

    def test_remove_hooks(self):
        """Verify remove hooks."""
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4)
        plugin.remove_hooks()

        moe = _get_moes(model)[0]
        x = torch.randn(8, 16)
        moe(x)

        metrics = plugin.collect(step=1, trainer=None)
        assert "moe/expert_load_cv" not in metrics or metrics.get("moe/expert_load_cv", 0) < 0.01

    def test_accumulation_across_forwards(self):
        """Verify accumulation across forwards."""
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4)
        moe = _get_moes(model)[0]

        _run_router_with_counts(moe, torch.tensor([10.0, 20.0, 30.0, 40.0]))
        _run_router_with_counts(moe, torch.tensor([20.0, 40.0, 60.0, 80.0]))

        metrics = plugin.collect(step=1, trainer=None)
        assert metrics["moe/expert_util_min"] > 0

    def test_router_hook_not_affected_by_tpe_zero(self):
        """Verify router hook not affected by tpe zero."""
        plugin, model = self._make_plugin(n_moe_layers=1, num_experts=4)
        moe = _get_moes(model)[0]

        _run_router_with_counts(moe, torch.tensor([10.0, 20.0, 30.0, 40.0]))

        # Simulate optimizer zeroing tokens_per_expert (as torchtitan does)
        moe.tokens_per_expert.zero_()

        _run_router_with_counts(moe, torch.tensor([15.0, 25.0, 35.0, 45.0]))

        metrics = plugin.collect(step=1, trainer=None)
        # Counter should have accumulated both forwards: [25, 45, 65, 85]
        assert metrics["moe/expert_util_min"] > 0
        assert metrics.get("moe/dead_expert_ratio", 0) == 0.0


class TestEPReduce:
    def _make_plugin(self, n_moe_layers=2, num_experts=4, ep_reduce_interval=50):
        model = MockModelPart(n_moe_layers=n_moe_layers, num_experts=num_experts)
        plugin = MoEHealthPlugin(
            model_parts=[model],
            ep_reduce_interval=ep_reduce_interval,
            parallel_dims=None,
        )
        return plugin, model

    def test_no_reduce_without_parallel_dims(self):
        plugin, model = self._make_plugin(ep_reduce_interval=50)
        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([100.0, 100.0, 100.0, 100.0]))

        metrics = plugin.collect(step=50, trainer=None)
        assert not any(k.startswith("moe_global/") for k in metrics)

    def test_no_reduce_when_interval_zero(self):
        model = MockModelPart(n_moe_layers=1, num_experts=4)
        plugin = MoEHealthPlugin(model_parts=[model], ep_reduce_interval=0)
        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([100.0, 100.0, 100.0, 100.0]))

        metrics = plugin.collect(step=50, trainer=None)
        assert not any(k.startswith("moe_global/") for k in metrics)

    def test_reduce_on_interval_step(self):
        from unittest.mock import MagicMock, patch

        model = MockModelPart(n_moe_layers=1, num_experts=4)
        plugin = MoEHealthPlugin(model_parts=[model], ep_reduce_interval=50)
        plugin._loss_pg = MagicMock()
        plugin._rank = 0

        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([10.0, 20.0, 30.0, 40.0]))

        with patch("titaninfra.monitoring.moe_health.dist.reduce"):
            metrics = plugin.collect(step=50, trainer=None)

        assert "moe_global/expert_load_cv" in metrics
        assert "moe_global/imbalance_ratio" in metrics
        assert "moe_global/dead_expert_ratio" in metrics

    def test_no_global_on_non_interval_step(self):
        from unittest.mock import MagicMock, patch

        model = MockModelPart(n_moe_layers=1, num_experts=4)
        plugin = MoEHealthPlugin(model_parts=[model], ep_reduce_interval=50)
        plugin._loss_pg = MagicMock()
        plugin._rank = 0

        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([100.0, 100.0, 100.0, 100.0]))

        with patch("titaninfra.monitoring.moe_health.dist.reduce") as mock_reduce:
            metrics = plugin.collect(step=33, trainer=None)

        mock_reduce.assert_not_called()
        assert not any(k.startswith("moe_global/") for k in metrics)

    def test_non_rank0_no_global_metrics(self):
        from unittest.mock import MagicMock, patch

        model = MockModelPart(n_moe_layers=1, num_experts=4)
        plugin = MoEHealthPlugin(model_parts=[model], ep_reduce_interval=50)
        plugin._loss_pg = MagicMock()
        plugin._rank = 3

        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([100.0, 100.0, 100.0, 100.0]))

        with patch("titaninfra.monitoring.moe_health.dist.reduce") as mock_reduce:
            metrics = plugin.collect(step=50, trainer=None)

        mock_reduce.assert_called_once()
        assert not any(k.startswith("moe_global/") for k in metrics)

    def test_global_metrics_correct_values(self):
        from unittest.mock import MagicMock, patch

        model = MockModelPart(n_moe_layers=1, num_experts=4)
        plugin = MoEHealthPlugin(model_parts=[model], ep_reduce_interval=10)
        plugin._loss_pg = MagicMock()
        plugin._rank = 0

        moe = _get_moes(model)[0]
        _run_router_with_counts(moe, torch.tensor([0.0, 100.0, 100.0, 200.0]))

        with patch("titaninfra.monitoring.moe_health.dist.reduce"):
            metrics = plugin.collect(step=10, trainer=None)

        assert metrics["moe_global/expert_load_cv"] > 0.5
        assert metrics["moe_global/dead_expert_ratio"] > 0


class TestComputeLayerLoadMetrics:
    def test_uniform_distribution(self):
        counts = [torch.tensor([100.0, 100.0, 100.0, 100.0])]
        m = _compute_layer_load_metrics(counts, num_experts=4)
        assert m is not None
        assert m.expert_load_cv < 0.01
        assert abs(m.expert_util_min - 1.0) < 0.01
        assert abs(m.expert_util_max - 1.0) < 0.01
        assert m.dead_expert_ratio == 0.0

    def test_skewed_distribution(self):
        counts = [torch.tensor([0.0, 0.0, 100.0, 200.0])]
        m = _compute_layer_load_metrics(counts, num_experts=4)
        assert m is not None
        assert m.expert_load_cv > 0.5
        assert m.expert_util_min == 0.0
        assert m.dead_expert_ratio > 0

    def test_imbalance_no_zeros(self):
        counts = [torch.tensor([10.0, 10.0, 10.0, 100.0])]
        m = _compute_layer_load_metrics(counts, num_experts=4)
        assert m is not None
        assert m.imbalance_ratio == 10.0

    def test_imbalance_with_zeros_uses_floor(self):
        counts = [torch.tensor([0.0, 10.0, 20.0, 30.0])]
        m = _compute_layer_load_metrics(counts, num_experts=4)
        assert m is not None
        assert m.imbalance_ratio <= 1000.0
        assert m.imbalance_ratio > 1.0

    def test_imbalance_cap_at_1000(self):
        counts = [torch.tensor([0.0001, 0.0, 0.0, 100000.0])]
        m = _compute_layer_load_metrics(counts, num_experts=4)
        assert m is not None
        assert m.imbalance_ratio == 1000.0

    def test_multiple_layers(self):
        counts = [
            torch.tensor([100.0, 100.0, 100.0, 100.0]),
            torch.tensor([10.0, 10.0, 10.0, 100.0]),
        ]
        m = _compute_layer_load_metrics(counts, num_experts=4)
        assert m is not None
        assert m.expert_load_cv > 0
        assert m.imbalance_ratio > 1.0

    def test_all_zeros(self):
        counts = [torch.tensor([0.0, 0.0, 0.0, 0.0])]
        m = _compute_layer_load_metrics(counts, num_experts=4)
        assert m is None

    def test_empty_list(self):
        m = _compute_layer_load_metrics([], num_experts=4)
        assert m is None

    def test_to_dict(self):
        m = LayerLoadMetrics(
            expert_load_cv=0.5,
            expert_util_min=0.2,
            expert_util_max=1.8,
            imbalance_ratio=9.0,
            dead_expert_ratio=0.25,
        )
        d = m.to_dict("moe")
        assert d["moe/expert_load_cv"] == 0.5
        assert d["moe/expert_util_min"] == 0.2
        assert d["moe/expert_util_max"] == 1.8
        assert d["moe/imbalance_ratio"] == 9.0
        assert d["moe/dead_expert_ratio"] == 0.25

    def test_to_dict_prefix(self):
        m = LayerLoadMetrics(0.1, 0.2, 0.3, 0.4, 0.5)
        d = m.to_dict("moe_global")
        assert all(k.startswith("moe_global/") for k in d)
        assert len(d) == 5


# ── _resolve_tokens_buffer: upstream buffer rename compat ──


def test_resolve_tokens_buffer_prefers_E_suffix():
    from titaninfra.monitoring.moe_health import _resolve_tokens_buffer

    class FakeMoE:
        pass

    m = FakeMoE()
    m.tokens_per_expert_E = torch.zeros(8)
    m.tokens_per_expert = torch.zeros(3)
    assert _resolve_tokens_buffer(m).shape[0] == 8


def test_resolve_tokens_buffer_fallback_old_name():
    from titaninfra.monitoring.moe_health import _resolve_tokens_buffer

    class FakeMoE:
        pass

    m = FakeMoE()
    m.tokens_per_expert = torch.zeros(4)
    assert _resolve_tokens_buffer(m).shape[0] == 4
