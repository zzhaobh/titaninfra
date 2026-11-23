"""Unit tests for state dict adapter."""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

# ============================================================
# ============================================================

_ADAPTER_PATH = Path(__file__).parent.parent / "titaninfra" / "models" / "qwen3" / "state_dict_adapter.py"


class MockMoE:
    """Minimal stand-in for torchtitan.models.common.moe.MoE.

    Only ``Config`` needs to be a real class: adapter.py does
    ``isinstance(layer.feed_forward, MoE.Config)`` to find the MoE layer among
    ``model_config.layers`` for the offline (no local_experts_indices) path.
    Module scope so both the module-injection below and ``_make_adapter()``'s
    fake layers can share the exact same class (MagicMock() attributes are
    never instances of a real class, so a mock layer's ``feed_forward`` must
    be constructed as ``MockMoE.Config()``, not left as an auto-mock).
    """

    class Config:
        def __init__(self, num_experts: int = 128):
            self.num_experts = num_experts


def _load_adapter_class():
    """Load adapter class."""
    # Mock base class
    class MockMoEStateDictAdapter:
        def __init__(self, model_config, hf_assets_path):
            self.model_config = model_config
            self.hf_assets_path = hf_assets_path
            self.grouped_expert_weight_placements = {}
            self.grouped_expert_weight_shape = {}
            self.grouped_expert_weight_mesh = {}
            self.local_experts_indices = {}

        def _get_local_experts_weights(self, *args, **kwargs):
            raise NotImplementedError

        def _split_experts_weights(self, *args, **kwargs):
            raise NotImplementedError

        def _concatenate_expert_weights_dtensor(self, *args, **kwargs):
            raise NotImplementedError

        def _concatenate_expert_weights(self, *args, **kwargs):
            raise NotImplementedError

        def _caculate_indices_from_placements(self, *args, **kwargs):
            raise NotImplementedError

        def _calculate_strided_shard_shard_indices(self, *args, **kwargs):
            raise NotImplementedError

        def _native_fused_linears_from_hf(self, state_dict, *, fuse_routed_experts=False):
            # Real base class re-stacks QKVLinear/GDN/DenseFeedForward/
            # RoutedExperts logical tensors by calling
            # ``self.model_config.traverse(...)``, which needs a real
            # Configurable config tree -- orthogonal to what
            # TestFromHfUsesCorrectPath exercises (online vs offline expert
            # routing). This passthrough matches its contract (same
            # signature, dict-in/dict-out) when there is nothing to fuse.
            return dict(state_dict)

        def _native_fused_linears_to_hf(self, state_dict, *, split_routed_experts=False):
            return dict(state_dict)

        def _validate_hf_rope_config(self, *args, **kwargs):
            return None

    # Inject mock modules. ``torchtitan.models.common`` must look like a real
    # package (have ``__path__``) for ``from torchtitan.models.common.moe
    # import MoE`` to resolve a submodule on it instead of raising
    # "'torchtitan.models.common' is not a package".
    mock_common = ModuleType("torchtitan.models.common")
    mock_common.__path__ = []
    mock_moe = ModuleType("torchtitan.models.common.moe")
    mock_moe.MoE = MockMoE
    mock_moe_state_dict_adapter = ModuleType(
        "torchtitan.models.common.moe.state_dict_adapter"
    )
    mock_moe_state_dict_adapter.MoEStateDictAdapter = MockMoEStateDictAdapter
    mock_moe.state_dict_adapter = mock_moe_state_dict_adapter
    mock_common.moe = mock_moe
    mock_rope = ModuleType("torchtitan.models.common.rope")
    mock_rope.CosSinRoPE = MagicMock()
    mock_common.rope = mock_rope

    mock_torchtitan_models = ModuleType("torchtitan.models")
    mock_torchtitan_models.common = mock_common
    mock_torchtitan = ModuleType("torchtitan")
    mock_torchtitan.models = mock_torchtitan_models

    # Mock .model (relative import inside state_dict_adapter.py)
    mock_qwen3_model_module = ModuleType("titaninfra.models.qwen3.model")
    mock_qwen3_model_module.Qwen3Model = MagicMock()

    # Temporarily inject mocks
    saved = {}
    to_inject = {
        "torchtitan": mock_torchtitan,
        "torchtitan.models": mock_torchtitan_models,
        "torchtitan.models.common": mock_common,
        "torchtitan.models.common.moe": mock_moe,
        "torchtitan.models.common.moe.state_dict_adapter": mock_moe_state_dict_adapter,
        "torchtitan.models.common.rope": mock_rope,
        "titaninfra.models.qwen3.model": mock_qwen3_model_module,
    }
    for key, mod in to_inject.items():
        saved[key] = sys.modules.get(key)
        sys.modules[key] = mod

    # Also ensure psutil is available
    try:
        import psutil  # noqa: F401
    except ImportError:
        saved["psutil"] = sys.modules.get("psutil")
        sys.modules["psutil"] = MagicMock()

    try:
        spec = importlib.util.spec_from_file_location(
            "titaninfra.models.qwen3.state_dict_adapter",
            str(_ADAPTER_PATH),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Qwen3StateDictAdapter
    finally:
        # Restore original sys.modules state
        for key, original in saved.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


# Load once at module level (avoid reloading on every test)
_Qwen3StateDictAdapter = _load_adapter_class()


# ============================================================
# ============================================================

def _make_adapter():
    """Make adapter."""
    mock_config = MagicMock()
    # if isinstance(l.feed_forward, MoE.Config)).feed_forward.num_experts`
    moe_layer = MagicMock()
    moe_layer.feed_forward = MockMoE.Config(num_experts=128)
    mock_config.layers = [moe_layer]
    mock_config.enable_weight_tying = False
    return _Qwen3StateDictAdapter(mock_config, hf_assets_path=None)


def _make_mock_dtensor(start_idx: int = 0, end_idx: int = 16, n_experts: int = 128):
    """Make mock dtensor."""
    from torch.distributed.tensor import DTensor

    mock_dtensor = MagicMock()
    mock_dtensor.__class__ = DTensor
    mock_dtensor.shape = (n_experts, 256, 256)
    mock_dtensor.placements = (MagicMock(),)
    mock_dtensor.device_mesh = MagicMock()
    mock_dtensor.dtype = torch.float32
    return mock_dtensor


# ============================================================
# ============================================================

class TestToHfBasicContract:
    """Tests for test to hf basic contract."""

    def test_local_experts_indices_empty_before_call(self):
        """Verify local experts indices empty before call."""
        adapter = _make_adapter()
        assert len(adapter.local_experts_indices) == 0


@pytest.mark.xfail(reason="for_load parameter not yet implemented in to_hf()")
class TestToHfForLoadPopulatesLocalExpertsIndices:
    """Tests for test to hf for load populates local experts indices."""

    def test_to_hf_for_load_populates_local_experts_indices(self):
        """Verify to hf for load populates local experts indices."""
        adapter = _make_adapter()
        mock_dtensor = _make_mock_dtensor(start_idx=0, end_idx=16)

        with patch.object(adapter, "_caculate_indices_from_placements", return_value=(0, 16)):
            state_dict = {"layers.0.moe.experts.w1": mock_dtensor}
            adapter.to_hf(state_dict, for_load=True)

        abstract_key = "layers.{}.moe.experts.w1"
        assert abstract_key in adapter.local_experts_indices, (
            "BUG: to_hf(for_load=True) 必须设置 local_experts_indices[abstract_key]。\n"
            "没有这行，from_hf() 无法走 DTensor-aware 在线加载路径，EP>1 时专家权重无法加载。"
        )
        assert adapter.local_experts_indices[abstract_key] == (0, 16)

    def test_to_hf_for_load_different_rank(self):
        """Verify to hf for load different rank."""
        adapter = _make_adapter()
        mock_dtensor = _make_mock_dtensor(start_idx=48, end_idx=64)

        with patch.object(adapter, "_caculate_indices_from_placements", return_value=(48, 64)):
            state_dict = {"layers.5.moe.experts.w2": mock_dtensor}
            adapter.to_hf(state_dict, for_load=True)

        abstract_key = "layers.{}.moe.experts.w2"
        assert abstract_key in adapter.local_experts_indices
        assert adapter.local_experts_indices[abstract_key] == (48, 64)

    def test_to_hf_for_load_all_expert_keys(self):
        """Verify to hf for load all expert keys."""
        adapter = _make_adapter()

        expert_keys = [
            "layers.0.moe.experts.w1",
            "layers.0.moe.experts.w2",
            "layers.0.moe.experts.w3",
        ]

        with patch.object(adapter, "_caculate_indices_from_placements", return_value=(0, 16)):
            state_dict = {k: _make_mock_dtensor() for k in expert_keys}
            adapter.to_hf(state_dict, for_load=True)

        for key in expert_keys:
            abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
            assert abstract_key in adapter.local_experts_indices, (
                f"to_hf(for_load=True) 没有设置 local_experts_indices[{abstract_key!r}]"
            )

    def test_to_hf_for_load_creates_placeholder_tensors(self):
        """Verify to hf for load creates placeholder tensors."""
        adapter = _make_adapter()
        mock_dtensor = _make_mock_dtensor(start_idx=0, end_idx=4, n_experts=8)

        with patch.object(adapter, "_caculate_indices_from_placements", return_value=(0, 4)):
            state_dict = {"layers.0.moe.experts.w1": mock_dtensor}
            hf_state_dict = adapter.to_hf(state_dict, for_load=True)

        for expert_id in range(4):
            key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            assert key in hf_state_dict, f"缺少 placeholder: {key}"
        assert "model.layers.0.mlp.experts.4.gate_proj.weight" not in hf_state_dict

    def test_to_hf_for_false_does_not_populate_indices(self):
        """Verify to hf for false does not populate indices."""
        adapter = _make_adapter()
        mock_dtensor = _make_mock_dtensor()

        with patch.object(adapter, "_get_local_experts_weights", return_value={}) as mock_fn:
            state_dict = {"layers.0.moe.experts.w1": mock_dtensor}
            adapter.to_hf(state_dict, for_load=False)
            mock_fn.assert_called_once()


# ============================================================
# ============================================================

class TestFromHfUsesCorrectPath:
    """Tests for test from hf uses correct path."""

    @staticmethod
    def _patch_psutil():
        """Patch log_memory to no-op since psutil is mocked/unavailable on macOS."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            g = _Qwen3StateDictAdapter.from_hf.__globals__
            original = g.get("log_memory")
            g["log_memory"] = lambda tag: None
            try:
                yield
            finally:
                if original is not None:
                    g["log_memory"] = original

        return _ctx()

    def test_from_hf_uses_dtensor_path_when_indices_set(self):
        """Verify from hf uses dtensor path when indices set."""
        adapter = _make_adapter()
        abstract_key = "layers.{}.feed_forward.routed_experts.w1_EFD"
        adapter.local_experts_indices[abstract_key] = (0, 16)

        hf_state_dict = {}
        for expert_id in range(16):
            key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.zeros(256, 256)

        fake_stacked = torch.zeros(16, 256, 256)
        with self._patch_psutil(), patch.object(
            adapter,
            "_concatenate_expert_weights_dtensor",
            return_value=fake_stacked,
        ) as mock_dtensor_path, patch.object(
            adapter,
            "_concatenate_expert_weights",
        ) as mock_offline_path:
            adapter.from_hf(hf_state_dict)

        assert mock_dtensor_path.called, (
            "from_hf() 应在 local_experts_indices 有值时调用 "
            "_concatenate_expert_weights_dtensor（DTensor-aware 路径）"
        )
        assert not mock_offline_path.called, (
            "from_hf() 不应在 local_experts_indices 有值时调用 "
            "_concatenate_expert_weights（离线路径需要全部 128 个专家）"
        )

    def test_from_hf_uses_offline_path_without_indices(self):
        """Verify from hf uses offline path without indices."""
        adapter = _make_adapter()

        hf_state_dict = {}
        for expert_id in range(128):
            key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.zeros(256, 256)

        fake_stacked = torch.zeros(128, 256, 256)
        with self._patch_psutil(), patch.object(
            adapter,
            "_concatenate_expert_weights",
            return_value=fake_stacked,
        ) as mock_offline_path, patch.object(
            adapter,
            "_concatenate_expert_weights_dtensor",
        ) as mock_dtensor_path:
            adapter.from_hf(hf_state_dict)

        assert mock_offline_path.called
        assert not mock_dtensor_path.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
