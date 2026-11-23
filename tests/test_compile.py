"""Tests for titaninfra.distributed.compile — MoE-safe per-submodule compile."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from titaninfra.distributed.compile import CompileConfig


def _make_mock_model(blocks):
    """Create a mock model with model.layers containing the given blocks."""
    model = MagicMock(spec=nn.Module)
    layers = MagicMock()
    layers.named_children.return_value = [(str(i), b) for i, b in enumerate(blocks)]
    model.layers = layers
    return model


def _make_block(submodule_names):
    """Create a mock TransformerBlock with named submodules."""
    block = MagicMock(spec=nn.Module)
    submodules = []
    for name in submodule_names:
        sub = MagicMock(spec=nn.Module)
        sub.compile = MagicMock()
        sub.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        submodules.append((name, sub))
    block.named_children.return_value = submodules
    return block, {name: sub for name, sub in submodules}


class TestApplyCompileMoeSafe:
    def setup_method(self):
        self.compile_config = CompileConfig(backend="inductor")

    def test_skips_moe_when_not_fp8(self):
        block, subs = _make_block(["attention", "attention_norm", "ffn_norm", "moe"])
        model = _make_mock_model([block])

        from titaninfra.distributed.compile import apply_compile_moe_safe

        apply_compile_moe_safe(model, self.compile_config)

        subs["attention"].compile.assert_called_once()
        subs["attention_norm"].compile.assert_called_once()
        subs["ffn_norm"].compile.assert_called_once()
        subs["moe"].compile.assert_not_called()

    def test_compiles_all_in_dense_model(self):
        block, subs = _make_block(["attention", "attention_norm", "ffn_norm", "feed_forward"])
        model = _make_mock_model([block])

        from titaninfra.distributed.compile import apply_compile_moe_safe

        apply_compile_moe_safe(model, self.compile_config)

        for name, sub in subs.items():
            sub.compile.assert_called_once(), f"{name} should be compiled"

    def test_compiles_moe_when_fp8_quantized(self):
        block, subs = _make_block(["attention", "moe"])
        model = _make_mock_model([block])

        with patch(
            "titaninfra.distributed.compile._is_fp8_quantized", return_value=True
        ):
            from titaninfra.distributed.compile import apply_compile_moe_safe

            apply_compile_moe_safe(model, self.compile_config)

        subs["attention"].compile.assert_called_once()
        subs["moe"].compile.assert_called_once()

    def test_dynamo_flags_set(self):
        block, _ = _make_block(["attention"])
        model = _make_mock_model([block])

        torch._dynamo.config.capture_scalar_outputs = False
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = False

        from titaninfra.distributed.compile import apply_compile_moe_safe

        apply_compile_moe_safe(model, self.compile_config)

        assert torch._dynamo.config.capture_scalar_outputs is True
        assert torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint is True

    def test_multiple_blocks_mixed(self):
        """Qwen3-style: some blocks have moe, some have feed_forward."""
        block_moe, subs_moe = _make_block(["attention", "attention_norm", "ffn_norm", "moe"])
        block_dense, subs_dense = _make_block(["attention", "attention_norm", "ffn_norm", "feed_forward"])
        model = _make_mock_model([block_moe, block_dense])

        from titaninfra.distributed.compile import apply_compile_moe_safe

        apply_compile_moe_safe(model, self.compile_config)

        subs_moe["moe"].compile.assert_not_called()
        subs_moe["attention"].compile.assert_called_once()
        subs_dense["feed_forward"].compile.assert_called_once()
        subs_dense["attention"].compile.assert_called_once()

    def test_skips_moe_held_under_feed_forward_by_type(self):
        """Post-migration families (deepseek_v3/v4, gpt_oss, qwen3, kimi_vl,
        qwen3_5) route MoE through the SAME `feed_forward` attribute dense
        layers use -- there is no separate `moe`-named submodule to catch by
        name. Regression test for the real bug this round found: before the
        isinstance(submodule, MoE) check was added, `apply_compile_moe_safe`
        would try to compile these families' MoE layers (incompatible with
        Dynamo/DeepEP), since name-matching alone never fires for them.
        """
        from torchtitan.models.common.moe import MoE

        block, subs = _make_block(["attention", "attention_norm", "ffn_norm", "feed_forward"])
        # Make the feed_forward submodule itself look like a real MoE instance
        # (spec= makes isinstance() succeed against a MagicMock).
        moe_sub = MagicMock(spec=MoE)
        moe_sub.compile = MagicMock()
        moe_sub.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        subs["feed_forward"] = moe_sub
        block.named_children.return_value = [
            (name, (moe_sub if name == "feed_forward" else sub))
            for name, sub in block.named_children.return_value
        ]
        model = _make_mock_model([block])

        from titaninfra.distributed.compile import apply_compile_moe_safe

        apply_compile_moe_safe(model, self.compile_config)

        subs["feed_forward"].compile.assert_not_called()
        subs["attention"].compile.assert_called_once()
        subs["attention_norm"].compile.assert_called_once()
        subs["ffn_norm"].compile.assert_called_once()

    def test_compiles_feed_forward_when_dense_even_though_moe_type_check_exists(self):
        """A dense feed_forward (NOT an MoE instance) must still compile --
        confirms the isinstance(submodule, MoE) check added for the test
        above doesn't over-match plain DenseFeedForward submodules."""
        block, subs = _make_block(["attention", "feed_forward"])
        model = _make_mock_model([block])

        from titaninfra.distributed.compile import apply_compile_moe_safe

        apply_compile_moe_safe(model, self.compile_config)

        subs["feed_forward"].compile.assert_called_once()

    def test_backend_passed_through(self):
        block, subs = _make_block(["attention"])
        model = _make_mock_model([block])
        config = CompileConfig(backend="aot_eager")

        from titaninfra.distributed.compile import apply_compile_moe_safe

        apply_compile_moe_safe(model, config)

        subs["attention"].compile.assert_called_once_with(backend="aot_eager", fullgraph=False)


class TestIsFP8Quantized:
    def test_returns_false_when_torchao_not_installed(self):
        from titaninfra.distributed.compile import _is_fp8_quantized

        module = nn.Linear(10, 10)
        with patch.dict("sys.modules", {"torchao.prototype.moe_training.scaled_grouped_mm": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert _is_fp8_quantized(module) is False

    def test_returns_false_for_regular_parameters(self):
        from titaninfra.distributed.compile import _is_fp8_quantized

        module = nn.Linear(10, 10)
        assert _is_fp8_quantized(module) is False
