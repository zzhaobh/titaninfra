"""Unit tests for loss."""

from titaninfra.data.loss import resolve_weight
from titaninfra.data.schema import (
    ImageBlock,
    LossConfig,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# ═══════════════════════════════════════════
# ═══════════════════════════════════════════


class TestResolveWeight:
    """Tests for test resolve weight."""

    def test_level1_block_weight_takes_priority(self):
        """Verify level1 block weight takes priority."""
        cfg = LossConfig(type_overrides={"text": 0.3})
        b = TextBlock(text="x", weight=0.7)
        assert resolve_weight(b, "assistant", cfg, msg_weight=1.0) == 0.7

    def test_level2_type_override(self):
        """Verify level2 type override."""
        cfg = LossConfig(type_overrides={"thinking": 0.5})
        b = ThinkingBlock(text="x")
        assert resolve_weight(b, "assistant", cfg, msg_weight=1.0) == 0.5

    def test_level3_role_default_assistant(self):
        """Verify level3 role default assistant."""
        cfg = LossConfig()
        b = TextBlock(text="x")
        assert resolve_weight(b, "assistant", cfg, msg_weight=1.0) == 1.0

    def test_level3_role_default_user(self):
        """Verify level3 role default user."""
        cfg = LossConfig()
        b = TextBlock(text="x")
        assert resolve_weight(b, "user", cfg, msg_weight=0.0) == 0.0

    def test_level3_role_default_system(self):
        cfg = LossConfig()
        b = TextBlock(text="x")
        assert resolve_weight(b, "system", cfg, msg_weight=0.0) == 0.0

    def test_level3_role_default_tool(self):
        cfg = LossConfig()
        b = ToolResultBlock(tool_use_id="c1", text="x")
        assert resolve_weight(b, "tool", cfg, msg_weight=0.0) == 0.0

    def test_msg_weight_multiplier_zero(self):
        """Verify msg weight multiplier zero."""
        cfg = LossConfig()
        b = TextBlock(text="x")
        assert resolve_weight(b, "assistant", cfg, msg_weight=0.0) == 0.0

    def test_msg_weight_multiplier_half(self):
        """Verify msg weight multiplier half."""
        cfg = LossConfig()
        b = TextBlock(text="x")
        assert resolve_weight(b, "assistant", cfg, msg_weight=0.5) == 0.5

    def test_msg_weight_none_uses_role_default(self):
        """Verify msg weight none uses role default."""
        cfg = LossConfig()
        b = TextBlock(text="x")
        # assistant default = 1, so 1 × 1 = 1
        assert resolve_weight(b, "assistant", cfg, msg_weight=None) == 1.0
        # user default = 0, so 0 × 0 = 0
        assert resolve_weight(b, "user", cfg, msg_weight=None) == 0.0

    def test_system_role_default(self):
        """Verify system role default."""
        cfg = LossConfig()
        b = TextBlock(text="x")
        assert resolve_weight(b, "system", cfg, msg_weight=1.0) == 0.0

    def test_type_override_with_msg_weight(self):
        """type_override × msg_weight"""
        cfg = LossConfig(type_overrides={"thinking": 0.5})
        b = ThinkingBlock(text="x")
        assert resolve_weight(b, "assistant", cfg, msg_weight=0.0) == 0.0
        assert resolve_weight(b, "assistant", cfg, msg_weight=1.0) == 0.5

    def test_block_weight_overrides_type_override(self):
        """Verify block weight overrides type override."""
        cfg = LossConfig(type_overrides={"thinking": 0.5})
        b = ThinkingBlock(text="x", weight=0.8)
        assert resolve_weight(b, "assistant", cfg, msg_weight=1.0) == 0.8

    def test_image_block(self):
        cfg = LossConfig(type_overrides={"image": 0})
        b = ImageBlock(source={"type": "url", "data": "x"})
        assert resolve_weight(b, "user", cfg, msg_weight=0.0) == 0.0

    def test_tool_use_block(self):
        cfg = LossConfig(type_overrides={"tool_use": 1.0})
        b = ToolUseBlock(id="c1", name="f", input={})
        assert resolve_weight(b, "assistant", cfg, msg_weight=1.0) == 1.0

    def test_unknown_role_defaults_to_zero(self):
        cfg = LossConfig()
        b = TextBlock(text="x")
        assert resolve_weight(b, "unknown_role", cfg, msg_weight=None) == 0.0


class TestWeightedCELossContract:
    """Tests for test weighted celoss contract."""

    def _make(self, vocab_size=8, seq_len=7):
        import torch

        torch.manual_seed(1)
        pred = torch.randn(seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (seq_len,))
        return pred, labels

    def test_no_weights_delegates_inner_and_gvt(self):
        """Verify no weights delegates inner and gvt."""
        import torch.nn.functional as F

        from titaninfra.data.loss import WeightedCELoss

        pred, labels = self._make()

        class _InnerCE:
            def __call__(self, p, lbl, gvt=None):
                s = F.cross_entropy(p.float(), lbl, reduction="sum", ignore_index=-100)
                return (s / gvt if gvt is not None else s), {}

        ce_sum = F.cross_entropy(pred.float(), labels, reduction="sum", ignore_index=-100)
        raw, aux = WeightedCELoss(_InnerCE())(pred, labels)
        assert aux == {}
        assert abs(raw.item() - ce_sum.item()) < 1e-4
        gvt = 5.0
        normed, _ = WeightedCELoss(_InnerCE())(pred, labels, gvt)
        assert abs(normed.item() - ce_sum.item() / gvt) < 1e-4

    def test_per_token_weights_and_gvt(self):
        """Verify per token weights and gvt."""
        import torch

        from titaninfra.data.loss import WeightedCELoss

        pred, labels = self._make(vocab_size=4, seq_len=6)
        wce = WeightedCELoss(inner_loss_fn=None)
        weights = torch.tensor([1.0, 0.5, 0.0, 2.0, 1.0, 0.0])
        gvt = 2.0
        loss, aux = wce(pred, labels, gvt, loss_weights=weights)
        assert aux == {}
        assert loss.shape == torch.Size([])

        import torch.nn.functional as F
        expected_per_token = F.cross_entropy(pred.float(), labels, reduction="none")
        expected = (expected_per_token * weights).sum() / gvt
        assert abs(loss.item() - expected.item()) < 1e-4

        weights2 = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        loss2, _ = wce(pred, labels, gvt, loss_weights=weights2)
        assert not torch.allclose(loss, loss2)

    def test_per_token_weights_rejects_flattened_double_pack(self):
        """Verify per token weights rejects flattened double pack."""
        import torch

        from titaninfra.data.loss import WeightedCELoss

        pred, labels = self._make(vocab_size=5, seq_len=3)
        wce = WeightedCELoss(inner_loss_fn=None)
        weights = torch.ones(3)
        loss, _ = wce(pred, labels, None, loss_weights=weights)
        import torch.nn.functional as F
        expected = F.cross_entropy(pred.float(), labels, reduction="sum")
        assert abs(loss.item() - expected.item()) < 1e-4
