"""Unit tests for neftune."""

import pytest
import torch
import torch.nn as nn

from tests.conftest import make_test_train_config
from titaninfra.utils.neftune import setup_neftune_hook


class FakeEmbedding(nn.Embedding):
    """Tests for fake embedding."""
    pass


class FakeModel(nn.Module):
    """Tests for fake model."""

    def __init__(self, vocab_size=100, embed_dim=64):
        super().__init__()
        self.tok_embeddings = FakeEmbedding(vocab_size, embed_dim)

    def forward(self, tokens):
        return self.tok_embeddings(tokens)


class TestSetupNeftuneHook:
    """Tests for test setup neftune hook."""

    def test_hook_registered(self):
        """Verify hook registered."""
        model = FakeModel()
        handle = setup_neftune_hook(model, alpha=5.0, seq_len=16)
        assert handle is not None
        handle.remove()

    def test_noise_injected_in_training_mode(self):
        """Verify noise injected in training mode."""
        torch.manual_seed(42)
        model = FakeModel(vocab_size=100, embed_dim=64)
        model.train()

        tokens = torch.randint(0, 100, (32,))

        with torch.no_grad():
            original = model(tokens).clone()

        handle = setup_neftune_hook(model, alpha=5.0, seq_len=16)

        with torch.no_grad():
            noisy = model(tokens)

        assert not torch.allclose(original, noisy), "训练模式下应注入噪声"
        handle.remove()

    def test_no_noise_in_eval_mode(self):
        """Verify no noise in eval mode."""
        torch.manual_seed(42)
        model = FakeModel(vocab_size=100, embed_dim=64)

        tokens = torch.randint(0, 100, (32,))

        handle = setup_neftune_hook(model, alpha=5.0, seq_len=16)

        model.eval()
        with torch.no_grad():
            out1 = model(tokens).clone()
            out2 = model(tokens)

        assert torch.allclose(out1, out2), "eval 模式下不应注入噪声"
        handle.remove()

    def test_noise_magnitude_scales_with_alpha(self):
        """Verify noise magnitude scales with alpha."""
        torch.manual_seed(42)
        model = FakeModel(vocab_size=100, embed_dim=64)
        model.train()

        tokens = torch.randint(0, 100, (32,))

        with torch.no_grad():
            original = model(tokens).clone()

        # alpha=1
        handle1 = setup_neftune_hook(model, alpha=1.0, seq_len=16)
        with torch.no_grad():
            out_small = model(tokens)
        diff_small = (out_small - original).abs().mean().item()
        handle1.remove()

        # alpha=100
        handle2 = setup_neftune_hook(model, alpha=100.0, seq_len=16)
        with torch.no_grad():
            out_large = model(tokens)
        diff_large = (out_large - original).abs().mean().item()
        handle2.remove()

        assert diff_large > diff_small, "更大的 alpha 应产生更大噪声"

    def test_hook_removable(self):
        """Verify hook removable."""
        torch.manual_seed(42)
        model = FakeModel(vocab_size=100, embed_dim=64)
        model.train()

        tokens = torch.randint(0, 100, (32,))

        handle = setup_neftune_hook(model, alpha=5.0, seq_len=16)
        handle.remove()

        with torch.no_grad():
            out1 = model(tokens).clone()
            out2 = model(tokens)

        assert torch.allclose(out1, out2), "移除 hook 后不应注入噪声"

    def test_raises_without_tok_embeddings(self):
        """Verify raises without tok embeddings."""
        model = nn.Linear(10, 10)
        with pytest.raises(AttributeError, match="tok_embeddings"):
            setup_neftune_hook(model, alpha=5.0, seq_len=16)

    def test_output_shape_unchanged(self):
        """Verify output shape unchanged."""
        model = FakeModel(vocab_size=100, embed_dim=64)
        model.train()
        tokens = torch.randint(0, 100, (32,))

        handle = setup_neftune_hook(model, alpha=5.0, seq_len=16)
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (32, 64)
        handle.remove()

    def test_magnitude_matches_seq_len_not_total_tokens(self):
        """Verify magnitude matches seq len not total tokens."""
        torch.manual_seed(7)
        embed_dim = 64
        seq_len = 16
        model = FakeModel(vocab_size=100, embed_dim=embed_dim)
        model.train()

        expected_mag = 5.0 / (seq_len * embed_dim) ** 0.5

        def _hook_free_embed(tokens):
            return torch.nn.functional.embedding(tokens, model.tok_embeddings.weight)

        tokens_1 = torch.randint(0, 100, (seq_len,))
        base1 = _hook_free_embed(tokens_1)
        handle = setup_neftune_hook(model, alpha=5.0, seq_len=seq_len)
        with torch.no_grad():
            out1 = model(tokens_1)
        handle.remove()
        delta1 = (out1 - base1).abs()
        assert delta1.max().item() <= expected_mag + 1e-6, (
            f"T=seq_len=16 时噪声上界应为 {expected_mag}，实测 max={delta1.max().item()}"
        )

        tokens_2 = torch.randint(0, 100, (seq_len * 2,))
        base2 = _hook_free_embed(tokens_2)
        handle = setup_neftune_hook(model, alpha=5.0, seq_len=seq_len)
        with torch.no_grad():
            out2 = model(tokens_2)
        handle.remove()
        delta2 = (out2 - base2).abs()
        assert delta2.max().item() <= expected_mag + 1e-6, (
            f"T=32（seq_len=16 不变）噪声上界仍应为 {expected_mag}，"
            f"实测 max={delta2.max().item()}（若此断言失败，说明实现仍在用 "
            "T 而非 seq_len 计算噪声幅度）"
        )


class TestNeftuneConfig:
    """Tests for test neftune config."""

    def test_train_config_accepts_neftune(self):
        """Verify train config accepts neftune."""
        config = make_test_train_config(neftune_noise_alpha=5.0)
        assert config.neftune_noise_alpha == 5.0

    def test_train_config_neftune_default_none(self):
        """Verify train config neftune default none."""
        config = make_test_train_config()
        assert config.neftune_noise_alpha is None

    def test_train_config_accepts_decay_ratio(self):
        """Verify train config accepts decay ratio."""
        config = make_test_train_config(decay_ratio=0.2)
        assert config.decay_ratio == 0.2

    def test_train_config_decay_ratio_default_none(self):
        """Verify train config decay ratio default none."""
        config = make_test_train_config()
        assert config.decay_ratio is None
