"""Tests for titaninfra.profiling.op_attribution (CPU-safe, no GPU/torchtitan).

Covers install_op_region_hooks on a toy module tree mimicking the
DeepSeekV3TransformerBlock layout (block.attention, block.moe.{router,experts,
shared_experts}), and verifies refmoe/<region> rows surface in a torch.profiler
key_averages() table.
"""

import torch
import torch.nn as nn

from titaninfra.profiling import install_op_region_hooks, remove_op_region_hooks


class _Leaf(nn.Module):
    def forward(self, x):
        return x + 1


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Leaf()
        self.experts = _Leaf()
        self.shared_experts = _Leaf()

    def forward(self, x):
        return self.shared_experts(self.experts(self.router(x)))


class _Block(nn.Module):
    """Mimics DeepSeekV3TransformerBlock attribute names."""

    def __init__(self):
        super().__init__()
        self.attention = _Leaf()
        self.attention_norm = _Leaf()
        self.ffn_norm = _Leaf()
        self.moe = _MoE()

    def forward(self, x):
        x = self.attention(self.attention_norm(x))
        return self.moe(self.ffn_norm(x))


class _Model(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(_Block() for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_install_returns_handles_and_counts():
    model = _Model(n_layers=2)
    handles = install_op_region_hooks([model])
    # 2 layers × (attn + router + experts + shared) = 8 modules × 2 hooks each
    assert len(handles) == 8 * 2
    remove_op_region_hooks(handles)


def test_include_norms_adds_norm_regions():
    model = _Model(n_layers=1)
    handles = install_op_region_hooks([model], include_norms=True)
    # 1 layer × (attn + router + experts + shared + attn_norm + ffn_norm) = 6
    assert len(handles) == 6 * 2
    remove_op_region_hooks(handles)


def test_no_match_returns_empty():
    handles = install_op_region_hooks([_Leaf()])
    assert handles == []


def test_regions_appear_in_profiler_table():
    model = _Model(n_layers=1)
    handles = install_op_region_hooks([model])
    x = torch.zeros(4)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        model(x)
    remove_op_region_hooks(handles)

    names = {evt.key for evt in prof.key_averages()}
    assert "refmoe/attn" in names
    assert "refmoe/moe.router" in names
    assert "refmoe/moe.experts" in names
    assert "refmoe/moe.shared" in names


def test_hooks_removed_leave_no_regions():
    model = _Model(n_layers=1)
    handles = install_op_region_hooks([model])
    remove_op_region_hooks(handles)
    x = torch.zeros(4)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        model(x)
    names = {evt.key for evt in prof.key_averages()}
    assert not any(n.startswith("refmoe/") for n in names)
