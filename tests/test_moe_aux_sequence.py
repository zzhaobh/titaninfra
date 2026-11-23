"""Unit tests for moe aux sequence."""

import torch

from titaninfra.monitoring._moe_aux_compute import compute_sequence_load_balancing_loss


def test_balanced_routing_gives_one():
    n, t = 4, 4
    probs = torch.full((t, n), 1.0 / n, requires_grad=True)
    topk = torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]])
    seq_ids = torch.zeros(t, dtype=torch.long)
    loss = compute_sequence_load_balancing_loss(probs, topk, seq_ids, n)
    assert abs(loss.item() - 1.0) < 1e-4


def test_collapsed_routing_exceeds_one():
    n, t = 4, 4
    base = torch.zeros((t, n))
    base[:, 0] = 0.5
    base[:, 1] = 0.5
    probs = base.clone().requires_grad_(True)
    topk = torch.tensor([[0, 1]] * t)
    seq_ids = torch.zeros(t, dtype=torch.long)
    loss = compute_sequence_load_balancing_loss(probs, topk, seq_ids, n)
    assert loss.item() > 1.5


def test_multi_sequence_averages():
    n = 4
    probs = torch.full((8, n), 1.0 / n, requires_grad=True)
    topk = torch.tensor([[0, 1], [2, 3]] * 4)
    seq_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss = compute_sequence_load_balancing_loss(probs, topk, seq_ids, n)
    assert abs(loss.item() - 1.0) < 1e-4


def test_gradient_flows_through_probs():
    n = 4
    probs = torch.full((4, n), 1.0 / n, requires_grad=True)
    topk = torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]])
    seq_ids = torch.zeros(4, dtype=torch.long)
    loss = compute_sequence_load_balancing_loss(probs, topk, seq_ids, n)
    loss.backward()
    assert probs.grad is not None and probs.grad.abs().sum() > 0


def test_empty_returns_zero_with_graph():
    n = 4
    probs = torch.full((0, n), 0.0, requires_grad=True)
    topk = torch.zeros((0, 2), dtype=torch.long)
    seq_ids = torch.zeros(0, dtype=torch.long)
    loss = compute_sequence_load_balancing_loss(probs, topk, seq_ids, n)
    assert loss.item() == 0.0


def test_negative_seq_id_tokens_skipped():
    """Verify negative seq id tokens skipped."""
    n = 4
    probs = torch.cat([
        torch.full((4, n), 1.0 / n),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
    ]).requires_grad_(True)
    topk = torch.cat([
        torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]]),
        torch.tensor([[0, 0]] * 4),
    ])
    seq_ids = torch.tensor([0, 0, 0, 0, -1, -1, -1, -1])
    loss = compute_sequence_load_balancing_loss(probs, topk, seq_ids, n)
    assert abs(loss.item() - 1.0) < 1e-4


def test_padding_does_not_pollute_neighbor_doc():
    """Verify padding does not pollute neighbor doc."""
    n = 4
    probs_clean = torch.full((4, n), 1.0 / n)
    probs_pad = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    probs = torch.cat([probs_clean, probs_pad]).requires_grad_(True)
    topk = torch.cat([
        torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]]),
        torch.tensor([[0, 0]] * 2),
    ])
    seq_ids = torch.tensor([0, 0, 0, 0, -1, -1])
    loss = compute_sequence_load_balancing_loss(probs, topk, seq_ids, n)
    assert abs(loss.item() - 1.0) < 1e-4


def test_build_seq_ids_marks_padding_negative():
    """Verify build seq ids marks padding negative."""
    from titaninfra.monitoring.moe_aux_loss import MoEAuxLossPlugin

    plugin = MoEAuxLossPlugin.__new__(MoEAuxLossPlugin)
    plugin._cur_positions = torch.tensor([[0, 1, 0, 1, -1, -1]])
    seq_ids = plugin._build_seq_ids()
    assert seq_ids is not None
    assert seq_ids[0].item() == 0 and seq_ids[1].item() == 0
    assert seq_ids[2].item() == 1 and seq_ids[3].item() == 1
    assert seq_ids[4].item() < 0 and seq_ids[5].item() < 0
