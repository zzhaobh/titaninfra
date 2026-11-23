"""Tests for the DeepSeek V4 MoE router (sqrtsoftplus scoring + hash routing).

Rewritten for the upstream-tip migration (see
docs/torchtitan_upgrade_plan_2026-11.md §38.1): ``score_func`` is now a real
``UnaryActivationFn.Config`` (``SqrtSoftplus.Config()``), not a string;
``num_hash_layers``/``is_hash`` are renamed ``n_hash_layers``/``hash``
(matching upstream's own field names); ``input_ids_T`` replaces
``input_ids=`` as the keyword (and is now REQUIRED as a keyword-only arg on
``_select_experts``, enforced structurally by ``DeepSeekV4Model.forward``'s
``input_ids_T`` threading, not by this router's own ``forward`` any more --
see model.py). The hash path's combine weight is no longer a fixed uniform
``1/top_k``: it now GATHERS the real (sqrtsoftplus-scored) gate output at the
hash-selected expert positions -- a genuine correctness fix over the old
Phase-1 simplification (§38.1), so these tests assert against the gate's
real output instead of a hardcoded uniform constant.
"""

import pytest

torch = pytest.importorskip("torch")
# torchtitan's top-level import succeeds without triton, but
# torchtitan.models.common.moe (which router.py imports) pulls in triton via
# the EP kernels. importorskip the actual dependency so this module skips
# cleanly on machines lacking triton (e.g. macOS dev) instead of erroring.
pytest.importorskip("torchtitan.models.common.moe")

from torchtitan.models.common import HiMidLoLinear, SqrtSoftplus  # noqa: E402

from titaninfra.models.deepseek_v4.router import DeepSeekV4Router  # noqa: E402

_DIM = 16


def _mk(num_experts=8, top_k=3, n_hash_layers=0, layer_id=0, vocab=32):
    cfg = DeepSeekV4Router.Config(
        num_experts=num_experts,
        gate=HiMidLoLinear.Config(
            in_features=_DIM, out_features=num_experts, backward_mode="hi_mid_lo", bias=False,
        ),
        top_k=top_k,
        score_func=SqrtSoftplus.Config(),
        n_hash_layers=n_hash_layers,
        layer_id=layer_id,
        vocab_size=vocab,
    )
    r = cfg.build()
    r.init_states(buffer_device=torch.device("cpu"))
    return r


def test_non_hash_router_score_path_topk_shape():
    r = _mk()
    assert not r.hash
    x_TD = torch.randn(8, _DIM)  # (T, D) — flat, no batch dim
    topk_scores, topk_indices, routing_map_TE = r(x_TD)
    assert topk_indices.shape == (8, 3)
    assert topk_scores.shape == (8, 3)
    # one-hot boolean routing map (T, E), matching upstream's
    # TokenChoiceTopKRouter.forward contract.
    assert routing_map_TE.shape == (8, 8)
    assert routing_map_TE.dtype == torch.bool
    assert torch.equal(
        routing_map_TE.sum(dim=-1), torch.full((8,), 3, dtype=torch.int64)
    )
    assert routing_map_TE.gather(dim=-1, index=topk_indices).all()


def test_hash_router_builds_real_table():
    r = _mk(n_hash_layers=1, layer_id=0, vocab=32, top_k=3)
    assert r.hash
    assert r.tid2eid.shape == (32, 3)
    # a real random hash table (torch.rand(...).topk(...)), not an all-zero
    # placeholder -- confirms _init_self_buffers/init_states materialized it.
    assert not torch.all(r.tid2eid == 0)


def test_hash_router_is_deterministic_per_token_id():
    r = _mk(n_hash_layers=1, layer_id=0, vocab=32, top_k=3)
    x_TD = torch.randn(8, _DIM)
    input_ids_T = torch.tensor([5, 5, 1, 2, 5, 0, 0, 5])
    _, idx_a, _ = r(x_TD, input_ids_T=input_ids_T)
    _, idx_b, _ = r(x_TD, input_ids_T=input_ids_T)
    assert torch.equal(idx_a, idx_b)
    assert torch.equal(idx_a[0], idx_a[1])  # same token id -> same experts
    assert torch.equal(idx_a[0], idx_a[4])
    assert torch.equal(idx_a, r.tid2eid[input_ids_T])


def test_hash_router_combine_weight_is_real_gate_score_not_uniform():
    """Correctness fix over the old Phase-1 router (§38.1): the combine
    weight must be the REAL sqrtsoftplus(gate(x)) score gathered at the
    hash-selected expert positions, not a hardcoded uniform 1/top_k."""
    r = _mk(n_hash_layers=1, layer_id=0, vocab=32, top_k=3)
    x_TD = torch.randn(8, _DIM)
    input_ids_T = torch.tensor([5, 5, 1, 2, 5, 0, 0, 5])
    topk_scores, topk_indices, routing_map_TE = r(x_TD, input_ids_T=input_ids_T)

    assert topk_scores.shape == (8, 3)
    assert torch.equal(topk_indices, r.tid2eid[input_ids_T])
    # Hand-compute the expected gathered score the same way the base
    # TokenChoiceTopKRouter.forward does: score_func(gate(x)), gathered at
    # the (hash-selected, here) expert indices.
    with torch.no_grad():
        expected_scores_TE = r.score_func(r.gate(x_TD))
    expected_scores = expected_scores_TE.gather(dim=-1, index=topk_indices)
    if r.route_norm:
        expected_scores = expected_scores / (
            expected_scores.sum(dim=-1, keepdim=True) + r.route_norm_epsilon
        )
    expected_scores = expected_scores * r.route_scale
    assert torch.allclose(topk_scores, expected_scores, atol=1e-5)
    # Explicitly NOT the old uniform-weight formula.
    uniform = torch.full_like(topk_scores, 1.0 / 3 * r.route_scale)
    assert not torch.allclose(topk_scores, uniform)

    assert routing_map_TE.shape == (8, 8)
    assert routing_map_TE.dtype == torch.bool
    assert routing_map_TE.gather(dim=-1, index=topk_indices).all()


def test_hash_router_without_input_ids_raises():
    r = _mk(n_hash_layers=1, layer_id=0, vocab=32, top_k=3)
    x_TD = torch.randn(8, _DIM)
    with pytest.raises(ValueError):
        r(x_TD)
