import pathlib

import pytest
import torch


def _grid_masks(positions):
    """Grid masks."""
    S = positions.shape[1]
    grid_q = torch.arange(S).view(S, 1)
    grid_k = torch.arange(S).view(1, S)
    doc_ids = (torch.cumsum((positions == 0).int(), dim=1) - 1)[0]
    is_pad = (positions < 0)[0]
    same_doc = doc_ids[grid_q] == doc_ids[grid_k]
    causal = grid_q >= grid_k
    not_pad = (~is_pad[grid_q]) & (~is_pad[grid_k])
    return same_doc & causal & not_pad, same_doc & causal, (positions[0] >= 0)


def test_not_pad_is_deadcode_qwen3():
    positions = torch.tensor([[0, 1, 2, 0, 1, -1, -1, -1]], dtype=torch.int32)
    with_np, without_np, real = _grid_masks(positions)
    assert torch.equal(
        with_np[real], without_np[real]
    ), "qwen3: real-token 行 not_pad 改变了 mask — 非死代码"


def test_not_pad_is_deadcode_qwen2_5():
    positions = torch.tensor([[0, 1, 0, 1, 2, 3, -1, -1]], dtype=torch.int32)
    with_np, without_np, real = _grid_masks(positions)
    assert torch.equal(
        with_np[real], without_np[real]
    ), "qwen2_5: real-token 行 not_pad 非死代码"


def test_not_pad_is_deadcode_gemma4():
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, -1, -1]], dtype=torch.int32)
    with_np, without_np, real = _grid_masks(positions)
    assert torch.equal(
        with_np[real], without_np[real]
    ), "gemma4: real-token 行 not_pad 非死代码"


def test_not_pad_is_deadcode_gpt_oss():
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, -1]], dtype=torch.int32)
    with_np, without_np, real = _grid_masks(positions)
    assert torch.equal(
        with_np[real], without_np[real]
    ), "gpt_oss: real-token 行 not_pad 非死代码"


def test_not_pad_is_deadcode_deepseek_v3():
    positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, -1]], dtype=torch.int32)
    with_np, without_np, real = _grid_masks(positions)
    assert torch.equal(
        with_np[real], without_np[real]
    ), "deepseek_v3: real-token 行 not_pad 非死代码"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
_FREE_FUNCS = (
    "apply_rotary_emb_cos_sin",
    "apply_rotary_emb_complex",
    "apply_rotary_emb_single_complex",
)
_MODELS_DIR = pathlib.Path(__file__).parent.parent / "titaninfra" / "models"

_DECOUPLED_FAMILIES = ["qwen3", "qwen2_5", "gemma4", "gpt_oss", "deepseek_v3"]


@pytest.mark.parametrize("family", _DECOUPLED_FAMILIES)
def test_family_has_no_rope_free_func_dependency(family):
    """Verify family has no rope free func dependency."""
    fam_dir = _MODELS_DIR / family
    for py in sorted(fam_dir.glob("*.py")):
        src = py.read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("(``"):
                continue
            hits = [f for f in _FREE_FUNCS if f in line]
            assert not hits, (
                f"{family}/{py.name} 仍依赖 rope free-func {hits}"
                f"（P3 须改走 upstream self.rope）: {stripped[:80]}"
            )
