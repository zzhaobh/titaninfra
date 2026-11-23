"""Unit tests for parallelism matrix."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

torchtitan = pytest.importorskip("torchtitan", reason="requires torchtitan for mesh construction")

from torchtitan.config.configs import CommConfig  # noqa: E402
from torchtitan.distributed.parallelism_context import ParallelismContext  # noqa: E402

from tests.conftest import make_test_model_config  # noqa: E402

pytestmark = pytest.mark.xdist_group("fake_pg_parallelism_matrix")


@dataclass(frozen=True)
class ParallelCombo:
    """Tests for parallel combo."""

    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1

    @property
    def world_size(self) -> int:
        """World size."""
        return self.dp_replicate * self.dp_shard * self.cp * self.tp * self.pp

    @property
    def edp_shard(self) -> int:
        """Edp shard."""
        return self.dp_shard * self.cp * self.tp // self.ep

    @property
    def is_ep_valid(self) -> bool:
        """Is ep valid."""
        return (self.dp_shard * self.cp * self.tp) % self.ep == 0

    @property
    def sparse_world_size(self) -> int:
        """Sparse world size."""
        return self.pp * self.dp_replicate * self.edp_shard * self.ep

    def label(self) -> str:
        return (
            f"dp_rep{self.dp_replicate}_dp_shard{self.dp_shard}"
            f"_cp{self.cp}_tp{self.tp}_pp{self.pp}_ep{self.ep}"
        )


# ============================================================
#
#   CP+PP / HSDP(dp_replicate+dp_shard)）；
# ============================================================

SINGLE_AXIS = [
    ParallelCombo(dp_shard=8),
    ParallelCombo(tp=8),
    ParallelCombo(cp=8),
    ParallelCombo(pp=8),
    ParallelCombo(dp_replicate=2, dp_shard=4),
]

PAIRWISE = [
    ParallelCombo(dp_shard=4, tp=2),
    ParallelCombo(dp_shard=4, cp=2),
    ParallelCombo(dp_shard=4, pp=2),
    ParallelCombo(tp=2, cp=4),
    ParallelCombo(tp=2, pp=4),
    ParallelCombo(cp=2, pp=4),
    ParallelCombo(dp_replicate=2, dp_shard=2, tp=2),
    # HSDP + PP
    ParallelCombo(dp_replicate=2, dp_shard=2, pp=2),
]

HIGHER_ORDER = [
    ParallelCombo(dp_shard=2, tp=2, cp=2),
    ParallelCombo(dp_shard=2, tp=2, pp=2),
    ParallelCombo(dp_shard=2, cp=2, pp=2),
    ParallelCombo(tp=2, cp=2, pp=2),
    ParallelCombo(dp_replicate=2, dp_shard=2, tp=2, pp=2),
    ParallelCombo(dp_shard=2, cp=2, tp=2, pp=2),
    ParallelCombo(dp_replicate=2, dp_shard=2, cp=2, tp=2, pp=2),
]

MOE_EP = [
    ParallelCombo(dp_shard=8, ep=8),
    # EP + TP
    ParallelCombo(dp_shard=4, tp=2, ep=4),
    # EP + CP
    ParallelCombo(dp_shard=4, cp=2, ep=4),
    # EP + PP
    ParallelCombo(dp_shard=4, pp=2, ep=4),
    # EP + TP + CP
    ParallelCombo(dp_shard=2, tp=2, cp=2, ep=4),
    ParallelCombo(dp_replicate=2, dp_shard=2, cp=2, tp=2, pp=2, ep=4),
]

ALL_COMBOS = SINGLE_AXIS + PAIRWISE + HIGHER_ORDER + MOE_EP


# ============================================================
# ============================================================


def _init_fake_pg(combo: ParallelCombo) -> object:
    """Init fake pg."""
    from torchtitan.distributed.utils import init_distributed

    os.environ["NGPU"] = str(combo.world_size)
    if combo.pp > 1:
        os.environ["FAKE_PP_RANK"] = "0"
    else:
        os.environ.pop("FAKE_PP_RANK", None)

    return init_distributed(
        CommConfig(backend="fake"),
        pipeline_parallel_degree=combo.pp,
    )


@pytest.fixture
def fake_pg():
    """Fake pg."""
    import torch.distributed as dist

    created: list[bool] = []

    def _make(combo: ParallelCombo):
        if dist.is_initialized():
            dist.destroy_process_group()
        topo = _init_fake_pg(combo)
        created.append(True)
        return topo

    try:
        yield _make
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _build_context(combo: ParallelCombo, topo) -> ParallelismContext:
    """Build context."""
    flavor = "debugmodel_moe" if combo.ep > 1 else "debugmodel"
    model_config = make_test_model_config(
        name=f"qwen3/{flavor}",
        dp_replicate=combo.dp_replicate,
        dp_shard=combo.dp_shard,
        cp=combo.cp,
        tp=combo.tp,
        pp=combo.pp,
        ep=combo.ep,
    )
    parallelism = model_config.to_torchtitan_parallelism_config()
    return ParallelismContext.from_config(
        parallelism, topo, dump_folder="/tmp/fake_pg_matrix"
    )


# ============================================================
# ============================================================


@pytest.mark.parametrize("combo", ALL_COMBOS, ids=lambda c: c.label())
def test_combo_arithmetic_is_self_consistent(combo: ParallelCombo):
    """Verify combo arithmetic is self consistent."""
    assert combo.world_size >= 1

    if combo.ep == 1:
        assert combo.edp_shard == combo.dp_shard * combo.cp * combo.tp
        assert combo.sparse_world_size == combo.world_size
    else:
        assert combo.is_ep_valid, (
            f"{combo.label()}: ep={combo.ep} 不整除 dp_shard*cp*tp="
            f"{combo.dp_shard * combo.cp * combo.tp}，该组合本身不合法，"
            "不应进入矩阵"
        )
        assert combo.sparse_world_size == combo.world_size, (
            f"{combo.label()}: 稀疏 mesh 世界大小 {combo.sparse_world_size} "
            f"!= 稠密 {combo.world_size}"
        )


# ============================================================
# ============================================================


@pytest.mark.parametrize("combo", ALL_COMBOS, ids=lambda c: c.label())
def test_mesh_builds_with_configured_degrees(combo: ParallelCombo, fake_pg):
    """Verify mesh builds with configured degrees."""
    topo = fake_pg(combo)
    ctx = _build_context(combo, topo)

    assert ctx.world_size == combo.world_size
    assert ctx.dp_replicate == combo.dp_replicate
    assert ctx.dp_shard == combo.dp_shard
    assert ctx.cp == combo.cp
    assert ctx.tp == combo.tp
    assert ctx.pp == combo.pp
    assert ctx.ep == combo.ep

    ctx.build_mesh()

    assert ctx.dp_shard_enabled is (combo.dp_shard > 1)
    assert ctx.dp_replicate_enabled is (combo.dp_replicate > 1)
    assert ctx.cp_enabled is (combo.cp > 1)
    assert ctx.tp_enabled is (combo.tp > 1)
    assert ctx.pp_enabled is (combo.pp > 1)
    assert ctx.ep_enabled is (combo.ep > 1)

    assert ctx.dp_enabled is (combo.dp_replicate > 1 or combo.dp_shard > 1)

    assert ctx.fsdp_enabled is (combo.dp_shard > 1 or combo.cp > 1)


# ============================================================
# ============================================================


MOE_COMBOS = [c for c in ALL_COMBOS if c.ep > 1]


@pytest.mark.parametrize("combo", MOE_COMBOS, ids=lambda c: c.label())
def test_ep_sparse_mesh_shape(combo: ParallelCombo, fake_pg):
    """Verify ep sparse mesh shape."""
    topo = fake_pg(combo)
    ctx = _build_context(combo, topo)
    ctx.build_mesh()

    fwdbwd_sparse = ctx.spmd_sparse_mesh()
    assert fwdbwd_sparse is not None, (
        f"{combo.label()}: ep={combo.ep}>1 时稀疏 mesh 不应为 None"
    )
    assert fwdbwd_sparse.mesh_dim_names == ("dp_replicate", "edp_shard", "ep"), (
        f"{combo.label()}: fwd/bwd 稀疏 mesh 轴名/顺序不符："
        f"{fwdbwd_sparse.mesh_dim_names}"
    )
    assert tuple(fwdbwd_sparse.shape) == (
        combo.dp_replicate,
        combo.edp_shard,
        combo.ep,
    ), (
        f"{combo.label()}: fwd/bwd 稀疏 mesh shape {tuple(fwdbwd_sparse.shape)} != "
        f"{(combo.dp_replicate, combo.edp_shard, combo.ep)}"
    )

    full_sparse = ctx.get_optional_mesh(
        ["pp", "dp_replicate", "edp_shard", "ep"], include_singleton_axes=True
    )
    assert full_sparse is not None, (
        f"{combo.label()}: 完整稀疏 mesh 应为 None 之外的值"
    )
    assert full_sparse.mesh_dim_names == (
        "pp",
        "dp_replicate",
        "edp_shard",
        "ep",
    ), f"{combo.label()}: 完整稀疏 mesh 轴名/顺序不符：{full_sparse.mesh_dim_names}"
    assert tuple(full_sparse.shape) == (
        combo.pp,
        combo.dp_replicate,
        combo.edp_shard,
        combo.ep,
    ), (
        f"{combo.label()}: 完整稀疏 mesh shape {tuple(full_sparse.shape)} != "
        f"{(combo.pp, combo.dp_replicate, combo.edp_shard, combo.ep)}"
    )


@pytest.mark.parametrize(
    "combo", [c for c in ALL_COMBOS if c.ep == 1], ids=lambda c: c.label()
)
def test_dense_combos_have_no_sparse_mesh(combo: ParallelCombo, fake_pg):
    """Verify dense combos have no sparse mesh."""
    topo = fake_pg(combo)
    ctx = _build_context(combo, topo)
    ctx.build_mesh()

    assert ctx.spmd_sparse_mesh() is None, (
        f"{combo.label()}: ep=1 却拿到了稀疏 mesh"
    )


# ============================================================
# ============================================================


def test_matrix_covers_every_axis_beyond_degree_one():
    """Verify matrix covers every axis beyond degree one."""
    for axis in ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep"):
        assert any(getattr(c, axis) > 1 for c in ALL_COMBOS), (
            f"矩阵里没有任何组合激活 {axis}>1"
        )

    assert any(c.pp > 1 for c in ALL_COMBOS), "缺少 pp>1 的组合"
    assert any(c.dp_replicate > 1 for c in ALL_COMBOS), "缺少 HSDP(dp_replicate>1)"
    assert any(c.tp > 1 and c.cp > 1 for c in ALL_COMBOS), "缺少 tp>1 且 cp>1"

    assert any(
        c.dp_replicate > 1
        and c.dp_shard > 1
        and c.cp > 1
        and c.tp > 1
        and c.pp > 1
        and c.ep == 1
        for c in ALL_COMBOS
    ), "缺少 5D 全开的 dense 组合"
    assert any(
        c.dp_replicate > 1
        and c.dp_shard > 1
        and c.cp > 1
        and c.tp > 1
        and c.pp > 1
        and c.ep > 1
        for c in ALL_COMBOS
    ), "缺少 6D 全开的 MoE 组合"
