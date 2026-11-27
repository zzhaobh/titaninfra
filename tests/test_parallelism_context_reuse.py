from types import SimpleNamespace

from titaninfra.engine import train_step


def _context(**overrides):
    values = {
        "dp_replicate": 1,
        "dp_shard": 2,
        "cp": 1,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        "world_size": 2,
        "enable_sequence_parallel": False,
        "_real_pp_group_for_fake_spmd": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _cache_and_share():
    cache = getattr(train_step, "_PARALLELISM_CONTEXTS", None)
    share = getattr(train_step, "_share_parallelism_context", None)
    assert cache is not None
    assert share is not None
    cache.clear()
    return cache, share


def test_same_topology_reuses_the_first_parallelism_context():
    cache, share = _cache_and_share()
    first = _context()
    second = _context()

    assert share(first) is first
    assert share(second) is first
    assert len(cache) == 1


def test_different_topology_does_not_reuse_parallelism_context():
    cache, share = _cache_and_share()
    first = _context()
    different_tp = _context(dp_shard=1, tp=2)
    different_sp = _context(enable_sequence_parallel=True)

    assert share(first) is first
    assert share(different_tp) is different_tp
    assert share(different_sp) is different_sp
    assert len(cache) == 3
