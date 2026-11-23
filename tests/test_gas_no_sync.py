"""CPU-only toggle tests for _set_microbatch_all_reduce.

These tests check the call sequence used to skip HSDP all-reduce until the last
gradient-accumulation microbatch. Numerical loss equivalence requires GPU tests.
"""

from titaninfra.engine.train_step import _set_microbatch_all_reduce


class _FakeFSDP:
    """Record set_requires_all_reduce values on a fake FSDP module."""

    def __init__(self):
        self.calls = []

    def set_requires_all_reduce(self, v, *, recurse=True):
        self.calls.append(v)


def test_disabled_always_noop():
    # Disabled mode never touches the module.
    m = _FakeFSDP()
    for last in (False, True):
        _set_microbatch_all_reduce([m], is_last_microbatch=last, enabled=False)
    assert m.calls == []


def test_enabled_skips_until_last():
    # Disable reduction for intermediate microbatches and enable it for the last.
    m = _FakeFSDP()
    _set_microbatch_all_reduce([m], is_last_microbatch=False, enabled=True)
    _set_microbatch_all_reduce([m], is_last_microbatch=True, enabled=True)
    assert m.calls == [False, True]


def test_no_fsdp_method_noop():
    # Objects without the FSDP method are ignored.
    _set_microbatch_all_reduce([object()], is_last_microbatch=False, enabled=True)


def test_multiple_model_parts_all_toggled():
    # Every model part receives the same toggle.
    a, b = _FakeFSDP(), _FakeFSDP()
    _set_microbatch_all_reduce([a, b], is_last_microbatch=False, enabled=True)
    assert a.calls == [False] and b.calls == [False]
