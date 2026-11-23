"""CPU-safe sentinel tests for the optional Helion dependency."""

import pytest


def test_helion_sentinel_importable():
    from titaninfra.kernels.ops import _helion

    assert isinstance(_helion.HELION_AVAILABLE, bool)
    if not _helion.HELION_AVAILABLE:
        assert _helion._HELION_IMPORT_ERROR is not None


def test_config_by_tokens_raises_without_helion():
    from titaninfra.kernels.ops import _helion

    if _helion.HELION_AVAILABLE:
        pytest.skip("helion installed; unavailable-branch not exercised")
    with pytest.raises(ImportError):
        _helion.helion_config_by_tokens(1024)


def test_run_tuned_exists():
    from titaninfra.kernels.ops import _helion

    assert callable(_helion.run_tuned)
