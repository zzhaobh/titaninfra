"""CPU-safe registration checks for the CSA gather kernel.

Importing titaninfra.kernels registers metadata without requiring the optional
Triton or Transformers implementations to be installed.
"""

from __future__ import annotations


def test_kernels_import_is_fault_tolerant():
    """Kernel imports remain fault-tolerant without Triton or TorchTitan."""
    import titaninfra.kernels  # noqa: F401


def test_csa_gather_registered():
    from titaninfra.kernels.registry import get_spec, is_registered

    assert is_registered("csa_gather")
    spec = get_spec("csa_gather")
    assert spec.min_cuda_capability == (9, 0)
    assert spec.rtol == 2e-3 and spec.atol == 2e-3


def test_csa_gather_visible_in_list():
    from titaninfra.kernels.registry import list_kernels

    assert "csa_gather" in list_kernels()
