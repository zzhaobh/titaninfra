"""Forward and backward parity checks between custom and PyTorch kernels.

Every custom Triton, Helion, or torch.compile implementation must pass this
check before registration. The reference path remains a proven-equivalent
fallback for every operator.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .registry import get_spec


def assert_parity(
    name: str,
    make_inputs: Callable[[], tuple[torch.Tensor, ...]],
    *,
    check_backward: bool = True,
) -> None:
    """Compare custom and reference outputs, optionally including gradients.

    Args:
        name: Registered operator whose custom_impl must be available.
        make_inputs: Zero-argument factory producing forward inputs.
        check_backward: Compare gradients for floating inputs requiring grad.

    Raises:
        AssertionError: Forward or backward values exceed spec.rtol/atol.
    """
    spec = get_spec(name)
    if spec.custom_impl is None:
        raise ValueError(f"kernel '{name}' 无 custom_impl，无法做 parity")

    # Generate inputs once and clone them for both implementations. Calling the
    # factory twice could produce different random inputs and a false failure.
    base_inputs = make_inputs()

    def _clone(t):
        if isinstance(t, torch.Tensor):
            c = t.detach().clone()
            c.requires_grad_(t.requires_grad)
            return c
        return t

    ref_inputs = tuple(_clone(t) for t in base_inputs)
    impl_inputs = tuple(_clone(t) for t in base_inputs)

    ref_out = spec.pytorch_ref(*ref_inputs)
    impl_out = spec.custom_impl(*impl_inputs)
    torch.testing.assert_close(impl_out, ref_out, rtol=spec.rtol, atol=spec.atol)

    if not check_backward:
        return

    # Normalize outputs to tuples so single- and multi-output operators share
    # the same backward path with matching gradient sequences.
    ref_outs = ref_out if isinstance(ref_out, (tuple, list)) else (ref_out,)
    impl_outs = impl_out if isinstance(impl_out, (tuple, list)) else (impl_out,)
    grad_outs = [torch.randn_like(o) for o in ref_outs]
    torch.autograd.backward(list(ref_outs), grad_outs)
    torch.autograd.backward(list(impl_outs), [g.clone() for g in grad_outs])

    for i, (ri, ii) in enumerate(zip(ref_inputs, impl_inputs, strict=True)):
        if isinstance(ri, torch.Tensor) and ri.requires_grad:
            assert ri.grad is not None, f"ref 输入#{i} grad 为 None"
            assert ii.grad is not None, f"impl 输入#{i} grad 为 None"
            torch.testing.assert_close(
                ii.grad, ri.grad, rtol=spec.rtol, atol=spec.atol,
                msg=f"{name} 输入#{i} 梯度不对齐",
            )
