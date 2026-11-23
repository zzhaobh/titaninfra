import pytest
import torch

from titaninfra.kernels.registry import list_kernels


def test_moonvit_kernels_registered():
    names = list_kernels()
    assert "moonvit_rope" in names and "moonvit_patch_merge" in names


def test_wire_moonvit_targets_live_rope_seam():
    """_wire_moonvit must swap VisionEncoderLayer._rope_impl (the live seam),
    not Rope2DPosEmb._rope_impl (dead). Guards the final-review miswiring bug."""
    from titaninfra.kernels.wiring import _wire_moonvit
    from titaninfra.models.kimi_vl.vision_encoder import MoonViT, VisionEncoderLayer

    model = MoonViT(hidden_dim=64, num_layers=2, num_heads=4, mlp_dim=128, attn_impl="eager")
    sentinel = object()
    # _wire_moonvit calls get_kernel imported from registry inside the function;
    # patch it at the source module so the sentinel flows through.
    import titaninfra.kernels.registry as R
    saved = R.get_kernel
    R.get_kernel = lambda name, enabled: sentinel if name == "moonvit_rope" else saved(name, enabled)
    try:
        handled = _wire_moonvit([model], ["moonvit_rope"])
    finally:
        R.get_kernel = saved
    assert "moonvit_rope" in handled
    # every VisionEncoderLayer (the LIVE seam) must now hold the sentinel:
    layers = [m for m in model.modules() if isinstance(m, VisionEncoderLayer)]
    assert len(layers) == 2
    assert all(layer._rope_impl is sentinel for layer in layers), "rope kernel not wired to live seam"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Helion kernel GPU-only")
def test_moonvit_rope_parity():
    pytest.importorskip("helion")
    from titaninfra.kernels import assert_parity
    from titaninfra.models.kimi_vl.rope import get_freqs_cis, precompute_freqs_cis

    def make_inputs():
        dim, L, H = 64, 22, 4
        cis = get_freqs_cis(precompute_freqs_cis(dim, 8, 8).cuda(), torch.tensor([[2, 3], [4, 4]]))
        xq = torch.randn(L, H, dim, device="cuda", requires_grad=True)
        xk = torch.randn(L, H, dim, device="cuda", requires_grad=True)
        return (xq, xk, cis)

    assert_parity("moonvit_rope", make_inputs, check_backward=True)
