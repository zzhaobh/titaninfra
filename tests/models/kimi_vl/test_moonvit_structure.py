import math

import pytest
import torch

from titaninfra.models.kimi_vl.rope import apply_rope, get_freqs_cis, precompute_freqs_cis


def test_freqs_cis_shape_and_dtype():
    dim = 64
    cis = precompute_freqs_cis(dim, max_h=8, max_w=8)
    assert cis.shape == (8, 8, dim // 2)
    assert cis.dtype == torch.complex64


def test_freqs_cis_requires_dim_div_4():
    with pytest.raises(AssertionError):
        precompute_freqs_cis(dim=62, max_h=8, max_w=8)


def test_get_freqs_cis_packs_by_grid():
    dim = 64
    cis = precompute_freqs_cis(dim, 8, 8)
    grid = torch.tensor([[2, 3], [4, 4]])  # 6 + 16 = 22
    assert get_freqs_cis(cis, grid).shape == (22, dim // 2)


def test_apply_rope_shape_preserved():
    dim, L, H = 64, 22, 4
    cis = precompute_freqs_cis(dim, 8, 8)
    packed = get_freqs_cis(cis, torch.tensor([[2, 3], [4, 4]]))
    xq, xk = torch.randn(L, H, dim), torch.randn(L, H, dim)
    oq, ok = apply_rope(xq, xk, packed)
    assert oq.shape == (L, H, dim) and ok.shape == (L, H, dim)


def test_freqs_cis_interleaves_x_then_y():
    # Pins HF Rope2DPosEmb interleaved layout (x0,y0,x1,y1,...); fails under block-concat.
    # dim=8 -> dim//4=2 freq bands; freqs[0]=theta**0=1.0. Slot0=x band0, slot1=y band0.
    dim = 8
    cis = precompute_freqs_cis(dim, max_h=4, max_w=4)  # (4,4,4) complex
    p = cis[0, 2]  # position (y=0, x=2): x_pos=2, y_pos=0; layout [x0,y0,x1,y1]
    # y-axis slot (odd index) at y_pos=0 -> angle 0 -> exactly 1+0j.
    # Under buggy block-concat this slot would be x band1 at x_pos=2 -> cos(0.02)!=1.
    assert torch.allclose(p[1].real, torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(p[1].imag, torch.tensor(0.0), atol=1e-5)
    # x-axis slot0 at x_pos=2, freq band0=1.0 -> angle 2.0 rad.
    assert torch.allclose(p[0].real, torch.tensor(math.cos(2.0)), atol=1e-5)
    assert torch.allclose(p[0].imag, torch.tensor(math.sin(2.0)), atol=1e-5)


def test_patch_embed_output_shape():
    from titaninfra.models.kimi_vl.vision_encoder import PatchEmbed
    pe = PatchEmbed(out_dim=64, in_dim=3, patch_size=14, pos_emb_hw=14)
    pixel_values = torch.randn(6, 3, 14, 14)
    out = pe(pixel_values, torch.tensor([[2, 3]]))
    assert out.shape == (6, 64)


def test_learnable_pos_emb_interp_branch():
    from titaninfra.models.kimi_vl.vision_encoder import Learnable2DInterpPosEmb
    pos = Learnable2DInterpPosEmb(height=14, width=14, dim=64)
    out = pos(torch.randn(6, 64), torch.tensor([[2, 3]]))  # != 14x14 → interpolate
    assert out.shape == (6, 64)


def test_vision_mlp_shape():
    from titaninfra.models.kimi_vl.vision_encoder import VisionMLP
    assert VisionMLP(hidden_dim=64, mlp_dim=128)(torch.randn(6, 64)).shape == (6, 64)


def test_build_cu_seqlens():
    from titaninfra.models.kimi_vl.vision_encoder import build_cu_seqlens
    cu = build_cu_seqlens(torch.tensor([[2, 3], [4, 4]]))
    assert cu.tolist() == [0, 6, 22] and cu.dtype == torch.int32


def test_encoder_layer_shape_eager():
    from titaninfra.models.kimi_vl.rope import get_freqs_cis, precompute_freqs_cis
    from titaninfra.models.kimi_vl.vision_encoder import VisionEncoderLayer
    dim, heads = 64, 4
    layer = VisionEncoderLayer(hidden_dim=dim, num_heads=heads, mlp_dim=128, attn_impl="eager")
    x = torch.randn(6, dim)
    # RoPE freqs are built per head_dim (HF oracle: rope on head_dim, not hidden).
    cis = get_freqs_cis(precompute_freqs_cis(dim // heads, 8, 8), torch.tensor([[2, 3]]))
    out = layer(x, freqs_cis=cis, cu_seqlens=torch.tensor([0, 6], dtype=torch.int32))
    assert out.shape == (6, dim)


def test_patch_merger_2x2_grouping():
    from titaninfra.models.kimi_vl.vision_encoder import patch_merger
    x = torch.arange(16, dtype=torch.float32).view(16, 1)  # h=w=4, d=1
    out = patch_merger(x, torch.tensor([[4, 4]]), merge_kernel_size=(2, 2))
    assert len(out) == 1 and out[0].shape == (4, 4)
    assert out[0][0].tolist() == [0.0, 1.0, 4.0, 5.0]
    assert out[0][1].tolist() == [2.0, 3.0, 6.0, 7.0]


def test_moonvit_full_tower_shape():
    from titaninfra.models.kimi_vl.vision_encoder import MoonViT
    m = MoonViT(hidden_dim=64, num_layers=2, num_heads=4, mlp_dim=128,
                patch_size=14, merge_kernel_size=(2, 2), attn_impl="eager")
    pixel_values = torch.randn(16, 3, 14, 14)
    out = m(pixel_values, torch.tensor([[4, 4]]))
    assert len(out) == 1 and out[0].shape == (4, 64 * 4)
