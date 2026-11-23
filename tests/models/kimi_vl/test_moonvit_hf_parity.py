"""Unit tests for moonvit hf parity."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HF Kimi-VL remote-code + weights GPU-only"
)


def _load_hf_vision():
    """Load hf vision."""
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained(
            "moonshotai/Kimi-VL-A3B-Instruct", trust_remote_code=True, torch_dtype=torch.float32
        )
    except ImportError as e:
        pytest.skip(f"Kimi-VL remote code 与当前 transformers 不兼容: {e}")
    return model.vision_tower


def _transfer_vision_via_adapter(hf_vision, jd) -> None:
    """Transfer vision via adapter."""
    import re

    from titaninfra.models.kimi_vl.state_dict_adapter import _VISION_FROM_HF_MAP

    jd_sd = jd.state_dict()
    for raw_key, tensor in hf_vision.state_dict().items():
        hf_key = "vision_tower." + raw_key
        abstract = re.sub(r"(\d+)", "{}", hf_key, count=1)
        if hf_key in _VISION_FROM_HF_MAP:
            jd_key = _VISION_FROM_HF_MAP[hf_key]
        elif abstract in _VISION_FROM_HF_MAP:
            idx = re.search(r"\d+", hf_key).group(0)
            jd_key = _VISION_FROM_HF_MAP[abstract].format(idx)
        else:
            continue
        jd_target = jd_key[len("vision_tower."):]
        jd_sd[jd_target].copy_(tensor)


def test_moonvit_single_image_parity():
    """Verify moonvit single image parity."""
    from titaninfra.models.kimi_vl import MOONVIT_CONFIG as C
    from titaninfra.models.kimi_vl.vision_encoder import MoonViT

    hf = _load_hf_vision().cuda().eval()
    jd = MoonViT(
        hidden_dim=C["hidden_size"],
        num_layers=C["num_hidden_layers"],
        num_heads=C["num_attention_heads"],
        mlp_dim=C["intermediate_size"],
        patch_size=C["patch_size"],
        merge_kernel_size=C["merge_kernel_size"],
        attn_impl="flex",
    ).cuda().eval()
    _transfer_vision_via_adapter(hf, jd)

    h = w = 28
    L = h * w
    pixel_values = torch.randn(L, 3, 14, 14, device="cuda")
    grid = torch.tensor([[h, w]], device="cuda")
    with torch.no_grad():
        jd_out = jd(pixel_values, grid)[0]
        hf_out = hf(pixel_values, grid)[0]
    torch.testing.assert_close(jd_out, hf_out, rtol=2e-3, atol=2e-3)
