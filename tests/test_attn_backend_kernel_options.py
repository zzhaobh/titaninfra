"""Tests for per-layer FlexInnerAttention kernel_options injection in _apply_attn_backend.

Background
----------
Gemma4 26B-A4B has hybrid attention: sliding layers use head_dim=256 (FlexInnerAttention
Triton kernel compiles fine) and full-attention layers use head_dim=512. At head_dim=512
the flash-style Triton template cannot stage Q/K/V tiles within per-SM shared memory for
any of inductor's default autotune block sizes -> every candidate is pruned ->
NoValidChoicesError at compile time.

Fix: when overriding attn_backend to "flex", inject a small-block kernel_options hint on
layers whose head_dim exceeds the FlexInnerAttention safe threshold (256). This constrains
inductor's autotune search to blocks small enough to fit SRAM, while layers at/below the
threshold keep an empty kernel_options (default autotune).
"""

import dataclasses

import pytest

torchtitan = pytest.importorskip("torchtitan", reason="requires torchtitan for ModelSpec")

from titaninfra.engine.pretrain import (  # noqa: E402
    _LARGE_HEAD_DIM_THRESHOLD,
    _apply_attn_backend,
)


def _make_fake_model_spec(head_dims):
    """Build a minimal fake model Config whose layers expose .attention.head_dim.

    We avoid constructing a real gemma4 config (heavy, GPU-leaning) and instead use
    lightweight dataclasses that mimic the structure _apply_attn_backend walks:
    model_spec.layers[i].attention.{inner_attention, head_dim}
    (ModelSpec wrapper was deleted upstream; the engine takes the bare Config).

    Note: no ``mask_type`` field — upstream P2 deleted it from all attention configs.
    Mask selection is now by ``inner_attention`` type + positions at runtime, not a
    per-config string field.
    """

    @dataclasses.dataclass
    class FakeAttention:
        head_dim: int
        inner_attention: object

    @dataclasses.dataclass
    class FakeLayer:
        attention: object

    @dataclasses.dataclass
    class FakeModel:
        layers: list

    layers = [
        FakeLayer(attention=FakeAttention(head_dim=hd, inner_attention=object()))
        for hd in head_dims
    ]
    return FakeModel(layers=layers)


def test_threshold_is_256():
    """The FlexInnerAttention safe head_dim threshold must be 256 (Triton SRAM ceiling)."""
    assert _LARGE_HEAD_DIM_THRESHOLD == 256


def test_large_head_dim_gets_kernel_options():
    """head_dim > 256 layers receive non-empty kernel_options with small blocks."""
    spec = _make_fake_model_spec([256, 512, 256, 512])
    new_spec = _apply_attn_backend(spec, "flex")

    layers = new_spec.layers
    # head_dim=512 layers (idx 1, 3) -> small-block kernel_options
    for idx in (1, 3):
        ko = layers[idx].attention.inner_attention.kernel_options
        assert ko, f"layer {idx} (head_dim=512) must get kernel_options, got {ko!r}"
        assert "BLOCK_M" in ko and "BLOCK_N" in ko
        assert ko["BLOCK_M"] <= 64 and ko["BLOCK_N"] <= 64


def test_small_head_dim_keeps_default_autotune():
    """head_dim <= 256 layers keep empty kernel_options (default autotune)."""
    spec = _make_fake_model_spec([256, 512, 256, 512])
    new_spec = _apply_attn_backend(spec, "flex")

    layers = new_spec.layers
    for idx in (0, 2):
        ko = layers[idx].attention.inner_attention.kernel_options
        assert not ko, f"layer {idx} (head_dim=256) must keep default autotune, got {ko!r}"


def test_flex_backend_sets_flexattention_inner_on_all_layers():
    """flex backend replaces every layer's inner_attention with a FlexInnerAttention.Config.

    Post-P2 there is no ``mask_type`` field: mask selection is by inner_attention type
    + positions at runtime. _apply_attn_backend's job is only to set the right
    inner_attention Config; block_causal vs causal is decided downstream from that type.
    """
    from torchtitan.models.common import FlexInnerAttention

    spec = _make_fake_model_spec([256, 512])
    new_spec = _apply_attn_backend(spec, "flex")
    for layer in new_spec.layers:
        assert isinstance(layer.attention.inner_attention, FlexInnerAttention.Config)


def test_immutability_original_spec_unchanged():
    """_apply_attn_backend must not mutate the input spec (immutable semantics)."""
    spec = _make_fake_model_spec([512])
    original_inner = spec.layers[0].attention.inner_attention
    _apply_attn_backend(spec, "flex")
    assert spec.layers[0].attention.inner_attention is original_inner


def test_unknown_backend_raises():
    """sdpa (and any backend other than flex) raises -- removed with torchtitan
    upstream-tip migration (ScaledDotProductInnerAttention no longer exists;
    config/model.py's ModelConfig.attn_backend is now Literal["flex"] only, so
    this ValueError is reachable only via direct calls like this test's, not
    through the validated production path)."""
    spec = _make_fake_model_spec([512, 512])
    with pytest.raises(ValueError, match="Unknown attn_backend"):
        _apply_attn_backend(spec, "sdpa")


def test_layer_without_head_dim_attr_is_safe():
    """Layers lacking a head_dim attribute (non-gemma4 families) default to no injection."""

    @dataclasses.dataclass
    class NoHeadDimAttention:
        inner_attention: object

    @dataclasses.dataclass
    class FakeLayer:
        attention: object

    @dataclasses.dataclass
    class FakeModel:
        layers: list

    spec = FakeModel(
        layers=[FakeLayer(attention=NoHeadDimAttention(inner_attention=object()))]
    )
    new_spec = _apply_attn_backend(spec, "flex")
    ko = new_spec.layers[0].attention.inner_attention.kernel_options
    assert not ko
