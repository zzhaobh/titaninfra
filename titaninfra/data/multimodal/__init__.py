"""Multimodal pretrain data pipeline, independent of SFT message streams.

The offline image-token estimate and runtime image-pad/ViT scatter must agree
for every image. Both paths share image_processor.smart_resize as the source
of truth. Pure sizing functions have no torch/torchvision/einops dependency;
decode and patchify import those libraries lazily when needed.
"""

from titaninfra.data.multimodal.image_processor import (
    calculate_vision_tokens,
    estimate_image_tokens,
    smart_resize,
)

__all__ = ["smart_resize", "calculate_vision_tokens", "estimate_image_tokens"]
