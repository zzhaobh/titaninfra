"""NEFTune: Noisy Embedding Fine-Tuning.

Adds uniform noise to embedding outputs during training to reduce SFT/CPT
overfitting. Noise is sampled from Uniform(-1, 1) and scaled by
alpha / sqrt(seq_len * embed_dim), following https://arxiv.org/abs/2310.05914.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_neftune_hook(
    model: nn.Module,
    alpha: float,
    seq_len: int,
) -> torch.utils.hooks.RemovableHandle:
    """Register a NEFTune forward hook on model.tok_embeddings.

    Noise is applied only in training mode. The caller supplies seq_len from
    TrainConfig because current collators flatten embedding output to
    (T, embed_dim); inferring the original batch and sequence dimensions from
    tensor shape is no longer valid. Using the configured length preserves the
    paper's normalization semantics for both packed and unpacked streams.

    Args:
        model: TorchTitan model exposing tok_embeddings.
        alpha: Noise strength, typically 5-15.
        seq_len: Tokens per sample before batch flattening.

    Returns:
        Removable hook handle.
    """
    embedding = getattr(model, "tok_embeddings", None)
    if embedding is None:
        raise AttributeError(
            f"{type(model).__name__} 没有 tok_embeddings 属性，"
            "无法注册 NEFTune hook"
        )

    def _neftune_hook(module: nn.Module, _input, output: torch.Tensor) -> torch.Tensor:
        if not module.training:
            return output
        # Output shape: (T, embed_dim), where T is the flattened token count.
        embed_dim = output.shape[-1]
        dims = seq_len * embed_dim
        mag = alpha / (dims ** 0.5)
        noise = torch.zeros_like(output).uniform_(-mag, mag)
        return output + noise

    handle = embedding.register_forward_hook(_neftune_hook)
    logger.info(f"NEFTune hook 已注册: alpha={alpha}, embedding={type(embedding).__name__}")
    return handle
