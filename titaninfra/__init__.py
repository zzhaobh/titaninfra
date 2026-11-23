"""TitanInfra public API for Pretrain, SFT, and CPT.

Typical usage:

    from titaninfra import ModelConfig, StreamConfig, TrainConfig, Trainer

    result = Trainer(
        model=ModelConfig(
            name="qwen3/8B",
            dtype="bf16",
            param_dtype="fp32",
            attn_backend="flex",
            attn_mask_type="block_causal",
            ac_mode="selective",
            compile=False,
        ),
        train=TrainConfig(
            local_batch_size=1,
            global_batch_size=-1,
            seq_len=4096,
            steps=100,
            optimizer_name="AdamW",
            lr=1e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
            optimizer_state_dtype="fp32",
            warmup_steps=10,
            min_lr_ratio=0.1,
            lr_decay_type="cosine",
            max_grad_norm=1.0,
        ),
        streams=[StreamConfig(local="/data/train", tokenizer_name="qwen3")],
    )
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("titaninfra")
except PackageNotFoundError:
    __version__ = "1.1.0"

from titaninfra.config import ModelConfig, PackStrategy, StreamConfig, TrainConfig
from titaninfra.config.train import MoEBiasSchedule  # noqa: E402
from titaninfra.convert import from_hf, to_hf
from titaninfra.data.templates import get_template, list_templates, register_template  # noqa: E402
from titaninfra.monitoring import MoEHealthPlugin  # noqa: E402
from titaninfra.tools import compare_losses, diagnose_hf_checkpoint, download_hf_assets
from titaninfra.trainer import Trainer, TrainResult

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "StreamConfig",
    "PackStrategy",
    "Trainer",
    "TrainResult",
    "to_hf",
    "from_hf",
    "download_hf_assets",
    "compare_losses",
    "diagnose_hf_checkpoint",
    "get_template",
    "list_templates",
    "register_template",
    "MoEHealthPlugin",
    "MoEBiasSchedule",
]
