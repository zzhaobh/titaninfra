"""Unified training entry point for Pretrain, SFT, and CPT."""

from collections.abc import Callable
from dataclasses import dataclass

from titaninfra.config.data import StreamConfig
from titaninfra.config.model import ModelConfig
from titaninfra.config.train import TrainConfig


@dataclass
class TrainResult:
    """Result returned after a training run finishes."""

    stage: str
    ckpt_dir: str
    steps: int
    model_name: str

    def __repr__(self) -> str:
        return (
            f"TrainResult(stage={self.stage!r}, "
            f"model={self.model_name!r}, "
            f"steps={self.steps}, "
            f"ckpt_dir={self.ckpt_dir!r})"
        )


def Trainer(
    model: ModelConfig,
    train: TrainConfig,
    streams: list[StreamConfig],
    *,
    hf_ckpt_path: str | None = None,
    task_name: str | None = None,
    _batch_validator: Callable | None = None,
) -> TrainResult:
    """Run Pretrain, SFT, or CPT using the TorchTitan-backed engine.

    Args:
        model: Model architecture and parallel strategy.
        train: Optimizer, scheduler, checkpoint, and runtime configuration.
        streams: MDS text and/or message streams used for training.
        hf_ckpt_path: Optional Hugging Face or DCP checkpoint path.
        task_name: Optional output-directory suffix.

    Returns:
        TrainResult describing the completed run.
    """
    if not isinstance(train, TrainConfig):
        raise TypeError(
            f"无法从 train 类型 {type(train).__name__} 推断引擎，"
            "请使用 TrainConfig（pretrain/SFT/CPT）"
        )

    resolved_task_name = task_name or train.task_name

    if isinstance(train, TrainConfig):
        from titaninfra.config.data import validate_scheduling_combination, validate_streams_mixing_mode

        validate_streams_mixing_mode(streams)
        validate_scheduling_combination(streams, train)

        if train.optimizer_state_dtype == "bf16" and model.param_dtype != "fp32":
            raise ValueError(
                "optimizer_state_dtype='bf16' requires ModelConfig.param_dtype='fp32'; "
                f"got param_dtype={model.param_dtype!r}."
            )

        from titaninfra.engine.pretrain import run_train

        return run_train(
            model,
            train,
            streams,
            hf_ckpt_path,
            resolved_task_name,
            _batch_validator=_batch_validator,
        )

    raise TypeError(
        f"Cannot infer an engine from train type {type(train).__name__}; "
        "use TrainConfig for Pretrain, SFT, or CPT."
    )
