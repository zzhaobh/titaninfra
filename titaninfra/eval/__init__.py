"""Pipeline / training-result evaluation utilities.

Shipped in the wheel so production training recipes (e.g.
``scripts/train/train_fsb_qwen3_v10.py``) and GPU pipeline tests can
import these checks after a non-editable ``pip install``.
"""
from titaninfra.eval.pretrain_checks import (
    check_checkpoint_dir,
    check_checkpoint_resume_continuity,
    check_eval_metrics,
    check_moe_health_metrics,
    check_per_stream_metrics,
    check_repeat_scheduling,
    check_resume_tokens_continuity,
    check_sft_data_health,
    check_train_result,
    check_training_health,
    check_wandb_offline_metrics,
    make_batch_validator,
    validate_sft_batch,
)

__all__ = [
    "check_checkpoint_dir",
    "check_checkpoint_resume_continuity",
    "check_eval_metrics",
    "check_moe_health_metrics",
    "check_per_stream_metrics",
    "check_repeat_scheduling",
    "check_resume_tokens_continuity",
    "check_sft_data_health",
    "check_train_result",
    "check_training_health",
    "check_wandb_offline_metrics",
    "make_batch_validator",
    "validate_sft_batch",
]
