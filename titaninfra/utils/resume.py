"""Unified checkpoint setup and resume helpers for Pretrain and SFT.

When the checkpoint folder exists, TorchTitan loads its latest DCP state;
otherwise initial_load_path supplies HF or DCP weights. This module resolves
resume paths, task-name suffixes, checkpoint copying, and HF validation.
"""

import logging
import os
import shutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointSetup:
    """Immutable checkpoint setup returned by resolve_checkpoint_config.

    Fields are ckpt_subdir, dump_folder, initial_load_path, and
    initial_load_in_hf (whether initial weights use safetensors).
    """

    ckpt_subdir: str
    dump_folder: str
    initial_load_path: str | None
    initial_load_in_hf: bool


def resolve_checkpoint_config(
    *,
    resume_from: str | None,
    ckpt_subdir: str,
    dump_folder: str,
    hf_ckpt_path: str | None,
    task_name: str | None,
    initial_load_path: str | None = None,
    initial_load_in_hf: bool = False,
    stage: str = "pretrain",
    is_hf_path_fn=None,
) -> CheckpointSetup:
    """Resolve checkpoint setup and return an immutable CheckpointSetup.

    resume_from may name a step directory or its parent. Validation raises
    ValueError for missing checkpoints and FileNotFoundError for missing HF
    assets. stage controls initialization logging only.
    """
    # Normalize paths before resolving checkpoint locations.
    ckpt_subdir = os.path.normpath(ckpt_subdir)
    dump_folder = os.path.normpath(dump_folder)

    # Append task_name only to dump_folder; ckpt_subdir remains explicit.
    effective_ckpt_subdir = ckpt_subdir
    effective_dump_folder = f"{dump_folder}_{task_name}" if task_name else dump_folder
    if task_name:
        logger.info(f"使用 task_name '{task_name}'，dump_folder：{effective_dump_folder}")

    # Handle resume or fresh initialization.
    if resume_from:
        _handle_resume(
            resume_from=resume_from,
            ckpt_subdir=effective_ckpt_subdir,
            dump_folder=effective_dump_folder,
        )
    else:
        _handle_no_resume(
            ckpt_subdir=effective_ckpt_subdir,
            dump_folder=effective_dump_folder,
            hf_ckpt_path=hf_ckpt_path,
            stage=stage,
        )

    # Resolve initial_load_path and initial_load_in_hf.
    using_default_checker = is_hf_path_fn is None
    hf_checker = is_hf_path_fn or _default_is_hf_path
    if initial_load_path is None:
        initial_load_path = hf_ckpt_path or None
        initial_load_in_hf = hf_checker(hf_ckpt_path) if hf_ckpt_path else False
    if initial_load_path and "://" not in initial_load_path:
        initial_load_path = os.path.abspath(initial_load_path)
    if (
        initial_load_path
        and not initial_load_in_hf
        and os.path.isdir(initial_load_path)
        and not os.path.isfile(os.path.join(initial_load_path, ".metadata"))
    ):
        completed_steps = [
            step
            for step in _list_checkpoints(initial_load_path)
            if os.path.isfile(os.path.join(initial_load_path, step, ".metadata"))
        ]
        if completed_steps:
            initial_load_path = os.path.join(
                initial_load_path,
                _find_latest_step(completed_steps),
            )

    # Validate HF assets only with the default checker; custom checkers own
    # their validation policy.
    if hf_ckpt_path and using_default_checker:
        # Verify that the supplied checkpoint path exists.
        if not os.path.exists(hf_ckpt_path):
            raise FileNotFoundError(
                f"❌ hf_ckpt_path 路径不存在: {hf_ckpt_path}\n"
                f"请检查：\n"
                f"  1. 路径是否正确（建议使用绝对路径）\n"
                f"  2. 训练节点是否可以访问该路径\n"
                f"  3. 是否已下载 HuggingFace 模型到该位置"
            )

        # Verify Hugging Face format to avoid accidental random initialization.
        if not initial_load_in_hf:
            logger.error(
                f"❌ hf_ckpt_path '{hf_ckpt_path}' 未被识别为 HuggingFace 格式\n"
                f"_is_hf_path() 返回 False，这可能是因为：\n"
                f"  1. 目录缺少关键文件（config.json 或 model.safetensors.index.json）\n"
                f"  2. 文件命名不符合 HuggingFace 标准\n"
                f"  3. 权重格式不是 safetensors\n"
                f"\n"
                f"这将导致 TorchTitan 跳过权重加载，使用随机初始化！\n"
                f"请检查目录内容：\n"
                f"  - 是否存在 config.json\n"
                f"  - 是否存在 model.safetensors 或 model.safetensors.index.json\n"
                f"  - 对于分片模型，是否存在 model-00001-of-000XX.safetensors 文件"
            )
            raise ValueError(
                f"无效的 HuggingFace checkpoint 目录: {hf_ckpt_path}\n"
                f"目录未被识别为 HuggingFace 格式。请检查文件完整性。"
            )
        else:
            # Detect whether the checkpoint uses safetensors shards.
            has_shards = any(
                f.startswith("model-") and f.endswith(".safetensors")
                for f in os.listdir(hf_ckpt_path)
            ) if os.path.isdir(hf_ckpt_path) else False

            if has_shards:
                logger.info(
                    f"✅ hf_ckpt_path 识别为 HuggingFace 分片模型格式\n"
                    f"   路径: {hf_ckpt_path}\n"
                    f"   TorchTitan 将自动合并加载分片权重"
                )
            else:
                logger.info(
                    f"✅ hf_ckpt_path 识别为 HuggingFace 格式\n"
                    f"   路径: {hf_ckpt_path}"
                )

    return CheckpointSetup(
        ckpt_subdir=effective_ckpt_subdir,
        dump_folder=effective_dump_folder,
        initial_load_path=initial_load_path,
        initial_load_in_hf=initial_load_in_hf,
    )


# Internal helpers.


def _handle_resume(
    *,
    resume_from: str,
    ckpt_subdir: str,
    dump_folder: str,
) -> None:
    """Resolve resume_from and copy the latest checkpoint when necessary."""
    # Require the resume source to exist.
    if not os.path.exists(resume_from):
        raise ValueError(
            f"resume_from 路径不存在: {resume_from}\n"
            f"请检查路径是否正确"
        )

    # Distinguish a step directory from a checkpoint root.
    resume_parent = _resolve_resume_parent(resume_from)

    # Require at least one valid checkpoint.
    _validate_has_checkpoints(resume_parent)

    # Construct the target checkpoint directory.
    full_ckpt_dir = os.path.join(dump_folder, ckpt_subdir)

    # Check whether source and target are already the same directory.
    if os.path.abspath(resume_parent) == os.path.abspath(full_ckpt_dir):
        # Same directory: load and continue saving in place.
        logger.info(
            f"断点续训配置：\n"
            f"  - resume_from: {resume_from}\n"
            f"  - checkpoint 目录: {full_ckpt_dir}\n"
            f"  - TorchTitan 将自动加载最新 checkpoint 并继续保存到同一目录"
        )
    else:
        # Different directory: copy the latest checkpoint to the target.
        _copy_latest_checkpoint(
            resume_parent=resume_parent,
            full_ckpt_dir=full_ckpt_dir,
        )


def _handle_no_resume(
    *,
    ckpt_subdir: str,
    dump_folder: str,
    hf_ckpt_path: str | None,
    stage: str,
) -> None:
    """Log initialization behavior when resume_from is not supplied."""
    full_ckpt_dir = os.path.join(dump_folder, ckpt_subdir)
    has_checkpoint = False

    if os.path.exists(full_ckpt_dir):
        checkpoints = _list_checkpoints(full_ckpt_dir)
        if checkpoints:
            has_checkpoint = True
            logger.info(
                f"检测到已有 checkpoint 目录: {full_ckpt_dir}\n"
                f"包含 {len(checkpoints)} 个 checkpoint\n"
                f"TorchTitan 将自动加载最新的 checkpoint 继续训练"
            )

    if has_checkpoint:
        return

    # With no existing checkpoint, log stage-appropriate initialization hints.
    if hf_ckpt_path is None:
        if stage == "sft":
            # SFT usually expects pretrained weights, so warn about random init.
            logger.warning(
                "SFT 通常需要从预训练权重初始化。\n"
                f"当前配置：\n"
                f"  - hf_ckpt_path: None\n"
                f"  - resume_from: None\n"
                f"  - checkpoint 目录: {full_ckpt_dir} (不存在或为空)\n"
                f"将从随机权重开始训练。如需从预训练权重开始，请设置 hf_ckpt_path。"
            )
        else:
            # Pretrain can legitimately start from random weights.
            logger.info(
                "Pretrain 初始化配置：\n"
                f"  - hf_ckpt_path: None\n"
                f"  - resume_from: None\n"
                f"  - checkpoint 目录: {full_ckpt_dir} (不存在或为空)\n"
                f"将从随机权重开始预训练。如需从已有权重继续训练，请设置 hf_ckpt_path 或 resume_from。"
            )
    else:
        if stage == "pretrain":
            logger.info(
                f"Pretrain 初始化配置：\n"
                f"  - 将从 HuggingFace 权重初始化: {hf_ckpt_path}\n"
                f"  - checkpoint 目录: {full_ckpt_dir}"
            )


def _resolve_resume_parent(resume_from: str) -> str:
    """Return the checkpoint root for a path or a step directory.

    step-* directories return their parent; other paths are returned unchanged.
    """
    stripped = resume_from.rstrip("/")
    basename = os.path.basename(stripped)
    if (
        os.path.isfile(resume_from)
        or basename.startswith("step-")
        or basename.startswith("step_")
    ):
        return os.path.dirname(stripped)
    return resume_from


def _validate_has_checkpoints(directory: str) -> None:
    """Raise ValueError when no step-* checkpoint exists in directory."""
    if not os.path.isdir(directory):
        return
    checkpoints = _list_checkpoints(directory)
    if not checkpoints:
        raise ValueError(
            f"resume_from 目录中没有找到 checkpoint: {directory}\n"
            f"请确保目录包含 step-xxx 或 step_xxx 格式的 checkpoint"
        )


def _list_checkpoints(directory: str) -> list[str]:
    """List step-* checkpoint directories in lexical order."""
    if not os.path.isdir(directory):
        return []
    return [
        f
        for f in os.listdir(directory)
        if f.startswith("step-") or f.startswith("step_")
    ]


def _copy_latest_checkpoint(
    *,
    resume_parent: str,
    full_ckpt_dir: str,
) -> None:
    """Copy the latest checkpoint when source and target differ.

    Copying works across filesystems and prevents new checkpoints or retention
    cleanup from modifying the source directory.
    """
    # Reuse an existing target checkpoint instead of copying over it.
    if os.path.exists(full_ckpt_dir):
        existing_checkpoints = _list_checkpoints(full_ckpt_dir)
        if existing_checkpoints:
            logger.info(
                f"目标 checkpoint 目录已有内容：\n"
                f"  - 目标目录: {full_ckpt_dir}\n"
                f"  - 包含 {len(existing_checkpoints)} 个 checkpoint\n"
                f"  - 跳过复制，使用现有目录继续训练"
            )
            return

    # Select the latest source step by numeric suffix.
    steps = _list_checkpoints(resume_parent)
    latest = _find_latest_step(steps)

    src = os.path.join(resume_parent, latest)
    dst = os.path.join(full_ckpt_dir, latest)
    os.makedirs(full_ckpt_dir, exist_ok=True)

    logger.info(
        f"断点续训配置（checkpoint 复制）：\n"
        f"  - 从: {src}\n"
        f"  - 到: {dst}\n"
        f"  - 复制中..."
    )
    shutil.copytree(src, dst, dirs_exist_ok=True)
    logger.info(
        f"  - 复制完成\n"
        f"  - TorchTitan 将从 {full_ckpt_dir} 加载\n"
        f"  - 新 checkpoint 将保存到 {full_ckpt_dir}"
    )


def _find_latest_step(step_names: list[str]) -> str:
    """Return the step name with the largest numeric suffix."""
    import re

    def _extract_step_num(name: str) -> int:
        m = re.search(r"step[-_](\d+)", name)
        return int(m.group(1)) if m else 0

    return max(step_names, key=_extract_step_num)


def _default_is_hf_path(path: str | None) -> bool:
    """Return whether path looks like a Hugging Face checkpoint.

    Recognizes model.safetensors, index.json, config.json, and sharded weights.
    DCP directories containing .distcp are rejected.
    """
    if not path:
        return False

    if not os.path.exists(path):
        # Missing paths fall back to a suffix heuristic.
        logger.debug(f"_is_hf_path: 路径不存在 {path}，根据后缀判断")
        return not path.endswith(".distcp")

    if os.path.isdir(path):
        file_names = {f for f in os.listdir(path)}

        # Check for standard HF metadata files.
        has_model_safetensors = "model.safetensors" in file_names
        has_model_index = "model.safetensors.index.json" in file_names
        has_config = "config.json" in file_names

        # Detect sharded model weights.
        has_shards = any(
            f.startswith("model-") and f.endswith(".safetensors")
            for f in file_names
        )

        # Detect DCP marker files.
        has_distcp = any(f.endswith(".distcp") for f in file_names)

        logger.debug(
            f"_is_hf_path 检查 {path}: "
            f"model.safetensors={has_model_safetensors}, "
            f"index.json={has_model_index}, "
            f"config.json={has_config}, "
            f"shards={has_shards}, "
            f"distcp={has_distcp}"
        )

        # Accept recognized HF layouts.
        if has_model_safetensors:
            logger.debug("✅ 识别为 HF 格式（model.safetensors）")
            return True
        if has_model_index:
            logger.debug("✅ 识别为 HF 格式（model.safetensors.index.json）")
            return True
        if has_config:
            logger.debug("✅ 识别为 HF 格式（config.json）")
            return True

        # Reject DCP layouts.
        if has_distcp:
            logger.debug("❌ 识别为 DCP 格式（包含 .distcp 文件）")
            return False

    return False
