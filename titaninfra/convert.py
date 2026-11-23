"""Convert between TorchTitan DCP and Hugging Face safetensors checkpoints.

to_hf exports DCP to HF; from_hf imports HF weights into DCP. Model registry
and state-dict adapters define weight mappings, while DCP storage readers and
writers handle serialization."""

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def to_hf(
    dcp_path: str,
    output_path: str,
    model_name: str,
    hf_assets_path: str | None = None,
    export_dtype: str = "float16",
    max_shard_size: str | int | None = "5GB",
) -> str:
    """Convert a DCP checkpoint to Hugging Face safetensors on CPU.

hf_assets_path is required for tokenizer/config assets and index mapping.
export_dtype and max_shard_size control output precision and sharding.
Returns output_path."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageWriter
    from torchtitan.components.checkpointer import ModelWrapper
    from torchtitan.config import TORCH_DTYPE_MAP

    family, flavor = _parse_model_name(model_name)

    # Validate required HF assets before conversion.
    if hf_assets_path is None:
        raise ValueError(
            "hf_assets_path 必须提供，用于获取 tokenizer、config.json 和 "
            "model.safetensors.index.json 映射文件。\n"
            "示例：hf_assets_path='/path/to/Qwen3-30B-A3B-Base'"
        )

    hf_assets_path_obj = Path(hf_assets_path)
    if not hf_assets_path_obj.exists():
        raise FileNotFoundError(f"hf_assets_path 不存在: {hf_assets_path}")

    # Create the output directory if necessary.
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"转换 DCP → HF：{dcp_path} → {output_path_obj}\n"
        f"  模型：{model_name}\n"
        f"  精度：{export_dtype}\n"
        f"  HF assets：{hf_assets_path}"
    )

    # 1. Resolve the model spec, preferring TitanInfra over TorchTitan.
    model_module = _import_model_module(family)
    model_spec = model_module.model_registry(flavor)
    model_config = model_spec.model

    # 2. Build the model on CPU.
    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    # 3. Resolve the state-dict adapter.
    sd_adapter = model_spec.state_dict_adapter(model_config, str(hf_assets_path_obj))
    if sd_adapter is None:
        raise RuntimeError(
            f"模型 {model_name} 的 state_dict_adapter 为 None，无法转换。"
            f"请检查 torchtitan.models.{family} 是否实现了 StateDictAdapter。"
        )

    # 4. Load the DCP checkpoint.
    state_dict = model._get_state_dict()
    logger.info(f"加载 DCP checkpoint: {dcp_path}")
    try:
        dcp.load(state_dict, checkpoint_id=str(dcp_path))
    except Exception as e:
        raise RuntimeError(
            f"加载 DCP checkpoint 失败: {dcp_path}\n"
            f"请检查 checkpoint 路径是否存在且格式完整（.distcp 目录应包含 __0_0.distcp 等文件）。\n"
            f"原始错误: {e}"
        ) from e

    # 5. Convert state-dict keys to HF layout.
    logger.info("转换 state dict 格式 (TT → HF)...")
    hf_state_dict = sd_adapter.to_hf(state_dict)

    # 6. Apply the requested export dtype.
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    # 7. Save safetensors with shard mapping when needed.
    fqn_mapping = sd_adapter.fqn_to_index_mapping
    if fqn_mapping is None and max_shard_size is not None:
        shard_bytes = _parse_size(max_shard_size)
        fqn_mapping = _auto_shard_mapping(hf_state_dict, max_shard_bytes=shard_bytes)
        n_shards = max(fqn_mapping.values())
        logger.info(f"自动分片：{n_shards} 个 shard（每个 ≤{max_shard_size}）")

    logger.info(f"保存 HF checkpoint: {output_path_obj}")
    storage_writer = HuggingFaceStorageWriter(
        path=str(output_path_obj),
        save_distributed=True,
        fqn_to_index_mapping=fqn_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    dcp.save(hf_state_dict, storage_writer=storage_writer)

    # 8. Remove the intermediate sharded directory after consolidation.
    sharded_dir = output_path_obj / "sharded"
    if sharded_dir.exists():
        import shutil
        shutil.rmtree(sharded_dir)
        logger.info(f"已清理 sharded 中间目录: {sharded_dir}")

    # 9. Copy tokenizer and config assets.
    _copy_hf_assets(hf_assets_path_obj, output_path_obj)

    logger.info(f"✅ 转换完成：{output_path_obj}")
    return str(output_path_obj)


def from_hf(
    hf_path: str,
    output_path: str,
    model_name: str,
) -> str:
    """Convert Hugging Face safetensors into a DCP checkpoint.

The model is built on CPU, its adapter maps weights to TorchTitan layout,
and the result is saved to output_path."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageReader
    from torchtitan.components.checkpointer import ModelWrapper

    family, flavor = _parse_model_name(model_name)

    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    logger.info(f"转换 HF → DCP：{hf_path} → {output_path_obj}（模型：{model_name}）")

    # 1. Resolve the model spec, preferring TitanInfra over TorchTitan.
    model_module = _import_model_module(family)
    model_spec = model_module.model_registry(flavor)
    model_config = model_spec.model

    # 2. Build the model on CPU.
    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    # 3. Resolve the state-dict adapter.
    sd_adapter = model_spec.state_dict_adapter(model_config, None)
    if sd_adapter is None:
        raise RuntimeError(
            f"模型 {model_name} 的 state_dict_adapter 为 None，无法转换。"
        )

    # 4. Prepare an empty state dict in HF layout for loading.
    state_dict = model._get_state_dict()
    hf_state_dict = sd_adapter.to_hf(state_dict)

    # 5. Load the HF checkpoint.
    logger.info(f"加载 HF checkpoint: {hf_path}")
    try:
        dcp.load(
            hf_state_dict,
            storage_reader=HuggingFaceStorageReader(path=str(hf_path)),
        )
    except Exception as e:
        raise RuntimeError(
            f"加载 HF checkpoint 失败: {hf_path}\n"
            f"请检查路径是否包含 model.safetensors 或 model-*.safetensors 文件。\n"
            f"原始错误: {e}"
        ) from e

    # 6. Convert state-dict keys back to TorchTitan layout.
    logger.info("转换 state dict 格式 (HF → TT)...")
    state_dict = sd_adapter.from_hf(hf_state_dict)

    # 7. Save the DCP checkpoint.
    logger.info(f"保存 DCP checkpoint: {output_path_obj}")
    dcp.save(state_dict, checkpoint_id=str(output_path_obj))

    logger.info(f"✅ 转换完成：{output_path_obj}")
    return str(output_path_obj)


def _import_model_module(family: str):
    """Import a family registry, trying TitanInfra before TorchTitan."""
    for pkg in (f"titaninfra.models.{family}", f"torchtitan.models.{family}"):
        try:
            return importlib.import_module(pkg)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        f"找不到模型 '{family}' 的 registry。"
        f"请确认 titaninfra.models.{family} 或 torchtitan.models.{family} 已安装。"
    )


def _parse_model_name(model_name: str) -> tuple[str, str]:
    """Split a model name into (family, flavor), requiring family/flavor form."""
    if "/" not in model_name:
        raise ValueError(
            f"model_name 格式错误：{model_name!r}。"
            f"应为 'family/flavor' 格式，如 'qwen3/8B' 或 'qwen3/30B-A3B'"
        )
    family, flavor = model_name.split("/", 1)
    return family, flavor


def _parse_size(size: str | int) -> int:
    """Parse human-readable size string ("5GB", "500MB") to bytes."""
    if isinstance(size, int):
        return size
    s = size.strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * 1024**3)
    if s.endswith("MB"):
        return int(float(s[:-2]) * 1024**2)
    if s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    return int(s)


def _auto_shard_mapping(state_dict: dict, max_shard_bytes: int = 5 * 1024**3) -> dict[str, int]:
    """Assign each weight to a shard index based on cumulative size (≤max_shard_bytes per shard)."""
    mapping: dict[str, int] = {}
    current_shard = 1
    current_bytes = 0
    for key, tensor in state_dict.items():
        nbytes = tensor.numel() * tensor.element_size()
        if current_bytes > 0 and current_bytes + nbytes > max_shard_bytes:
            current_shard += 1
            current_bytes = 0
        mapping[key] = current_shard
        current_bytes += nbytes
    return mapping


def _copy_hf_assets(hf_assets_path: Path, output_path: Path) -> None:
    """Copy available tokenizer and config assets to the export directory."""
    import shutil

    asset_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]

    copied = []
    for filename in asset_files:
        src = hf_assets_path / filename
        if src.exists():
            dst = output_path / filename
            shutil.copy2(src, dst)
            copied.append(filename)

    if copied:
        logger.info(f"复制 HF assets 文件: {', '.join(copied)}")
    else:
        logger.warning(
            f"未找到任何 HF assets 文件在 {hf_assets_path}。"
            f"输出目录可能缺少 tokenizer 配置。"
        )
