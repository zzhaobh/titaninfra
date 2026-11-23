"""Download selected Hugging Face model assets with glob filters."""

import logging
import os
from fnmatch import fnmatch
from pathlib import Path

logger = logging.getLogger(__name__)

ASSET_PATTERNS: dict[str, list[str]] = {
    "tokenizer": [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ],
    "safetensors": ["*.safetensors", "*model.safetensors.index.json"],
    "index": ["*model.safetensors.index.json"],
    "config": ["config.json", "generation_config.json"],
}


def _resolve_download_patterns(
    asset_types: list[str],
    additional_patterns: list[str] | None,
) -> list[str]:
    """Resolve asset-type names and additional globs into a pattern list."""
    patterns: list[str] = []
    for asset_type in asset_types:
        if asset_type not in ASSET_PATTERNS:
            raise ValueError(
                f"未知资产类型 '{asset_type}'。"
                f"可用类型：{list(ASSET_PATTERNS.keys())}"
            )
        patterns.extend(ASSET_PATTERNS[asset_type])
    if additional_patterns:
        patterns.extend(additional_patterns)
    return patterns


def _list_files_to_download(
    repo_id: str,
    token: str | None,
    patterns: list[str],
    download_all: bool,
) -> list[str]:
    """List repository files and filter them by the requested patterns."""
    from huggingface_hub import list_repo_files

    try:
        available_files = list(list_repo_files(repo_id=repo_id, token=token))
    except Exception as e:
        if "401" in str(e):
            raise PermissionError(
                f"访问 {repo_id} 需要有效的 HF token。"
                "请设置 hf_token 参数或 HF_TOKEN 环境变量。"
            ) from e
        raise

    if download_all:
        return available_files

    def _should_download(filename: str) -> bool:
        basename = os.path.basename(filename)
        for pattern in patterns:
            p = pattern.lower()
            if basename == p:
                return True
            if ("*" in p or "?" in p) and fnmatch(basename, p):
                return True
        return False

    return [f for f in available_files if _should_download(f)]


def _download_files(
    files: list[str],
    repo_id: str,
    model_dir: str,
    token: str | None,
) -> tuple[list[str], list[str]]:
    """Download files and return (downloaded, failed_missing_404)."""
    from huggingface_hub import hf_hub_download

    try:
        from tqdm import tqdm
        iterator = tqdm(files, desc="下载中", unit="file")
        use_tqdm = True
    except ImportError:
        iterator = files
        use_tqdm = False

    downloaded: list[str] = []
    failed: list[str] = []

    for filename in iterator:
        try:
            if use_tqdm:
                iterator.set_description(f"下载 {os.path.basename(filename)}")  # type: ignore[attr-defined]
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
                token=token,
            )
            downloaded.append(filename)
        except Exception as e:
            if "404" in str(e):
                logger.warning(f"文件 {filename} 不存在，跳过")
                failed.append(filename)
            else:
                raise

    return downloaded, failed


def download_hf_assets(
    repo_id: str,
    local_dir: str = "./assets/hf",
    asset_types: str | list[str] = "tokenizer",
    download_all: bool = False,
    hf_token: str | None = None,
    additional_patterns: list[str] | None = None,
) -> str:
    """Download model assets from a Hugging Face Hub repository.

    asset_types supports tokenizer, config, safetensors, and index, plus custom
    glob patterns. Returns local_dir/<model-name>.
    """
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "download_hf_assets 需要 huggingface_hub 和 requests：\n"
            "pip install huggingface_hub requests"
        ) from e

    if "/" not in repo_id:
        raise ValueError(
            f"无效的 repo_id 格式：'{repo_id}'。期望格式：'organization/model-name'"
        )

    token = hf_token or os.environ.get("HF_TOKEN")
    model_name = repo_id.split("/")[-1].strip()
    model_dir = os.path.join(local_dir, model_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(asset_types, str):
        asset_types = [asset_types]

    patterns = _resolve_download_patterns(asset_types, additional_patterns)
    logger.info(f"扫描仓库 {repo_id}...")
    files = _list_files_to_download(repo_id, token, patterns, download_all)

    if not files:
        logger.warning(f"在 {repo_id} 中未找到匹配文件")
        return model_dir

    logger.info(f"找到 {len(files)} 个文件，开始下载...")
    downloaded, failed = _download_files(files, repo_id, model_dir, token)

    logger.info(f"下载完成：{len(downloaded)} 个文件 → {model_dir}")
    if failed:
        logger.warning(f"以下文件下载失败：{failed}")

    return model_dir
