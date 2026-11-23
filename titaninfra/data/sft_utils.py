"""Shared SFT dataset-loading and tokenization helpers.

Load HF Hub, Parquet, JSON, JSONL, or directories into datasets.Dataset, and
convert conversations to input IDs and loss labels through the native dispatch
path. A legacy template path remains for backward compatibility.
"""

import glob as _glob
import logging
import os

from titaninfra.data.templates import ChatTemplate, encode_with_template, validate_messages

# Avoid initializing Transformers during module import.
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


def load_sft_dataset(
    dataset: str | None = None,
    train_files: str | None = None,
    split: str = "train",
    dp_rank: int = 0,
    dp_size: int = 1,
):
    """Load an SFT dataset from HF Hub, a file, or a directory.

    When dp_size is greater than one, shard the resulting dataset by dp_rank.
    The datasets package is imported lazily so argument validation still works
    when it is not installed.
    """
    # Validate arguments before importing the optional datasets dependency.
    if train_files is None and not dataset:
        raise ValueError(
            "load_sft_dataset 必须指定 dataset 或 train_files 其中之一"
        )

    try:
        from datasets import Dataset, load_dataset
    except ImportError as e:
        raise ImportError("SFT 数据加载需要 datasets 库：pip install datasets") from e

    if train_files is not None:
        if os.path.isdir(train_files):
            raw = _load_from_directory(train_files)
        elif train_files.endswith(".parquet"):
            raw = Dataset.from_parquet(train_files)
        elif train_files.endswith(".jsonl") or train_files.endswith(".json"):
            raw = Dataset.from_json(train_files)
        else:
            # Try Parquet for files without a recognized suffix.
            raw = Dataset.from_parquet(train_files)
    else:
        raw = load_dataset(dataset, split=split)

    if dp_size > 1:
        raw = raw.shard(num_shards=dp_size, index=dp_rank)

    return raw


def _load_from_directory(dir_path: str):
    """Load and concatenate all supported files in a directory."""
    from datasets import Dataset, concatenate_datasets

    parts = []
    for pattern, loader in [
        ("*.parquet", lambda p: Dataset.from_parquet(p)),
        ("*.jsonl", lambda p: Dataset.from_json(p)),
        ("*.json", lambda p: Dataset.from_json(p)),
    ]:
        for file_path in sorted(_glob.glob(os.path.join(dir_path, pattern))):
            try:
                ds = loader(file_path)
                parts.append(ds)
                logger.info(f"加载数据文件：{file_path} ({len(ds)} 条)")
            except Exception as e:
                logger.warning(f"跳过文件 {file_path}：{e}")

    if not parts:
        raise ValueError(
            f"目录 {dir_path} 中没有找到有效数据文件"
            "（支持 .parquet、.json、.jsonl）"
        )

    return concatenate_datasets(parts)


def tokenize_conversation(
    messages: list[dict],
    tokenizer,
    template: ChatTemplate | None = None,
    max_length: int = 4096,
    train_on_inputs: bool = False,
    enable_thinking: bool | None = None,
    default_system: str | None = None,
    debug: bool = False,
    family: str = "qwen3",
    tools: list | None = None,
    loss_config: dict | None = None,
) -> tuple[list[int], list[int]]:
    """Tokenize a conversation and produce loss labels.

    With template=None, route through dispatch.tokenize_sample. Supplying a
    template uses the legacy formatter for backward compatibility. Prompt
    positions use IGNORE_INDEX unless train_on_inputs is enabled.
    """
    # Legacy path: use an explicitly supplied template.
    if template is not None:
        template_name = template.name
        validation_result = validate_messages(messages, template_name)
        for warning in validation_result.warnings:
            logger.warning(f"数据完整性警告: {warning}")
        if not validation_result.is_valid:
            raise ValueError(f"数据格式错误: {'; '.join(validation_result.errors)}")

        return encode_with_template(
            tokenizer=tokenizer,
            template=template,
            messages=messages,
            max_length=max_length,
            train_on_inputs=train_on_inputs,
            enable_thinking=enable_thinking,
            default_system=default_system,
            debug=debug,
        )

    # Preferred path: route through the unified native dispatcher.
    from titaninfra.data.sft import dispatch

    sample = {"messages": messages, "system": default_system, "tools": tools}
    encoded = dispatch.tokenize_sample(
        sample,
        tokenizer,
        family,
        max_length=max_length,
        loss_config=loss_config,
        train_on_inputs=train_on_inputs,
    )
    return list(encoded.input_ids), list(encoded.labels)


def load_hf_tokenizer(tokenizer_path: str):
    """Load a trusted HF tokenizer and ensure pad_token is set."""
    if AutoTokenizer is None:
        raise ImportError("SFT 需要 transformers 库：pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,  # Only load tokenizer code from a trusted source.
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Supply a basic ChatML template only when the tokenizer lacks one.
    if not getattr(tokenizer, "chat_template", None):
        logger.warning(
            f"Tokenizer '{tokenizer_path}' 缺少 chat_template，自动设置 ChatML 格式模板。"
            "建议使用带有内置 chat_template 的模型 tokenizer。"
        )
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
        )

    return tokenizer
