"""Convert raw JSONL, JSON, and Parquet data into balanced MDS datasets.

DataConverter handles pretrain text and convert_messages_to_mds handles SFT
messages with the same native encoders used by runtime training. Conversion
computes exact token statistics, length histograms, and stream_stats.json."""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from titaninfra.data.validator import validate_or_raise

logger = logging.getLogger(__name__)


# ============================================================================
# ============================================================================

OutputPath = str | tuple[str, str]
"""Output path: local, remote, or a (local, remote) tuple."""

WriterFormat = Literal["mds", "json", "csv", "tsv"]
"""Supported output formats."""


# ============================================================================
# ============================================================================


class ShardValidationError(Exception):
    """Shard validation error configuration and behavior."""

    def __init__(self, message: str, *, errors: list[str]):
        super().__init__(message)
        self.errors = errors


# ============================================================================
# ============================================================================


@dataclass(kw_only=True, slots=True)
class DataConvertConfig:
    """Configuration for MDS conversion.

source and output are required. tokenizer_path and tokenizer_name define token
statistics; mode selects text or messages, and messages additionally requires
family. Sharding, worker counts, checkpoint/resume, validation, and overwrite
behavior are configurable."""

    source: str
    """Input source file or directory."""

    output: OutputPath
    """Output path: local, remote, or a (local, remote) tuple."""

    file_format: str | None = None
    """Input format override (parquet, json, jsonl); None auto-detects it."""

    output_format: WriterFormat = "mds"
    """Output format: mds, json, csv, or tsv."""

    columns: list[str] | None = None
    """Columns to convert; None selects all columns."""

    field_mapping: dict[str, str] | None = None
    """Mapping from output fields to possibly nested source paths."""

    mds_columns: dict[str, str] | None = None
    """MDS column types for scalar, array, image, JSON, and pickle data."""

    tokenizer_path: str | None = None
    """Tokenizer path used for exact token counts and balanced shards."""

    tokenizer_name: str | None = None
    """Tokenizer key in stream_stats.json; inferred from the path when omitted."""

    text_key: str = "text"
    """Source field containing pretrain text."""

    add_bos: bool = True
    """Add a BOS token during conversion."""

    add_eos: bool = True
    """Add an EOS token during conversion."""

    row_filter: dict[str, str | list[str]] | None = None
    """Declarative row filter applied as early as possible in the reader."""

    num_shards: int = 16
    """Target shard count; use enough shards for all data-parallel ranks."""

    store_token_count: bool = False
    """Store an additional token_count column in MDS."""

    compression: str | None = None
    """Optional shard compression algorithm supported by the MDS writer."""

    size_limit: int | str = "256MB"
    """Shard size limit as bytes or a human-readable string such as 256MB."""

    hashes: list[str] | None = None
    """Hash algorithms used for output integrity metadata."""

    keep_local: bool = False
    """Keep the local copy when writing to remote storage."""

    progress_bar: bool = False
    """Show the upload progress bar."""

    max_workers: int | None = None
    """Parallel upload workers; defaults to min(32, cpu_count + 4)."""

    retry: int = 2
    """Number of upload retries."""

    num_workers: int = 4
    """Parallel reader processes for source files."""

    num_tokenizer_workers: int | None = None
    """Tokenizer workers; None falls back to num_workers."""

    batch_size: int = 1000
    """Number of records processed per batch."""

    overwrite: bool = False
    """Allow replacement of an existing output directory."""

    resume: bool = False
    """Resume from the conversion checkpoint instead of starting over."""

    checkpoint_interval: int | None = None
    """Files per checkpoint; None disables incremental checkpoints."""

    preload: bool = False
    """Read source files into memory before starting the tokenizer pipeline."""

    stream_id: str = ""
    """Human-readable stream ID stored in MDS; inferred from source when empty."""

    stream_type: str = "pretrain"
    """Stream type: pretrain text or sft messages."""

    family: str | None = None
    """Native SFT encoder family used for message tokenization."""

    validate_after_convert: bool = True
    """Run shard quality validation after conversion."""

    validate_sample_ratio: float = 0.01
    """Validation sampling ratio in [0, 1]; 1.0 validates all samples."""

    validate_min_samples: int = 2
    """Minimum validation samples per shard regardless of ratio."""


# ============================================================================
# ============================================================================


def _write_stream_stats(
    output_dir: str,
    num_samples: int,
    total_tokens: int,
    stream_id: str = "",
    stream_type: str = "pretrain",
    tokenizer_name: str | None = None,
    tokenizer_path: str = "",
    *,
    max_token_length: int | None = None,
    min_token_length: int | None = None,
    token_length_dist: dict[str, int] | None = None,
    fine_token_length_dist: dict[str, int] | None = None,
) -> None:
    """Write stream stats."""
    from datetime import UTC, datetime

    from titaninfra.data.schema import StreamStats, TokenizerStats

    now_iso = datetime.now(UTC).isoformat()
    resolved_name = tokenizer_name
    if not resolved_name and tokenizer_path:
        resolved_name = os.path.basename(tokenizer_path.rstrip("/"))
        logger.warning(
            f"tokenizer_name 未显式指定，从路径自动推导为 '{resolved_name}'。"
            f"建议在配置中显式设置 tokenizer_name 以保持与 StreamConfig 一致。"
        )
    if not resolved_name:
        resolved_name = ""
    avg_tokens = total_tokens / max(num_samples, 1)

    tok_stats: dict[str, TokenizerStats] = {}
    if resolved_name:
        tok_stats[resolved_name] = TokenizerStats(
            total_tokens=total_tokens,
            avg_tokens_per_sample=avg_tokens,
            created_at=now_iso,
            max_token_length=max_token_length,
            min_token_length=min_token_length,
            token_length_dist=token_length_dist,
            fine_token_length_dist=fine_token_length_dist,
        )

    stats = StreamStats(
        stream_id=stream_id,
        stream_type=stream_type,
        num_samples=num_samples,
        default_tokenizer=resolved_name,
        tokenizer_stats=tok_stats,
        created_at=now_iso,
    )
    stats.save(output_dir)
    logger.info(f"stream_stats.json (v3) 已写入: {output_dir}")


def _try_import_dependencies():
    """Try import dependencies."""
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "DataConverter 需要 pyarrow。" "请安装: pip install pyarrow"
        ) from e

    try:
        from streaming.base import MDSWriter  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "DataConverter 需要 mosaicml-streaming。" "请安装: pip install mosaicml-streaming"
        ) from e


def _resolve_stream_id(config: DataConvertConfig) -> str:
    """Resolve stream id."""
    if config.stream_id:
        return config.stream_id
    return Path(config.source).stem


def _get_nested_value(obj: dict, path: str) -> Any:
    """Return the requested value."""
    keys = path.split(".")
    value = obj
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def _infer_columns_from_sample(sample: dict) -> dict[str, str]:
    """Infer columns from sample."""
    columns = {}
    for key, value in sample.items():
        if isinstance(value, str):
            columns[key] = "str"
        elif isinstance(value, bool):
            columns[key] = "int"
        elif isinstance(value, int):
            columns[key] = "int"
        elif isinstance(value, float):
            columns[key] = "float64"
        elif isinstance(value, bytes):
            columns[key] = "bytes"
        else:
            columns[key] = "str"
    return columns


def _is_shard_completed(shard_dir: str | Path) -> bool:
    """Is shard completed."""
    shard_path = Path(shard_dir)

    if not shard_path.exists():
        return False

    index_file = shard_path / "index.json"
    if not index_file.exists():
        return False

    try:
        with open(index_file, encoding="utf-8") as f:
            index = json.load(f)

        shards = index.get("shards", [])
        if not shards:
            return False

        for shard_info in shards:
            raw_data = shard_info.get("raw_data", {})
            filename = raw_data.get("basename")
            if filename:
                data_file = shard_path / filename
                if not data_file.exists():
                    return False

        return True
    except (json.JSONDecodeError, KeyError, OSError):
        return False


# ============================================================================
# ============================================================================


def _get_writer_class(format: WriterFormat):
    """Return the requested value."""
    from streaming.base import CSVWriter, JSONWriter, MDSWriter, TSVWriter

    writers = {
        "mds": MDSWriter,
        "json": JSONWriter,
        "csv": CSVWriter,
        "tsv": TSVWriter,
    }
    if format not in writers:
        raise ValueError(f"不支持的输出格式: {format}，支持: {list(writers.keys())}")
    return writers[format]


def _create_writer(
    format: WriterFormat,
    output: OutputPath,
    columns: dict[str, str],
    config: DataConvertConfig,
):
    """Create writer."""
    WriterClass = _get_writer_class(format)

    return WriterClass(
        out=output,
        columns=columns,
        compression=config.compression,
        hashes=config.hashes,
        size_limit=config.size_limit,
        keep_local=config.keep_local,
        progress_bar=config.progress_bar,
        max_workers=config.max_workers,
        retry=config.retry,
        exist_ok=config.overwrite,
    )


# ============================================================================
# Tokenization Writer
# ============================================================================




# ============================================================================
# ============================================================================


class _DataReader:
    """Data reader configuration and behavior."""

    PARQUET_EXTENSIONS = {".parquet"}
    JSON_EXTENSIONS = {".json", ".jsonl"}

    def __init__(self, config: DataConvertConfig):
        self.config = config
        _try_import_dependencies()

    def _get_file_format(self, path: str) -> str:
        """Return the requested value."""
        if self.config.file_format:
            return self.config.file_format.lower()

        ext = Path(path).suffix.lower()
        if ext in self.PARQUET_EXTENSIONS:
            return "parquet"
        elif ext in self.JSON_EXTENSIONS:
            return "jsonl" if ext == ".jsonl" else "json"
        else:
            raise ValueError(f"无法推断文件格式: {path}，请通过 file_format 参数指定")

    def list_files(self) -> list[str]:
        """List files."""
        source = self.config.source

        if source.startswith(("s3://", "gs://", "azure://", "oci://")):
            return self._list_cloud_files(source)
        else:
            return self._list_local_files(source)

    def _list_cloud_files(self, uri: str) -> list[str]:
        """List cloud files."""
        import pyarrow.fs as fs

        parsed = uri.split("://", 1)
        if len(parsed) != 2:
            raise ValueError(f"无效的 URI: {uri}")

        scheme, path = parsed
        bucket, _, prefix = path.partition("/")

        if scheme == "s3":
            filesystem = fs.S3FileSystem()
        elif scheme == "gs":
            filesystem = fs.GcsFileSystem()
        elif scheme == "azure":
            filesystem = fs.AzureFileSystem()
        else:
            raise ValueError(f"不支持的云存储: {scheme}")

        selector = fs.FileSelector(prefix, recursive=True) if prefix else fs.FileSelector(bucket, recursive=True)
        files_info = filesystem.get_file_info(selector)

        all_extensions = self.PARQUET_EXTENSIONS | self.JSON_EXTENSIONS
        valid_files = []

        for info in files_info:
            if info.type == fs.FileType.File:
                ext = Path(info.path).suffix.lower()
                if ext in all_extensions:
                    valid_files.append(f"{scheme}://{bucket}/{info.path}")

        logger.info(f"找到 {len(valid_files)} 个文件")
        return valid_files

    def _list_local_files(self, path: str) -> list[str]:
        """List local files."""
        p = Path(path)
        if p.is_file():
            return [str(p)]

        all_extensions = self.PARQUET_EXTENSIONS | self.JSON_EXTENSIONS
        valid_files = [
            str(f)
            for f in p.glob("**/*")
            if f.is_file() and f.suffix.lower() in all_extensions
        ]

        logger.info(f"找到 {len(valid_files)} 个文件")
        return valid_files

    def read_file(self, path: str) -> tuple[Any, str]:
        """Read file."""
        file_format = self._get_file_format(path)

        if file_format == "parquet":
            return self._read_parquet(path), file_format
        else:
            return self._read_json(path), file_format

    def _read_parquet(self, path: str) -> Any:
        """Read parquet."""
        import pyarrow.parquet as pq

        if path.startswith(("s3://", "gs://", "azure://")):
            import pyarrow.fs as fs

            parsed = path.split("://", 1)
            scheme, full_path = parsed
            bucket, _, key = full_path.partition("/")

            if scheme == "s3":
                filesystem = fs.S3FileSystem()
            elif scheme == "gs":
                filesystem = fs.GcsFileSystem()
            elif scheme == "azure":
                filesystem = fs.AzureFileSystem()
            else:
                raise ValueError(f"不支持的云存储: {scheme}")

            table = pq.read_table(f"{bucket}/{key}", filesystem=filesystem)
        else:
            table = pq.read_table(path)

        if self.config.columns:
            available_columns = set(table.column_names)
            selected_columns = [c for c in self.config.columns if c in available_columns]
            if len(selected_columns) < len(self.config.columns):
                missing = set(self.config.columns) - available_columns
                logger.warning(f"以下列不存在: {missing}")
            table = table.select(selected_columns)

        return table

    def _read_json(self, path: str) -> list[dict]:
        """Read json."""
        if path.startswith(("s3://", "gs://", "azure://")):
            import pyarrow.fs as fs

            parsed = path.split("://", 1)
            scheme, full_path = parsed
            bucket, _, key = full_path.partition("/")

            if scheme == "s3":
                filesystem = fs.S3FileSystem()
            elif scheme == "gs":
                filesystem = fs.GcsFileSystem()
            elif scheme == "azure":
                filesystem = fs.AzureFileSystem()
            else:
                raise ValueError(f"不支持的云存储: {scheme}")

            with filesystem.open_input_stream(f"{bucket}/{key}") as f:
                content = f.read().decode("utf-8")
        else:
            with open(path, encoding="utf-8") as f:
                content = f.read()

        records = []
        content = content.strip()

        file_format = self._get_file_format(path)
        if file_format == "jsonl" or content.startswith("{"):
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            data = json.loads(content)
            if isinstance(data, list):
                records = data
            else:
                records = [data]

        if self.config.field_mapping:
            records = self._apply_field_mapping(records)

        return records

    def _apply_field_mapping(self, records: list[dict]) -> list[dict]:
        """Apply the requested transformation."""
        mapped_records = []
        for record in records:
            mapped_record = {}
            for output_key, source_path in self.config.field_mapping.items():
                value = _get_nested_value(record, source_path)
                mapped_record[output_key] = value
            mapped_records.append(mapped_record)
        return mapped_records

    def iter_samples(self, data: Any, file_format: str) -> Any:
        """Iter samples."""
        if file_format == "parquet":
            for record in data.to_pylist():
                yield record
        else:
            for record in data:
                yield record


# ============================================================================
# ============================================================================


def validate_mds_shards(
    output_dir: str,
    *,
    sample_ratio: float = 0.01,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Validate the supplied configuration."""
    output_path = Path(output_dir)
    index_file = output_path / "index.json"

    if not index_file.exists():
        raise ShardValidationError(
            "index.json 不存在",
            errors=["index.json 不存在"],
        )

    try:
        with open(index_file, encoding="utf-8") as f:
            index = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ShardValidationError(
            f"index.json 解析失败: {e}",
            errors=[f"index.json 解析失败: {e}"],
        ) from e

    shards = index.get("shards", [])

    if not shards:
        raise ShardValidationError(
            "shards 列表为空",
            errors=["shards 列表为空"],
        )

    errors: list[str] = []
    total_samples = 0
    schemas: list[dict | None] = []

    for idx, shard_info in enumerate(shards):
        shard_errors = _validate_single_shard(output_path, idx, shard_info)
        errors.extend(shard_errors)

        total_samples += shard_info.get("samples", 0)
        schemas.append(shard_info.get("column_encodings"))

    non_null_schemas = [s for s in schemas if s is not None]
    if len(non_null_schemas) >= 2:
        first = non_null_schemas[0]
        for i, s in enumerate(non_null_schemas[1:], start=1):
            if s != first:
                errors.append(
                    f"shard {i} 的 column schema 与 shard 0 不一致: "
                    f"{s} != {first}"
                )

    if total_samples == 0 and not errors:
        errors.append("样本总数为 0")

    if errors:
        raise ShardValidationError(
            f"分片校验失败，共 {len(errors)} 个错误:\n"
            + "\n".join(f"  - {e}" for e in errors),
            errors=errors,
        )

    return {
        "passed": True,
        "total_shards": len(shards),
        "total_samples": total_samples,
    }


def _validate_single_shard(
    output_path: Path,
    idx: int,
    shard_info: dict,
) -> list[str]:
    """Validate the supplied configuration."""
    errors: list[str] = []
    raw_data = shard_info.get("raw_data", {})
    basename = raw_data.get("basename", "")

    if not basename:
        errors.append(f"shard {idx}: raw_data.basename 为空")
        return errors

    p = Path(basename)
    if p.is_absolute() or ".." in p.parts:
        errors.append(f"shard {idx}: basename 包含非法路径字符 ({basename!r})")
        return errors

    data_file = output_path / basename

    zip_data = shard_info.get("zip_data", {})
    zip_basename = zip_data.get("basename", "")
    if zip_basename:
        zip_file = output_path / zip_basename
        if zip_file.exists():
            expected_bytes = zip_data.get("bytes")
            if expected_bytes is not None:
                actual_bytes = zip_file.stat().st_size
                if actual_bytes != expected_bytes:
                    errors.append(
                        f"shard {idx}: 压缩文件大小不匹配 ({zip_basename}): "
                        f"期望 {expected_bytes}, 实际 {actual_bytes}"
                    )
            zip_hashes = zip_data.get("hashes")
            if zip_hashes:
                errors.extend(_verify_file_hashes(zip_file, zip_hashes, idx, zip_basename))
            return errors

    if not data_file.exists():
        errors.append(f"shard {idx}: 数据文件缺失 ({basename})")
        return errors

    expected_bytes = raw_data.get("bytes")
    if expected_bytes is not None:
        actual_bytes = data_file.stat().st_size
        if actual_bytes != expected_bytes:
            errors.append(
                f"shard {idx}: 文件大小不匹配 ({basename}): "
                f"期望 {expected_bytes}, 实际 {actual_bytes}"
            )

    hashes = raw_data.get("hashes")
    if hashes:
        hash_errors = _verify_file_hashes(data_file, hashes, idx, basename)
        errors.extend(hash_errors)

    return errors


def _verify_file_hashes(
    data_file: Path,
    hashes: dict[str, str],
    idx: int,
    basename: str,
) -> list[str]:
    """Verify file hashes."""
    errors: list[str] = []
    _ALLOWED_HASH_ALGOS = {"md5", "sha1", "sha256", "sha512", "xxh64"}
    file_bytes = data_file.read_bytes()

    for algo, expected_hex in hashes.items():
        if algo not in _ALLOWED_HASH_ALGOS:
            logger.warning(f"不支持的哈希算法: {algo}，跳过（允许列表: {_ALLOWED_HASH_ALGOS}）")
            continue

        if algo == "xxh64":
            try:
                import xxhash

                actual_hex = xxhash.xxh64(file_bytes).hexdigest()
            except ImportError:
                logger.debug("xxhash 未安装，跳过 xxh64 校验")
                continue
        else:
            h = hashlib.new(algo)
            h.update(file_bytes)
            actual_hex = h.hexdigest()

        if actual_hex != expected_hex:
            errors.append(
                f"shard {idx}: 哈希不匹配 ({basename}, {algo}): "
                f"期望 {expected_hex}, 实际 {actual_hex}"
            )

    return errors


# ============================================================================
# ============================================================================

_SENTINEL = None
_WORKER_ERROR_TAG = "__TITANINFRA_WORKER_ERROR__"

_HIST_BOUNDARIES: tuple[int, ...] = (0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def _tc_to_bucket(tc: int) -> str:
    for i in range(len(_HIST_BOUNDARIES) - 1):
        if tc < _HIST_BOUNDARIES[i + 1]:
            return f"{_HIST_BOUNDARIES[i]}-{_HIST_BOUNDARIES[i + 1]}"
    return f"{_HIST_BOUNDARIES[-1]}+"


def _empty_hist() -> dict[str, int]:
    hist: dict[str, int] = {}
    for i in range(len(_HIST_BOUNDARIES) - 1):
        hist[f"{_HIST_BOUNDARIES[i]}-{_HIST_BOUNDARIES[i + 1]}"] = 0
    hist[f"{_HIST_BOUNDARIES[-1]}+"] = 0
    return hist


_FINE_BUCKET_WIDTH = 64
_FINE_MAX_TOKEN = 65536


def _tc_to_fine_bucket(tc: int) -> str:
    idx = tc // _FINE_BUCKET_WIDTH
    lo = idx * _FINE_BUCKET_WIDTH
    hi = lo + _FINE_BUCKET_WIDTH
    if hi > _FINE_MAX_TOKEN:
        return f"{_FINE_MAX_TOKEN}+"
    return f"{lo}-{hi}"



_SUPPORTED_EXTS = {".jsonl", ".json", ".parquet"}


def _list_data_files(source: str) -> list[str]:
    """List data files."""
    p = Path(source)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(
            str(f) for f in p.rglob("*")
            if f.suffix in _SUPPORTED_EXTS
        )
        if not files:
            raise ValueError(f"目录中无 jsonl/json/parquet 文件: {source}")
        return files
    raise ValueError(f"路径不存在: {source}")


def _read_records_from_file(file_path: str) -> list[dict]:
    """Read records from file."""
    p = Path(file_path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(str(p)).to_pylist()
    # JSONL / JSON
    records: list[dict] = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _validate_columns(
    files: list[str],
    mode: str,
    text_key: str = "text",
    messages_key: str = "messages",
    row_filter: dict | None = None,
) -> None:
    """Validate the supplied configuration."""
    if not files:
        return

    sample_records = _read_records_from_file(files[0])
    if not sample_records:
        return

    available_cols = set(sample_records[0].keys())

    required_key = text_key if mode == "pretrain" else messages_key
    if required_key not in available_cols:
        key_param = "text_key" if mode == "pretrain" else "messages_key"
        raise ValueError(
            f"数据中不存在列 '{required_key}'（当前 {key_param}='{required_key}'）。\n"
            f"可用列: {sorted(available_cols)}\n"
            f"请检查 text_key/messages_key 配置是否与数据字段匹配。"
        )

    if row_filter:
        missing_filter_cols = set(row_filter.keys()) - available_cols
        if missing_filter_cols:
            raise ValueError(
                f"row_filter 引用的列不存在: {sorted(missing_filter_cols)}。\n"
                f"可用列: {sorted(available_cols)}\n"
                f"列名拼写错误会导致 100% 数据被静默过滤。"
            )



FilterSpec = dict[str, str | list[str]]


def _build_checker(expr: str | list[str]) -> tuple[bool, object]:
    """Build and return the configured objects."""
    import re as _re
    from operator import eq, ge, gt, le, lt, ne

    def _to_float(val: object) -> float:
        """To float."""
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).strip())

    # list → set membership (string)
    if isinstance(expr, list):
        allowed = {str(v) for v in expr}
        return False, lambda val: val in allowed

    expr = expr.strip()

    # "re:pattern" → regex (string)
    if expr.startswith("re:"):
        pattern = _re.compile(expr[3:])
        return False, lambda val: pattern.search(val) is not None

    # "lo ~ hi" → closed range (numeric)
    if "~" in expr:
        lo_s, hi_s = expr.split("~", 1)
        lo, hi = float(lo_s.strip()), float(hi_s.strip())
        return True, lambda val, _f=_to_float: lo <= _f(val) <= hi

    # "> 0.5", ">= 0.7", "< 1.0", "!= 0", "== 1" → numeric comparison
    m = _re.match(r"^(>=|<=|!=|>|<|==)\s*(.+)$", expr)
    if m:
        op_str, num_str = m.group(1), m.group(2).strip()
        threshold = float(num_str)
        ops = {">": gt, ">=": ge, "<": lt, "<=": le, "==": eq, "!=": ne}
        op_fn = ops[op_str]
        return True, lambda val, _op=op_fn, _t=threshold, _f=_to_float: _op(_f(val), _t)

    # fallback: exact string match
    return False, lambda val: val == expr


def _compile_row_filter(spec: FilterSpec | None):
    """Compile row filter."""
    if not spec:
        return None

    checkers: list[tuple[str, bool, object]] = []
    for col, expr in spec.items():
        is_numeric, check_fn = _build_checker(expr)
        checkers.append((col, is_numeric, check_fn))

    def _filter(row: dict) -> bool:
        for col, is_numeric, check in checkers:
            val = row.get(col)
            if val is None:
                return False
            try:
                if is_numeric:
                    if not check(val):
                        return False
                else:
                    if not check(str(val)):
                        return False
            except (ValueError, TypeError):
                return False
        return True

    return _filter



def _generic_reader_worker(
    file_paths: list[str],
    read_q: mp.Queue,
    batch_size: int = 1000,
    row_filter_spec: FilterSpec | None = None,
    message_converter_fn: Callable[[dict], dict] | None = None,
) -> None:
    """Generic reader worker."""
    filter_fn = _compile_row_filter(row_filter_spec) if row_filter_spec else None
    for file_path in file_paths:
        try:
            records = _read_records_from_file(file_path)
        except Exception as e:
            logger.error(f"跳过文件 {file_path}: {e}")
            continue
        batch: list[dict] = []
        for record in records:
            if filter_fn and not filter_fn(record):
                continue
            if message_converter_fn is not None:
                record = message_converter_fn(record)
                if not record:
                    continue
            batch.append(record)
            if len(batch) >= batch_size:
                read_q.put(batch)
                batch = []
        if batch:
            read_q.put(batch)



def _generic_tokenizer_worker(
    tokenizer_path: str,
    read_q: mp.Queue,
    pack_q: mp.Queue,
    mode: str,
    text_key: str = "text",
    messages_key: str = "messages",
    stream_id: str = "",
    add_bos: bool = True,
    add_eos: bool = True,
    store_token_count: bool = False,
    family: str | None = None,
    resolution_policy=None,
    spatial_merge_size: int | None = None,
) -> None:
    """Generic tokenizer worker."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    try:
        while True:
            batch = read_q.get()
            if batch is _SENTINEL:
                pack_q.put(_SENTINEL)
                return

            for record in batch:
                if mode == "messages":
                    payload, tc = _prepare_messages_item(
                        record, tokenizer, messages_key, stream_id, family,
                        resolution_policy, spatial_merge_size,
                    )
                else:
                    payload, tc = _prepare_pretrain_item(
                        record, tokenizer, text_key, stream_id,
                        add_bos, add_eos, store_token_count,
                    )
                if tc > 0:
                    pack_q.put((payload, tc))
    except BaseException as exc:
        # Fail-fast: any worker failure (validator raise, OOM, tokenizer
        # crash) must surface in the orchestrator instead of deadlocking
        # pack_q.get(). Push an ERROR sentinel the main loop re-raises,
        # then propagate so the thread dies loud.
        pack_q.put((_WORKER_ERROR_TAG, exc))
        raise


def _prepare_pretrain_item(
    record: dict,
    tokenizer,
    text_key: str,
    stream_id: str,
    add_bos: bool,
    add_eos: bool,
    store_token_count: bool,
) -> tuple[dict, int]:
    """Prepare pretrain item."""
    text = str(record.get(text_key, ""))
    if not text:
        return {}, 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tc = len(tokens)
    if add_bos:
        tc += 1
    if add_eos:
        tc += 1
    payload: dict[str, Any] = {"text": text, "stream_id": stream_id}
    if store_token_count:
        payload["token_count"] = tc
    return payload, tc


def _prepare_messages_item(
    record: dict,
    tokenizer,
    messages_key: str,
    stream_id: str,
    family: str | None,
    resolution_policy=None,
    spatial_merge_size: int | None = None,
) -> tuple[dict, int]:
    """Prepare messages item."""
    if not family:
        raise ValueError(
            "_prepare_messages_item requires a non-empty `family` "
            "(messages-mode converter must specify the SFT encoder family)."
        )

    from titaninfra.data.sft.capabilities import EmptyAssistantError
    from titaninfra.data.sft.dispatch import tokenize_sample
    from titaninfra.data.sft.message_converter import to_ir_messages

    msgs = record.get(messages_key)
    if msgs is None:
        return {}, 0

    if isinstance(msgs, list):
        msgs_parsed = msgs
    elif isinstance(msgs, str):
        msgs_parsed = json.loads(msgs)
    else:
        return {}, 0

    # ── HF / loose-shape → titaninfra IR ────────────────────────────────────
    # Adapt {role, content} (str or list) and HF-standard tool_calls /
    # tool_call_id into IR ``blocks`` form. Idempotent on rows that are
    # already IR. Single chokepoint between historical / public JSONL
    # fixtures and the strict msgspec validator below; on-disk MDS rows
    # are therefore always normalized IR.
    msgs_parsed = to_ir_messages(msgs_parsed)
    msgs_str = json.dumps(msgs_parsed, ensure_ascii=False)

    sample_for_dispatch: dict[str, Any] = {"messages": msgs_parsed}
    if record.get("system"):
        sample_for_dispatch["system"] = record["system"]
    if record.get("tools"):
        sample_for_dispatch["tools"] = record["tools"]
    if record.get("loss_config"):
        sample_for_dispatch["loss_config"] = record["loss_config"]
    if record.get("thinking_intent"):
        sample_for_dispatch["thinking_intent"] = record["thinking_intent"]

    validate_or_raise(sample_for_dispatch, source=f"stream_id={stream_id}")

    # VLM SFT: assembler no longer decodes inline (2026-07-29 unified decode
    # layer) — offline converter must pre-decode its own images so token_count
    # (expanded image_pad tokens) matches runtime byte-for-byte. Synchronous
    # decode here: the converter already parallelizes across tokenizer worker
    # threads, so no per-sample pool.
    _decoded_images = None
    if resolution_policy is not None:
        from titaninfra.data.multimodal.image_processor import preprocess_image, sft_images_of
        _raw_imgs = sft_images_of(sample_for_dispatch, messages_key="messages")
        if _raw_imgs:
            _merge = spatial_merge_size if spatial_merge_size is not None else 2
            _decoded_images = [
                preprocess_image(
                    _b,
                    min_pixels=resolution_policy.min_pixels,
                    max_pixels=resolution_policy.max_pixels,
                    spatial_merge_size=_merge,
                )
                for _b in _raw_imgs
            ]

    try:
        encoded = tokenize_sample(
            sample_for_dispatch, tokenizer, family,
            resolution_policy=resolution_policy,
            spatial_merge_size=spatial_merge_size,
            decoded_images=_decoded_images,
        )
    except EmptyAssistantError as exc:
        logger.warning(
            "Dropping sample (stream_id=%s family=%s): %s",
            stream_id, family, exc,
        )
        return {}, 0

    tc = max(1, len(encoded.input_ids))

    payload: dict[str, Any] = {
        "messages": msgs_str,
        "stream_id": stream_id,
        "token_count": tc,
    }
    lc = record.get("loss_config")
    payload["loss_config"] = (
        json.dumps(lc, ensure_ascii=False) if lc and not isinstance(lc, str) else (lc or "{}")
    )
    sys_val = record.get("system", "")
    payload["system"] = sys_val if isinstance(sys_val, str) else ""
    tools_val = record.get("tools")
    if tools_val and not isinstance(tools_val, str):
        payload["tools"] = json.dumps(tools_val, ensure_ascii=False)
    else:
        payload["tools"] = tools_val or ""
    return payload, tc




def _preload_and_feed(
    files: list[str],
    read_q: mp.Queue,
    num_tokenizers: int,
    batch_size: int,
    filter_fn: Callable | None,
    message_converter_fn: Callable[[dict], dict] | None,
) -> None:
    """Preload and feed."""
    failed = 0
    for file_path in files:
        try:
            records = _read_records_from_file(file_path)
        except Exception as e:
            logger.error(f"Preload 跳过文件 {file_path}: {e}")
            failed += 1
            continue
        if filter_fn:
            records = [r for r in records if filter_fn(r)]
        if message_converter_fn is not None:
            records = [message_converter_fn(r) for r in records]
            records = [r for r in records if r]
        for i in range(0, len(records), batch_size):
            read_q.put(records[i:i + batch_size])
    if failed:
        logger.warning(f"Preload: {failed} 个文件读取失败，已跳过")
    for _ in range(num_tokenizers):
        read_q.put(_SENTINEL)


def _start_data_feeding(
    files: list[str],
    read_q: mp.Queue,
    num_tokenizers: int,
    num_workers: int,
    batch_size: int,
    preload: bool,
    row_filter: FilterSpec | None,
    message_converter_fn: Callable[[dict], dict] | None,
) -> threading.Thread:
    """Start data feeding."""
    if preload:
        filter_fn = _compile_row_filter(row_filter) if row_filter else None
        return threading.Thread(
            target=_preload_and_feed,
            args=(files, read_q, num_tokenizers, batch_size, filter_fn, message_converter_fn),
            daemon=True,
        )

    num_readers = min(num_workers, len(files))
    reader_file_groups: list[list[str]] = [[] for _ in range(num_readers)]
    for i, f in enumerate(files):
        reader_file_groups[i % num_readers].append(f)

    reader_procs: list[mp.Process] = []
    for file_group in reader_file_groups:
        p = mp.Process(
            target=_generic_reader_worker,
            args=(file_group, read_q, batch_size, row_filter, message_converter_fn),
            daemon=True,
        )
        p.start()
        reader_procs.append(p)

    def _send_sentinels():
        for rp in reader_procs:
            rp.join()
        for _ in range(num_tokenizers):
            read_q.put(_SENTINEL)

    return threading.Thread(target=_send_sentinels, daemon=True)


def _run_heap_packing_loop(
    pack_q: mp.Queue | queue.Queue,
    num_tokenizers: int,
    writers: list,
    initial_shard_tokens: list[int] | None,
    num_shards: int,
    initial_max_tc: int = 0,
    initial_min_tc: int = 0,
    initial_hist: dict[str, int] | None = None,
    initial_fine_hist: dict[str, int] | None = None,
) -> tuple[int, list[int], int, int, dict[str, int], dict[str, int]]:
    """Run heap packing loop."""
    if initial_shard_tokens and len(initial_shard_tokens) == num_shards:
        heap: list[tuple[int, int]] = [(t, i) for i, t in enumerate(initial_shard_tokens)]
        shard_tokens = list(initial_shard_tokens)
    else:
        heap: list[tuple[int, int]] = [(0, i) for i in range(num_shards)]
        shard_tokens = [0] * num_shards
    heapq.heapify(heap)

    max_tc = initial_max_tc
    min_tc = initial_min_tc
    hist = dict(initial_hist) if initial_hist is not None else _empty_hist()
    fine_hist: dict[str, int] = dict(initial_fine_hist) if initial_fine_hist is not None else {}

    total_samples = 0
    done_count = 0

    while True:
        item = pack_q.get()
        if item is _SENTINEL:
            done_count += 1
            if done_count >= num_tokenizers:
                break
            continue

        # Fail-fast: worker thread reported an exception.
        if isinstance(item, tuple) and len(item) == 2 and item[0] == _WORKER_ERROR_TAG:
            raise RuntimeError(
                "tokenizer worker failed; aborting balanced pipeline"
            ) from item[1]

        payload, tc = item
        if tc > max_tc:
            max_tc = tc
        if min_tc == 0 or tc < min_tc:
            min_tc = tc
        hist[_tc_to_bucket(tc)] += 1
        fine_key = _tc_to_fine_bucket(tc)
        fine_hist[fine_key] = fine_hist.get(fine_key, 0) + 1

        tokens, idx = heapq.heappop(heap)
        heapq.heappush(heap, (tokens + tc, idx))
        shard_tokens[idx] += tc
        writers[idx].write(payload)
        total_samples += 1

    return total_samples, shard_tokens, max_tc, min_tc, hist, fine_hist


def _balanced_fast_pipeline(
    files: list[str],
    output: str,
    columns: dict[str, str],
    tokenizer_path: str,
    mode: str,
    num_shards: int,
    num_workers: int = 4,
    num_tokenizer_workers: int | None = None,
    batch_size: int = 1000,
    stream_id: str = "",
    text_key: str = "text",
    messages_key: str = "messages",
    add_bos: bool = True,
    add_eos: bool = True,
    store_token_count: bool = False,
    queue_size: int = 4096,
    row_filter: FilterSpec | None = None,
    initial_shard_tokens: list[int] | None = None,
    output_subdir: str | None = None,
    preload: bool = False,
    compression: str | None = None,
    message_converter_fn: Callable[[dict], dict] | None = None,
    size_limit: int | str | None = "256MB",
    initial_max_tc: int = 0,
    initial_min_tc: int = 0,
    initial_hist: dict[str, int] | None = None,
    initial_fine_hist: dict[str, int] | None = None,
    family: str | None = None,
    resolution_policy=None,
    spatial_merge_size: int | None = None,
) -> dict[str, Any]:
    """Balanced fast pipeline."""
    from streaming.base import MDSWriter
    from streaming.base.util import merge_index

    start_time = time.time()

    _validate_columns(files, mode, text_key, messages_key, row_filter)

    num_tokenizers = num_tokenizer_workers if num_tokenizer_workers is not None else num_workers
    preload_hint = "preload" if preload else f"{min(num_workers, len(files))} readers"
    logger.info(
        f"Balanced fast pipeline: {len(files)} files, {num_shards} shards, "
        f"{preload_hint}, {num_tokenizers} tokenizer threads, mode={mode}"
    )

    read_q: mp.Queue = mp.Queue(maxsize=queue_size)
    pack_q: queue.Queue = queue.Queue(maxsize=queue_size)

    if mode == "messages" and not family:
        raise ValueError(
            "_balanced_fast_pipeline(mode='messages') requires `family`; "
            "messages-mode converter must specify the SFT encoder family."
        )

    tok_threads: list[threading.Thread] = []
    for _ in range(num_tokenizers):
        t = threading.Thread(
            target=_generic_tokenizer_worker,
            args=(
                tokenizer_path, read_q, pack_q, mode,
                text_key, messages_key, stream_id,
                add_bos, add_eos, store_token_count, family,
                resolution_policy, spatial_merge_size,
            ),
            daemon=True,
        )
        t.start()
        tok_threads.append(t)

    sentinel_thread = _start_data_feeding(
        files, read_q, num_tokenizers, num_workers, batch_size,
        preload, row_filter, message_converter_fn,
    )
    sentinel_thread.start()

    actual_output = os.path.join(output, output_subdir) if output_subdir else output
    os.makedirs(actual_output, exist_ok=True)

    shard_dirs = [
        os.path.join(actual_output, f"shard_{shard_id:05d}") if num_shards > 1 else actual_output
        for shard_id in range(num_shards)
    ]
    writers = [
        MDSWriter(out=sd, columns=columns, compression=compression, size_limit=size_limit)
        for sd in shard_dirs
    ]

    total_samples, shard_tokens, max_tc, min_tc, hist, fine_hist = _run_heap_packing_loop(
        pack_q, num_tokenizers, writers, initial_shard_tokens, num_shards,
        initial_max_tc=initial_max_tc,
        initial_min_tc=initial_min_tc,
        initial_hist=initial_hist,
        initial_fine_hist=initial_fine_hist,
    )

    for w in writers:
        w.finish()

    sentinel_thread.join()
    for t in tok_threads:
        t.join()

    if num_shards > 1 and not output_subdir:
        logger.info(f"合并 {num_shards} 个 shard 的 index 到 {output}")
        merge_index(output, keep_local=True)

    initial = initial_shard_tokens or [0] * num_shards
    batch_shard_tokens = [shard_tokens[i] - initial[i] for i in range(num_shards)]
    batch_total_tokens = sum(batch_shard_tokens)
    elapsed = time.time() - start_time

    if total_samples > 0:
        avg_tokens = sum(shard_tokens) / num_shards
        max_dev = max(abs(st - avg_tokens) / avg_tokens * 100 for st in shard_tokens)
    else:
        avg_tokens = 0
        max_dev = 0.0

    logger.info(
        f"Balanced fast pipeline 完成: {total_samples} samples, "
        f"{num_shards} shards, avg {avg_tokens:.0f} tokens/shard, "
        f"deviation {max_dev:.1f}%, time {elapsed:.2f}s"
    )

    return {
        "files": len(files),
        "samples": total_samples,
        "shards": num_shards,
        "total_tokens": batch_total_tokens,
        "tokens_per_shard": batch_shard_tokens,
        "cumulative_shard_tokens": shard_tokens,
        "max_deviation_pct": round(max_dev, 2),
        "time": elapsed,
        "max_token_length": max_tc,
        "min_token_length": min_tc,
        "token_length_hist": hist,
        "fine_token_length_hist": fine_hist,
    }



def _greedy_bin_pack(
    items: list[tuple[Any, int]],
    n_bins: int,
) -> tuple[list[list[tuple[Any, int]]], list[int]]:
    """Greedy bin pack."""
    items.sort(key=lambda x: -x[1])

    bins: list[list[tuple[Any, int]]] = [[] for _ in range(n_bins)]
    heap: list[tuple[int, int]] = [(0, i) for i in range(n_bins)]
    heapq.heapify(heap)

    for payload, token_count in items:
        tokens, idx = heapq.heappop(heap)
        bins[idx].append((payload, token_count))
        heapq.heappush(heap, (tokens + token_count, idx))

    bin_tokens = [sum(tc for _, tc in b) for b in bins]
    return bins, bin_tokens




# ── Checkpoint helpers ─────────────────────────────────────

_CHECKPOINT_VERSION = 1


def _save_checkpoint(path: str, data: dict) -> None:
    """Persist the supplied data."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _load_checkpoint(path: str) -> dict | None:
    """Load and return the requested data."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("version") != _CHECKPOINT_VERSION:
            logger.warning(f"Checkpoint 版本不兼容: {data.get('version')}")
            return None
        return data
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Checkpoint 损坏，忽略: {e}")
        return None


def _clean_batch_dirs(output: str) -> None:
    """Clean batch dirs."""
    for d in Path(output).glob("_batch_*"):
        if d.is_dir():
            shutil.rmtree(d)


def _safe_merge_index(output: str) -> None:
    """Safe merge index."""
    shards: list[dict] = []
    for dirpath, _, files in os.walk(output):
        if "index.json" not in files:
            continue
        rel_dir = os.path.relpath(dirpath, output)
        if rel_dir == ".":
            continue
        idx_path = os.path.join(dirpath, "index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        for shard in idx.get("shards", []):
            for key in ("raw_data", "zip_data", "raw_meta", "zip_meta"):
                if shard.get(key):
                    shard[key]["basename"] = os.path.join(rel_dir, shard[key]["basename"])
            shards.append(shard)

    merged = {"version": 2, "shards": shards}
    root_index = os.path.join(output, "index.json")
    with open(root_index, "w") as f:
        json.dump(merged, f)
    logger.info(f"Merged {len(shards)} shards → {root_index}")


def _load_or_init_checkpoint_state(
    ckpt_path: str,
    output: str,
    num_shards: int,
    resume: bool,
    files: list[str],
    format_result_fn: Callable,
) -> tuple[list[str], list[int] | None, int, int, int, list[str], int, int, dict[str, int] | None] | dict:
    """Load and return the requested data."""
    ckpt = _load_checkpoint(ckpt_path) if resume else None
    if ckpt:
        if ckpt["num_shards"] != num_shards:
            raise ValueError(
                f"checkpoint num_shards={ckpt['num_shards']} 与当前 "
                f"num_shards={num_shards} 不一致。"
                "请使用相同的 num_shards 或删除 checkpoint 重新开始。"
            )
        completed_set = set(ckpt["completed_files"])
        remaining = [f for f in files if f not in completed_set]
        for d in Path(output).glob("_batch_*"):
            try:
                idx = int(d.name.split("_")[-1])
            except ValueError:
                continue
            if idx >= ckpt["batch_count"] and d.is_dir():
                logger.info(f"清理不完整的 batch 目录: {d}")
                shutil.rmtree(d)
        logger.info(
            f"Resume: 跳过 {len(completed_set)} 个已完成文件，"
            f"剩余 {len(remaining)} 个文件，从 batch {ckpt['batch_count']} 继续"
        )
        return (remaining, ckpt["shard_tokens"], ckpt["total_samples"],
                ckpt["total_tokens"], ckpt["batch_count"], list(ckpt["completed_files"]),
                ckpt.get("max_token_length", 0),
                ckpt.get("min_token_length", 0),
                ckpt.get("token_length_hist", None),
                ckpt.get("fine_token_length_hist", None))

    existing_batches = list(Path(output).glob("_batch_*"))
    if existing_batches and resume:
        stats_path = os.path.join(output, "stream_stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path) as f:
                    prev_stats = json.load(f)
                logger.info(
                    f"上次转换已完成（{prev_stats.get('num_samples', '?')} samples），"
                    f"跳过重复运行。如需重跑请使用 overwrite=True"
                )
                return format_result_fn(
                    len(files), prev_stats.get("num_samples", 0),
                    prev_stats.get("total_tokens", 0), 0, 0,
                )
            except (json.JSONDecodeError, KeyError):
                pass
        raise FileExistsError(
            f"输出目录 {output} 已包含 {len(existing_batches)} 个 batch 目录，"
            f"但 _checkpoint.json 不存在（转换可能未完成）。\n"
            f"如确认要重新转换，请使用 overwrite=True 清除旧数据。\n"
            f"如只需补跑部分文件，请用新的输出目录。"
        )

    _clean_batch_dirs(output)
    return (files, None, 0, 0, 0, [], 0, 0, None, None)


def _run_checkpoint_pipeline(
    files: list[str],
    output: str,
    num_shards: int,
    checkpoint_interval: int,
    resume: bool,
    overwrite: bool,
    pipeline_kwargs: dict,
    write_stats_fn: Callable,
    format_result_fn: Callable[[int, int, int, int, float], dict],
) -> dict:
    """Run checkpoint pipeline."""
    ckpt_path = os.path.join(output, "_checkpoint.json")

    if overwrite and os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)

    state = _load_or_init_checkpoint_state(
        ckpt_path, output, num_shards, resume, files, format_result_fn,
    )
    if isinstance(state, dict):
        return state
    (files, shard_tokens, total_samples, total_tokens, batch_count, completed_files,
     cum_max_tc, cum_min_tc, cum_hist, cum_fine_hist) = state

    if not files:
        logger.info("所有文件已转换完成，无需继续")
        if num_shards > 1:
            _safe_merge_index(output)
        return format_result_fn(len(completed_files), total_samples, total_tokens, batch_count, 0)

    batches = [files[i:i + checkpoint_interval] for i in range(0, len(files), checkpoint_interval)]
    start_time = time.time()

    for batch_files in batches:
        subdir = f"_batch_{batch_count:03d}"
        logger.info(f"Batch {batch_count}: 处理 {len(batch_files)} 个文件 → {subdir}/")

        result = _balanced_fast_pipeline(
            files=batch_files,
            output=output,
            output_subdir=subdir,
            initial_shard_tokens=shard_tokens,
            initial_max_tc=cum_max_tc,
            initial_min_tc=cum_min_tc,
            initial_hist=cum_hist,
            initial_fine_hist=cum_fine_hist,
            **pipeline_kwargs,
        )

        shard_tokens = result["cumulative_shard_tokens"]
        total_samples += result["samples"]
        total_tokens += result["total_tokens"]
        cum_max_tc = result["max_token_length"]
        cum_min_tc = result["min_token_length"]
        cum_hist = result["token_length_hist"]
        cum_fine_hist = result["fine_token_length_hist"]
        completed_files.extend(batch_files)
        batch_count += 1

        _save_checkpoint(ckpt_path, {
            "version": _CHECKPOINT_VERSION,
            "num_shards": num_shards,
            "checkpoint_interval": checkpoint_interval,
            "completed_files": completed_files,
            "shard_tokens": shard_tokens,
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "batch_count": batch_count,
            "max_token_length": cum_max_tc,
            "min_token_length": cum_min_tc,
            "token_length_hist": cum_hist,
            "fine_token_length_hist": cum_fine_hist,
        })
        logger.info(
            f"Batch {batch_count - 1} 完成: {result['samples']} samples, "
            f"checkpoint 已保存 ({len(completed_files)} 文件已完成)"
        )

    if num_shards > 1:
        _safe_merge_index(output)

    if total_samples > 0:
        write_stats_fn(output, total_samples, total_tokens, cum_max_tc, cum_min_tc, cum_hist, cum_fine_hist)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("转换完成，checkpoint 已删除")

    elapsed = time.time() - start_time
    return format_result_fn(len(completed_files), total_samples, total_tokens, batch_count, elapsed)


# ============================================================================
# ============================================================================


class DataConverter:
    """Convert records to balanced MDS shards with resumable statistics.

The converter tokenizes records, balances shards by token count, validates
samples, and writes index.json and stream_stats.json without altering the
runtime native encoding contract."""

    Config = DataConvertConfig

    def __init__(self, config: DataConvertConfig):
        self.config = config

        _try_import_dependencies()

    def convert_single_file(
        self,
        input_path: str,
        output: OutputPath,
        file_index: int,
    ) -> int:
        """Convert single file."""
        reader = _DataReader(self.config)
        data, file_format = reader.read_file(input_path)

        if file_format == "parquet":
            num_samples = data.num_rows
        else:
            num_samples = len(data)

        logger.info(f"处理文件 [{file_index}] {input_path}，共 {num_samples} 条样本")

        return self._convert_raw(data, output, file_format)

    def _convert_raw(
        self,
        data: Any,
        output: OutputPath,
        file_format: str,
    ) -> int:
        """Convert raw."""
        reader = _DataReader(self.config)

        if self.config.mds_columns:
            columns = self.config.mds_columns
        else:
            first_sample = next(reader.iter_samples(data, file_format))
            columns = _infer_columns_from_sample(first_sample)
            logger.info(f"自动推断列类型: {columns}")

        samples_written = 0

        with _create_writer(
            self.config.output_format,
            output,
            columns,
            self.config,
        ) as writer:
            for sample in reader.iter_samples(data, file_format):
                writer.write(sample)
                samples_written += 1

        logger.info(f"写入 {samples_written} 条样本")
        return samples_written

    def _prepare_output_dir(self, output: OutputPath) -> None:
        """Prepare output dir."""
        import shutil

        if not isinstance(output, str) or output.startswith(("s3://", "gs://", "azure://")):
            return

        output_path = Path(output)
        if not output_path.exists():
            return

        if self.config.overwrite:
            logger.info(f"overwrite=True，删除输出目录: {output}")
            shutil.rmtree(output_path)
        elif self.config.resume:
            logger.info("resume=True，检查已完成的 shard")
        else:
            raise FileExistsError(
                f"输出目录已存在: {output}。"
                f"使用 overwrite=True 覆盖，或 resume=True 续传。"
            )

    def _build_pending_tasks(
        self, files: list[str], output: OutputPath
    ) -> tuple[list[tuple[int, str, OutputPath]], int]:
        """Build and return the configured objects."""
        pending = []
        completed = 0
        for idx, file_path in enumerate(files):
            shard_output = self._get_subprocess_output(output, idx)
            if self.config.resume and _is_shard_completed(shard_output):
                logger.info(f"[Resume] 跳过已完成的 shard {idx}: {file_path}")
                completed += 1
            else:
                pending.append((idx, file_path, shard_output))
        return pending, completed

    def _run_parallel(
        self, files: list[str], output: OutputPath,
    ) -> tuple[int, int]:
        """Run parallel."""
        pending_tasks, completed_count = self._build_pending_tasks(files, output)
        total_samples = 0

        if pending_tasks:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {
                    executor.submit(
                        self._convert_file_in_subprocess,
                        file_path, shard_output, idx,
                    ): idx
                    for idx, file_path, shard_output in pending_tasks
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        samples = future.result()
                        total_samples += samples
                        logger.info(f"文件 {idx} 完成，共 {samples} 条样本")
                    except Exception as e:
                        logger.error(f"文件 {idx} 转换失败: {e}")
                        raise

        if isinstance(output, str) and not output.startswith(("s3://", "gs://", "azure://")):
            self._merge_shards(output, len(files))

        return total_samples, completed_count

    def _run_sequential(
        self, files: list[str], output: OutputPath,
    ) -> tuple[int, int]:
        """Run sequential."""
        total_samples = 0
        completed_count = 0

        for idx, file_path in enumerate(files):
            shard_output = self._get_subprocess_output(output, idx) if len(files) > 1 else output

            if self.config.resume and _is_shard_completed(shard_output):
                logger.info(f"[Resume] 跳过已完成的 shard {idx}: {file_path}")
                completed_count += 1
                continue

            samples = self.convert_single_file(file_path, shard_output, idx)
            total_samples += samples

        return total_samples, completed_count

    def convert(self) -> dict[str, Any]:
        """Convert."""
        return self._convert_balanced()

        start_time = time.time()

        reader = _DataReader(self.config)

        files = reader.list_files()
        if not files:
            logger.warning(f"未找到支持的文件: {self.config.source}")
            return {"files": 0, "samples": 0, "time": 0}

        output = self.config.output
        self._prepare_output_dir(output)

        logger.info(f"开始转换 {len(files)} 个文件，使用 {self.config.num_workers} 个进程")

        if self.config.num_workers > 1 and len(files) > 1:
            total_samples, completed_count = self._run_parallel(files, output)
        else:
            total_samples, completed_count = self._run_sequential(files, output)

        elapsed = time.time() - start_time

        result = {
            "files": len(files),
            "samples": total_samples,
            "time": elapsed,
            "output": str(output),
            "completed_shards": completed_count,
            "new_shards": len(files) - completed_count,
        }

        if completed_count > 0:
            logger.info(
                f"转换完成：跳过 {completed_count} 个已完成 shard，"
                f"新处理 {len(files) - completed_count} 个文件，"
                f"{total_samples} 条样本，耗时 {elapsed:.2f}s"
            )
        else:
            logger.info(
                f"转换完成：{len(files)} 个文件，"
                f"{total_samples} 条样本，耗时 {elapsed:.2f}s"
            )

        if self.config.validate_after_convert:
            local_output = self._resolve_local_output(output)
            if local_output is not None:
                validation = self.validate_shards(local_output)
                result["validation"] = validation

        return result

    def _convert_file_in_subprocess(
        self,
        input_path: str,
        output: OutputPath,
        file_index: int,
    ) -> int:
        """Convert file in subprocess."""
        converter = DataConverter(self.config)
        return converter.convert_single_file(input_path, output, file_index)

    def _get_subprocess_output(self, output: OutputPath, idx: int) -> OutputPath:
        """Return the requested value."""
        if isinstance(output, tuple):
            local, remote = output
            return (f"{local}/shard_{idx:05d}", f"{remote}/shard_{idx:05d}")
        else:
            return f"{output}/shard_{idx:05d}"

    @staticmethod
    def _resolve_local_output(output: OutputPath) -> str | None:
        """Resolve local output."""
        if isinstance(output, tuple):
            return output[0]
        if isinstance(output, str) and not output.startswith(("s3://", "gs://", "azure://", "oci://")):
            return output
        return None

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def validate_shards(self, output_dir: str) -> dict[str, Any]:
        """Validate the supplied configuration."""
        return validate_mds_shards(
            output_dir,
            sample_ratio=self.config.validate_sample_ratio,
            min_samples=self.config.validate_min_samples,
        )

    def _convert_balanced(self) -> dict[str, Any]:
        """Convert balanced."""
        return self._convert_balanced_fast()

    def _convert_balanced_fast(self) -> dict[str, Any]:
        """Convert balanced fast."""
        config = self.config

        if not config.tokenizer_path:
            raise ValueError("tokenizer_path 是必选参数，用于精确 token 计数和均衡分片")

        reader = _DataReader(config)
        files = sorted(reader.list_files())
        if not files:
            return {"files": 0, "samples": 0, "time": 0}

        output = str(config.output)

        columns = {"text": "str", "stream_id": "str"}
        if config.store_token_count:
            columns["token_count"] = "int"

        stream_id = _resolve_stream_id(config)

        pipeline_kwargs = dict(
            columns=columns,
            tokenizer_path=config.tokenizer_path,
            mode="pretrain",
            num_shards=config.num_shards,
            num_workers=config.num_workers,
            num_tokenizer_workers=config.num_tokenizer_workers,
            batch_size=config.batch_size,
            stream_id=stream_id,
            text_key=config.text_key,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
            store_token_count=config.store_token_count,
            row_filter=config.row_filter,
            preload=config.preload,
            compression=config.compression,
            size_limit=config.size_limit,
        )

        if config.checkpoint_interval is None:
            if os.path.exists(output) and config.overwrite:
                shutil.rmtree(output)
            os.makedirs(output, exist_ok=True)

            result = _balanced_fast_pipeline(files=files, output=output, **pipeline_kwargs)
            result["output"] = output

            if result["samples"] > 0:
                _write_stream_stats(
                    output, result["samples"], result["total_tokens"],
                    stream_id=stream_id, stream_type=config.stream_type,
                    tokenizer_name=config.tokenizer_name,
                    tokenizer_path=config.tokenizer_path or "",
                    max_token_length=result["max_token_length"],
                    min_token_length=result["min_token_length"],
                    token_length_dist=result["token_length_hist"],
                    fine_token_length_dist=result["fine_token_length_hist"],
                )
            return result

        def _write_stats(out, samples, tokens, max_tc, min_tc, hist, fine_hist):
            _write_stream_stats(
                out, samples, tokens,
                stream_id=stream_id, stream_type=config.stream_type,
                tokenizer_name=config.tokenizer_name,
                tokenizer_path=config.tokenizer_path or "",
                max_token_length=max_tc,
                min_token_length=min_tc,
                token_length_dist=hist,
                fine_token_length_dist=fine_hist,
            )

        def _format_result(num_files, samples, tokens, batches, elapsed):
            return {
                "files": num_files, "samples": samples, "total_tokens": tokens,
                "shards": config.num_shards, "batches": batches, "output": output, "time": elapsed,
            }

        return _run_checkpoint_pipeline(
            files=files, output=output, num_shards=config.num_shards,
            checkpoint_interval=config.checkpoint_interval, resume=config.resume,
            overwrite=config.overwrite, pipeline_kwargs=pipeline_kwargs,
            write_stats_fn=_write_stats, format_result_fn=_format_result,
        )

    def _merge_shards(self, output: str, num_shards: int) -> None:
        """Merge shards."""
        from streaming.base.util import merge_index

        logger.info(f"合并 {num_shards} 个 shard 的 index 到 {output}")
        merge_index(output, keep_local=True)


# ============================================================================
# ============================================================================


def convert_to_mds(
    source: str,
    output: str,
    *,
    columns: list[str] | None = None,
    field_mapping: dict[str, str] | None = None,
    # Tokenization
    tokenizer_path: str | None = None,
    text_key: str = "text",
    # Streaming Writer
    output_format: WriterFormat = "mds",
    compression: str | None = None,
    size_limit: int | str = "256MB",
    hashes: list[str] | None = None,
    keep_local: bool = False,
    progress_bar: bool = False,
    num_shards: int = 16,
    num_workers: int = 4,
    overwrite: bool = False,
    resume: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Convert to mds."""
    config = DataConvertConfig(
        source=source,
        output=output,
        columns=columns,
        field_mapping=field_mapping,
        tokenizer_path=tokenizer_path,
        text_key=text_key,
        output_format=output_format,
        compression=compression,
        size_limit=size_limit,
        hashes=hashes,
        keep_local=keep_local,
        progress_bar=progress_bar,
        num_shards=num_shards,
        num_workers=num_workers,
        overwrite=overwrite,
        resume=resume,
        **kwargs,
    )

    converter = DataConverter(config)
    return converter.convert()


def _extract_messages_content(messages: list[dict]) -> str:
    """Extract messages content."""
    parts: list[str] = []
    for msg in messages:
        if "blocks" in msg:
            for block in msg["blocks"]:
                text = block.get("text", "")
                if text:
                    parts.append(text)
        elif "content" in msg:
            content = msg["content"]
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
    return "\n".join(parts)


def convert_messages_to_mds(
    source: str,
    output: str,
    tokenizer_path: str,
    family: str,
    messages_key: str = "messages",
    stream_id: str | None = None,
    num_shards: int = 16,
    num_workers: int = 4,
    num_tokenizer_workers: int | None = None,
    batch_size: int = 1000,
    queue_size: int = 4096,
    overwrite: bool = False,
    row_filter: dict[str, str | list[str]] | None = None,
    checkpoint_interval: int | None = None,
    resume: bool = False,
    preload: bool = False,
    compression: str | None = None,
    message_converter_fn: Callable[[dict], dict] | None = None,
    tokenizer_name: str | None = None,
    size_limit: int | str | None = "256MB",
    resolution_policy=None,
    spatial_merge_size: int | None = None,
) -> dict:
    """Convert messages to mds."""
    if not family:
        raise ValueError(
            "convert_messages_to_mds requires `family` (e.g. 'qwen3', 'llama3', 'gpt_oss'). "
            "Token counts must be computed via the family's native encoder so that "
            "stream_stats.json aligns with runtime dispatch.tokenize_sample. See "
            "feedback_jinja_is_reference_not_truth + feedback_token_stats_consistency."
        )
    _try_import_dependencies()

    files = sorted(_list_data_files(source))

    resolved_sid = stream_id or Path(source).stem

    columns: dict[str, str] = {
        "messages": "str",
        "stream_id": "str",
        "token_count": "int",
        "loss_config": "str",
        "system": "str",
        "tools": "str",
    }

    pipeline_kwargs = dict(
        columns=columns,
        tokenizer_path=tokenizer_path,
        mode="messages",
        num_shards=num_shards,
        num_workers=num_workers,
        num_tokenizer_workers=num_tokenizer_workers,
        batch_size=batch_size,
        stream_id=resolved_sid,
        messages_key=messages_key,
        queue_size=queue_size,
        row_filter=row_filter,
        preload=preload,
        compression=compression,
        message_converter_fn=message_converter_fn,
        size_limit=size_limit,
        family=family,
        resolution_policy=resolution_policy,
        spatial_merge_size=spatial_merge_size,
    )

    if checkpoint_interval is None:
        out_path = Path(output)
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"输出目录已存在: {output}，使用 overwrite=True 覆盖")
        if out_path.exists() and overwrite:
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        result = _balanced_fast_pipeline(files=files, output=output, **pipeline_kwargs)

        total_tokens = result["total_tokens"]
        total_samples = result["samples"]

        _write_stream_stats(
            output, total_samples, total_tokens,
            stream_id=resolved_sid, stream_type="messages",
            tokenizer_name=tokenizer_name,
            tokenizer_path=tokenizer_path,
            max_token_length=result["max_token_length"],
            min_token_length=result["min_token_length"],
            token_length_dist=result["token_length_hist"],
            fine_token_length_dist=result["fine_token_length_hist"],
        )

        avg_tokens = total_tokens / max(1, total_samples)
        logger.info(
            f"Messages→MDS 转换完成: {total_samples} samples, {result['shards']} shards, "
            f"avg_tokens={avg_tokens:.0f}, deviation {result.get('max_deviation_pct', 0):.1f}%, "
            f"output={output}"
        )
        return {
            "num_samples": total_samples,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": round(avg_tokens, 1),
            "max_deviation_pct": result.get("max_deviation_pct", 0),
        }

    def _write_stats(out, samples, tokens, max_tc, min_tc, hist, fine_hist):
        _write_stream_stats(
            out, samples, tokens,
            stream_id=resolved_sid, stream_type="messages",
            tokenizer_name=tokenizer_name,
            tokenizer_path=tokenizer_path,
            max_token_length=max_tc,
            min_token_length=min_tc,
            token_length_dist=hist,
            fine_token_length_dist=fine_hist,
        )

    def _format_result(num_files, samples, tokens, _batches, elapsed):  # noqa: ARG001
        avg = tokens / max(1, samples)
        return {
            "num_samples": samples, "total_tokens": tokens,
            "avg_tokens_per_sample": round(avg, 1), "max_deviation_pct": 0,
        }

    return _run_checkpoint_pipeline(
        files=files, output=output, num_shards=num_shards,
        checkpoint_interval=checkpoint_interval, resume=resume,
        overwrite=overwrite, pipeline_kwargs=pipeline_kwargs,
        write_stats_fn=_write_stats, format_result_fn=_format_result,
    )
