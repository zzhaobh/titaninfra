"""MDS data loading, tokenization, packing, mixing, and DP sharding.

Each StreamConfig builds an independent StreamingDataset and PackedStream,
then StreamMixer combines them using proportion, repeat, or choose scheduling.
Token statistics drive epoch sizing and total-step estimation. All ranks use
the same statistics to avoid divergent iteration lengths."""

import json
import logging
import math
import os
import random
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.tokenizer import BaseTokenizer

from titaninfra.config.data import StreamConfig
from titaninfra.data.packer import MMDoc, PackedSample, PackMode, StreamingPacker, strategy_to_pack_mode

logger = logging.getLogger(__name__)


# ============================================================================
# ============================================================================


class PackedStream:
    """Packed stream configuration and behavior."""

    def __init__(
        self,
        source: Iterator[list[tuple[int, int]]],
        packer: StreamingPacker,
        name: str | None = None,
        *,
        source_factory: Any | None = None,
    ):
        self._source = source
        self._source_factory = source_factory
        self._packer = packer
        self._name = name or "unnamed"
        self._iterator: Iterator[PackedSample] | None = None

    def __iter__(self) -> Iterator[PackedSample]:
        source = self._source_factory() if self._source_factory else self._source
        self._packer.reset()
        for doc in source:
            if isinstance(doc, MMDoc):
                self._packer.add_document(doc.tokens, images=doc.images)
            else:
                self._packer.add_document(doc)
            yield from self._packer.flush_ready()
        yield from self._packer.flush_all()

    def __next__(self) -> PackedSample:
        if self._iterator is None:
            self._iterator = iter(self)
        return next(self._iterator)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if hasattr(self._source, "state_dict"):
            state["source"] = self._source.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "source" in state and hasattr(self._source, "load_state_dict"):
            self._source.load_state_dict(state["source"])
        self._iterator = None


# ============================================================================
# ============================================================================


class StreamMixer:
    """Stream mixer configuration and behavior."""

    def __init__(
        self,
        streams: list[PackedStream],
        weights: list[float],
        rng_seed: int = 9176,
    ):
        if len(streams) != len(weights):
            raise ValueError(
                f"streams 和 weights 长度不匹配: {len(streams)} vs {len(weights)}"
            )
        self._streams = streams
        self._weights = weights
        self._rng = random.Random(rng_seed)
        self._stream_iters: list[Iterator[PackedSample]] = []
        self._iterator: Iterator[PackedSample] | None = None

    def __iter__(self) -> Iterator[PackedSample]:
        self._stream_iters = [iter(stream) for stream in self._streams]
        weights = list(self._weights)

        while True:
            stream_idx = self._rng.choices(
                range(len(self._streams)), weights=weights, k=1,
            )[0]
            try:
                sample = next(self._stream_iters[stream_idx])
                yield sample
            except StopIteration:
                weights[stream_idx] = 0.0
                if sum(weights) == 0.0:
                    return

    def __next__(self) -> PackedSample:
        if self._iterator is None:
            self._iterator = iter(self)
        return next(self._iterator)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self._rng.getstate(),
            "stream_states": [s.state_dict() for s in self._streams],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])
        if "stream_states" in state:
            for stream, saved_state in zip(self._streams, state["stream_states"], strict=False):
                stream.load_state_dict(saved_state)
        self._iterator = None


# ============================================================================
# ============================================================================


class StreamingTextDataset(IterableDataset, Stateful):
    """Streaming text dataset configuration and behavior."""

    def __init__(
        self,
        streaming_dataset,
        tokenizer: BaseTokenizer,
        seq_len: int,
        text_key: str = "text",
        default_stream_id: str = "unknown",
        skip_corrupted_samples: bool = True,
        max_consecutive_errors: int = 10,
        sentence_end_token_ids: list[int] | None = None,
        rng_seed: int = 9176,
        pretrain_stream_id: str = "pretrain",
        pre_built_streams: list[PackedStream] | None = None,
        pre_built_weights: list[float] | None = None,
        pre_built_stream_ids: list[str] | None = None,
        infinite: bool = True,
        add_bos_eos: bool = False,
    ) -> None:
        self._streaming_ds = streaming_dataset
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self._text_key = text_key
        self._default_stream_id = default_stream_id

        self._pad_id = self._resolve_pad_id(tokenizer)

        packing_mode = PackMode.PRETRAIN if sentence_end_token_ids else PackMode.SIMPLE
        self._packer = StreamingPacker(
            seq_len=seq_len,
            mode=packing_mode,
            sentence_end_tokens=sentence_end_token_ids or [],
        )

        self._streaming_samples_consumed: int = 0

        self._per_stream_tokens: dict[str, int] = {}
        self._per_stream_samples: dict[str, int] = {}

        self._skip_corrupted = skip_corrupted_samples
        self._max_consecutive_errors = max_consecutive_errors
        self._consecutive_errors = 0
        self._error_stats: dict[str, int] = {}
        self._corrupted_samples: list[dict] = []
        self._infinite = infinite
        self._add_bos_eos = add_bos_eos

        self._mixer: StreamMixer | None = None
        self._packed_streams: list[PackedStream] = []
        self._stream_ids: list[str] = []

        if pre_built_streams:
            self._packed_streams = list(pre_built_streams)
            self._stream_ids = list(pre_built_stream_ids or [s._name for s in pre_built_streams])
            weights = list(pre_built_weights or [1.0 / len(pre_built_streams)] * len(pre_built_streams))
            self._mixer = StreamMixer(
                streams=self._packed_streams,
                weights=weights,
                rng_seed=rng_seed,
            )

    @staticmethod
    def _resolve_pad_id(tokenizer) -> int:
        """Resolve pad id."""
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None:
            if hasattr(inner, "pad_token_id") and inner.pad_token_id is not None:
                return inner.pad_token_id

        tokenizer_path = getattr(tokenizer, "tokenizer_path", None)
        if tokenizer_path:
            import json as _json
            from pathlib import Path
            cfg_path = Path(tokenizer_path) / "tokenizer_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = _json.load(f)
                pad_token = cfg.get("pad_token")
                # {"__type": "AddedToken", "content": "<pad>", "lstrip": false, ...}。
                if isinstance(pad_token, dict):
                    pad_token = pad_token.get("content")
                if isinstance(pad_token, str) and pad_token and inner is not None and hasattr(inner, "token_to_id"):
                    pid = inner.token_to_id(pad_token)
                    if pid is not None:
                        return pid

        # fallback: eos_id
        eos = getattr(tokenizer, "eos_id", None)
        if eos is not None:
            return eos
        return 0

    _SHARD_ERROR_SENTINEL = object()
    _EXHAUSTED_SENTINEL = object()

    def _fetch_next_sample(self, iterator):
        """Fetch next sample."""
        try:
            return next(iterator)
        except StopIteration:
            return self._EXHAUSTED_SENTINEL
        except ValueError as e:
            if self._skip_corrupted:
                self._handle_shard_error(e, self._streaming_samples_consumed)
                return self._SHARD_ERROR_SENTINEL
            raise
        except Exception as e:
            if self._skip_corrupted and self._is_shard_error(e):
                self._handle_shard_error(e, self._streaming_samples_consumed)
                return self._SHARD_ERROR_SENTINEL
            raise

    def _tokenize_sample(self, sample) -> list[int] | None:
        """Tokenize sample."""
        if not self._validate_sample(sample):
            return None

        text = sample.get(self._text_key, "") if isinstance(sample, dict) else str(sample)

        try:
            tokens = self._tokenizer.encode(text, add_bos=self._add_bos_eos, add_eos=self._add_bos_eos)
        except Exception as enc_err:
            if self._skip_corrupted:
                self._handle_sample_error(enc_err, sample, "tokenize_error")
                return None
            raise

        if not tokens:
            if self._skip_corrupted:
                self._handle_sample_error(
                    ValueError(f"tokenize 结果为空（文本长度={len(text)}）"),
                    sample, "empty_tokens",
                )
                return None
            raise ValueError("tokenize 结果为空")

        return tokens

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Iter."""
        while True:
            if self._mixer is not None:
                yield from self._iter_mixed()
            else:
                yield from self._iter_pretrain()
            if not self._infinite:
                break

    def _pad_to_seq_len(
        self,
        input_ids: list[int],
        labels: list[int],
        positions: list[int] | None,
        loss_weights: list[float] | None = None,
    ) -> tuple[list[int], list[int], list[int] | None, list[float] | None]:
        """Pad to seq len."""
        target_len = self.seq_len
        actual_len = len(input_ids)
        if actual_len >= target_len:
            return input_ids, labels, positions, loss_weights

        pad_len = target_len - actual_len
        input_ids = input_ids + [self._pad_id] * pad_len
        labels = labels + [-100] * pad_len
        if positions is not None:
            positions = positions + [-1] * pad_len
        if loss_weights is not None:
            loss_weights = loss_weights + [0.0] * pad_len
        return input_ids, labels, positions, loss_weights

    def _iter_pretrain(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Iter pretrain."""
        iterator = iter(self._streaming_ds)

        while True:
            sample = self._fetch_next_sample(iterator)
            if sample is self._EXHAUSTED_SENTINEL:
                logger.warning(
                    "epoch_size 不足，数据耗尽（已消费 %d 样本）。训练停止。"
                    "如需更多数据：增大数据集，或使用 repeat 配置。",
                    self._streaming_samples_consumed,
                )
                return
            if sample is self._SHARD_ERROR_SENTINEL:
                continue

            tokens = self._tokenize_sample(sample)
            if tokens is None:
                continue

            stream_id = (
                sample.get("stream_id", self._default_stream_id)
                if isinstance(sample, dict)
                else self._default_stream_id
            )
            self._per_stream_tokens[stream_id] = (
                self._per_stream_tokens.get(stream_id, 0) + len(tokens)
            )
            self._per_stream_samples[stream_id] = (
                self._per_stream_samples.get(stream_id, 0) + 1
            )

            self._streaming_samples_consumed += 1
            self._packer.add_document([(t, t) for t in tokens])
            self._consecutive_errors = 0

            for packed_sample in self._packer.flush_ready():
                input_ids = packed_sample.input_ids[:-1]
                labels = packed_sample.labels[1:]
                pos = packed_sample.positions[:-1] if packed_sample.positions is not None else None
                weights = packed_sample.loss_weights[1:] if packed_sample.loss_weights is not None else None

                input_ids, labels, pos, weights = self._pad_to_seq_len(input_ids, labels, pos, weights)

                output = {"input": torch.LongTensor(input_ids)}
                if pos is not None:
                    output["positions"] = torch.LongTensor(pos)
                if weights is not None:
                    output["loss_weights"] = torch.tensor(weights, dtype=torch.float32)

                yield output, torch.LongTensor(labels)

    def _iter_mixed(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Iter mixed."""
        assert self._mixer is not None
        for packed_sample in self._mixer:
            input_ids = packed_sample.input_ids[:-1]
            labels = packed_sample.labels[1:]
            pos = packed_sample.positions[:-1] if packed_sample.positions is not None else None
            weights = packed_sample.loss_weights[1:] if packed_sample.loss_weights is not None else None

            input_ids, labels, pos, weights = self._pad_to_seq_len(input_ids, labels, pos, weights)

            output = {"input": torch.LongTensor(input_ids)}
            if pos is not None:
                output["positions"] = torch.LongTensor(pos)
            if weights is not None:
                output["loss_weights"] = torch.tensor(weights, dtype=torch.float32)

            if packed_sample.pixel_values is not None:
                output["pixel_values"] = packed_sample.pixel_values
                output["grid_thw"] = packed_sample.grid_thw

            yield output, torch.LongTensor(labels)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """State dict."""
        state: dict[str, Any] = {
            "streaming_samples_consumed": self._streaming_samples_consumed,
            "per_stream_tokens": dict(self._per_stream_tokens),
            "per_stream_samples": dict(self._per_stream_samples),
        }

        if self._mixer is not None:
            state["mixer"] = self._mixer.state_dict()
        elif self._streaming_ds is not None:
            state["streaming"] = self._streaming_ds.state_dict(
                num_samples=self._streaming_samples_consumed,
                from_beginning=False,
            )

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load and return the requested data."""
        if self._mixer is not None:
            if "mixer" in state_dict:
                self._mixer.load_state_dict(state_dict["mixer"])
        elif self._streaming_ds is not None:
            if "streaming" in state_dict:
                self._streaming_ds.load_state_dict(state_dict["streaming"])
                logger.info(f"StreamingDataset 状态已恢复: {state_dict['streaming']}")

        self._streaming_samples_consumed = state_dict.get("streaming_samples_consumed", 0)
        self._per_stream_tokens = dict(state_dict.get("per_stream_tokens", {}))
        self._per_stream_samples = dict(state_dict.get("per_stream_samples", {}))

    def get_total_tokens_seen(self) -> int:
        """Return the requested value."""
        if self._packed_streams:
            return sum(ps._packer.get_total_tokens_seen() for ps in self._packed_streams)
        return self._packer.get_total_tokens_seen()

    def get_total_valid_tokens(self) -> int:
        """Return the requested value."""
        num_samples = self._streaming_ds.num_samples
        return num_samples * self.seq_len

    def get_num_original_samples(self) -> int:
        """Return the requested value."""
        return self._streaming_ds.num_samples

    def get_per_stream_stats(self) -> dict[str, dict[str, int]]:
        """Return the requested value."""
        result = {}
        all_ids = set(self._per_stream_tokens) | set(self._per_stream_samples)
        for sid in all_ids:
            result[sid] = {
                "tokens": self._per_stream_tokens.get(sid, 0),
                "samples": self._per_stream_samples.get(sid, 0),
            }
        return result

    def get_epoch_progress(
        self, stream_total_tokens: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """Return the requested value."""
        if not stream_total_tokens:
            return {}
        result = {}
        for sid, consumed in self._per_stream_tokens.items():
            total = stream_total_tokens.get(sid, 0)
            if total > 0:
                result[sid] = consumed / total
        return result

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    def _reject_sample(self, error: Exception, sample, error_type: str) -> bool:
        """Reject sample."""
        if self._skip_corrupted:
            self._handle_sample_error(error, sample, error_type)
            return False
        raise error

    def _validate_sample(self, sample) -> bool:
        """Validate the supplied configuration."""
        if sample is None:
            return self._reject_sample(ValueError("样本为 None"), sample, "null_sample")

        if not isinstance(sample, (dict, str)):
            return self._reject_sample(
                TypeError(f"样本类型不合法: {type(sample).__name__}，期望 dict 或 str"),
                sample, "invalid_type",
            )

        if isinstance(sample, str):
            if not sample.strip():
                return self._reject_sample(ValueError("字符串样本为空"), sample, "empty_text")
            return True

        if not sample:
            return self._reject_sample(ValueError("样本为空字典"), sample, "empty_dict")

        raw_text = sample.get(self._text_key)

        if raw_text is None:
            return self._reject_sample(
                ValueError(f"字段 '{self._text_key}' 不存在，可用字段: {list(sample.keys())}"),
                sample, "missing_key",
            )

        if not isinstance(raw_text, str):
            return self._reject_sample(
                TypeError(f"字段 '{self._text_key}' 类型不合法: {type(raw_text).__name__}，期望 str"),
                sample, "invalid_text_type",
            )

        if not raw_text.strip():
            return self._reject_sample(
                ValueError(f"字段 '{self._text_key}' 为空"), sample, "empty_text",
            )

        return True

    def _handle_sample_error(self, error: Exception, sample, error_type: str):
        """Handle sample error."""
        self._consecutive_errors += 1
        self._error_stats[error_type] = self._error_stats.get(error_type, 0) + 1

        if len(self._corrupted_samples) < 100:
            self._corrupted_samples.append({
                "sample_idx": self._streaming_samples_consumed,
                "error_type": error_type,
                "error": str(error),
                "sample_preview": str(sample)[:200] if sample else None,
            })

        logger.warning(
            f"数据加载错误 [{error_type}]: {error}\n"
            f"样本索引: {self._streaming_samples_consumed}\n"
            f"样本预览: {str(sample)[:100] if sample else 'None'}\n"
            f"当前错误统计: {self._error_stats}"
        )

        if self._consecutive_errors >= self._max_consecutive_errors:
            self._log_error_summary()
            raise RuntimeError(
                f"数据损坏严重：连续 {self._consecutive_errors} 个样本失败。\n"
                f"错误统计: {self._error_stats}\n"
                f"建议检查数据源完整性。"
            )

    def _handle_shard_error(self, error: Exception, sample_idx: int):
        """Handle shard error."""
        shard_count = self._error_stats.get("shard_errors", 0) + 1
        self._error_stats["shard_errors"] = shard_count

        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        relevant_frames = [
            line for line in tb_lines
            if any(kw in line for kw in ("streaming", "mds", "reader", "shard", "array"))
        ]
        short_tb = "".join(relevant_frames) if relevant_frames else "（无相关堆栈）"

        logger.error(
            f"[Shard 错误 #{shard_count}] 已跳过，训练继续\n"            f"  异常类型: {type(error).__name__}\n"
            f"  错误信息: {error}\n"
            f"  样本索引: {sample_idx}\n"
            f"  相关堆栈:\n{short_tb}\n"
            f"  累积 shard 错误: {shard_count}"
        )

        if len(self._corrupted_samples) < 100:
            self._corrupted_samples.append({
                "sample_idx": sample_idx,
                "error_type": "shard_error",
                "error": f"{type(error).__name__}: {error}",
                "sample_preview": None,
            })

        self._consecutive_errors += 1
        if self._consecutive_errors >= self._max_consecutive_errors:
            self._log_error_summary()
            raise RuntimeError(
                f"数据损坏严重：连续 {self._consecutive_errors} 次错误（含 shard 级）。\n"
                f"错误统计: {self._error_stats}\n"
                f"建议检查数据源完整性。"
            )

    def _is_shard_error(self, error: Exception) -> bool:
        """Is shard error."""
        shard_error_types = (
            OSError,
            IOError,
            FileNotFoundError,
        )
        if isinstance(error, shard_error_types):
            return True

        error_type_name = type(error).__name__
        if error_type_name == "error" and type(error).__module__ == "struct":
            return True

        error_str = str(error)
        shard_error_patterns = [
            "not enough values to unpack",
            "No such file",
            "I/O operation",
            "frombuffer",
            "shard_corrupt",
            "Invalid shard",
        ]
        return any(pattern in error_str for pattern in shard_error_patterns)

    def _log_error_summary(self):
        """Log error summary."""
        total_errors = sum(self._error_stats.values())
        if total_errors > 0:
            logger.warning(
                f"StreamingTextDataset 错误统计摘要:\n"
                f"  总错误数: {total_errors}\n"
                f"  错误类型分布: {self._error_stats}\n"
                f"  损坏样本数: {len(self._corrupted_samples)}\n"
                f"  成功处理样本数: {self._streaming_samples_consumed}"
            )

            if len(self._corrupted_samples) > 10:
                logger.error(
                    f"检测到大量损坏样本（{len(self._corrupted_samples)} 个），"
                    f"建议检查数据源完整性！"
                )

    def get_error_stats(self) -> dict[str, int]:
        """Return the requested value."""
        return self._error_stats.copy()

    def get_corrupted_samples(self) -> list[dict]:
        """Return the requested value."""
        return self._corrupted_samples.copy()

    def reset_error_stats(self):
        """Reset error stats."""
        self._error_stats.clear()
        self._corrupted_samples.clear()
        self._consecutive_errors = 0
        logger.info("错误统计信息已重置")


# ============================================================================
# ============================================================================


def _read_single_stream_stats(local_path: str, tokenizer_name: str | None = None) -> dict | None:
    """Read single stream stats."""
    if not local_path:
        return None
    from titaninfra.data.schema import StreamStats

    stats = StreamStats.load(local_path)
    if stats is None:
        return None
    total_tokens, avg_tokens = stats.resolve_tokens(tokenizer_name)

    fine_dist = None
    resolved_name = tokenizer_name or stats.default_tokenizer
    if resolved_name and resolved_name in stats.tokenizer_stats:
        fine_dist = stats.tokenizer_stats[resolved_name].fine_token_length_dist

    return {
        "num_samples": stats.num_samples,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": avg_tokens,
        "stream_id": stats.stream_id,
        "fine_token_length_dist": fine_dist,
    }


def _compute_per_stream_weights(
    streams: list,
    seq_len: int | None = None,
) -> list[float]:
    """Compute and return the result."""
    from titaninfra.data.token_estimator import _is_sample_based, estimate_packing_efficiency

    raw_weights: list[float] = []
    for s in streams:
        if s.proportion is not None:
            raw_weights.append(s.proportion)
        elif s.repeat is not None:
            stats = _read_single_stream_stats(s.local, tokenizer_name=getattr(s, "tokenizer_name", None))
            n = stats["num_samples"] if stats else 1000
            avg = stats["avg_tokens_per_sample"] if stats else 1.0
            avg = avg or 1.0
            sample_based = _is_sample_based(s.pack_strategy)
            w = s.repeat * n * (1.0 if sample_based else avg)
            if not sample_based and seq_len is not None and stats:
                pe = estimate_packing_efficiency(
                    stats.get("fine_token_length_dist"), seq_len, s.pack_strategy,
                )
                if pe > 0:
                    w /= pe
                w /= seq_len
            raw_weights.append(w)
        elif s.choose is not None:
            stats = _read_single_stream_stats(s.local, tokenizer_name=getattr(s, "tokenizer_name", None))
            avg = stats["avg_tokens_per_sample"] if stats else 1.0
            avg = avg or 1.0
            sample_based = _is_sample_based(s.pack_strategy)
            w = float(s.choose) * (1.0 if sample_based else avg)
            if not sample_based and seq_len is not None and stats:
                pe = estimate_packing_efficiency(
                    stats.get("fine_token_length_dist"), seq_len, s.pack_strategy,
                )
                if pe > 0:
                    w /= pe
                w /= seq_len
            raw_weights.append(w)
        else:
            raw_weights.append(1.0)
    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / len(raw_weights)] * len(raw_weights)
    return [w / total for w in raw_weights]


def _warn_if_margin_insufficient(
    epoch_size: int,
    dp_world_size: int,
    training_steps: int | None,
    local_batch_size: int,
    gas: int,
    seq_len: int,
    normalized_weight: float,
    pack_strategy,
    avg_tokens: float,
    margin: float,
    stream_local: str,
) -> None:
    if training_steps is None or avg_tokens <= 0:
        return
    w = max(normalized_weight, 1e-6)
    val = pack_strategy.value if hasattr(pack_strategy, "value") else str(pack_strategy)
    if val == "no_pack":
        consumed = training_steps * local_batch_size * gas * w
    else:
        consumed = training_steps * local_batch_size * gas * seq_len * w / avg_tokens
    available = epoch_size / dp_world_size
    if available < consumed:
        logger.warning(
            f"epoch_size margin 不足: epoch/dp={available:.0f} < "
            f"consumed={consumed:.0f}, 建议增大 epoch_size_margin "
            f"(当前 {margin}, 推荐 ≥0.5) ({stream_local})"
        )


def _compute_per_stream_epoch_size(
    *,
    stream_cfg,
    stats: dict | None,
    normalized_weight: float,
    training_steps: int | None,
    seq_len: int,
    local_batch_size: int,
    dp_world_size: int,
    margin: float,
    mosaicml_world_size: int | None = None,
    gas: int = 1,
    calibration_steps: int = 0,
    prefetch_batches: int = 0,
) -> int | None:
    """Compute and return the result."""
    if stream_cfg.repeat is not None:
        if stats is None:
            logger.warning(f"repeat 模式但无 stats: {stream_cfg.local}，跳过 epoch_size 计算")
            return None
        base_epoch_size = math.ceil(stats["num_samples"] * stream_cfg.repeat)
        epoch_size = math.ceil(base_epoch_size * (1.0 + margin))
        logger.info(
            f"[per-stream] repeat={stream_cfg.repeat}, margin={margin} → "
            f"epoch_size={epoch_size:,} samples ({stream_cfg.local})"
        )
        _warn_if_margin_insufficient(
            epoch_size, dp_world_size, training_steps, local_batch_size,
            gas, seq_len, normalized_weight, stream_cfg.pack_strategy,
            stats["avg_tokens_per_sample"], margin, stream_cfg.local,
        )
        return epoch_size

    if stream_cfg.choose is not None:
        epoch_size = math.ceil(stream_cfg.choose * (1.0 + margin))
        logger.info(
            f"[per-stream] choose={stream_cfg.choose}, margin={margin} → "
            f"epoch_size={epoch_size:,} samples ({stream_cfg.local})"
        )
        if stats is not None:
            _warn_if_margin_insufficient(
                epoch_size, dp_world_size, training_steps, local_batch_size,
                gas, seq_len, normalized_weight, stream_cfg.pack_strategy,
                stats.get("avg_tokens_per_sample", 0), margin, stream_cfg.local,
            )
        return epoch_size

    if stats is None:
        logger.warning(f"proportion 模式但无 stats: {stream_cfg.local}，跳过 epoch_size 计算")
        return None

    avg_tokens = stats["avg_tokens_per_sample"]
    num_samples = stats["num_samples"]

    if avg_tokens <= 0:
        logger.warning(f"avg_tokens <= 0: {stream_cfg.local}，跳过 epoch_size 计算")
        return None

    from titaninfra.data.token_estimator import estimate_epoch_size, estimate_packing_efficiency

    pe = estimate_packing_efficiency(
        stats.get("fine_token_length_dist"), seq_len, stream_cfg.pack_strategy,
    )
    effective_world = mosaicml_world_size if mosaicml_world_size is not None else dp_world_size
    epoch_size = estimate_epoch_size(
        num_samples=num_samples,
        avg_tokens_per_sample=avg_tokens,
        pack_strategy=stream_cfg.pack_strategy,
        normalized_weight=normalized_weight,
        training_steps=training_steps,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        gas=gas,
        dp_world_size=dp_world_size,
        mosaicml_world_size=effective_world,
        margin=margin,
        calibration_steps=calibration_steps,
        prefetch_batches=prefetch_batches,
        packing_efficiency=pe,
    )

    if epoch_size is not None:
        passes = epoch_size / num_samples
        mode = f"steps={training_steps}"
        pe_str = f", PE={pe:.3f}" if pe < 1.0 else ""
        logger.info(
            f"[per-stream] {mode}, weight={normalized_weight:.3f} → "
            f"epoch_size={epoch_size:,} samples "
            f"(avg_tokens={avg_tokens:.0f}, ≈{passes:.1f}× data{pe_str}) "
            f"({stream_cfg.local})"
        )
    return epoch_size


def _encode_text_to_pairs(
    text: str,
    tokenizer: "BaseTokenizer",
    add_bos_eos: bool = False,
) -> list[tuple[int, int]]:
    """Encode text to pairs."""
    tokens = tokenizer.encode(text, add_bos=add_bos_eos, add_eos=add_bos_eos)
    return [(t, t) for t in tokens]


def _make_pretrain_doc_iter(
    streaming_ds,
    tokenizer: "BaseTokenizer",
    text_key: str,
    stream_id: str,
    add_bos_eos: bool = False,
) -> Iterator[list[tuple[int, int]]]:
    """Create the configured objects."""
    for sample in streaming_ds:
        text = sample.get(text_key, "") if isinstance(sample, dict) else str(sample)
        yield _encode_text_to_pairs(text, tokenizer, add_bos_eos)


def _sft_images_of(sample: dict, messages_key: str = "messages") -> list[bytes]:
    """Sft images of."""
    from titaninfra.data.multimodal.image_processor import sft_images_of

    return sft_images_of(sample, messages_key=messages_key)


def _make_messages_doc_iter(
    streaming_ds,
    hf_tokenizer,
    stream_cfg: "StreamConfig",
    seq_len: int,
    image_decode_workers: int | None = None,
) -> Iterator[list[tuple[int, int]] | list[tuple[int, int, float]]]:
    """Create the configured objects."""
    from titaninfra.data.sft.assembler import TruncatedToEmptyError
    from titaninfra.data.sft.dispatch import tokenize_sample

    messages_key = stream_cfg.messages_key
    max_turns = getattr(stream_cfg, "max_turns", None)
    thinking_scale = getattr(stream_cfg, "thinking_scale", None)
    thinking_mode = getattr(stream_cfg, "thinking_mode", None)
    validate_alignment = getattr(stream_cfg, "validate_template_alignment", False)
    _alignment_checked = 0

    sid = stream_cfg.resolve_stream_id()
    _family = getattr(stream_cfg, "family", None) or getattr(stream_cfg, "tokenizer_name", None)
    if not _family:
        raise ValueError(
            f"[{sid}] StreamConfig.family 与 tokenizer_name 均未设置，"
            "无法路由到 SFT encoder。请显式设置 family（如 'qwen3' / 'llama3' / 'gemma4'）。"
        )
    _resolution_policy = getattr(stream_cfg, "resolution_policy", None)
    _spatial_merge_size = 2 if _resolution_policy is not None else None
    _truncate_at_last_segment = getattr(stream_cfg, "truncate_at_last_segment", False)
    # full_sample semantics: skip samples that exceed seq_len entirely (don't truncate).
    # Other strategies (hard_truncate, sentence_boundary) intentionally truncate.
    _is_full_sample = strategy_to_pack_mode(stream_cfg.pack_strategy.value) == PackMode.SFT
    if _is_full_sample:
        _max_len = seq_len + 1 if _truncate_at_last_segment else None
    else:
        _max_len = seq_len

    if _resolution_policy is not None:
        from titaninfra.data.multimodal.decode_pool import ImageDecodePool

        def _sample_source_gen():
            with ImageDecodePool(
                workers=image_decode_workers,
                preprocess_kwargs={
                    "min_pixels": _resolution_policy.min_pixels,
                    "max_pixels": _resolution_policy.max_pixels,
                    "spatial_merge_size": _spatial_merge_size,
                },
            ) as pool:
                yield from pool.map_docs(streaming_ds, _sft_images_of)

        _sample_source = _sample_source_gen()
    else:
        _sample_source = ((s, None) for s in streaming_ds)

    for sample, _decoded in _sample_source:
        raw = sample.get(messages_key, "")
        if not raw:
            continue

        if isinstance(raw, str):
            try:
                messages = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"[{sid}] 无法解析 messages JSON，跳过")
                continue
        elif isinstance(raw, list):
            messages = raw
        else:
            continue

        if not messages:
            continue

        if max_turns is not None:
            messages = _truncate_turns(messages, max_turns)

        lc_raw = sample.get("loss_config", "")
        loss_config = None
        if lc_raw:
            try:
                loss_config = json.loads(lc_raw) if isinstance(lc_raw, str) else lc_raw
            except json.JSONDecodeError:
                logger.warning(
                    "[%s] malformed loss_config JSON, ignoring "
                    "(thinking_scale/type_overrides will not apply). raw=%r",
                    sid, str(lc_raw)[:200],
                )
                loss_config = None

        if thinking_scale is not None:
            if loss_config is None:
                loss_config = {}
            loss_config.setdefault("type_overrides", {})["thinking"] = thinking_scale

        system_raw = sample.get("system", "")
        system = system_raw if system_raw else None

        tools_raw = sample.get("tools", "")
        tools = None
        if tools_raw:
            try:
                tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "[%s] malformed tools JSON, ignoring "
                    "(tool declarations will not be rendered). raw=%r",
                    sid, str(tools_raw)[:200],
                )
                tools = None

        sample_json = {"messages": messages, "system": system, "tools": tools}

        if thinking_mode is not None:
            ti_raw = sample.get("thinking_intent", None)
            if isinstance(ti_raw, str) and ti_raw:
                try:
                    ti = json.loads(ti_raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "[%s] malformed thinking_intent JSON, "
                        "overriding mode from StreamConfig only. raw=%r",
                        sid, str(ti_raw)[:200],
                    )
                    ti = {}
            elif isinstance(ti_raw, dict):
                ti = dict(ti_raw)
            else:
                ti = {}
            ti["mode"] = thinking_mode
            sample_json["thinking_intent"] = ti

        try:
            encoded = tokenize_sample(
                sample_json,
                hf_tokenizer,
                family=_family,
                max_length=_max_len,
                loss_config=loss_config,
                truncate_at_last_segment=_truncate_at_last_segment,
                sft_mask_mode=getattr(stream_cfg, "sft_mask_mode", "auto"),
                history_policy=getattr(stream_cfg, "history_policy", None),
                spatial_merge_size=_spatial_merge_size,
                decoded_images=_decoded,
            )
        except TruncatedToEmptyError:
            continue
        input_ids = encoded.input_ids
        labels = encoded.labels
        loss_weights = encoded.loss_weights

        if not input_ids or (_is_full_sample and len(input_ids) > seq_len):
            continue

        if validate_alignment and (_alignment_checked < 5 or _alignment_checked % 1000 == 0):
            from titaninfra.data.sft.alignment_check import validate_hf_alignment
            validate_hf_alignment(messages, input_ids, hf_tokenizer)
        _alignment_checked += 1

        if encoded.pixel_values is not None:
            if loss_weights is not None:
                raise NotImplementedError(
                    f"[{sid}] VLM SFT + per-token loss_weights 尚未支持"
                    "（图流用 sft_mask_mode=auto，无 type_overrides/thinking_scale）"
                )
            yield MMDoc(
                tokens=list(zip(input_ids, labels, strict=False)),
                images=(encoded.pixel_values, encoded.grid_thw),
            )
        elif loss_weights is not None:
            yield [(tid, lbl, w) for tid, lbl, w in zip(input_ids, labels, loss_weights, strict=False)]
        else:
            yield list(zip(input_ids, labels, strict=False))


def _truncate_turns(messages: list[dict], max_turns: int) -> list[dict]:
    """Truncate turns."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= max_turns:
        return messages

    truncated_non_system = non_system[-max_turns:]
    return system_msgs + truncated_non_system


# ============================================================================
# _FlatTokenCollator — builds upstream's TokenizedTrainingMicrobatch
# ============================================================================


class _FlatTokenCollator:
    """Flat token collator configuration and behavior."""

    def __init__(self, *, num_tokens_per_microbatch: int, max_context_length: int, pad_id: int) -> None:
        self._size = num_tokens_per_microbatch
        self._max_context_length = max_context_length
        self._pad_id = pad_id

    def __call__(
        self, batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> TokenizedTrainingMicrobatch:
        outputs = [b[0] for b in batch]
        labels_list = [b[1] for b in batch]
        has_positions = "positions" in outputs[0]
        has_weights = "loss_weights" in outputs[0]

        num_tokens = sum(len(o["input"]) for o in outputs)
        if num_tokens > self._size:
            raise ValueError(
                f"batch 的总 token 数 {num_tokens} 超过配置的 "
                f"num_tokens_per_microbatch={self._size}（每条样本理应已被 "
                "packer pad 到 seq_len，总数应恰好等于 "
                "local_batch_size * seq_len；请检查 batch_size/seq_len 配置）"
            )

        size = self._size
        input_ids = torch.full((size,), self._pad_id, dtype=torch.int64)
        labels = torch.full((size,), -100, dtype=torch.int64)
        positions = torch.full((size,), -1, dtype=torch.int64)
        loss_weights = torch.zeros((size,), dtype=torch.float32) if has_weights else None

        torch.cat([o["input"] for o in outputs], out=input_ids[:num_tokens])
        torch.cat(labels_list, out=labels[:num_tokens])
        if has_positions:
            torch.cat([o["positions"] for o in outputs], out=positions[:num_tokens])
        else:
            offset = 0
            for o in outputs:
                n = len(o["input"])
                positions[offset : offset + n] = torch.arange(n)
                offset += n
        if has_weights:
            torch.cat([o["loss_weights"] for o in outputs], out=loss_weights[:num_tokens])

        pad_len = size - num_tokens
        if pad_len:
            positions[num_tokens:] = (
                torch.arange(pad_len) % self._max_context_length
            )

        padding_mask = positions == -1
        if pad_len:
            padding_mask[num_tokens:] = True

        loss_token_counts = (labels != -100).sum()
        routing_token_counts = torch.stack([(~padding_mask).sum()])

        model_kwargs: dict[str, Any] = {}
        if has_weights:
            model_kwargs["loss_kwargs"] = {"loss_weights": loss_weights}

        return TokenizedTrainingMicrobatch(
            input=input_ids,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
            loss_token_counts=loss_token_counts,
            routing_token_counts=routing_token_counts,
            model_kwargs=model_kwargs,
        )


# ============================================================================
# StreamingTextDataLoader - StatefulDataLoader + torchtitan BaseDataLoader
# ============================================================================


class StreamingTextDataLoader(StatefulDataLoader, BaseDataLoader):
    """Streaming text data loader configuration and behavior."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        """Config configuration and behavior."""

        streams: list[StreamConfig]
        """StreamConfig entries with independent source, tokenization, and packing."""

        repeat: bool = True
        """Whether to repeat iteration; Pretrain defaults to True and validation to False."""

        shuffle: bool = True
        """Whether to shuffle samples."""

        shuffle_seed: int = 9176
        """Seed for deterministic shuffling."""

        num_canonical_nodes: int | None = None
        """Canonical node count used for deterministic resume."""

        batch_size_per_device: int | None = None
        """Per-device batch size used by StreamingDataset partitioning."""

        training_steps: int | None = None
        """Total optimizer steps used to calculate epoch_size."""

        gas: int = 1
        """Gradient-accumulation microbatches per optimizer step."""

        calibration_steps: int = 0
        """MoE calibration steps included in the epoch_size budget."""

        skip_corrupted_samples: bool = True
        """Skip corrupted samples instead of failing immediately."""

        max_consecutive_errors: int = 10
        """Consecutive-error threshold before training stops."""

        predownload: int | None = None
        """Number of samples prefetched by StreamingDataset."""

        shuffle_block_size: int | None = None
        """Shuffle block size."""

        shuffle_algo: str | None = None
        """MosaicML shuffle algorithm forwarded from TrainConfig."""

        hf_tokenizer_path: str = ""
        """Hugging Face tokenizer path for native message tokenization."""

        pp: int = 1
        """Pipeline Parallelism degree used by the legacy data-distribution strategy."""

        # --- DataLoader ---
        num_workers: int = 0
        persistent_workers: bool = False
        pin_memory: bool = False
        prefetch_factor: int | None = None

        image_decode_workers: int | None = None
        """Image decode thread-pool size; None uses a CPU-derived default."""
        image_decode_prefetch: int | None = None
        """Image decode future queue depth; None defaults to twice workers."""

        def __post_init__(self):
            """Post init."""
            super(StreamingTextDataLoader.Config, self).__post_init__()
            if not self.streams:
                raise ValueError("streams 不能为空，至少需要一个数据流")

            proportions = [
                s.proportion for s in self.streams
                if s.proportion is not None
            ]
            if proportions and sum(proportions) > 1.0 + 1e-6:
                raise ValueError(
                    f"数据流 proportion 总和 ({sum(proportions):.2f}) 超过 1.0"
                )

    def __init__(
        self,
        config: "StreamingTextDataLoader.Config",
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_microbatch: int,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Init."""
        try:
            from streaming import Stream, StreamingDataset
        except ImportError as e:
            raise ImportError(
                "StreamingTextDataLoader 需要 mosaicml-streaming。\n"
                "请安装：pip install mosaicml-streaming"
            ) from e

        seq_len = max_context_length
        local_batch_size = num_tokens_per_microbatch // max_context_length

        self._per_stream_total_tokens: dict[str, int] = {}

        hf_tokenizer = None
        has_messages = any(s.data_type == "messages" for s in config.streams)
        if has_messages:
            from titaninfra.data.sft_utils import load_hf_tokenizer
            hf_tokenizer = load_hf_tokenizer(config.hf_tokenizer_path)

        packed_streams: list[PackedStream] = []
        stream_ids: list[str] = []
        weights = _compute_per_stream_weights(config.streams, seq_len=seq_len)
        base_kwargs = self._streaming_dataset_kwargs(config, local_batch_size)

        # --- Data distribution strategy ---
        # Non-DP dimensions (CP, TP, PP) must see identical data.
        # Two strategies depending on mesh topology:
        #
        # Strategy A (pp==1): Use MosaicML's `replication` parameter.
        #   Keeps real RANK/WORLD_SIZE env vars → unique shared memory per process.
        #   replication=non_dp_degree groups consecutive CP/TP ranks for same data.
        #
        # Strategy B (pp>1): Override RANK/WORLD_SIZE env vars to dp_rank.
        #   PP ranks are non-consecutive in mesh → replication can't group them.
        #   Safe because PP ranks span different nodes (no shared memory collision).
        mosaicml_world_size = dp_world_size
        _actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        non_dp_degree = max(1, _actual_world_size // dp_world_size)
        _use_replication = non_dp_degree > 1 and config.pp <= 1
        _env_override_active = False
        _replication: int | None = None

        if _use_replication:
            base_kwargs["num_canonical_nodes"] = dp_world_size
            _replication = non_dp_degree
        elif non_dp_degree > 1:
            base_kwargs["num_canonical_nodes"] = dp_world_size
            _env_override_active = True
            _env_keys = ("RANK", "WORLD_SIZE")
            _saved_env = {k: os.environ.get(k) for k in _env_keys}
            os.environ["RANK"] = str(dp_rank)
            os.environ["WORLD_SIZE"] = str(dp_world_size)
        else:
            base_kwargs["num_canonical_nodes"] = dp_world_size

        if _replication is not None:
            logger.info(
                f"Data distribution: replication={_replication} "
                f"(world_size={_actual_world_size}, dp={dp_world_size}, pp={config.pp})"
            )
        elif _env_override_active:
            logger.info(
                f"Data distribution: env var override "
                f"(RANK={dp_rank}, WORLD_SIZE={dp_world_size}, pp={config.pp})"
            )

        try:
            for i, stream_cfg in enumerate(config.streams):
                sid = stream_cfg.resolve_stream_id()

                stats_i = _read_single_stream_stats(stream_cfg.local, tokenizer_name=stream_cfg.tokenizer_name)
                if stats_i is None and getattr(stream_cfg, "remote", None):
                    stats_i = _read_single_stream_stats(stream_cfg.remote, tokenizer_name=stream_cfg.tokenizer_name)
                if stats_i:
                    self._per_stream_total_tokens[sid] = stats_i["total_tokens"]

                prefetch = config.num_workers * (config.prefetch_factor or 2) if config.num_workers > 0 else 0
                epoch_size_i = _compute_per_stream_epoch_size(
                    stream_cfg=stream_cfg,
                    stats=stats_i,
                    normalized_weight=weights[i],
                    training_steps=config.training_steps,
                            seq_len=seq_len,
                    local_batch_size=local_batch_size,
                    dp_world_size=dp_world_size,
                    margin=stream_cfg.epoch_size_margin,
                    mosaicml_world_size=mosaicml_world_size,
                    gas=config.gas,
                    calibration_steps=config.calibration_steps,
                    prefetch_batches=prefetch,
                )

                mosaicml_kwargs = stream_cfg.to_mosaicml_kwargs()
                build_kwargs = {**base_kwargs}
                build_kwargs["shuffle_seed"] = config.shuffle_seed + i
                if epoch_size_i is not None:
                    build_kwargs["epoch_size"] = epoch_size_i
                if _replication is not None:
                    build_kwargs["replication"] = _replication
                if stream_cfg.cache_limit is not None:
                    build_kwargs["cache_limit"] = stream_cfg.cache_limit
                if stream_cfg.resolution_policy is not None:
                    build_kwargs["allow_unsafe_types"] = True
                ds_i = StreamingDataset(
                    streams=[Stream(**mosaicml_kwargs)],
                    **build_kwargs,
                )

                sent_ids: list[int] = []
                for s in stream_cfg.sentence_end_tokens:
                    ids = tokenizer.encode(s, add_bos=False, add_eos=False)
                    if len(ids) == 1:
                        sent_ids.append(ids[0])
                    else:
                        logger.warning(
                            f"[{sid}] sentence_end_token '{s}' 编码为 {len(ids)} 个 token，跳过"
                        )

                pack_mode = strategy_to_pack_mode(stream_cfg.pack_strategy.value)
                packer_i = StreamingPacker(
                    seq_len=seq_len,
                    mode=pack_mode,
                    sentence_end_tokens=sent_ids if pack_mode == PackMode.PRETRAIN else [],
                )

                if stream_cfg.data_type == "messages":
                    if hf_tokenizer is None:
                        raise ValueError(
                            f"[{sid}] data_type='messages' 需要 hf_tokenizer_path"
                        )
                    if (
                        stream_cfg.resolution_policy is not None
                        and pack_mode != PackMode.SFT
                    ):
                        raise ValueError(
                            f"[{sid}] 带图 messages 流（resolution_policy 非 None）"
                            f"必须 pack_strategy='full_sample'，实得 {stream_cfg.pack_strategy.value}"
                        )
                    _hf_tok = hf_tokenizer
                    _scfg = stream_cfg
                    _slen = seq_len
                    _idw = config.image_decode_workers

                    def _msg_iter(_ds=ds_i, _t=_hf_tok, _s=_scfg, _l=_slen, _w=_idw):
                        return _make_messages_doc_iter(_ds, _t, _s, _l, image_decode_workers=_w)

                    doc_factory = _msg_iter
                elif stream_cfg.resolution_policy is not None:
                    if pack_mode != PackMode.SFT:
                        raise ValueError(
                            f"[{sid}] 多模态 pretrain 必须 pack_strategy='full_sample'"
                            f"（文档级 bin-packing 保证图 span 不被切开），实得 {stream_cfg.pack_strategy.value}"
                        )
                    from titaninfra.data.multimodal.decode_pool import ImageDecodePool
                    from titaninfra.data.multimodal.mm_doc_iter import (
                        _make_multimodal_pretrain_doc_iter,
                    )

                    _tok = tokenizer
                    _abe = stream_cfg.add_bos_eos
                    _rp = stream_cfg.resolution_policy
                    _dw = config.image_decode_workers
                    _dp = config.image_decode_prefetch

                    def _mm_iter(_ds=ds_i, _t=_tok, _a=_abe, _rp=_rp, _dw=_dw, _dp=_dp):
                        pool = ImageDecodePool(
                            workers=_dw,
                            prefetch=_dp,
                            preprocess_kwargs={
                                "min_pixels": _rp.min_pixels,
                                "max_pixels": _rp.max_pixels,
                            },
                        )
                        return _make_multimodal_pretrain_doc_iter(
                            _ds, _t, pool, add_bos_eos=_a
                        )

                    doc_factory = _mm_iter
                else:
                    _tok = tokenizer
                    _tkey = stream_cfg.text_key
                    _sid = sid
                    _abe = stream_cfg.add_bos_eos

                    def _pretrain_iter(_ds=ds_i, _t=_tok, _k=_tkey, _s=_sid, _a=_abe):
                        return _make_pretrain_doc_iter(_ds, _t, _k, _s, add_bos_eos=_a)

                    doc_factory = _pretrain_iter

                packed_stream_i = PackedStream(
                    source=doc_factory(),
                    packer=packer_i,
                    name=sid,
                    source_factory=doc_factory,
                )
                packed_streams.append(packed_stream_i)
                stream_ids.append(sid)
        finally:
            if _env_override_active:
                for k in _env_keys:
                    if _saved_env[k] is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = _saved_env[k]

        logger.info(
            f"Per-stream 构建完成：{len(packed_streams)} 流 "
            f"({', '.join(f'{sid}({w:.2f})' for sid, w in zip(stream_ids, weights, strict=False))})"
        )

        default_stream_id = (
            stream_ids[0] if len(stream_ids) == 1 else "multi"
        )

        dataset = StreamingTextDataset(
            streaming_dataset=None,
            tokenizer=tokenizer,
            seq_len=seq_len,
            text_key=config.streams[0].text_key if config.streams else "text",
            default_stream_id=default_stream_id,
            skip_corrupted_samples=config.skip_corrupted_samples,
            max_consecutive_errors=config.max_consecutive_errors,
            rng_seed=config.shuffle_seed,
            pretrain_stream_id=default_stream_id,
            pre_built_streams=packed_streams,
            pre_built_weights=weights,
            pre_built_stream_ids=stream_ids,
            infinite=config.repeat,
        )

        # DataLoader kwargs
        dataloader_kwargs: dict[str, Any] = {
            "batch_size": local_batch_size,
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
        }
        if config.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = config.prefetch_factor

        #
        _has_mm = any(getattr(s, "resolution_policy", None) is not None for s in config.streams)
        if _has_mm:
            from titaninfra.data.multimodal.mm_collate import MMCollator

            dataloader_kwargs["collate_fn"] = MMCollator(
                tokenizer,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
                max_context_length=max_context_length,
                pad_id=StreamingTextDataset._resolve_pad_id(tokenizer),
            )
        else:
            dataloader_kwargs["collate_fn"] = _FlatTokenCollator(
                num_tokens_per_microbatch=num_tokens_per_microbatch,
                max_context_length=max_context_length,
                pad_id=StreamingTextDataset._resolve_pad_id(tokenizer),
            )

        super().__init__(
            dataset,
            **dataloader_kwargs,
        )
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self._rank_id = f"dp_rank_{dp_rank}"

        logger.info(
            f"StreamingTextDataLoader 初始化完成 "
            f"(dp_rank={dp_rank}/{dp_world_size}, "
            f"seq_len={seq_len}, batch={local_batch_size}, "
            f"streams={len(config.streams)})"
        )

    def __bool__(self) -> bool:
        """Bool."""
        return True

    def state_dict(self) -> dict[str, Any]:
        import pickle

        return {
            self._rank_id: pickle.dumps(StatefulDataLoader.state_dict(self)),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        import pickle

        if not state_dict:
            return
        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                f"expected key {self._rank_id}"
            )
            return
        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        StatefulDataLoader.load_state_dict(self, pickle.loads(state_dict[self._rank_id]))

    def get_total_tokens_seen(self) -> int:
        """Return the requested value."""
        return self.dataset.get_total_tokens_seen()

    def get_per_stream_stats(self) -> dict[str, dict[str, int]]:
        """Return the requested value."""
        return self.dataset.get_per_stream_stats()

    def get_epoch_progress(
        self, stream_total_tokens: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """Return the requested value."""
        if stream_total_tokens is None:
            stream_total_tokens = self._per_stream_total_tokens
        return self.dataset.get_epoch_progress(stream_total_tokens)

    def get_error_stats(self) -> dict[str, int]:
        """Return the requested value."""
        return self.dataset.get_error_stats()

    def get_corrupted_samples(self) -> list[dict]:
        """Return the requested value."""
        return self.dataset.get_corrupted_samples()

    def reset_error_stats(self):
        """Reset error stats."""
        self.dataset.reset_error_stats()

    @staticmethod
    def _streaming_dataset_kwargs(
        config: "StreamingTextDataLoader.Config",
        local_batch_size: int,
    ) -> dict[str, Any]:
        """Streaming dataset kwargs."""
        kwargs: dict[str, Any] = {
            "shuffle": config.shuffle,
            "shuffle_seed": config.shuffle_seed,
            "sampling_method": "balanced",
            "batch_size": config.batch_size_per_device or local_batch_size,
        }
        if config.predownload is not None:
            kwargs["predownload"] = config.predownload
        if config.shuffle_block_size is not None:
            kwargs["shuffle_block_size"] = config.shuffle_block_size
        if config.shuffle_algo is not None:
            kwargs["shuffle_algo"] = config.shuffle_algo
        return kwargs

    #   ({"input": Tensor[batch, seq_len]}, Tensor[batch, seq_len])
