"""SFT message schemas, JSON decoding, and stream_stats models.

Msgspec block types define text, thinking, tools, images, and history summaries
with cascading loss weights and history policies. Pydantic StreamStats and
TokenizerStats models standardize stream_stats.json reads and writes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Literal

import msgspec
from pydantic import BaseModel, Field

# Block types.

class TextBlock(msgspec.Struct, tag="text", tag_field="type"):
    """Plain text content."""
    text: str
    weight: float | None = None


class ThinkingBlock(msgspec.Struct, tag="thinking", tag_field="type"):
    """Reasoning content (chain of thought)."""
    text: str
    weight: float | None = None


class ToolUseBlock(msgspec.Struct, tag="tool_use", tag_field="type"):
    """Tool invocation content."""
    id: str
    name: str
    input: dict
    weight: float | None = None


class ToolResultBlock(msgspec.Struct, tag="tool_result", tag_field="type"):
    """Tool response content."""
    tool_use_id: str
    text: str
    is_error: bool = False
    weight: float | None = None


class ImageBlock(msgspec.Struct, tag="image", tag_field="type"):
    """Image source with type path, bytes, base64, or PIL."""
    source: dict  # Example: {"type": "base64", "data": "..."}
    weight: float | None = None


class HistorySummaryBlock(msgspec.Struct, tag="history_summary", tag_field="type"):
    """First-class compressed-history block.

    text is supplied by an upstream summarizer or human; normalize never
    generates it. compressed_turn_count records replaced turns. style selects
    DSv4 latest_reminder, system_prefix, or Gemma4 ephemeral_block rendering.
    """
    text: str
    compressed_turn_count: int = 0
    style: Literal["latest_reminder", "system_prefix", "ephemeral_block"] = "system_prefix"
    weight: float | None = None


Block = (
    TextBlock
    | ThinkingBlock
    | ToolUseBlock
    | ToolResultBlock
    | ImageBlock
    | HistorySummaryBlock
)


# HistoryPolicy with per-block-type granularity.

class HistoryAction(msgspec.Struct):
    """Action applied to one block type in historical turns.

    A historical turn occurs strictly before the last user turn. Current-turn
    blocks are never modified. action is keep, drop, or summarize;
    summary_template is required for summarize and uses {content}.
    """
    action: Literal["keep", "drop", "summarize"]
    summary_template: str | None = None


class HistoryPolicy(msgspec.Struct):
    """Per-block-type policy for turns before the last user turn.

    StreamConfig owns this recipe-level choice, not SFTSample. per_type maps
    block tags to HistoryAction and defaults to keep. keep_last_n retains the
    latest N assistant turns as current. insert_summary replaces the entire
    history region with supplied summary blocks and is mutually exclusive with
    keep_last_n; per_type is not used on that path.
    """
    per_type: dict[str, HistoryAction] = msgspec.field(default_factory=dict)
    insert_summary: bool = False
    keep_last_n: int | None = None


# Native family defaults used when StreamConfig.history_policy is unset.
# Qwen3, GPT-OSS, and Gemma4 drop historical thinking to match inference;
# other families keep history by default.
DEFAULT_HISTORY_POLICY: dict[str, HistoryPolicy] = {
    "qwen3":       HistoryPolicy(per_type={"thinking": HistoryAction(action="drop")}),
    "gpt_oss":     HistoryPolicy(per_type={"thinking": HistoryAction(action="drop")}),
    "gemma4":      HistoryPolicy(per_type={"thinking": HistoryAction(action="drop")}),
    "qwen2_5":     HistoryPolicy(),
    "deepseek_v3": HistoryPolicy(),
    "llama3":      HistoryPolicy(),
    "deepseek_v4": HistoryPolicy(),
}


# LossConfig.

class LossConfig(msgspec.Struct):
    """Per-sample, per-block-type loss-weight overrides."""
    type_overrides: dict[str, float] = msgspec.field(default_factory=dict)
    # Example: {"thinking": 0.5, "tool_use": 0, "image": 0}


# Tool definitions.

class Tool(msgspec.Struct):
    """Tool definition using JSON Schema-style fields."""
    name: str
    description: str
    input_schema: dict = msgspec.field(default_factory=dict)


# Message.

class Message(msgspec.Struct):
    """One message with role, blocks, and optional weight/name/task."""
    role: str
    blocks: list[Block]
    weight: float | None = None
    name: str | None = None
    task: str | None = None  # DSv4 action tag inserts task-transition tokens


# Family-agnostic ThinkingIntent.

class ThinkingIntent(msgspec.Struct):
    """Sample-level thinking intent shared by all messages in a sample.

    mode enables or disables the thinking branch. effort controls depth only
    for GPT-OSS; other families warn and ignore it. Literal
    fields reject invalid values during decoding.
    """
    mode: Literal["on", "off"] | None = None
    effort: Literal["low", "medium", "high"] | None = None


# Top-level SFTSample.

class SFTSample(msgspec.Struct):
    """One SFT training sample."""
    messages: list[Message]
    system: str | None = None
    tools: list[Tool] | None = None
    loss_config: LossConfig = msgspec.field(default_factory=LossConfig)
    thinking_intent: ThinkingIntent | None = None
    # history_policy belongs to StreamConfig, not SFTSample. Legacy on-disk
    # fields are ignored for backward compatibility; dispatch injects the
    # effective policy into NormalizedSample before encoding.


# JSON decoding ignores _comment, _scenario, and other unknown fields.
_decoder = msgspec.json.Decoder(SFTSample, strict=False)


def decode_sample(data: bytes | str) -> SFTSample:
    """Decode one JSON object into SFTSample."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _decoder.decode(data)


def decode_jsonl(path: str | Path) -> list[SFTSample]:
    """Decode one SFTSample per JSONL line."""
    samples = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(_decoder.decode(line))
    return samples


def decode_json_array(path: str | Path) -> list[SFTSample]:
    """Decode a JSON array of SFTSample records."""
    with open(path, "rb") as f:
        raw_list = json.load(f)
    return [decode_sample(json.dumps(item).encode()) for item in raw_list]


# stream_stats.json Pydantic models.

_stats_logger = logging.getLogger(__name__)

_STREAM_STATS_FILENAME = "stream_stats.json"
_TOKENIZERS_DIR = "tokenizers"
_CURRENT_STATS_VERSION = 3


class TokenizerStats(BaseModel):
    """Token statistics for one tokenizer."""
    total_tokens: int
    avg_tokens_per_sample: float
    created_at: str = ""
    max_token_length: int | None = None
    min_token_length: int | None = None
    token_length_dist: dict[str, int] | None = None
    fine_token_length_dist: dict[str, int] | None = None


class StreamStats(BaseModel):
    """Versioned stream_stats.json read/write model.

    Token metrics live under tokenizer_stats[name]; top-level fields provide
    sample count and the default tokenizer key.
    """
    version: int = _CURRENT_STATS_VERSION
    stream_id: str = ""
    stream_type: str = "pretrain"
    num_samples: int = 0
    created_at: str = ""

    default_tokenizer: str = ""
    tokenizer_stats: dict[str, TokenizerStats] = Field(default_factory=dict)

    @classmethod
    def load(cls, mds_dir: str) -> StreamStats | None:
        """Load v3 stream_stats.json, returning None if absent or invalid."""
        stats_path = os.path.join(mds_dir, _STREAM_STATS_FILENAME)
        try:
            with open(stats_path) as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict) or raw.get("version") != _CURRENT_STATS_VERSION:
            return None
        if raw.get("num_samples", 0) <= 0:
            return None
        return cls(**raw)

    def save(self, mds_dir: str) -> None:
        """Write stream_stats.json in v3 format."""
        stats_path = os.path.join(mds_dir, _STREAM_STATS_FILENAME)
        with open(stats_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def resolve_tokens(self, tokenizer_name: str | None = None) -> tuple[int, float]:
        """Return (total_tokens, avg_tokens_per_sample) for a tokenizer.

        Explicit names must exist in tokenizer_stats. None selects
        default_tokenizer; a missing entry returns (0, 0.0).
        """
        if tokenizer_name is not None:
            if tokenizer_name not in self.tokenizer_stats:
                available = list(self.tokenizer_stats.keys())
                raise ValueError(
                    f"tokenizer_name='{tokenizer_name}' 在 stream_stats.json 中不存在。"
                    f"可用的 tokenizer: {available}。"
                    f"修复：使用可用名称之一，或检查数据转换时的 --name 参数。"
                )
            entry = self.tokenizer_stats[tokenizer_name]
            return entry.total_tokens, entry.avg_tokens_per_sample
        if self.default_tokenizer and self.default_tokenizer in self.tokenizer_stats:
            entry = self.tokenizer_stats[self.default_tokenizer]
            return entry.total_tokens, entry.avg_tokens_per_sample
        return 0, 0.0

    def add_tokenizer_stats(
        self,
        tokenizer_name: str,
        total_tokens: int,
        created_at: str = "",
        *,
        max_token_length: int | None = None,
        min_token_length: int | None = None,
        token_length_dist: dict[str, int] | None = None,
        fine_token_length_dist: dict[str, int] | None = None,
    ) -> None:
        """Add or replace statistics for one tokenizer."""
        avg = total_tokens / max(self.num_samples, 1)
        self.tokenizer_stats[tokenizer_name] = TokenizerStats(
            total_tokens=total_tokens,
            avg_tokens_per_sample=avg,
            created_at=created_at,
            max_token_length=max_token_length,
            min_token_length=min_token_length,
            token_length_dist=token_length_dist,
            fine_token_length_dist=fine_token_length_dist,
        )


_WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}


def embed_tokenizer(tokenizer_path: str, mds_dir: str, tokenizer_name: str) -> None:
    """Copy tokenizer assets into an MDS directory, excluding model weights."""
    if not tokenizer_path or not os.path.isdir(tokenizer_path):
        _stats_logger.warning(f"tokenizer_path 不存在或不是目录: {tokenizer_path}，跳过嵌入")
        return
    dest = os.path.join(mds_dir, _TOKENIZERS_DIR, tokenizer_name)
    if os.path.exists(dest):
        shutil.rmtree(dest)

    def _ignore_weights(directory: str, contents: list[str]) -> set[str]:
        ignored = set()
        for name in contents:
            full_path = os.path.join(directory, name)
            if os.path.isfile(full_path):
                _, ext = os.path.splitext(name)
                if ext in _WEIGHT_EXTENSIONS:
                    ignored.add(name)
        return ignored

    shutil.copytree(tokenizer_path, dest, dirs_exist_ok=True, ignore=_ignore_weights)
