"""Stream-level data-source, packing, and mixing configuration.

Each StreamConfig declares one MDS stream. Text and message streams share the
same loader and scheduling API, while family selects the native SFT encoder."""


from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from titaninfra.config.train import TrainConfig

from pydantic import BaseModel, Field, model_validator

from titaninfra.data.schema import HistoryPolicy

# ============================================================================
# ============================================================================


class ResolutionPolicy(BaseModel):
    """Resolution policy configuration and behavior."""

    min_pixels: int = Field(default=65536, ge=1, description="最小像素预算（低于则放大）")
    max_pixels: int = Field(default=16777216, ge=1, description="最大像素预算（高于则缩小）")

    @model_validator(mode="after")
    def _check_order(self) -> "ResolutionPolicy":
        if self.min_pixels > self.max_pixels:
            raise ValueError(
                f"min_pixels({self.min_pixels}) 不能大于 max_pixels({self.max_pixels})"
            )
        return self


class PackStrategy(str, Enum):
    """Pack strategy configuration and behavior."""

    NO_PACK = "no_pack"
    FULL_SAMPLE = "full_sample"
    SENTENCE_BOUNDARY = "sentence_boundary"
    HARD_TRUNCATE = "hard_truncate"


# ============================================================================
# ============================================================================


class StreamConfig(BaseModel):
    """Declarative MDS stream configuration.

local and tokenizer_name are required. data_type is text or messages; messages
requires family. proportion, repeat, and choose are mutually exclusive and all
streams in one run must use the same mode. SFT rejects hard truncation."""

    model_config = {"arbitrary_types_allowed": True}

    local: str = Field(description="MDS 本地路径")
    remote: str | None = Field(default=None, description="MDS 远程路径")
    split: str | None = Field(default=None, description="数据集分片名称")
    cache_limit: str | None = Field(
        default=None,
        description=(
            "本流 local 缓存盘的上限（per-stream，作用于该流独立的 StreamingDataset）。\n"
            "remote+local 模式下，shard 流式下载到 local，超过此上限按 LRU 驱逐最久未用的\n"
            "shard，使 local 当滑动缓存窗口而非被单调填满。\n"
            "None（默认）= 禁用驱逐：shard 永久留 local，盘会单调增长直到装下所有访问过的\n"
            "shard（小 大数据 → [Errno 28] 撑爆）。多流数据量差异大时应按各流体量分别设\n"
            "（如 crawl-syn 设大、math_cc 设小）。\n"
            "约束（MosaicML 强制）：必须 > 本流 index 大小，且 ≥ 4×max_shard_size（16MB shard\n"
            "→ ≥64MB），否则启动 ValueError。支持 '100gb' / '10gb' / 整数字节。"
        ),
    )
    resolution_policy: ResolutionPolicy | None = Field(
        default=None,
        description=(
            "多模态 pretrain 专用：本流图像分辨率预算（min/max pixels）。\n"
            "None → 用 image_processor 默认（qwen3_5：65536 / 16777216）。\n"
            "支持分辨率递进（不同 stream 不同预算）。纯文本流忽略。"
        ),
    )

    data_type: Literal["text", "messages"] = Field(
        default="text",
        description=(
            "数据类型。\n"
            "- 'text': 纯文本，MDS 中读取 text_key 字段，encode 后 label=input\n"
            "- 'messages': 对话格式，MDS 中读取 messages_key 字段（JSON 字符串），"
            "apply_chat_template 后按角色 mask"
        ),
    )
    text_key: str = Field(default="text", description="text 模式下的 MDS 字段名")
    messages_key: str = Field(default="messages", description="messages 模式下的 MDS 字段名")

    family: str | None = Field(
        default=None,
        description=(
            "模型家族名（如 'qwen3' / 'llama3' / 'gemma4'）。\n"
            "决定 SFT pipeline 路由到哪一个 segment encoder（titaninfra/data/sft/encoders/）。\n"
            "None 时回退到 tokenizer_name。两者都为 None 时 fail-loud。"
        ),
    )
    thinking_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "<think>...</think> 内部 token 的 loss 权重缩放因子 (0.0~1.0)。\n"
            "None = 不缩放（thinking 和 answer 同等权重）。\n"
            "0.3 = thinking token loss 乘以 0.3。\n"
            "0.0 = thinking token 完全不参与 loss。\n"
            "覆盖 per-sample loss_config.type_overrides.thinking（StreamConfig 级优先）。\n\n"
            "⚠ 多轮对话注意事项（Qwen3）：HF Jinja 模板只保留最后一轮的 <think>，\n"
            "历史轮次的 reasoning_content 会被静默丢弃。因此 thinking_scale 仅作用于\n"
            "最后一轮 assistant 的 thinking 内容，历史轮次不受影响。\n"
            "thinking marker 从 ChatTemplate.thought_words 自动解析，支持 DeepSeek R1 等不同词表。"
        ),
    )
    thinking_mode: Literal["on", "off"] | None = Field(
        default=None,
        description=(
            "覆盖 per-sample thinking_intent.mode（StreamConfig 级优先）。\n"
            "None=用 per-sample 或 family 默认；on/off=强制整个 stream 的 thinking 分支。\n"
            "仅 data_type='messages' 有效。effort 维度不在 StreamConfig 暴露（per-sample only，\n"
            "因为只 gpt_oss 有 wire 级 effort dial）。\n"
            "优先级与 thinking_scale 一致：StreamConfig > per-sample。"
        ),
    )
    validate_template_alignment: bool = Field(
        default=False,
        description=(
            "数据管道层对齐检查：前 N 个样本验证 titaninfra tokenize 结果与 HF "
            "apply_chat_template(tokenize=True) 完全一致。\n"
            "True = 启用（推荐在 GPU test case 中使用）。\n"
            "False = 关闭（生产训练无需开启，已有离线测试覆盖）。"
        ),
    )

    sft_mask_mode: Literal["auto", "manual"] = Field(
        default="auto",
        description=(
            "SFT loss mask 模式。\n"
            "- auto: 所有 assistant 轮次算 loss，非 assistant 不算（默认）\n"
            "- manual: 完全使用数据中的 message.weight 标注"
        ),
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        description=(
            "最大对话轮数（None=不限制）。超过时从头部整轮丢弃，system 永远保留。\n"
            "截断按轮次粒度，不在消息中间硬切。"
        ),
    )
    truncate_at_last_segment: bool = Field(
        default=False,
        description=(
            "SFT-style segment-aligned truncation（仅 data_type='messages' 有效）。\n"
            "False（默认）：样本超过 max_length 时 fail-loud（ValueError），\n"
            "  推荐模式 — 超长样本应在离线转换阶段过滤。\n"
            "True：在最后一个完整 segment 边界处截断，并 emit warnings.warn。\n"
            "  适用于无法离线预过滤的数据，或需要强制适配固定 seq_len 的场景。\n"
            "  注意：截断会丢弃超长部分的 assistant 内容，影响 loss 分布。"
        ),
    )

    # === Pretrain special tokens ===
    add_bos_eos: bool = Field(
        default=True,
        description=(
            "Runtime pretrain tokenize 时是否在每条样本首尾分别插入 BOS / EOS token（仅 data_type='text' 有效）。\n"
            "False：token 流为 [t1, ..., tN]，不注入任何特殊 token。\n"
            "True（默认）：token 流为 [BOS, t1, ..., tN, EOS]，BOS / EOS 在 downstream 完全等同于普通 token。\n\n"
            "重要不变量（两种模式均成立）：\n"
            "  - 文档边界**始终**仅由 positions==0 信号传递给 FlexInnerAttention，与 BOS/EOS **无关**。\n"
            "  - Packer / mask / loss 不识别 BOS / EOS 作为任何特殊标记。\n"
            "  - 超长样本被切分时（包括 [BOS, ..., EOS] 跨 window 被切成两段），\n"
            "    按 packer 通用切分规则处理，BOS / EOS 不享有保护。\n\n"
            "⚠ 启用后注意：\n"
            "  1. 实际 token 数比 stream_stats.json 中的 avg_tokens_per_sample 多 2，\n"
            "     epoch_size 估算偏低，可适当增大 epoch_size_margin（如 0.1）避免 NCCL hang。\n"
            "  2. 该字段是 *runtime* 通道，与 DataConvertConfig.add_bos / add_eos\n"
            "     （离线 MDS 转换时的 token_count 统计开关）相互独立。\n"
            "  3. SFT (data_type='messages') 由 chat template / segment encoder 控制特殊 token，\n"
            "     此字段对 messages 流无效，设置非 False 会 fail-loud。"
        ),
    )

    # === Packing ===
    pack_strategy: PackStrategy = Field(
        default=PackStrategy.HARD_TRUNCATE,
        description=(
            "打包策略。\n"
            "- hard_truncate: 固定长度硬切（默认，最高效）\n"
            "- sentence_boundary: 标点/句子边界切分\n"
            "- full_sample: 保留完整样本，超长跨 batch（推荐 COT）\n"
            "- no_pack: 不打包，每个样本独立 pad/truncate（必须用 sdpa+causal）"
        ),
    )
    sentence_end_tokens: list[str] = Field(
        default_factory=list,
        description=(
            "句子边界 token 字符串列表（仅 sentence_boundary 策略使用）。\n"
            "示例：['。', '\\n', '.', '！', '？']"
        ),
    )

    proportion: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="比例权重（与 repeat/choose 互斥）",
    )
    repeat: float | None = Field(
        default=None, gt=0.0,
        description="绝对倍数（与 proportion/choose 互斥）",
    )
    choose: int | None = Field(
        default=None, gt=0,
        description="精确样本数（与 proportion/repeat 互斥）",
    )

    stream_id: str | None = Field(
        default=None,
        description="数据流 ID（metrics/日志用）。None 时从 local 路径推断。",
    )
    tokenizer_name: str = Field(
        description=(
            "tokenizer 短名（如 'qwen3'），必须与数据转换时 --name 参数一致。\n"
            "用于从 stream_stats.json 中精确选取对应的 tokenizer_stats 条目。"
        ),
    )

    epoch_size_margin: float = Field(
        default=0.05,
        ge=0.0,
        le=10.0,
        description=(
            "epoch_size 冗余因子。实际 epoch_size = 估算值 × (1 + margin)。\n"
            "覆盖 packing 尾部浪费、shard 分片波动、token 长度方差等。\n"
            "有 max_turns 截断的流建议增大（如 3.0），因为 stats 中的 avg_tokens\n"
            "是截断前的估算值，runtime 截断后实际 token 数更少。"
        ),
    )

    history_policy: HistoryPolicy | None = Field(
        default=None,
        description=(
            "历史轮（最后一个 user turn / keep_last_n 之前）的处理策略。\n"
            "None（默认）：走 family native 默认（qwen3/gpt_oss/gemma4 drop 历史\n"
            "  thinking 对齐推理；qwen2_5/dsv3/llama3 keep）。\n"
            "设置时：keep_last_n（保留最近 N 个 assistant 回合为 current）与\n"
            "  per_type（每类 block keep/drop）必须一起给（全或无防呆）。\n"
            "推荐压缩配置：{keep_last_n:1, per_type:{thinking:keep, tool_use:drop,\n"
            "  tool_result:drop}} — think 已含 tool 语义，丢 tool JSON 省 token。"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_history_policy(cls, data):
        """Coerce history policy."""
        if isinstance(data, dict):
            hp = data.get("history_policy")
            if isinstance(hp, dict):
                import msgspec
                data = {**data, "history_policy": msgspec.convert(hp, type=HistoryPolicy)}
        return data

    @model_validator(mode="after")
    def _validate_history_policy(self) -> "StreamConfig":
        """Validate the supplied configuration."""
        hp = self.history_policy
        if hp is None:
            return self
        errs: list[str] = []
        # insert_summary replaces the entire history region using the last-user
        # boundary. It does not use keep_last_n or per_type, but conflicts with
        # keep_last_n because their history boundaries differ.
        if hp.insert_summary:
            if hp.keep_last_n is not None:
                errs.append("keep_last_n 与 insert_summary 不能同时设置（边界语义冲突）")
            if errs:
                raise ValueError("StreamConfig.history_policy 配置非法:\n  - " + "\n  - ".join(errs))
            return self
        if hp.keep_last_n is None:
            errs.append("缺 keep_last_n；keep_last_n 与 per_type 必须一起给（防呆）")
        elif hp.keep_last_n < 1:
            errs.append(f"keep_last_n 必须 >=1，得到 {hp.keep_last_n}")
        if not hp.per_type:
            errs.append("缺 per_type；keep_last_n 与 per_type 必须一起给（防呆）")
        if errs:
            raise ValueError(
                "StreamConfig.history_policy 配置非法:\n  - " + "\n  - ".join(errs)
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _default_pack_strategy_for_sft(cls, data):
        """Default pack strategy for sft."""
        if isinstance(data, dict):
            if data.get("data_type") == "messages" and "pack_strategy" not in data:
                data = {**data, "pack_strategy": PackStrategy.FULL_SAMPLE}
        return data

    @model_validator(mode="after")
    def _validate_mixing_mode(self) -> "StreamConfig":
        """Validate the supplied configuration."""
        set_fields = [
            f for f in ("proportion", "repeat", "choose")
            if getattr(self, f) is not None
        ]
        if len(set_fields) > 1:
            raise ValueError(
                f"proportion/repeat/choose 互斥，不能同时设置: {set_fields}"
            )
        return self

    @model_validator(mode="after")
    def _validate_sentence_tokens(self) -> "StreamConfig":
        """Validate the supplied configuration."""
        if (
            self.sentence_end_tokens
            and self.pack_strategy != PackStrategy.SENTENCE_BOUNDARY
        ):
            raise ValueError(
                f"sentence_end_tokens 仅在 pack_strategy='sentence_boundary' 下有效，"
                f"当前 pack_strategy='{self.pack_strategy.value}'"
            )
        return self

    @model_validator(mode="after")
    def _forbid_hard_truncate_for_sft(self) -> "StreamConfig":
        """Forbid hard truncate for sft."""
        if (
            self.data_type == "messages"
            and self.pack_strategy == PackStrategy.HARD_TRUNCATE
        ):
            raise ValueError(
                "SFT 数据 (data_type='messages') 不能使用 "
                "pack_strategy='hard_truncate'：会切坏 chat template 边界。"
                "请改用 pack_strategy='full_sample'（超长样本会被 skip）。"
            )
        return self

    @model_validator(mode="after")
    def _forbid_truncate_at_last_segment_for_text(self) -> "StreamConfig":
        """Forbid truncate at last segment for text."""
        if self.truncate_at_last_segment and self.data_type != "messages":
            raise ValueError(
                "truncate_at_last_segment=True 只对 data_type='messages' 有效，"
                f"当前 data_type='{self.data_type}'。"
            )
        return self

    @model_validator(mode="after")
    def _forbid_add_bos_eos_for_messages(self) -> "StreamConfig":
        """Forbid add bos eos for messages."""
        if (
            "add_bos_eos" in self.model_fields_set
            and self.add_bos_eos
            and self.data_type != "text"
        ):
            raise ValueError(
                "add_bos_eos=True 仅对 data_type='text' 有效，"
                f"当前 data_type='{self.data_type}'。\n"
                "SFT (messages) 流的特殊 token 由 segment encoder 管理，"
                "请在数据转换或 chat template 层处理。"
            )
        return self

    def to_mosaicml_kwargs(self) -> dict:
        """To mosaicml kwargs."""
        d: dict = {"local": self.local}
        for key in ("remote", "split"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d

    def resolve_stream_id(self) -> str:
        """Resolve stream id."""
        if self.stream_id:
            return self.stream_id
        parts = self.local.rstrip("/").split("/")
        return parts[-1] if parts else "unknown"


def _mode_of(s: "StreamConfig") -> str:
    if s.proportion is not None:
        return "proportion"
    if s.repeat is not None:
        return "repeat"
    if s.choose is not None:
        return "choose"
    return "none"


def validate_streams_mixing_mode(streams: list["StreamConfig"]) -> None:
    """Validate the supplied configuration."""
    if len(streams) <= 1:
        return

    modes = {_mode_of(s) for s in streams}
    if len(modes) > 1:
        detail = ", ".join(
            f"{s.resolve_stream_id()}={_mode_of(s)}" for s in streams
        )
        raise ValueError(
            f"所有数据流必须使用同一种混合模式（proportion / repeat / choose / 全部不设）。\n"
            f"当前配置混用了多种模式: {detail}\n"
            f"原因：不同模式的权重量纲不同，混用导致归一化后权重失真。"
        )


def validate_scheduling_combination(
    streams: list["StreamConfig"],
    train_config: "TrainConfig",
) -> None:
    """Validate the supplied configuration."""
    modes = {_mode_of(s) for s in streams}
    mixing = modes.pop() if len(modes) == 1 else "none"
    if mixing == "none":
        mixing = "proportion"

    if mixing in ("choose", "repeat"):
        return

    if "steps" not in train_config.model_fields_set:
        raise ValueError(
            f"非法配置组合：proportion 模式未显式设置 steps。\n"
            f"当前 steps={train_config.steps}（默认值），系统无法确定训练长度。\n"
            "修复（任选其一）：\n"
            "  - 设置 steps=<总步数>（推荐预训练场景）\n"
            "  - 改用 repeat=N 让系统自动推导 total_steps\n"
            "  - 改用 choose=K 指定每流精确样本数\n\n"
            "合法组合：\n"
            "  choose + auto       → 每流精确 K 条（SFT）\n"
            "  repeat + auto       → 每流精确 R 遍（CPT）\n"
            "  proportion + steps  → 按 token 比例混合（Pretrain）"
        )
