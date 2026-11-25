# 弹性数据管道与格式

[English](../en/data.md) · [README](../../README.zh-CN.md) · [Message 系统完整规范](message-system.md)

我们把每种训练输入都视为声明式数据流。Pretrain、SFT 和 CPT 共用同一套逐流契约：转换、统计、packing、混合、分布式分片与恢复训练。

## 端到端流程

```text
原始 JSONL/Parquet
  → MDS 转换 + stream_stats.json
  → 每流 StreamingDataset
  → 每流 packing
  → StreamMixer
  → Trainer 与 DCP checkpoint
```

`stream_stats.json` 保存 token 数量和长度分布。运行时利用这些统计估算 packing 效率、epoch size 和总步数，而不是根据文件数或字节数猜测。

## Pretrain 文本输入

JSONL 文本记录默认包含 `text` 字段：

```json
{"text": "Language models learn statistical structure from data."}
```

转换方式：

```python
from titaninfra.data.converter import DataConvertConfig, DataConverter

config = DataConvertConfig(
    source="/data/raw/pretrain.jsonl",
    output="/data/mds/general/train",
    tokenizer_path="/models/Qwen3-8B",
    tokenizer_name="qwen3",
    text_key="text",
    num_shards=16,
    num_workers=4,
    checkpoint_interval=50,
    resume=True,
    overwrite=False,
)
stats = DataConverter(config).convert()
```

`source` 可以是文件或目录。`output` 会生成 MDS 分片、`index.json` 和 `stream_stats.json`。`tokenizer_name` 必须与后续 `StreamConfig.tokenizer_name` 一致。

文本流运行时可以设置 `add_bos_eos=True`。Attention 的文档边界来自 `positions == 0`，不是 BOS/EOS token。

## SFT 消息输入

**我们的 Message 系统是为训练而生的。** 它承载原生 family encoder、loss
mask、history policy、精确 token 统计和 packing 所需的信息，而不只是聊天界面
中显示的文本。

消息使用富 block，而不是把所有内容塞进单个字符串：

```json
{
  "system": "You are a helpful assistant.",
  "messages": [
    {
      "role": "user",
      "blocks": [{"type": "text", "text": "Solve 2+2."}]
    },
    {
      "role": "assistant",
      "blocks": [
        {"type": "thinking", "text": "I should add the numbers."},
        {"type": "text", "text": "4"}
      ]
    }
  ]
}
```

### Message 系统 schema

规范 schema 定义在 `titaninfra.data.schema`，通过 `decode_sample` /
`decode_jsonl` 校验。新数据集建议使用 `blocks`；旧 `content` 和 HF 风格
字段会在 normalize 阶段兼容转换，但 rich blocks 是稳定的训练表示。

| 字段 | 含义 |
|---|---|
| `messages` | 有序的 `Message`，包含 `role`、`blocks` 和可选 `weight`、`name`、`task`。 |
| `system` | 可选顶层 system prompt，默认不参与 loss。 |
| `tools` | 工具声明，字段为 `name`、`description` 和 JSON Schema `input_schema`。 |
| `loss_config.type_overrides` | 按 block 类型设置 loss 倍率。 |
| `thinking_intent` | 样本级 `mode: on/off` 和 `effort: low/medium/high`。 |

支持的 block：

| Block | 必需内容 | 默认 loss 行为 |
|---|---|---|
| `text` | `text` | Assistant 文本参与训练；system/user/tool 文本默认屏蔽。 |
| `thinking` | `text` | 推理内容，使用同一套 weight 瀑布。 |
| `tool_use` | `id`、`name`、`input` 对象 | 与后续 `tool_result.tool_use_id` 配对。 |
| `tool_result` | `tool_use_id`、`text`，可选 `is_error` | 工具输出默认不参与 loss。 |
| `image` | `source`，例如 `{type: base64, data: ...}` | 图像 token 不参与 loss。 |
| `history_summary` | `text`，可选 `compressed_turn_count`、`style` | 上游提供的压缩历史；normalize 不会自行生成摘要。 |

完整 tool-loop 示例：

```json
{
  "tools": [{
    "name": "search",
    "description": "Search an index",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
  }],
  "messages": [
    {"role": "user", "blocks": [{"type": "text", "text": "Find the answer."}]},
    {"role": "assistant", "blocks": [
      {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "TitanInfra"}}
    ]},
    {"role": "tool", "blocks": [
      {"type": "tool_result", "tool_use_id": "call_1", "text": "TitanInfra documentation"}
    ]},
    {"role": "assistant", "blocks": [{"type": "text", "text": "Use the documentation."}]}
  ]
}
```

### Loss weight 与 history

原生训练路径分两层应用 loss policy：

- **`sft_mask_mode="auto"`（默认）：** assistant 消息权重为 1，其他 role
  为 0；显式 `weight: 0` 始终生效。非零的自定义 assistant 权重会归一化为 1。
- **`sft_mask_mode="manual"`：** 逐消息权重原样保留，因此 `0.5` 等连续值
  可以缩放该消息的 loss 贡献。
- **`loss_config.type_overrides`：** family encoder 可以缩放特定 block
  类型，例如 thinking 或 tool-call segment。是否支持取决于 family，不能把它
  当作适用于所有 block 的通用倍率。

`block.weight` 虽然存在于 schema，但不是原生训练路径的通用开关。只有明确
支持逐 block 权重的 family 才能使用它；其他 family 会警告并忽略。直接调用
`tokenize_sample` 时还可以传入 `train_on_inputs=True`，使所有消息权重为 1；
`StreamConfig` 不提供这个开关。

History policy 属于 **`StreamConfig`**，不是落盘 sample。可在 recipe 中配置
`per_type` 动作（`keep`、`drop`，或 family 支持的 `summarize`）和
`keep_last_n`。未显式配置时，Qwen3、GPT-OSS 和 Gemma4 默认丢弃历史
thinking，其他原生 family 默认保留。Current-turn block 不受修改。
`insert_summary=True` 会将整个 history 区域替换为上游提供的
`HistorySummaryBlock` 内容，不能与 `keep_last_n` 同用；该路径不消费
`per_type`。如果没有 summary block，history 区域会被丢弃。

### 校验与直接使用

```python
import json
from titaninfra.data.schema import decode_sample
from titaninfra.data.sft.dispatch import tokenize_sample
from transformers import AutoTokenizer

raw = open("/data/raw/one_sft_record.json").read()
decode_sample(raw)  # schema 校验
encoded = tokenize_sample(
    json.loads(raw),
    tokenizer=AutoTokenizer.from_pretrained("/models/Qwen3-8B"),
    family="qwen3",
)
```

工具 ID 配对错误、不支持的 block 结构、空 assistant 内容都会明确失败，
不会静默生成零 loss 样本。`thinking_intent` 作用于整个样本：`mode` 控制
推理分支；`effort` 仅 GPT-OSS 使用，其他 family 会警告并忽略。

使用与训练一致的原生 family encoder 转换：

```python
from titaninfra.data.converter import convert_messages_to_mds

stats = convert_messages_to_mds(
    source="/data/raw/sft.jsonl",
    output="/data/mds/sft/train",
    tokenizer_path="/models/Qwen3-8B",
    tokenizer_name="qwen3",
    family="qwen3",
    messages_key="messages",
    num_shards=16,
    overwrite=True,
)
```

原生 family 包括 `qwen3`、`qwen2_5`、`llama3`、`deepseek_v3`、`deepseek_v4`、`gpt_oss` 和 `gemma4`。Encoder 决定协议标记、assistant loss、thinking/tool block 和家族专属渲染；生产训练不会回退到 Jinja 渲染。

工具调用样本把 `tools` 放在样本顶层，`tool_use.id` 与 `tool_result.tool_use_id` 配对。可选 `loss_config.type_overrides` 可以缩放或屏蔽某类 block 的 loss。

## 两层 packing

1. **离线分片均衡：** 转换阶段按 token 量分配 MDS shard，降低分片间差异。
2. **运行时序列 packing：** 每个流应用自己的 `PackStrategy`，并生成在文档边界重置的 `positions`。

| 策略 | 用途 |
|---|---|
| `HARD_TRUNCATE` | 高利用率的文本预训练。 |
| `SENTENCE_BOUNDARY` | 优先保持句子边界的文本窗口。 |
| `FULL_SAMPLE` | SFT 和长推理样本，保留模板结构。 |
| `NO_PACK` | 每个序列一个样本，用于特殊实验。 |

消息流拒绝 `HARD_TRUNCATE`。`epoch_size_margin` 为 token 长度方差和 packing 尾部提供余量；使用截断或样本长度差异很大时可适当增大。

## 混合与调度

同一次训练中的所有流必须使用相同模式。

| 模式 | 配置 | 步数计算 |
|---|---|---|
| `proportion` | `StreamConfig(proportion=...)` + 显式 `TrainConfig.steps` | 按 token 权重采样固定步数。 |
| `repeat` | `StreamConfig(repeat=...)` | 根据每流 epoch 和重复倍数推导总步数。 |
| `choose` | `StreamConfig(choose=...)` | 精确抽取指定样本数并推导总步数。 |

估算器使用平均 token、细粒度长度直方图、packing 效率、local batch size、梯度累积、序列长度和 DP world size。`choose`/`repeat` 不能与 `proportion` 混用。

## 分布式数据分片

TorchTitan 使用 `dp_replicate × dp_shard` 计算数据并行度。TitanInfra 将 MosaicML 看到的 rank/world size 映射到该 DP 网格，使同一 DP 数据列中的 TP/CP rank 看到相同样本。所有 rank 根据同一统计计算 epoch，避免迭代长度分叉导致集合同步死锁。

`remote + local` 允许每个节点把 MDS shard 缓存到本地，适合共享 NFS/S3 数据源。生成的缓存不是训练 checkpoint，不应提交到仓库。

## Checkpoint 与恢复

DCP resume 可以恢复模型、优化器、LR scheduler、训练状态和 dataloader 状态。默认从 `dump_folder/ckpt_subdir` 查找最新 checkpoint；`resume_from` 可以选择其他路径用于迁移。只有在明确需要时，才通过 `exclude_from_loading` 排除受支持的状态类别。

## 测试夹具

公开分支不跟踪可下载的 tokenizer 文件。通过以下命令生成测试素材：

```bash
python tests/fixtures/tokenizers/download_all.py
# 或只下载指定家族：
python tests/fixtures/tokenizers/download_all.py qwen3 qwen2_5
```

`HF_ENDPOINT` 可以覆盖下载端点。部分上游仓库需要访问授权或可能不可用，这些外部 fixture 是集成测试前提，不是随仓库分发的模型权重。单元测试需要的小型合成数据集保留在 `tests/fixtures/`。
