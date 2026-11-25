# TitanInfra Message 系统规范

[English](../en/message-system.md) · [README](../../README.zh-CN.md) · [API 参考](api.md) · [数据管道](data.md)

**我们的 Message 系统是为训练而生的。** 它是 SFT 和 Pretrain+SFT 混合训练的
规范表示，不仅承载文本，还承载推理、工具、图像、loss policy 和 history
语义。

本文完整描述当前公开的 Message IR、原生编码路径和 MDS 集成。函数签名也可
在 [API 参考](api.md#message-格式转换-api)中查阅。

## 1. 范围与设计原则

Message 系统覆盖：

- 规范的 `SFTSample` 与 `Message` schema；
- text、thinking、tool、image 和 history-summary blocks；
- assistant loss mask，以及受支持的逐消息/逐 block 权重；
- `thinking_intent`、history policy 和 turn 截断；
- 原生 per-family 编码与能力检查；
- MDS 转换、直接 tokenize、packing 和校验。

生产训练路径使用原生 Python encoder。Jinja 不作为训练渲染器，只在适用的
测试路径中作为 oracle。本版本不提供公开 RL 引擎或 RL Message API。

核心不变量是：

> **Token 与其 loss 语义必须在同一条路径中共同生成。**

不要先渲染字符串，再通过字符串匹配反推 assistant mask。

## 2. 规范对象模型

### `SFTSample`

```python
SFTSample(
    messages=[...],                 # 必填，有序 Message 列表
    system="You are helpful.",     # 可选顶层 system prompt
    tools=[...],                    # 可选工具声明
    loss_config=LossConfig(...),    # 可选 block 类型覆盖
    thinking_intent=ThinkingIntent(...),
)
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `messages` | `list[Message]` | 有序对话轮次，必填且非空。 |
| `system` | `str | None` | 顶层 system prompt，默认不参与 loss。 |
| `tools` | `list[Tool] | None` | 由原生 family encoder 渲染的工具声明。 |
| `loss_config` | `LossConfig` | 通过 `type_overrides` 设置 block 类型 loss 覆盖。 |
| `thinking_intent` | `ThinkingIntent | None` | 样本级推理 mode 和 effort。 |

`history_policy` 刻意配置在 **`StreamConfig`**，而不是每条 sample 上。这样
同一份落盘数据可以在不同 recipe 中复用。

### `Message`

```python
Message(
    role="assistant",
    blocks=[...],
    weight=1.0,       # 可选
    name=None,        # 可选兼容字段
    task=None,        # DSv4 专用 action/task 转换标记
)
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `role` | `str` | 通常是 `system`、`user`、`assistant` 或 `tool`；DSv4 内部还建模 `developer` 和 `latest_reminder`。 |
| `blocks` | `list[Block]` | 有序富内容 blocks。 |
| `weight` | `float | None` | 可选消息级权重；规范 normalized message 契约要求其归一化到 `[0, 1]`。 |
| `name` | `str | None` | 可选兼容名称字段。 |
| `task` | `str | None` | DSv4 专用 task 标签，例如 `action`。 |

新数据集应使用 `blocks`。旧 `content` 和 HF 风格字段仍由 normalize 和转换
路径兼容处理，但 rich blocks 是稳定的训练表示。

## 3. Block 类型

### `TextBlock`

```json
{"type": "text", "text": "Solve 2+2.", "weight": null}
```

普通文本。Assistant 文本通常参与训练，system/user/tool 文本通常被屏蔽。

### `ThinkingBlock`

```json
{"type": "thinking", "text": "Add the numbers.", "weight": null}
```

推理内容，根据最终 policy 和 family 支持决定是否参与 loss。不要在 `text`
中自行加入 `<think>` 标签，wire envelope 由原生 encoder 负责。

### `ToolUseBlock`

```json
{
  "type": "tool_use",
  "id": "call_1",
  "name": "search",
  "input": {"query": "TitanInfra"},
  "weight": null
}
```

Assistant 发起的工具调用。`id` 与后续 tool result 配对；`input` 是字典，
不是 JSON 字符串。

### `ToolResultBlock`

```json
{
  "type": "tool_result",
  "tool_use_id": "call_1",
  "text": "TitanInfra documentation",
  "is_error": false,
  "weight": null
}
```

工具返回结果，默认不参与 loss。`tool_use_id` 必须匹配此前的
`ToolUseBlock.id`。

### `ImageBlock`

```json
{
  "type": "image",
  "source": {"type": "path", "data": "/data/example.jpg"},
  "weight": null
}
```

图像引用。内部处理路径支持 path、bytes、base64 或受支持的 PIL-like 对象。
图像 token 不参与 loss，也不会被当成待生成文本。当前公开原生图像支持
取决于 family，Qwen3.5 是 VLM 路径。

### `HistorySummaryBlock`

```json
{
  "type": "history_summary",
  "text": "Earlier turns established the task constraints.",
  "compressed_turn_count": 3,
  "style": "system_prefix",
  "weight": null
}
```

上游生成的历史摘要。`normalize` 不会自行创造摘要文本。`style` 可取
`latest_reminder`、`system_prefix` 或 `ephemeral_block`，是否生效取决于
family。`compressed_turn_count` 记录摘要替代的历史轮数。

### Block weight 不是通用能力

Block schema 中虽然存在 `weight`，但它不是原生训练路径的通用开关。只有
目标 family 明确声明 per-block loss 支持时才能使用；否则 dispatch 会警告，
encoder 可能忽略它。文档推荐的训练路径使用 message weight、
`sft_mask_mode` 和受支持的 `type_overrides`。

## 4. Loss 与 mask 语义

### `sft_mask_mode="auto"`

默认行为：

- assistant 消息权重为 1；
- system、user、tool 消息权重为 0；
- 显式 `weight: 0` 始终生效；
- 非零自定义 assistant 权重会归一化为 1。

因此 auto 是 assistant-only 二值 loss，唯一例外是显式屏蔽。

### `sft_mask_mode="manual"`

逐消息权重原样保留。`0.5` 等连续值可以缩放消息贡献。规范 normalized
message 契约要求权重位于 `[0, 1]`。

### `train_on_inputs=True`

直接调用 `tokenize_sample` 和底层工具路径时，可以把包括 prompt 在内的所有
消息权重设为 1。`StreamConfig` 不提供这个开关。

### `loss_config.type_overrides`

Family encoder 可以应用 block 类型倍率，例如：

```json
{"loss_config": {"type_overrides": {"thinking": 0.5, "tool_use": 0.0}}}
```

是否支持取决于 family。不能假设所有 block 类型或所有 family 都支持相同
override。

### Labels 与 positions

最终 `EncodedSample` 包含：

- `input_ids`：token ID；
- `labels`：参与 loss 的位置为 token ID，否则为 `IGNORE_INDEX = -100`；
- `positions`：每个文档开头为 `0`，文档内递增；
- 可选 `loss_weights`：当前 policy 提供的逐 token 倍率；
- 可选 `pixel_values` 和 `grid_thw`：多模态输入。

Padding 属于 packer/dataloader，使用 `-1` position，encoder 不做 padding。
FlexAttention 文档 mask 由下游根据 `positions == 0` 推导，而不是使用
BOS/EOS 作为文档边界。

## 5. Thinking intent

```json
{
  "thinking_intent": {"mode": "on", "effort": "high"},
  "messages": [...]
}
```

| 字段 | 取值 | 语义 |
|---|---|---|
| `mode` | `on`、`off` 或省略 | 在 family 支持时控制样本推理分支开关。 |
| `effort` | `low`、`medium`、`high` 或省略 | 推理深度。**当前仅 GPT-OSS 消费该字段**，其他 family 会警告并忽略。 |

该 intent 作用于整个样本，而不是单条 message。这与推理时选择下一次回答
的推理行为相符，而不是逐条修改历史消息。

## 6. History policy

在 `StreamConfig(history_policy=...)` 上配置：

```python
history_policy = {
    "keep_last_n": 1,
    "per_type": {
        "thinking": {"action": "drop"},
        "tool_result": {"action": "drop"},
    },
}
```

### 字段

| 字段 | 含义 |
|---|---|
| `per_type` | 将 block 类型映射到 `HistoryAction(action="keep" | "drop" | "summarize")`；未声明类型默认 keep。 |
| `keep_last_n` | 将最近 N 个 assistant 回合保留为 current，更早轮次为 history。 |
| `insert_summary` | 用上游提供的 `HistorySummaryBlock` 内容替换整个 history 区域。 |

`summarize` 需要 `summary_template` 和 family 支持。Current-turn block 不受
修改，history turn 的权重为 0。

`insert_summary=True` 与 `keep_last_n` 互斥，且不消费 `per_type`。如果源数据
没有 summary block，history 区域会被丢弃。其他 history policy 需要同时给出
`keep_last_n` 和 `per_type`，避免隐藏继承。

未显式配置时，Qwen3、GPT-OSS 和 Gemma4 默认丢弃历史 thinking；Qwen2.5、
Llama3、DeepSeek V3 和 DeepSeek V4 默认保留。这些家族默认值属于训推兼容
契约，不应静默修改。

## 7. Tool 声明与转换

### Tool schema

```python
Tool(
    name="search",
    description="Search an index",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
)
```

字段名是 `input_schema`，不是 `parameters`。原生 encoder 会将其投影到该
family 所需的声明位置和格式。

### Rich → HF messages

```python
from titaninfra.data.sft.message_converter import normalize_to_hf_messages

hf_messages, weights = normalize_to_hf_messages(
    rich_messages,
    system="You are a helpful assistant.",
)
```

返回 `(hf_messages, weights)`。Text/thinking block 渲染为 HF content，推理
内容可保留在 `reasoning_content`，tool call 转为 HF `tool_calls`，tool
result 使用 `tool_call_id`。返回的 weights 与消息逐条对齐，不是最终逐 token
mask。

### HF messages → rich IR

```python
from titaninfra.data.sft.message_converter import to_ir_messages

rich_messages = to_ir_messages(hf_messages)
```

HF text、`reasoning_content`、`tool_calls` 和 `tool_call_id` 会转换为 rich
blocks。反向转换保留普通 `thinking + text` 的顺序和内容，不会将其合并成
单个 text block。

这些函数只转换消息结构，不负责 tokenize 或写 MDS。

## 8. 原生 encoder family 与能力边界

每个原生 encoder 声明其支持的 blocks。下表用于导航，feature 级声明还会
控制 tools、并行调用、history 和 reasoning 行为。

| Family | Text | Thinking | Tool use/result | Image |
|---|---|---|---|---|
| Qwen3 | 是 | 是 | 是 | 否 |
| Qwen2.5 | 是 | 否 | 否 | 否 |
| Llama3 | 是 | 否 | 否 | 否 |
| DeepSeek V3 | 是 | 否 | 是 | 否 |
| DeepSeek V4 | 是 | 是 | 是 | 否 |
| GPT-OSS | 是 | 是 | 是 | 否 |
| Gemma4 | 是 | 是 | 是 | 否 |
| Qwen3.5 | 是 | 是 | 是 | 是 |

其他 feature 声明覆盖 system prompt、multi-turn、parallel tool calls、
thinking history、intent mode/effort、history summary 风格和权重覆盖。
Sample 使用不支持的 feature 时，dispatch 会按约定警告或报错，而不是静默
假装支持。

新增 family 需要实现原生 encoder、声明 blocks/features，并添加 oracle/parity
覆盖。不要通过通用 Jinja fallback 绕过这些工作。

## 9. 运行时编码管线

```text
raw sample dict
  → dispatch.tokenize_sample
      → block/feature 能力检查
      → normalize 为 NormalizedSample
      → LossPolicy.resolve_weights
      → 原生 family router/encoder
      → Segment 列表
      → Assembler
  → EncodedSample
```

### Normalized 结构

`NormalizedSample` 携带规范化消息、system、tools、thinking intent 和
history policy。`NormalizedMessage` 携带 role、blocks、解析后的 weight、
turn role 和可选 task。这些结构不可变，使非法 encoder 输出在边界处失败。

### Segment 与 token class

`Segment` 描述一个 wire-level 单元，包括：

- text 或预编码 token IDs；
- kind：`MARKER`、`STOP_MARKER`、`CONTENT`、`THINKING`、`TOOL_CALL`、
  `TOOL_RESULT`、`SYSTEM`、`TOOLS_HEADER` 或 `IMAGE`；
- role、generation flag 和 token class（`PROTOCOL`、`DECODED`、`METADATA`）；
- scale factor 和派生的 `loss_scale`。

结构性 base mask 只训练模型自行解码的 assistant content、thinking、tool
call 和 stop marker。Protocol framing、tool result、system 文本和图像被
屏蔽。Metadata 与 protocol token 不同，不能静默当作可训练内容。

Assembler 在 tokenize 前合并兼容的相邻 text segments，以保留 BPE 边界，
然后输出对齐的 IDs、labels、positions 和可选权重。除非调用方显式允许
segment 对齐截断，否则截断会明确失败。

## 10. MDS 转换

`convert_messages_to_mds` 使用与运行时训练相同的原生 encoder 写入 rich
messages，保证 token 统计与 loss mask 共用同一实现。

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

必填参数为 `source`、`output`、`tokenizer_path` 和 `family`。
`tokenizer_name` 应显式设置为与 `StreamConfig.tokenizer_name` 一致。

MDS 列：

| 列 | 内容 |
|---|---|
| `messages` | JSON 消息负载 |
| `stream_id` | 数据流 ID |
| `token_count` | 原生 token 数 |
| `loss_config` | 序列化 loss 覆盖 |
| `system` | 序列化 system prompt |
| `tools` | 序列化工具声明 |

返回字典包含 sample/token/shard 统计，同样的数据写入
`stream_stats.json`，供调度使用。

## 11. 直接 tokenize_sample API

```python
tokenize_sample(
    sample_json,
    tokenizer,
    family,
    *,
    max_length=None,
    loss_config=None,
    truncate_at_last_segment=False,
    sft_mask_mode="auto",
    train_on_inputs=False,
    history_policy=None,
    resolution_policy=None,
    spatial_merge_size=None,
    decoded_images=None,
) -> EncodedSample
```

`sample_json` 是原始字典，不是已解码的 `SFTSample` 对象。需要直接 tokenize
时，先用 `decode_sample` 做 schema 校验，再把 `json.loads(raw)` 传给
`tokenize_sample`。

## 12. 校验与失败行为

系统会拒绝或报告非法输入，不会静默生成零 loss 样本：

- schema 或 block 字段错误会在 decode 时失败；
- 不支持的 block 按能力规则警告并裁剪；
- 裁剪后所有 assistant 内容为空时抛出 `EmptyAssistantError`；
- 非法 tool ID 配对会失败；
- 未显式启用 segment 对齐截断时，超长 segment 会失败；
- 首个 segment 超过 `max_length` 时抛出 `TruncatedToEmptyError`，由 streaming
  调用方警告并跳过。

不要通过在原生管线之外渲染字符串绕过这些检查。

## 13. 完整示例

### 推理 SFT

```json
{
  "system": "You are a careful assistant.",
  "thinking_intent": {"mode": "on", "effort": "high"},
  "messages": [
    {"role": "user", "blocks": [{"type": "text", "text": "Solve 2+2."}]},
    {"role": "assistant", "blocks": [
      {"type": "thinking", "text": "Add the numbers."},
      {"type": "text", "text": "4"}
    ]}
  ]
}
```

当前版本中，`effort` 仅对 GPT-OSS 生效。

### Tool loop

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

### 图文交错消息

```json
{
  "role": "user",
  "blocks": [
    {"type": "text", "text": "Compare these images."},
    {"type": "image", "source": {"type": "path", "data": "/data/first.jpg"}},
    {"type": "text", "text": "Then describe the second."},
    {"type": "image", "source": {"type": "path", "data": "/data/second.jpg"}}
  ]
}
```

## 14. 兼容与扩展规则

- 新字段以追加方式加入，不改变既有字段含义。
- Family 专属 wire 行为保留在该 family encoder 内。
- 新能力必须显式声明，并为每种行为添加测试。
- 按 token position 保持 `input_ids`、`labels`、`positions` 和可选
  `loss_weights` 对齐。
- 保证转换时统计与运行时原生编码一致。
- 跨环境比较行为时，固定同一 TitanInfra 和 TorchTitan revision。

使用说明继续阅读[数据格式](data.md)、[API 参考](api.md)和
[训练 recipe](training.md)。设计动机见[研究说明](../en/research.md)。
