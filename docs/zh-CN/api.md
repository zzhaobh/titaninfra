# API 参考

[English](../en/api.md) · [README](../../README.zh-CN.md) · [配置参考](config.md)

我们为 TitanInfra 提供统一的数据转换、模型配置、训练和 checkpoint 管理 API。Pretrain、SFT 和 CPT 共用 `Trainer`。我们的自研 RL 模式仍在开发中，尚未提供公开 API。

## 根导出

```python
from titaninfra import (
    ModelConfig, TrainConfig, StreamConfig, PackStrategy,
    Trainer, TrainResult,
    to_hf, from_hf,
    download_hf_assets, compare_losses, diagnose_hf_checkpoint,
    get_template, list_templates, register_template,
    MoEHealthPlugin, MoEBiasSchedule,
)
```

## ModelConfig

`ModelConfig(name="qwen3/8B", ...)` 选择 `<family>/<flavor>` 及其分布式策略。

### 必填算法字段

| 字段 | 常用值 | 含义 |
|---|---|---|
| `dtype` | `"bf16"` | 前向/反向计算精度。 |
| `param_dtype` | `"fp32"` | 参数/master weight 存储精度。 |
| `attn_backend` | `"flex"` | Flex Attention 后端，是当前唯一合法后端。 |
| `attn_mask_type` | `"block_causal"` 或 `"causal"` | 文档打包掩码或标准 causal 掩码。 |
| `ac_mode` | `"none"`、`"selective"`、`"full"`、`"memory_budget"` | 激活重计算策略。 |
| `compile` | `False` | 是否启用 `torch.compile`。 |

常用可选字段包括 `dp_shard=-1`、`dp_replicate=1`、`tp=1`、`cp=1`、`ep=1`、`cpu_offload=False` 和 `kernel_impls=[]`。配置模型中虽然存在 `pp`，但 Pipeline Parallelism 不在当前支持范围内。

推荐以 `dtype="bf16"`、`param_dtype="fp32"`、`optimizer_state_dtype="fp32"` 作为混合精度起点。纯 bf16 master weight 可能丢失很小的优化器更新。

## TrainConfig

`TrainConfig` 定义优化器、学习率调度、运行时和 checkpoint 行为。

### 必填字段

```text
local_batch_size, global_batch_size, seq_len,
optimizer_name, lr, beta1, beta2, eps, weight_decay,
optimizer_state_dtype, min_lr_ratio, lr_decay_type, max_grad_norm
```

还必须显式选择 `warmup_steps` 或 `warmup_ratio` 之一。`proportion` 调度必须显式设置 `steps`；`repeat` 和 `choose` 会从数据流推导总步数。

常用可选字段包括 `ckpt_interval`、`keep_latest_k`、`resume_from`、`dump_folder`、`task_name`、`log_interval`、`seed`、`last_save_model_only` 和 `last_save_in_hf`。

## StreamConfig

每个流独立声明 MDS 数据源、tokenizer、数据类型、packing 和混合方式。

```python
StreamConfig(
    local="/data/mds/general/train",
    tokenizer_name="qwen3",
    data_type="text",
    proportion=0.7,
    pack_strategy=PackStrategy.HARD_TRUNCATE,
)
```

必填字段：

- `local`：本地 MDS 目录或缓存目录。
- `tokenizer_name`：必须与转换时写入 `stream_stats.json` 的 key 一致。

重要字段：

- `data_type`：`"text"`（默认）或 `"messages"`。
- `family`：原生消息编码必填；支持 `qwen3`、`qwen2_5`、`llama3`、`deepseek_v3`、`deepseek_v4`、`gpt_oss` 和 `gemma4` 等。
- `remote`：可选远程 MDS 数据源，与 `local` 节点缓存配合使用。
- `pack_strategy`：`HARD_TRUNCATE`、`SENTENCE_BOUNDARY`、`FULL_SAMPLE` 或 `NO_PACK`。
- `proportion`、`repeat`、`choose`：互斥的混合控制；同一次训练的所有流必须使用同一种模式。
- `epoch_size_margin`：用于 token 长度方差和 packing 尾部的安全余量。
- `history_policy`：可选的逐 block 类型 history `keep`/`drop`/family 支持的 `summarize` 和 `keep_last_n`；配置在流上，而不是 sample 上。`insert_summary=True` 会用上游提供的 summary block 替换 history，且不能与 `keep_last_n` 同用。
- `add_bos_eos`：仅文本运行时选项；不要在消息流上显式设置为 `True`。

消息流默认使用 `FULL_SAMPLE`；`HARD_TRUNCATE` 会被拒绝，因为它可能切断 chat template 与 loss mask 的边界。

## PackStrategy

| 值 | 行为 |
|---|---|
| `HARD_TRUNCATE` | 固定长度 token 窗口，利用率最高，仅用于文本。 |
| `SENTENCE_BOUNDARY` | 优先在标点或句子边界切分。 |
| `FULL_SAMPLE` | 尽可能保留完整样本，是常规 SFT 选择。 |
| `NO_PACK` | 每个序列只放一个样本，进行 padding/truncation。 |

## Trainer 与 TrainResult

```python
result = Trainer(
    model=model,
    train=train,
    streams=streams,
    hf_ckpt_path="/models/Qwen3-8B",  # 可选
    task_name="my-run",               # 可选
)
```

对于 `TrainConfig + list[StreamConfig]`，`Trainer` 先校验调度与精度兼容性，再调用 TorchTitan 训练引擎。纯文本流执行 Pretrain，消息流执行 SFT，文本/消息混合流执行 CPT。

`TrainResult` 字段：

- `stage: str`
- `ckpt_dir: str`
- `steps: int`
- `model_name: str`

## Checkpoint 转换

```python
to_hf(
    dcp_path="/outputs/run/checkpoints/step-1000",
    output_path="/exports/qwen3-8b",
    model_name="qwen3/8B",
    hf_assets_path="/models/Qwen3-8B",
    export_dtype="float16",
    max_shard_size="5GB",
)

from_hf(
    hf_path="/models/Qwen3-8B",
    output_path="/outputs/qwen3-8b-dcp",
    model_name="qwen3/8B",
)
```

`to_hf` 必须提供 `hf_assets_path`，用于 tokenizer/config 资产以及 HF index 映射。

## 资产与诊断工具

- `download_hf_assets(repo_id, local_dir="./assets/hf", asset_types="tokenizer", download_all=False, hf_token=None, additional_patterns=None) -> str`
- `compare_losses(log_dirs, labels=None, metric="loss", output_path=None, show=True) -> dict`
- `diagnose_hf_checkpoint(hf_path, model_name=None, verbose=False) -> DiagnosisResult`

## 模板与监控

- `get_template`、`list_templates`、`register_template` 管理已注册的 prompt 模板。
- `MoEHealthPlugin` 提供 MoE 专家利用率监控 hook。
- `MoEBiasSchedule(calibration_steps=0, calibration_multiplier=50.0, freeze_after=None)` 配置 expert bias 校准和可选的末期冻结。

## MDS 格式转换 API

先安装固定版本 TorchTitan 和 `requirements.txt`，再在 MosaicML Streaming
源码目录执行 `python -m pip install --no-build-isolation --no-deps .`，最后执行
`python -m pip install --no-deps -e ".[data]"` 安装数据运行时。

### `DataConvertConfig` 与 `DataConverter`

```python
from titaninfra.data.converter import DataConvertConfig, DataConverter

config = DataConvertConfig(
    source="/data/raw/pretrain.jsonl",       # 必填：文件或目录
    output="/data/mds/general/train",        # 必填
    tokenizer_path="/models/Qwen3-8B",
    tokenizer_name="qwen3",                 # 必须与 StreamConfig 一致
    text_key="text",
    add_bos=True,
    add_eos=True,
    num_shards=16,
    num_workers=4,
    checkpoint_interval=50,
    resume=True,
    overwrite=False,
)
stats = DataConverter(config).convert()
```

`source` 和 `output` 必填。Pretrain 应显式提供 `tokenizer_path` 与
`tokenizer_name`，以便 `stream_stats.json` 记录精确 token 统计。常用可选字段：

| 分组 | 字段 |
|---|---|
| 输入选择 | `file_format`、`columns`、`field_mapping`、`text_key`、`row_filter` |
| 输出/schema | `output_format="mds"`、`mds_columns`、`compression`、`size_limit`、`hashes`、`keep_local` |
| Tokenize | `add_bos=True`、`add_eos=True`、`store_token_count=False`、`num_tokenizer_workers` |
| 并行转换 | `num_shards=16`、`num_workers=4`、`max_workers`、`batch_size=1000`、`retry=2`、`preload` |
| 恢复与安全 | `checkpoint_interval`、`resume=False`、`overwrite=False`、`validate_after_convert=True`、`validate_sample_ratio=0.01`、`validate_min_samples=2` |
| 流元数据 | `stream_id`、`stream_type="pretrain"`；消息感知转换使用 `family` |

`convert() -> dict[str, Any]` 返回聚合统计，并写出均衡 MDS shard、
`index.json` 和 `stream_stats.json`。已有输出不会被静默覆盖。远程输出遵循
`OutputPath` 约定，`keep_local` 控制是否保留本地副本。

### `convert_messages_to_mds`

```python
from titaninfra.data.converter import convert_messages_to_mds

stats = convert_messages_to_mds(
    source="/data/raw/sft.jsonl",
    output="/data/mds/sft/train",
    tokenizer_path="/models/Qwen3-8B",
    tokenizer_name="qwen3",
    family="qwen3",
    messages_key="messages",
    stream_id="instruction_v1",
    num_shards=16,
    num_workers=4,
    checkpoint_interval=50,
    resume=True,
    overwrite=False,
)
```

`source`、`output`、`tokenizer_path` 和 `family` 必填。`family` 必须指定训练
使用的原生 SFT encoder，从而保证转换和运行时的 token 数、协议标记及 loss
mask 一致。省略 `tokenizer_name` 时会从 tokenizer 目录名推导，但建议显式
设置为与 `StreamConfig.tokenizer_name` 相同的值。

其他参数包括 `num_tokenizer_workers`、`batch_size=1000`、`queue_size=4096`、
`row_filter`、`preload`、`compression`、`message_converter_fn`、
`size_limit="256MB"`、`resolution_policy` 和 `spatial_merge_size`。
`message_converter_fn(record) -> record` 可在原生 tokenize 前把特定数据源
schema 归一化。恢复转换使用 `checkpoint_interval` 与 `resume=True`，不要与
`overwrite` 语义误用。

MDS 列为 `messages`、`stream_id`、`token_count`、`loss_config`、`system`
和 `tools`。返回字典包含样本/token/shard 统计，同样的数据写入
`stream_stats.json`。

## Message 格式转换 API

完整 schema、loss/history 规则、能力矩阵、MDS 契约和原生编码管线见
[Message 系统完整规范](message-system.md)。

以下函数位于 `titaninfra.data.sft.message_converter`，只转换消息结构，
不负责 tokenize 或写入 MDS。

### `normalize_to_hf_messages`

```python
from titaninfra.data.sft.message_converter import normalize_to_hf_messages

hf_messages, weights = normalize_to_hf_messages(
    messages=rich_messages,
    system="You are a helpful assistant.",
)
```

返回 `(hf_messages, weights)`。Text/thinking block 渲染为 HF `content`，
推理文本可保留到 `reasoning_content`，`tool_use` 转为 HF `tool_calls`，
工具结果 ID 使用 `tool_call_id`。`weights` 与返回消息逐条对齐；传入的
system 会插入第一条且权重为 0。这是兼容格式转换，不是最终逐 token loss mask。

### `to_ir_messages`

```python
from titaninfra.data.sft.message_converter import to_ir_messages

rich_messages = to_ir_messages(hf_messages)
```

将 HF 或宽松格式消息转回 TitanInfra `blocks` IR。已经是 IR 的消息会保持
不变；HF `tool_calls`、`tool_call_id`、`reasoning_content` 和文本 content
会转成对应 rich block。结果可放入 SFT sample 的 `messages` 字段，再交给
`convert_messages_to_mds` 或 `tokenize_sample`。

完整 schema、block 表、loss weight 瀑布、history policy 和 tool-loop 示例见
[Message 系统与数据格式](data.md#message-系统-schema)。
