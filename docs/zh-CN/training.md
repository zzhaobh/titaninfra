# 训练与完整 recipe

[English](../en/training.md) · [README](../../README.zh-CN.md)

我们使用同一个 TorchTitan 引擎执行 Pretrain、SFT 和 CPT。训练阶段由数据流决定，因此你可以在不同 recipe 中共用模型与分布式配置。

## 1. 准备 MDS 数据

按照[数据指南](data.md)生成文本和/或消息 MDS，并包含 `stream_stats.json`。有效训练目录必须包含 MDS 分片和 `index.json`，tokenizer 名称必须与 recipe 一致。

## 2. 完整 Pretrain recipe

打开 [examples/pretrain.py](../../examples/pretrain.py) 并替换中性路径。其 `build_config()` 返回 `(model, train, streams)`，没有副作用。

关键设置：

- 模型：`qwen3/8B`，`dp_shard=-1`，bf16 计算，fp32 master weight，flex + block-causal attention，选择性激活重计算。
- 优化器：AdamW，`lr=3e-4`，`beta1=0.9`，`beta2=0.95`，`eps=1e-8`，`weight_decay=0.1`，fp32 optimizer state。
- 调度：`steps=100_000`，`warmup_steps=2_000`，cosine decay，`min_lr_ratio=0.1`，`max_grad_norm=1.0`。
- 数据：两路文本流，`proportion=0.7/0.3`，`HARD_TRUNCATE`，16K 序列长度，显式 epoch 余量。
- Checkpoint：每 1,000 步保存一次，保留最近三份。

文件显式设置了全部必填字段：

```text
Model: dtype, param_dtype, attn_backend, attn_mask_type, ac_mode, compile
Train: local_batch_size, global_batch_size, seq_len, optimizer_name,
       lr, beta1, beta2, eps, weight_decay, optimizer_state_dtype,
       warmup_steps, min_lr_ratio, lr_decay_type, max_grad_norm
```

本地或分布式启动：

```bash
python examples/pretrain.py
# 分布式示例；请按节点实际 GPU 数调整：
torchrun --nproc_per_node=8 examples/pretrain.py
```

`proportion` 必须显式设置 `steps`，因为 token 比例本身不能决定训练应在何时结束。

## 3. 完整 SFT recipe

打开 [examples/sft.py](../../examples/sft.py)。该 recipe 使用原生 Qwen3 消息编码和 `repeat=3.0`，总步数由 MDS 统计自动推导。

关键设置：

- 模型和精度设置与 Pretrain 相同。
- AdamW 使用 `lr=1e-5`，fp32 master weight 和 optimizer state。
- 序列长度 8,192，200 warmup steps，cosine decay。
- 消息流使用 `family="qwen3"`、`data_type="messages"` 和 `FULL_SAMPLE` packing。
- 每 500 步保存 checkpoint。

```bash
python examples/sft.py
# 或：
torchrun --nproc_per_node=8 examples/sft.py
```

`FULL_SAMPLE` 保护 chat template 和 assistant loss 边界。不要把 SFT 流改为 `HARD_TRUNCATE`，配置校验会拒绝该设置。

## 4. CPT：混合文本和消息

CPT 使用同一引擎混合不同 stream 类型。所有流使用同一种调度模式，例如都使用 `repeat`：

```python
streams = [
    StreamConfig(
        local="/data/mds/general/train",
        tokenizer_name="qwen3",
        data_type="text",
        repeat=2.0,
        pack_strategy=PackStrategy.HARD_TRUNCATE,
    ),
    StreamConfig(
        local="/data/mds/sft/train",
        tokenizer_name="qwen3",
        data_type="messages",
        family="qwen3",
        repeat=1.0,
        pack_strategy=PackStrategy.FULL_SAMPLE,
    ),
]
```

不要在一次训练中混用 `proportion`、`repeat` 和 `choose`。

## 5. 分布式策略

`ModelConfig` 提供：

- `dp_replicate`：用于 HSDP 布局的数据复制。
- `dp_shard`：FSDP 分片，`-1` 根据 mesh 自动推导。
- `tp`：Tensor Parallelism。
- `cp`：用于长序列的 Context Parallelism。
- `ep`：用于 MoE 模型的 Expert Parallelism。

Pipeline Parallelism 不在当前支持范围内。存在某个配置字段或模型能力，不代表所有 GPU/模型组合都已验证。请先用 debug 模型验证目标拓扑、tokenizer 和 checkpoint 路径。

## 6. Checkpoint 恢复

Trainer 默认在 `dump_folder/ckpt_subdir` 下查找最新 DCP checkpoint。若要从其他 checkpoint 迁移，设置：

```python
train.resume_from = "/outputs/run/checkpoints/step-1000"
```

模型、优化器、调度器和 dataloader 状态可以一起恢复。只有在明确需要部分恢复时，才使用 `exclude_from_loading`。

## 7. 导出 Hugging Face 权重

训练后可将 DCP 转成 safetensors：

```python
from titaninfra import to_hf

to_hf(
    dcp_path="/outputs/run/checkpoints/step-1000",
    output_path="/exports/qwen3-8b",
    model_name="qwen3/8B",
    hf_assets_path="/models/Qwen3-8B",
    export_dtype="bfloat16",
    max_shard_size="5GB",
)
```

如果需要从 HF 权重以 DCP 形式开始训练，使用 `from_hf(...)`。详见 [API 参考](api.md)。

## 8. 未来 self-developed RL

自研 RL 模式仍在开发中，目前尚未提供 RL 配置、引擎、安装方式或 recipe。本版本支持的训练模式是 Pretrain、SFT 和 CPT。
