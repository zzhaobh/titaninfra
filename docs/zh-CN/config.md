# 配置参考

[English](../en/config.md) · [README](../../README.zh-CN.md) · [API 参考](api.md)

我们使用三个 Pydantic 配置类描述一次训练。算法相关字段必须显式写出，
让 recipe 成为可复现的规格，而不是依赖隐藏默认值。

| 配置类 | 职责 |
|---|---|
| `ModelConfig` | 模型 family/flavor、精度、attention、激活重计算和并行策略。 |
| `TrainConfig` | 优化器、LR 调度、batch/step、checkpoint、验证、监控和 profiling。 |
| `StreamConfig` | MDS 数据源、tokenizer、文本/消息行为、packing、混合、history 和 epoch sizing。 |

辅助类型包括 `MoEBiasSchedule`、`DataConvertConfig`、`TrainResult`，以及
[数据格式](data.md)中的 Message schema。恢复训练使用当前新配置，checkpoint
提供权重和状态，不会静默覆盖 recipe。

## ModelConfig

### 核心字段

| 字段 | 默认值 | 含义 |
|---|---|---|
| `name` | 必填 | `family/flavor`，如 `qwen3/8B` 或 `deepseek_v3/debugmodel`。 |
| `dtype` | 必填 | 前向/反向计算精度，标准 recipe 使用 `bf16`。 |
| `param_dtype` | 必填 | 参数/master weight 存储精度，标准混合精度使用 `fp32`。 |
| `attn_backend` | 必填 | `flex`，当前唯一支持的 attention 后端。 |
| `attn_mask_type` | 必填 | 普通 causal 使用 `causal`，打包文档使用 `block_causal`。 |
| `ac_mode` | 必填 | `none`、`selective`、`full` 或 `memory_budget`。 |
| `compile` | 必填 | `True` 时启用 `torch.compile`。 |

`dtype`、`param_dtype` 和 `TrainConfig.optimizer_state_dtype` 相互独立：
计算精度、master 权重和 optimizer moment 不应由一个开关隐式决定。推荐起点
是 bf16 计算 + fp32 master + fp32 moment。纯 bf16 master 可能把小更新舍入为零。
`dtype="mxfp8"` 在支持的 Hopper+ GPU 上启用 MXFP8 linear transform，计算基底
仍为 bf16，不改变 master weight 的语义。

### 并行与显存

| 字段 | 默认值 | 含义 |
|---|---|---|
| `dp_shard` | `-1` | FSDP shard 度，`-1` 由设备 mesh 推导。 |
| `dp_replicate` | `1` | HSDP/DDP 复制度；`dp_replicate × dp_shard` 是 DP world size。 |
| `tp` | `1` | Tensor Parallel 度。 |
| `pp` | `1` | Pipeline Parallel 字段保留，但 PP 不在当前支持范围内。 |
| `cp` | `1` | 长序列 Context Parallel 度。 |
| `ep` | `1` | MoE Expert Parallel 度。 |
| `ep_backend` | `"standard"` | Expert 通信后端；`deepep`/`hybridep` 需要支持的硬件和库。 |
| `fsdp_reshard_after_forward` | `"default"` | FSDP reshard 策略；`always` 偏显存，`never` 偏通信。 |
| `fsdp_explicit_prefetch` | `False` | 在支持的路径上显式启用 FSDP 前向/反向预取。 |
| `pipeline_parallel_schedule` | `"1F1B"` | PP 调度字段，不代表当前支持 PP。 |
| `context_parallel_load_balancer` | `"ptrr"` | FlexAttention 的 CP 负载均衡策略。 |
| `enable_async_tensor_parallel` | `False` | 异步 TP，需要 `compile=True`。 |
| `ac_memory_budget` | `0.5` | `ac_mode="memory_budget"` 且 `compile=True` 时的显存/速度平衡。 |
| `cpu_offload` | `False` | 将 optimizer state offload 到 CPU。 |
| `kernel_impls` | `[]` | 按算子名启用已注册自定义 kernel，未知名称会明确报错。 |

Batch 语义：

```text
dp_world_size = dp_replicate × dp_shard
gradient_accumulation_steps = global_batch_size / (local_batch_size × dp_world_size)
tokens_per_step = global_batch_size × seq_len
```

### MoE 模型选项

| 字段 | 默认值 | 含义 |
|---|---|---|
| `moe_load_balance_coeff` | `None` | 覆盖家族声明的 expert-bias 步长；不能给未声明均衡机制的家族强行启用它。 |
| `moe_aux_loss_coeff` | `None` | 覆盖家族声明的 aux-loss 权重，作用域来自模型 balance spec。 |
| `moe_top_k` | `None` | 覆盖每个 token 选择的 expert 数。 |

均衡机制是模型家族属性，不是可以任意打开的通用开关。详见下方 MoE 部分。

## TrainConfig

### 必填字段

```text
local_batch_size、global_batch_size、seq_len、
optimizer_name、lr、beta1、beta2、eps、weight_decay、
optimizer_state_dtype、min_lr_ratio、lr_decay_type、max_grad_norm
```

`warmup_steps` 和 `warmup_ratio` 必须二选一。`proportion` 模式必须显式给出
`steps`；`repeat` 和 `choose` 从流统计推导步数。

### 优化器与 LR 调度

| 字段 | 默认值 | 含义 |
|---|---|---|
| `steps` | `100` | 总 optimizer steps；`proportion` 必须显式给出。 |
| `optimizer_implementation` | `"fused"` | `fused`、`foreach` 或 `for-loop`，GPU 通常使用 fused。 |
| `optimizer_state_dtype` | 必填 | 通常 `fp32`；`bf16` moment 要求 `param_dtype="fp32"`。 |
| `warmup_steps` | `200` | 显式 warmup 步数，也可改用 `warmup_ratio`。 |
| `warmup_ratio` | `None` | `(0, 1]` 的 warmup 比例，设置后覆盖按步数指定的 warmup。 |
| `min_lr_ratio` | 必填 | 最终 LR 相对 base LR 的比例。 |
| `lr_decay_type` | 必填 | `cosine`、`linear`、`sqrt` 或 `constant`。 |
| `decay_ratio` | `None` | Warmup-Stable-Decay 比例，例如 `0.2` 表示最后 20% 步数才衰减。 |
| `max_grad_norm` | 必填 | 梯度裁剪阈值，`0` 表示不裁剪。 |
| `chunked_loss` | `False` | 分块计算 loss，降低 logits 峰值显存。 |
| `loss_num_chunks` | `1` | 分块 loss 的序列段数，常用 4–8。 |
| `neftune_noise_alpha` | `None` | NEFTune embedding 噪声强度 α，SFT 常用 5–15。 |

NEFTune 使用 `Embedding(input) + Uniform(-1, 1) × α / √(L × d)`，其中 `L`
是序列长度，`d` 是 hidden size。可与 router 冻结或分层 LR 倍数组合。

### MuonAdamW

MuonAdamW 将符合条件的 2D linear 权重交给 Muon，其余参数交给 AdamW。
公开控制项如下：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `muon_lr` | `0.02` | Muon 学习率，通常大于 AdamW LR。 |
| `muon_momentum` | `0.95` | 动量 EMA 系数。 |
| `muon_nesterov` | `True` | 在 Newton–Schulz 正交化前使用 Nesterov 动量。 |
| `muon_ns_steps` | `5` | Newton–Schulz 迭代次数，范围 1–20。 |
| `muon_weight_decay` | `0.0` | Muon 解耦权重衰减。 |
| `muon_min_ndim_size` | `4096` | 2D 权重进入 Muon 的最小 numel。 |
| `muon_include_router` | `False` | router/gate 默认走 AdamW；是否改走 Muon 取决于模型。 |
| `muon_ns_dtype` | `"bf16"` | Newton–Schulz 计算精度，也可选 `fp32`。 |
| `muon_scale_factor` | `1.0` | 更新缩放：KellerJordan 用 `1.0`，Moonlight/Kimi K2 用 `0.2`。 |
| `muon_gather_mode` | `False` | 对 2D shard 聚合成全矩阵做精确 NS，而不是 per-shard NS。 |
| `muon_qk_clip_tau` | `None` | 启用 QK-Clip 的阈值 τ，常用 30–100。 |

两种常见配方是独立 Muon LR（`muon_lr=0.02`、`muon_scale_factor=1.0`）
和共享 LR（`muon_lr == lr`、`muon_scale_factor=0.2`）。除非已经专门验证
其他精度配方，否则保持 `optimizer_state_dtype="fp32"` 和
`param_dtype="fp32"`。

### 参数组

`freeze_params` 完全禁止匹配参数更新，`layer_lr_overrides` 则只调整其 LR
倍数，两者都跟随 LR scheduler。

```python
freeze_params=["router"]
layer_lr_overrides={"embed_tokens": 0.1, "norm": 5.0}
```

常用 pattern 包括 `router`、`embed_tokens`、`output`、`norm` 和 `experts`。
SFT 保护预训练 router 时用 `freeze_params`；希望慢速适应而不是完全冻结时，
使用 `layer_lr_overrides`。

### MoE 健康度与 bias 调度

| 字段 | 默认值 | 含义 |
|---|---|---|
| `enable_moe_health_monitor` | `False` | 采集 expert 利用率、CV、不均衡和 dead-expert 指标。 |
| `moe_health_ep_reduce_interval` | `0` | EP 全局归约间隔，`0` 仅保留本地指标。 |
| `enable_sv_entropy_monitor` | `False` | 采样 Muon 权重并上报奇异值熵。 |
| `sv_entropy_interval` | `100` | 采样间隔，必须能整除 `log_interval`。 |
| `sv_entropy_max_layers` | `4` | 每次最多采样的层数。 |
| `moe_bias_schedule` | `None` | `MoEBiasSchedule(calibration_steps, calibration_multiplier, freeze_after)`。 |

`MoEBiasSchedule` 有三个字段：

- `calibration_steps=0`：参数 `lr=0` 的校准步数，让 expert bias 加速收敛；
- `calibration_multiplier=50.0`：校准期间放大家族 bias γ；
- `freeze_after=None`：可选 1-based 步数，到达后停止更新 expert bias。

HF 冷启动不包含 expert bias，需要从 0 校准；DCP resume 会从 checkpoint
恢复 bias，通常无需重新校准。只有家族声明了 bias 机制，γ 加速才有意义。

### 验证、数据加载与 checkpoint

| 字段 | 默认值 | 含义 |
|---|---|---|
| `val_streams` | `None` | 与训练相同的 `StreamConfig` 验证流。 |
| `eval_freq` | `500` | 验证间隔（optimizer steps）。 |
| `eval_steps` | `-1` | 使用 `val_streams` 时必须为正数。 |
| `tokenizer_path` | `None` | tokenizer 路径，通常与模型资产目录一致。 |
| `num_workers` | `1` | DataLoader worker 数，多流训练不要随意增大。 |
| `shuffle` | `False` | 是否启用 StreamingDataset shuffle。 |
| `shuffle_seed` | `9176` | 数据 shuffle 种子，与模型 `seed` 不同。 |
| `persistent_workers` | `False` | worker > 0 时跨 epoch 保持 worker。 |
| `pin_memory` | `True` | 使用 CUDA pinned memory 加速主机到 GPU 拷贝。 |
| `predownload` | `None` | StreamingDataset 预下载 sample 数。 |
| `shuffle_block_size` | `None` | 可选 shuffle block 大小。 |
| `shuffle_algo` | `"py1br"` | 适合恢复的 MosaicML shuffle 算法。 |
| `skip_corrupted_samples` | `True` | 对损坏样本警告并跳过，而不是立即失败。 |
| `max_consecutive_errors` | `10` | 连续样本错误达到该值后停止。 |
| `ckpt_interval` | `500` | checkpoint 保存间隔。 |
| `keep_latest_k` | `3` | 保留最近 k 个 checkpoint；`0` 全保留，不支持 `1`。 |
| `resume_from` | `None` | 用于迁移的可选 checkpoint 路径，而不是默认加载最新 checkpoint。 |
| `exclude_from_loading` | `[]` | 跳过的状态类别：`optimizer`、`lr_scheduler`、`dataloader`、`train_state`。 |
| `async_checkpoint` | `"disabled"` | `disabled`、`async` 或 `async_with_pinned_mem`。 |
| `last_save_model_only` | `False` | 结束时仅保存模型权重。 |
| `last_save_in_hf` | `False` | 同时导出最终 HF safetensors，要求 `last_save_model_only=True`。 |
| `export_dtype` | `"float32"` | HF 导出精度：`float16`、`bfloat16` 或 `float32`。 |
| `enable_first_step_checkpoint` | `False` | step 1 保存一次，提前验证 checkpoint 存储。 |

### 日志与 profiling

| 字段 | 默认值 | 含义 |
|---|---|---|
| `seed` | `None` | 可选全局训练随机种子。 |
| `log_interval` | `10` | 指标/日志间隔。 |
| `dump_folder` | `./outputs` | 输出根目录。 |
| `task_name` | `None` | 输出和 checkpoint 目录的可选后缀。 |
| `comm_init_timeout_seconds` | `3600` | 进程组初始化超时。 |
| `comm_train_timeout_seconds` | `3600` | 训练集合通信超时。 |
| `fsdp_gas_no_sync` | `False` | 在支持的 non-PP GAS>1 路径上，仅最后一个 microbatch 做跨 replica all-reduce。 |
| `enable_wandb` | `False` | 启用 Weights & Biases。 |
| `wandb_project/entity/run_name/run_id` | `None` | WandB 项目、团队、显示名和恢复 ID。 |
| `wandb_tags/notes` | `None` | 可选逗号分隔标签和备注。 |
| `wandb_mode` | `"offline"` | `offline`、`dryrun` 或 `online`。 |
| `enable_profiling` | `False` | 启用 torch.profiler trace。 |
| `profile_freq` | `100` | trace cycle 频率。 |
| `profile_active` | `2` | 每 cycle 活跃采集步数。 |
| `profile_warmup` | `3` | 正式采集前 warmup 步数。 |
| `profile_skip_first` | `5` | 首轮前跳过的步数。 |
| `profile_export_dir` | `"profile_traces"` | `dump_folder` 下的 trace 输出子目录。 |
| `profile_memory` | `False` | OOM 诊断时启用 memory snapshot。 |
| `profile_with_stack` | `False` | 记录 Python 调用栈，开销较大。 |
| `profile_with_flops` | `True` | 采集 FLOPs 估算用于 MFU 分析。 |

## StreamConfig

### 数据源与 tokenization

| 字段 | 默认值 | 含义 |
|---|---|---|
| `local` | 必填 | 本地 MDS 目录或本地缓存目录。 |
| `remote` | `None` | 可选远程 MDS 数据源，与 `local` 缓存配合。 |
| `split` | `None` | 数据集 split 名称。 |
| `cache_limit` | `None` | 可选本地缓存上限，需满足 index 和 shard 空间要求。 |
| `tokenizer_name` | 必填 | 与 `stream_stats.json.tokenizer_stats` 匹配的 key。 |
| `data_type` | `"text"` | `text` 或 `messages`。 |
| `text_key` | `"text"` | MDS 中的文本字段。 |
| `messages_key` | `"messages"` | MDS 中的 JSON messages 字段。 |
| `family` | `None` | 原生 Message encoder 家族，省略时回退到 `tokenizer_name`。 |
| `resolution_policy` | `None` | 可选多模态图像分辨率预算。 |
| `add_bos_eos` | `True` | 仅文本流插入 BOS/EOS；消息流不要显式设为 `True`。 |
| `validate_template_alignment` | `False` | 诊断用：将前若干样本与 HF tokenize 结果比较。 |

文本流和消息流共享转换、调度、packing 与恢复机制。Message tokenization
使用原生 family encoder，不回退到 Jinja 训练渲染。文档边界来自
`positions == 0`，不是 BOS/EOS。

### Packing、混合与 history

| 字段 | 默认值 | 含义 |
|---|---|---|
| `pack_strategy` | `HARD_TRUNCATE` | `HARD_TRUNCATE`、`SENTENCE_BOUNDARY`、`FULL_SAMPLE` 或 `NO_PACK`。 |
| `sentence_end_tokens` | `[]` | `SENTENCE_BOUNDARY` 的自定义句子结束符。 |
| `proportion` | `None` | 按 token 权重分配比例，需要显式 `TrainConfig.steps`。 |
| `repeat` | `None` | 重复倍数，从流统计推导总步数。 |
| `choose` | `None` | 精确样本数，从流统计推导总步数。 |
| `stream_id` | `None` | metrics/log ID，省略时从 `local` 推断。 |
| `epoch_size_margin` | `0.05` | epoch sizing 膨胀系数，防止 rank 数据分叉。 |
| `sft_mask_mode` | `"auto"` | assistant 二值 loss，显式 0 始终生效；`manual` 保留连续消息权重。 |
| `thinking_scale` | `None` | 流级 thinking loss 缩放，覆盖 sample 的 thinking override。 |
| `thinking_mode` | `None` | 流级 `on`/`off`，覆盖 thinking 分支。 |
| `max_turns` | `None` | 只保留最后 N 轮，保留 system 且不按消息中间切分。 |
| `truncate_at_last_segment` | `False` | 允许按 segment 边界截断并告警，而不是超长即失败。 |
| `history_policy` | `None` | 按类型 `keep`/`drop`/`summarize`、`keep_last_n` 或 `insert_summary`。 |

同一训练中的所有流必须使用相同混合模式。`insert_summary=True` 用上游提供的
`HistorySummaryBlock` 替换 history，与 `keep_last_n` 互斥，且不消费
`per_type`。其他 history policy 需要同时给出 `keep_last_n` 和 `per_type`。
Current-turn block 不会被修改。小数据、shard 不均衡或 turn 截断时增大
`epoch_size_margin`，`max_turns` 可从 `3.0–5.0` 起步。

## DataConvertConfig 速查

`DataConvertConfig(source, output, ...)` 配置 `DataConverter.convert()`。
`source` 和 `output` 必填，主要字段分组如下：

| 分组 | 字段 |
|---|---|
| 输入 | `file_format`、`columns`、`field_mapping`、`text_key`、`row_filter` |
| 输出 | `output_format="mds"`、`mds_columns`、`compression`、`size_limit`、`hashes`、`keep_local` |
| Tokenization | `tokenizer_path`、`tokenizer_name`、`add_bos`、`add_eos`、`store_token_count`、`num_tokenizer_workers` |
| 并行转换 | `num_shards=16`、`num_workers=4`、`max_workers`、`batch_size=1000`、`retry=2`、`preload` |
| 恢复与安全 | `checkpoint_interval`、`resume=False`、`overwrite=False`、`validate_after_convert=True`、`validate_sample_ratio=0.01`、`validate_min_samples=2` |
| 元数据 | `stream_id`、`stream_type="pretrain"`、`family` |

消息数据应使用 `convert_messages_to_mds(...)`，不要直接套文本转换器。
其必填参数为 `source`、`output`、`tokenizer_path` 和 `family`；
`tokenizer_name` 应显式设置并与 `StreamConfig` 一致。完整签名和 Message
格式转换函数见 [MDS 与 Message 转换 API](api.md#mds-格式转换-api)。
