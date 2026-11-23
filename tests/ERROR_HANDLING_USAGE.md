# Streaming 错误处理功能使用指南

## 快速开始

### 1. 启用错误处理（推荐）

```python
from titaninfra.data.streaming_dataset import StreamingTextDataLoader

config = StreamingTextDataLoader.Config(
    streams=[
        {"local": "/data/zh", "proportion": 0.7},
        {"local": "/data/en", "proportion": 0.3},
    ],

    # 错误处理配置（生产环境推荐）
    skip_corrupted_samples=True,   # 自动跳过损坏样本
    max_consecutive_errors=20,     # 连续错误阈值
)

dataloader = config.build(
    dp_rank=0,
    dp_world_size=8,
    tokenizer=tokenizer,
    seq_len=4096,
    local_batch_size=8,
)
```

### 2. 监控数据质量

```python
# 训练循环中监控错误
for batch_idx, batch in enumerate(dataloader):
    # 正常训练
    loss = train_step(model, optimizer, batch)

    # 定期检查错误统计
    if batch_idx % 100 == 0:
        error_stats = dataloader.get_error_stats()
        total_errors = sum(error_stats.values())

        if total_errors > 0:
            logger.warning(f"检测到 {total_errors} 个数据错误: {error_stats}")

# 训练结束后生成报告
error_stats = dataloader.get_error_stats()
total_samples = dataloader.dataset.get_num_original_samples()
error_rate = sum(error_stats.values()) / total_samples * 100

print(f"\n数据质量报告:")
print(f"总样本数: {total_samples}")
print(f"错误数: {sum(error_stats.values())}")
print(f"错误率: {error_rate:.4f}%")
print(f"错误分布: {error_stats}")
```

## 配置说明

### skip_corrupted_samples

- **类型**: `bool`
- **默认值**: `True`
- **说明**: 是否自动跳过损坏样本
- **建议**:
  - 生产环境：`True`（防止训练中断）
  - 调试阶段：`False`（快速发现数据问题）

### max_consecutive_errors

- **类型**: `int`
- **默认值**: `10`
- **说明**: 连续错误阈值，超过则终止训练
- **目的**: 防止因数据严重损坏导致无限循环
- **建议**:
  - 大规模训练：`10-20`（容错性强）
  - 小数据集：`5-10`（快速失败）

## 错误类型

### 样本级错误（自动跳过）

| 错误类型 | 描述 | 示例 |
|----------|------|------|
| `null_sample` | 样本为 `None` | `None` |
| `empty_dict` | 样本为空字典 | `{}` |
| `empty_text` | 文本字段为空 | `{"text": ""}` |
| `missing_text_key` | 缺少文本字段 | `{"content": "..."}` |
| `tokenization` | Tokenization 失败 | 编码异常 |
| `processing` | 数据处理失败 | 打包异常 |

### Shard 级错误

| 错误类型 | 描述 | 触发条件 |
|----------|------|----------|
| `shard_errors` | Shard 损坏 | MDS 文件损坏 |
| `iterator_errors` | 迭代器错误 | 顶层异常 |

## 故障排查

### 问题 1：错误率过高

**诊断步骤**：

```bash
# 1. 运行诊断工具
python diagnose_mds_error.py /path/to/streaming_data

# 2. 检查诊断报告
cat /path/to/streaming_data/diagnosis_report.json
```

**解决方案**：

```python
# 重新生成数据，启用完整性校验
from titaninfra.data import convert_to_mds

result = convert_to_mds(
    source="data.parquet",
    output="./streaming_data",

    # 启用校验
    validate_shards=True,
    validate_hashes=True,
    validate_samples=True,
    max_samples_to_validate=500,

    # 覆盖已有数据
    overwrite=True,
)
```

### 问题 2：训练意外终止

**原因**：连续错误超过阈值

**处理方式**：

```python
try:
    for batch in dataloader:
        train_step(batch)
except RuntimeError as e:
    if "数据损坏严重" in str(e):
        logger.error("数据严重损坏，保存检查点后退出")
        save_checkpoint()
        # 可选：重新生成数据后继续
    else:
        raise
```

## 监控集成

### WandB 集成

```python
import wandb

wandb.init(project="training", name="pretrain")

if batch_idx % 100 == 0:
    error_stats = dataloader.get_error_stats()
    wandb.log({
        'data/total_errors': sum(error_stats.values()),
        'data/null_samples': error_stats.get('null_sample', 0),
        'data/empty_text': error_stats.get('empty_text', 0),
        'data/shard_errors': error_stats.get('shard_errors', 0),
    })
```

### TensorBoard 集成

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

if batch_idx % 100 == 0:
    error_stats = dataloader.get_error_stats()
    writer.add_scalar('data/total_errors', sum(error_stats.values()), batch_idx)
    writer.add_scalar('data/shard_errors', error_stats.get('shard_errors', 0), batch_idx)
```

## 运行测试

```bash
# 运行所有错误处理测试
pytest tests/test_streaming_error_handling.py -v

# 运行特定测试
pytest tests/test_streaming_error_handling.py::test_sample_error_skipping -v

# 运行集成测试（需要真实数据）
pytest -m integration tests/test_streaming_error_handling.py
```

## 最佳实践

### 1. 数据预处理阶段

```python
# 在生成 MDS 数据时添加验证
def validate_text(text: str) -> bool:
    """验证文本有效性"""
    if not text or not text.strip():
        return False
    if len(text) < 10:  # 过短文本
        return False
    return True

# 过滤无效样本
valid_samples = [s for s in samples if validate_text(s.get('text', ''))]
```

### 2. 训练配置

```python
# 推荐配置
config = StreamingTextDataLoader.Config(
    streams=[...],
    skip_corrupted_samples=True,      # 启用错误跳过
    max_consecutive_errors=20,        # 适当的容错阈值
    num_workers=4,                    # 多 worker 加速
    persistent_workers=True,          # 保持 worker 存活
)
```

### 3. 检查点恢复

```python
# 恢复训练时也恢复错误统计
checkpoint = torch.load('checkpoint.pth')
dataloader.load_state_dict(checkpoint['dataloader'])

# 可选：重置错误统计
# dataloader.reset_error_stats()
```

## 性能影响

| 场景 | 性能影响 | 说明 |
|------|----------|------|
| 正常数据 | < 0.1% | 仅增加验证步骤 |
| 损坏数据 | < 1ms/sample | 跳过速度极快 |
| 内存占用 | < 100 个样本 | 损坏样本缓存限制 |
| 日志开销 | 可忽略 | 仅 WARNING 级别 |
