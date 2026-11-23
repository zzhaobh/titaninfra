# Training and Complete Recipes

[简体中文](../zh-CN/training.md) · [README](../../README.md)

We use one TorchTitan-backed engine for Pretrain, SFT, and CPT. The data streams determine the training stage, so you can keep the same model and distributed configuration across recipes.

## 1. Prepare MDS data

Use the [data guide](data.md) to produce text and/or message MDS with `stream_stats.json`. A valid training directory contains MDS shards and `index.json`; the tokenizer name must match the recipe.

## 2. Complete Pretrain recipe

Open [examples/pretrain.py](../../examples/pretrain.py) and replace the neutral paths. Its `build_config()` returns `(model, train, streams)` without side effects.

Key settings:

- Model: `qwen3/8B`, `dp_shard=-1`, bf16 compute, fp32 master weights, flex + block-causal attention, selective activation recomputation.
- Optimizer: AdamW, `lr=3e-4`, `beta1=0.9`, `beta2=0.95`, `eps=1e-8`, `weight_decay=0.1`, fp32 optimizer states.
- Schedule: `steps=100_000`, `warmup_steps=2_000`, cosine decay, `min_lr_ratio=0.1`, `max_grad_norm=1.0`.
- Data: two text streams with `proportion=0.7/0.3`, `HARD_TRUNCATE`, 16K sequence length, and explicit epoch margins.
- Checkpoints: every 1,000 steps with the latest three retained.

The file explicitly sets all required fields:

```text
Model: dtype, param_dtype, attn_backend, attn_mask_type, ac_mode, compile
Train: local_batch_size, global_batch_size, seq_len, optimizer_name,
       lr, beta1, beta2, eps, weight_decay, optimizer_state_dtype,
       warmup_steps, min_lr_ratio, lr_decay_type, max_grad_norm
```

Run locally or through a distributed launcher:

```bash
python examples/pretrain.py
# Example distributed launch; choose the count for your node:
torchrun --nproc_per_node=8 examples/pretrain.py
```

`proportion` requires an explicit `steps` value. This is intentional: token proportions do not determine when training should stop.

## 3. Complete SFT recipe

Open [examples/sft.py](../../examples/sft.py). It uses native Qwen3 message encoding and `repeat=3.0`, so total steps are derived from the MDS statistics.

Key settings:

- Model and precision are the same as the Pretrain recipe.
- AdamW uses `lr=1e-5` with fp32 master weights and optimizer states.
- Sequence length is 8,192 with 200 warmup steps and cosine decay.
- The message stream uses `family="qwen3"`, `data_type="messages"`, and `FULL_SAMPLE` packing.
- Checkpoints are written every 500 steps.

```bash
python examples/sft.py
# Or:
torchrun --nproc_per_node=8 examples/sft.py
```

`FULL_SAMPLE` protects chat-template and assistant-loss boundaries. Do not change an SFT stream to `HARD_TRUNCATE`; configuration validation rejects it.

## 4. CPT: mix text and messages

CPT uses the same engine with mixed stream types. Use a common scheduling mode—for example, `repeat` on both streams:

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

Do not mix `proportion`, `repeat`, and `choose` in one run.

## 5. Distributed strategies

`ModelConfig` exposes:

- `dp_replicate`: data replication for HSDP-style layouts.
- `dp_shard`: FSDP sharding; `-1` derives the degree from the mesh.
- `tp`: tensor parallelism.
- `cp`: context parallelism for long sequences.
- `ep`: expert parallelism for MoE models.

Pipeline Parallelism is outside the current supported scope. A configuration field or model-family capability does not mean every combination has been verified on every GPU/model. Start with a small debug model and validate the exact topology, tokenizer, and checkpoint path.

## 6. Checkpoint resume

The trainer checks `dump_folder/ckpt_subdir` for the latest DCP checkpoint by default. To migrate from another checkpoint, set:

```python
train.resume_from = "/outputs/run/checkpoints/step-1000"
```

Model, optimizer, scheduler, and dataloader state can resume together. Use `exclude_from_loading` only for deliberate partial restoration.

## 7. Export Hugging Face weights

After training, convert DCP to safetensors:

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

Use `from_hf(...)` when a training run must start from HF weights in DCP form. See the [API reference](api.md).

## 8. Future self-developed RL

A self-developed RL mode is under development. RL configuration, engine support, installation, and recipes are not yet public. Pretrain, SFT, and CPT are the supported training modes in this release.
