# TitanInfra Architecture

[README](../../README.md) · [API reference](api.md) · [Configuration reference](config.md) · [Research rationale](research.md)

We built TitanInfra around one idea: **training behavior should be composed from
data streams, not split into unrelated stage-specific systems**. Pretrain, SFT,
and CPT share a model, distributed strategy, optimizer, checkpoint format, and
training engine. Changing the data composition is enough to change the training
stage.

This guide explains how the pieces fit together. For exact fields and function
signatures, use the linked API and configuration references.

## Data streams as the composition boundary

A `StreamConfig` describes one source of training data. Text streams and message
streams use the same scheduling and packing interfaces:

```python
streams = [
    StreamConfig(local="/data/mds/general/train", tokenizer_name="qwen3",
                 data_type="text", proportion=0.7),
    StreamConfig(local="/data/mds/sft/train", tokenizer_name="qwen3",
                 data_type="messages", family="qwen3", proportion=0.3),
]
result = Trainer(model=model, train=train, streams=streams)
```

The stage follows the stream types:

- text only → Pretrain;
- messages only → SFT;
- text + messages → CPT.

There is no separate public RL engine in this release. A self-developed RL mode
is under development and intentionally has no public API yet.

## End-to-end path

```text
Raw JSONL / JSON / Parquet
  → MDS conversion and native token statistics
  → per-stream StreamingDataset
  → per-stream PackedStream
  → StreamMixer
  → TorchTitan training engine
  → DCP checkpoint
  → optional Hugging Face export
```

The same path handles text, rich messages, and multimodal inputs. The data
pipeline owns conversion, token statistics, packing, mixing, DP sharding, and
resume; the model and optimizer consume the resulting token stream.

## Elastic data pipeline

### Conversion and statistics

`DataConverter` handles text records. `convert_messages_to_mds` handles rich
messages through the same native encoder used at training time. Both produce
balanced MDS shards, `index.json`, and `stream_stats.json`.

`stream_stats.json` records the sample count, total and average token counts,
and coarse/fine token-length histograms per tokenizer. These statistics are the
input to token-aware scheduling rather than an informational side file.

Conversion is resumable and validated. Existing outputs are not silently
overwritten, and message conversion requires an explicit native `family` so the
statistics match runtime encoding.

### Two-layer packing

1. Offline conversion distributes records across shards using token counts.
2. Runtime packing combines documents into `seq_len` sequences.

Available strategies are:

| Strategy | Behavior |
|---|---|
| `HARD_TRUNCATE` | Fixed-length text windows with high utilization. |
| `SENTENCE_BOUNDARY` | Prefer sentence boundaries when cutting text. |
| `FULL_SAMPLE` | Preserve complete messages where possible; the normal SFT choice. |
| `NO_PACK` | One sample per sequence for specialized experiments. |

Message streams reject `HARD_TRUNCATE` because it can split protocol and loss-mask
boundaries. `positions == 0` marks a document start and drives document-aware
attention masks; BOS/EOS tokens are not used as the boundary oracle.

### Mixing and epoch sizing

All streams in one run use one mixing mode:

- `proportion`: token-weighted sampling with explicit `TrainConfig.steps`;
- `repeat`: per-stream repeat factors with derived total steps;
- `choose`: exact sample counts with derived total steps.

The estimator uses average tokens, fine length histograms, packing efficiency,
local/global batch sizes, gradient accumulation, sequence length, and DP world
size. `epoch_size_margin` protects against packing tails and shard imbalance.
Every rank computes the same epoch sizing from the same statistics, preventing
divergent iteration lengths and collective deadlocks.

See [Message system specification](message-system.md) and [Configuration reference](config.md) for the
exact controls.

## Message system for training

**Our Message system is built for training, not only for chat formatting.** Rich
blocks preserve the information needed to produce the correct tokens, labels,
loss weights, history behavior, and token statistics.

The schema supports text, thinking, tool_use, tool_result, image, and
history_summary blocks. A sample can also carry system text, tool declarations,
loss overrides, and `thinking_intent`.

The runtime path is:

```text
raw message JSON
  → capability checks
  → normalize to canonical blocks
  → native per-family encoder
  → Segment list
  → Assembler
  → EncodedSample(input_ids, labels, loss_weights)
```

Native encoders own family-specific wire formats and protocol markers. Jinja is
not a production training renderer; retained Jinja implementations are used as
test oracles where applicable.

Loss policy is explicit:

- `sft_mask_mode="auto"` gives binary assistant-only loss while respecting an
  explicit message `weight: 0`;
- `sft_mask_mode="manual"` preserves continuous per-message weights;
- `loss_config.type_overrides` applies family-supported per-block scaling;
- history policies can keep, drop, or summarize supported historical blocks.

`thinking_intent.effort` is currently consumed by GPT-OSS. Other families warn
and ignore it. This distinction matters: a schema field is not automatically a
training guarantee for every model family.

## Model adaptation

We support heterogeneous model families on a TorchTitan base without assuming
that every layer is identical. The adaptation ladder is:

1. **Configure:** express the model with existing nested configs when the base
   architecture already supports it.
2. **Subclass and override:** replace only the methods that differ, such as
   attention, masks, norms, or forward control flow.
3. **Self-own a component:** maintain a family-specific component when its
   parameter layout or forward signature is structurally incompatible with the
   generic one.

We self-own a component only when reuse would silently produce incorrect
numerics, sharding, or checkpoint behavior. Numeric differences alone usually
need only a subclass and a test.

Important adaptation axes include per-layer attention types, RoPE variants,
normalization counts, QKV biases, expert parameter layout, multimodal towers,
and state-dict FQNs. Checkpoint adapters and distributed sharding must preserve
the model's real parameter names and shapes.

See [Model families](../../README.md#model-families) for the supported navigation
map. Support for TP/CP/EP and custom kernels is family-specific and must be
validated for the exact configuration.

## Distributed execution

### Parallel dimensions

| Dimension | Config | Role |
|---|---|---|
| FSDP/HSDP | `dp_shard`, `dp_replicate` | Shard or replicate parameters and optimizer state. |
| Tensor Parallel | `tp` | Shard large per-layer matrices. |
| Context Parallel | `cp` | Split long sequences across ranks. |
| Expert Parallel | `ep` | Distribute MoE experts. |

Pipeline Parallel fields exist for compatibility, but PP is outside the current
supported scope.

The device mesh constrains the degrees. Data parallelism for the loader is:

```text
dp_world_size = dp_replicate × dp_shard
```

MosaicML's rank/world-size view is mapped to that DP grid so TP/CP peers in the
same DP column receive identical samples. This avoids input/label mismatches and
keeps collective operations aligned.

Batch semantics are:

```text
gradient_accumulation_steps =
    global_batch_size / (local_batch_size × dp_world_size)
tokens_per_step = global_batch_size × seq_len
```

Packing means a batch item is a `seq_len` token window, not necessarily one raw
document. Token-based scheduling therefore measures actual token consumption
instead of estimating steps from file counts.

## Custom kernels and optimizers

Custom GPU kernels are opt-in and registered by operator name. The kernel layer
provides wiring, parity checks, and standard fallback paths. Current registered
paths cover DeepSeek V4 hyper-connection/CSA gather and Kimi MoonViT RoPE/
patch-merge operations; this is not a blanket claim of equal speedup across all
models.

MuonAdamW splits parameters between Muon and AdamW. Eligible 2D matrices and
named 3D expert tensors use Newton–Schulz orthogonalized momentum, while
embeddings, norms, lm_head, and routers normally remain in AdamW. Routers can
be moved to Muon explicitly with `muon_include_router=True`. `muon_scale_factor`, NS precision, gather mode,
and QK-Clip control the numerical recipe. The [research guide](research.md)
explains the two LR conventions and their rationale.

Precision is deliberately split into compute (`ModelConfig.dtype`), master
weights (`param_dtype`), and optimizer state (`TrainConfig.optimizer_state_dtype`).
The standard starting point is bf16 compute + fp32 master + fp32 moments.

## MoE stability and monitoring

MoE stability is a family-level architectural property, not a universal switch.
DeepSeek-style families can declare bias and auxiliary-loss balancing; families
without those declarations do not acquire them by setting a coefficient.

Available mechanisms include:

- family-declared expert-bias updates and optional auxiliary loss;
- `MoEBiasSchedule` for HF cold-start calibration and optional late freezing;
- `freeze_params=["router"]` to preserve pretrained routing during SFT;
- `MoEHealthPlugin` for utilization CV, min/max load, imbalance ratio, and a
  sliding-window dead-expert ratio;
- optional EP-global metric reduction and WandB reporting.

Expert bias is persistent state in DCP checkpoints. DCP resume therefore does
not need to repeat HF cold-start calibration.

## Checkpoints, conversion, and observability

DCP resume restores model, optimizer, LR scheduler, train state, and dataloader
state. `resume_from` selects a checkpoint for migration; the normal path uses the
latest checkpoint under `dump_folder/ckpt_subdir`.

`to_hf` and `from_hf` convert DCP and Hugging Face safetensors using the model's
state-dict adapter. HF assets are required for tokenizer/config metadata and
index mapping.

WandB, metric plugins, MoE monitoring, torch.profiler traces, and operator
attribution use the same run configuration across Pretrain, SFT, and CPT.

## Current validation boundary

The public tests cover configuration, conversion, native messages, packing,
scheduling, resume, model construction, and feasible CPU/GPU regressions. GPU
and kernel support still depends on the exact model family, topology, and
backend.

For rolling development, pin the TitanInfra commit and the documented TorchTitan
revision together. The API and configuration documents in that same commit are
the contract for that revision.

Continue with [Training and complete recipes](training.md) or
[Research rationale](research.md).
