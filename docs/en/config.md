# Configuration Reference

[简体中文](../zh-CN/config.md) · [README](../../README.md) · [API reference](api.md)

We use three Pydantic configuration classes to describe a run. Algorithm-defining
fields are explicit so a recipe can be reproduced without hidden defaults.

| Class | Responsibility |
|---|---|
| `ModelConfig` | Model family/flavor, precision, attention, activation recomputation, and parallel strategy. |
| `TrainConfig` | Optimizer, LR schedule, batch/step semantics, checkpoint, validation, monitoring, and profiling. |
| `StreamConfig` | MDS source, tokenizer, text/message behavior, packing, mixing, history, and epoch sizing. |

Supporting classes are `MoEBiasSchedule`, `DataConvertConfig`, `TrainResult`, and
the message schemas documented in [Data formats](data.md). Resume uses the new
configuration; checkpoints provide weights and state rather than silently
overriding the recipe.

## ModelConfig

### Core fields

| Field | Default | Meaning |
|---|---|---|
| `name` | required | `family/flavor`, such as `qwen3/8B` or `deepseek_v3/debugmodel`. |
| `dtype` | required | Forward/backward compute precision; use `bf16` for the standard recipe. |
| `param_dtype` | required | Parameter/master-weight storage; use `fp32` for normal mixed precision. |
| `attn_backend` | required | `flex`; it is the only supported attention backend. |
| `attn_mask_type` | required | `causal` for ordinary attention or `block_causal` for packed documents. |
| `ac_mode` | required | `none`, `selective`, `full`, or `memory_budget`. |
| `compile` | required | Enables `torch.compile` when `True`. |

`dtype`, `param_dtype`, and `TrainConfig.optimizer_state_dtype` are independent:
compute, master weights, and optimizer moments should not be driven implicitly by
one flag. The usual starting point is bf16 compute + fp32 master + fp32 moments.
Pure-bf16 master weights can round away small optimizer updates.
`dtype="mxfp8"` enables the MXFP8 linear transform on supported Hopper+ GPUs;
its compute base remains bf16 and it does not change master-weight semantics.

### Parallelism and memory

| Field | Default | Meaning |
|---|---|---|
| `dp_shard` | `-1` | FSDP shard degree; `-1` derives it from the device mesh. |
| `dp_replicate` | `1` | HSDP/DDP replica degree. `dp_replicate × dp_shard` is the DP world size. |
| `tp` | `1` | Tensor Parallel degree. |
| `pp` | `1` | Pipeline Parallel degree exists in the config but PP is outside the current supported scope. |
| `cp` | `1` | Context Parallel degree for long sequences. |
| `ep` | `1` | Expert Parallel degree for MoE. |
| `ep_backend` | `"standard"` | Expert communication backend; `deepep`/`hybridep` need supported hardware and libraries. |
| `fsdp_reshard_after_forward` | `"default"` | FSDP reshard policy; `always` favors memory and `never` favors communication. |
| `fsdp_explicit_prefetch` | `False` | Opt-in explicit FSDP forward/backward prefetch where supported. |
| `pipeline_parallel_schedule` | `"1F1B"` | PP schedule field; not a claim of current PP support. |
| `context_parallel_load_balancer` | `"ptrr"` | CP load-balancing strategy for FlexAttention. |
| `enable_async_tensor_parallel` | `False` | Async TP; requires `compile=True`. |
| `ac_memory_budget` | `0.5` | Memory/speed balance for `ac_mode="memory_budget"` and `compile=True`. |
| `cpu_offload` | `False` | Offload optimizer states to CPU. |
| `kernel_impls` | `[]` | Opt into registered custom kernels by operator name; unknown names fail loudly. |

Batch semantics are:

```text
dp_world_size = dp_replicate × dp_shard
gradient_accumulation_steps = global_batch_size / (local_batch_size × dp_world_size)
tokens_per_step = global_batch_size × seq_len
```

### MoE model options

| Field | Default | Meaning |
|---|---|---|
| `moe_load_balance_coeff` | `None` | Override the family-declared expert-bias step size. It cannot enable balancing on a family without that declaration. |
| `moe_aux_loss_coeff` | `None` | Override family-declared auxiliary-loss weight; scope comes from the model's balance spec. |
| `moe_top_k` | `None` | Override the number of experts selected per token. |

Balancing is an architectural property of the model family, not a switch that can
be enabled arbitrarily on every model. See the MoE section below.

## TrainConfig

### Required fields

```text
local_batch_size, global_batch_size, seq_len,
optimizer_name, lr, beta1, beta2, eps, weight_decay,
optimizer_state_dtype, min_lr_ratio, lr_decay_type, max_grad_norm
```

Also choose exactly one of `warmup_steps` or `warmup_ratio`. `steps` is required
for `proportion`; `repeat` and `choose` derive it from stream statistics.

### Optimizer and LR scheduling

| Field | Default | Meaning |
|---|---|---|
| `steps` | `100` | Total optimizer steps; required explicitly for `proportion`. |
| `optimizer_implementation` | `"fused"` | `fused`, `foreach`, or `for-loop`; fused is the normal GPU choice. |
| `optimizer_state_dtype` | required | `fp32` normally; `bf16` moments require `param_dtype="fp32"`. |
| `warmup_steps` | `200` | Explicit warmup steps, or use `warmup_ratio` instead. |
| `warmup_ratio` | `None` | Warmup fraction in `(0, 1]`; overrides step-derived warmup when set. |
| `min_lr_ratio` | required | Final LR as a fraction of the base LR. |
| `lr_decay_type` | required | `cosine`, `linear`, `sqrt`, or `constant`. |
| `decay_ratio` | `None` | Warmup-Stable-Decay fraction; e.g. `0.2` decays only in the final 20% of steps. |
| `max_grad_norm` | required | Gradient clipping threshold; `0` disables clipping. |
| `chunked_loss` | `False` | Compute loss in chunks to reduce peak logits memory. |
| `loss_num_chunks` | `1` | Sequence chunks when chunked loss is enabled; typical values are 4–8. |
| `neftune_noise_alpha` | `None` | Add NEFTune embedding noise with strength α; typical SFT values are 5–15. |

NEFTune uses `Embedding(input) + Uniform(-1, 1) × α / √(L × d)`, where `L` is
sequence length and `d` is hidden size. It can be combined with router freezing
or per-layer LR multipliers.

### MuonAdamW

MuonAdamW sends eligible 2D linear weights to Muon and the remaining parameters
to AdamW. The table lists the public controls:

| Field | Default | Meaning |
|---|---|---|
| `muon_lr` | `0.02` | Muon learning rate, normally larger than the AdamW LR. |
| `muon_momentum` | `0.95` | Momentum EMA factor. |
| `muon_nesterov` | `True` | Apply Nesterov momentum before Newton–Schulz orthogonalization. |
| `muon_ns_steps` | `5` | Newton–Schulz iterations, from 1 to 20. |
| `muon_weight_decay` | `0.0` | Muon decoupled weight decay. |
| `muon_min_ndim_size` | `4096` | Minimum numel for a 2D weight to enter Muon. |
| `muon_include_router` | `False` | Keep router/gate weights in AdamW; enabling it is model-specific. |
| `muon_ns_dtype` | `"bf16"` | Newton–Schulz compute precision; `fp32` is also available. |
| `muon_scale_factor` | `1.0` | Update scaling: `1.0` for KellerJordan or `0.2` for Moonlight/Kimi K2. |
| `muon_gather_mode` | `False` | Gather 2D shards for exact full-matrix Newton–Schulz instead of per-shard NS. |
| `muon_qk_clip_tau` | `None` | Enable QK-Clip at threshold τ, normally 30–100. |

Two useful recipes are independent Muon LR (`muon_lr=0.02`,
`muon_scale_factor=1.0`) and shared LR (`muon_lr == lr`,
`muon_scale_factor=0.2`). Keep `optimizer_state_dtype="fp32"` and
`param_dtype="fp32"` unless you have specifically validated another precision
recipe.

### Parameter groups

`freeze_params` completely disables updates for matching parameters;
`layer_lr_overrides` multiplies their LR instead. Both follow the scheduler.

```python
freeze_params=["router"]
layer_lr_overrides={"embed_tokens": 0.1, "norm": 5.0}
```

Common patterns include `router`, `embed_tokens`, `output`, `norm`, and
`experts`. Use `freeze_params` to protect a pretrained router during SFT and
`layer_lr_overrides` when you want slower adaptation rather than a hard freeze.

### MoE health and bias scheduling

| Field | Default | Meaning |
|---|---|---|
| `enable_moe_health_monitor` | `False` | Collect per-expert utilization, CV, imbalance, and dead-expert metrics. |
| `moe_health_ep_reduce_interval` | `0` | EP-global reduction interval; `0` keeps local metrics only. |
| `enable_sv_entropy_monitor` | `False` | Sample Muon weights and report singular-value entropy. |
| `sv_entropy_interval` | `100` | Sampling interval; it must divide `log_interval`. |
| `sv_entropy_max_layers` | `4` | Maximum sampled layers per collection. |
| `moe_bias_schedule` | `None` | `MoEBiasSchedule(calibration_steps, calibration_multiplier, freeze_after)`. |

`MoEBiasSchedule` has three fields:

- `calibration_steps=0`: steps with `lr=0` while expert bias converges faster;
- `calibration_multiplier=50.0`: multiplies the family bias γ during calibration;
- `freeze_after=None`: optional 1-based step after which expert bias stops updating.

Calibration matters for HF cold starts, where expert bias is absent and starts
from zero. DCP resume restores it from the checkpoint and normally needs no new
calibration. A family must declare the bias mechanism for γ acceleration to have
an effect.

### Validation, data loading, and checkpoint

| Field | Default | Meaning |
|---|---|---|
| `val_streams` | `None` | Validation streams using the same `StreamConfig` type. |
| `eval_freq` | `500` | Validation interval in optimizer steps. |
| `eval_steps` | `-1` | Must be positive when `val_streams` is used. |
| `tokenizer_path` | `None` | Tokenizer path, usually the model asset directory. |
| `num_workers` | `1` | DataLoader workers; multi-stream runs should not increase this casually. |
| `shuffle` | `False` | Enable StreamingDataset shuffling. |
| `shuffle_seed` | `9176` | Data-shuffle seed, distinct from model `seed`. |
| `persistent_workers` | `False` | Keep workers alive across epochs when workers > 0. |
| `pin_memory` | `True` | Use CUDA pinned memory for host-to-GPU copies. |
| `predownload` | `None` | Samples prefetched by StreamingDataset. |
| `shuffle_block_size` | `None` | Optional shuffle block size. |
| `shuffle_algo` | `"py1br"` | Resume-efficient MosaicML shuffle algorithm. |
| `skip_corrupted_samples` | `True` | Warn and skip corrupt samples instead of failing immediately. |
| `max_consecutive_errors` | `10` | Stop after this many consecutive sample errors. |
| `ckpt_interval` | `500` | Checkpoint interval in steps. |
| `keep_latest_k` | `3` | Keep the latest k checkpoints; `0` keeps all, `1` is not supported. |
| `resume_from` | `None` | Optional checkpoint path for migration instead of the normal latest checkpoint. |
| `exclude_from_loading` | `[]` | State categories to skip: `optimizer`, `lr_scheduler`, `dataloader`, `train_state`. |
| `async_checkpoint` | `"disabled"` | `disabled`, `async`, or `async_with_pinned_mem`. |
| `last_save_model_only` | `False` | Save only model weights at the end. |
| `last_save_in_hf` | `False` | Also export final HF safetensors; requires `last_save_model_only=True`. |
| `export_dtype` | `"float32"` | HF export precision: `float16`, `bfloat16`, or `float32`. |
| `enable_first_step_checkpoint` | `False` | Save once at step 1 to validate checkpoint storage early. |

### Logging and profiling

| Field | Default | Meaning |
|---|---|---|
| `seed` | `None` | Optional global training RNG seed. |
| `log_interval` | `10` | Metric/log interval in steps. |
| `dump_folder` | `./outputs` | Output root. |
| `task_name` | `None` | Optional suffix for output and checkpoint directories. |
| `comm_init_timeout_seconds` | `3600` | Process-group initialization timeout. |
| `comm_train_timeout_seconds` | `3600` | Collective-communication timeout during training. |
| `fsdp_gas_no_sync` | `False` | For supported non-PP GAS>1 runs, skip HSDP cross-replica all-reduce until the last microbatch. |
| `enable_wandb` | `False` | Enable Weights & Biases logging. |
| `wandb_project/entity/run_name/run_id` | `None` | WandB project, team, display name, and resume ID. |
| `wandb_tags/notes` | `None` | Optional comma-separated tags and notes. |
| `wandb_mode` | `"offline"` | `offline`, `dryrun`, or `online`. |
| `enable_profiling` | `False` | Enable torch.profiler traces. |
| `profile_freq` | `100` | Trace cycle frequency. |
| `profile_active` | `2` | Active steps per cycle. |
| `profile_warmup` | `3` | Warmup steps before active profiling. |
| `profile_skip_first` | `5` | Steps skipped before the first cycle. |
| `profile_export_dir` | `"profile_traces"` | Trace output subdirectory under `dump_folder`. |
| `profile_memory` | `False` | Enable memory snapshots for OOM diagnosis. |
| `profile_with_stack` | `False` | Record Python stacks; significant overhead. |
| `profile_with_flops` | `True` | Collect FLOPs estimates for MFU analysis. |

## StreamConfig

### Source and tokenization

| Field | Default | Meaning |
|---|---|---|
| `local` | required | Local MDS directory or local cache. |
| `remote` | `None` | Optional remote MDS source used with `local` as a cache. |
| `split` | `None` | Dataset split name. |
| `cache_limit` | `None` | Optional local cache limit; use enough space for index and shard requirements. |
| `tokenizer_name` | required | Key matching `stream_stats.json.tokenizer_stats`. |
| `data_type` | `"text"` | `text` or `messages`. |
| `text_key` | `"text"` | Text field in the MDS record. |
| `messages_key` | `"messages"` | JSON messages field in the MDS record. |
| `family` | `None` | Native message encoder family; falls back to `tokenizer_name` when omitted. |
| `resolution_policy` | `None` | Optional multimodal image-resolution budget. |
| `add_bos_eos` | `True` | Text-only BOS/EOS insertion; do not explicitly set it to `True` for messages. |
| `validate_template_alignment` | `False` | Compare early samples with HF tokenization as a diagnostic. |

Text and message streams share conversion, scheduling, packing, and resume.
Message tokenization uses native family encoders, not a Jinja training fallback.
Document boundaries come from `positions == 0`, not from BOS/EOS markers.

### Packing, mixing, and history

| Field | Default | Meaning |
|---|---|---|
| `pack_strategy` | `HARD_TRUNCATE` | `HARD_TRUNCATE`, `SENTENCE_BOUNDARY`, `FULL_SAMPLE`, or `NO_PACK`. |
| `sentence_end_tokens` | `[]` | Custom sentence terminators for `SENTENCE_BOUNDARY`. |
| `proportion` | `None` | Token-weighted share; requires explicit `TrainConfig.steps`. |
| `repeat` | `None` | Repeat factor; derives steps from stream statistics. |
| `choose` | `None` | Exact sample count; derives steps from stream statistics. |
| `stream_id` | `None` | Metrics/log ID, inferred from `local` when omitted. |
| `epoch_size_margin` | `0.05` | Multiplier added to epoch sizing to prevent rank divergence. |
| `sft_mask_mode` | `"auto"` | Binary assistant loss with explicit zero respected; `manual` preserves continuous message weights. |
| `thinking_scale` | `None` | Stream-level thinking loss scale; overrides per-sample thinking override. |
| `thinking_mode` | `None` | Stream-level `on`/`off` override for the reasoning branch. |
| `max_turns` | `None` | Keep only the last N turns, preserving system and whole-message boundaries. |
| `truncate_at_last_segment` | `False` | Allow segment-aligned truncation with a warning instead of fail-loud overflow. |
| `history_policy` | `None` | Per-type `keep`/`drop`/`summarize`, `keep_last_n`, or `insert_summary`. |

All streams in one run must use the same mixing mode. `insert_summary=True`
replaces history with supplied `HistorySummaryBlock` content and is mutually
exclusive with `keep_last_n`; it does not consume `per_type`. Other history
policies require `keep_last_n` and `per_type` together. Current-turn blocks are
never modified. Increase `epoch_size_margin` for small/imbalanced datasets or
turn truncation; `3.0–5.0` is a useful starting range for `max_turns`.

## DataConvertConfig quick reference

`DataConvertConfig(source, output, ...)` configures `DataConverter.convert()`.
`source` and `output` are required. Important groups are:

| Group | Fields |
|---|---|
| Input | `file_format`, `columns`, `field_mapping`, `text_key`, `row_filter` |
| Output | `output_format="mds"`, `mds_columns`, `compression`, `size_limit`, `hashes`, `keep_local` |
| Tokenization | `tokenizer_path`, `tokenizer_name`, `add_bos`, `add_eos`, `store_token_count`, `num_tokenizer_workers` |
| Parallel conversion | `num_shards=16`, `num_workers=4`, `max_workers`, `batch_size=1000`, `retry=2`, `preload` |
| Resume/safety | `checkpoint_interval`, `resume=False`, `overwrite=False`, `validate_after_convert=True`, `validate_sample_ratio=0.01`, `validate_min_samples=2` |
| Metadata | `stream_id`, `stream_type="pretrain"`, `family` |

For messages, use `convert_messages_to_mds(...)` instead of constructing the
text converter directly. Its required arguments are `source`, `output`,
`tokenizer_path`, and `family`; `tokenizer_name` should be explicit and match
`StreamConfig`. See [MDS and Message conversion APIs](api.md#mds-conversion-apis)
for the complete signatures and message-format conversion functions.
