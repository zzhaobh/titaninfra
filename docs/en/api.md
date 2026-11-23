# API Reference

[简体中文](../zh-CN/api.md) · [README](../../README.md)

We provide a unified API for data conversion, model configuration, training, and checkpoint management. Pretrain, SFT, and CPT share `Trainer`. Our self-developed RL mode is under development and is not yet part of the public API.

## Root exports

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

`ModelConfig(name="qwen3/8B", ...)` selects a `<family>/<flavor>` and its distributed strategy.

### Required algorithm fields

| Field | Typical value | Meaning |
|---|---|---|
| `dtype` | `"bf16"` | Forward/backward compute precision. |
| `param_dtype` | `"fp32"` | Parameter/master-weight storage precision. |
| `attn_backend` | `"flex"` | Flex Attention backend; it is the only accepted backend. |
| `attn_mask_type` | `"block_causal"` or `"causal"` | Packed-document or standard causal masking. |
| `ac_mode` | `"none"`, `"selective"`, `"full"`, `"memory_budget"` | Activation recomputation policy. |
| `compile` | `False` | Whether to enable `torch.compile`. |

Common optional fields include `dp_shard=-1`, `dp_replicate=1`, `tp=1`, `cp=1`, `ep=1`, `cpu_offload=False`, and `kernel_impls=[]`. `pp` exists in the configuration model but Pipeline Parallelism is outside the current supported scope.

Use `dtype="bf16"`, `param_dtype="fp32"`, and `optimizer_state_dtype="fp32"` as the default mixed-precision starting point. Pure-bf16 master weights can lose small optimizer updates.

## TrainConfig

`TrainConfig` defines optimizer, scheduler, runtime, and checkpoint behavior.

### Required fields

```text
local_batch_size, global_batch_size, seq_len,
optimizer_name, lr, beta1, beta2, eps, weight_decay,
optimizer_state_dtype, min_lr_ratio, lr_decay_type, max_grad_norm
```

You must also explicitly choose **one** of `warmup_steps` or `warmup_ratio`. `steps` is required for `proportion` scheduling; `repeat` and `choose` derive total steps from the streams.

Frequently used optional fields include `ckpt_interval`, `keep_latest_k`, `resume_from`, `dump_folder`, `task_name`, `log_interval`, `seed`, `last_save_model_only`, and `last_save_in_hf`.

## StreamConfig

Each stream independently declares its MDS source, tokenizer, data type, packing, and mixing mode.

```python
StreamConfig(
    local="/data/mds/general/train",
    tokenizer_name="qwen3",
    data_type="text",
    proportion=0.7,
    pack_strategy=PackStrategy.HARD_TRUNCATE,
)
```

Required fields:

- `local`: local MDS directory or cache directory.
- `tokenizer_name`: key matching the conversion-time `stream_stats.json` entry.

Important fields:

- `data_type`: `"text"` (default) or `"messages"`.
- `family`: required for native message encoding; supported values include `qwen3`, `qwen2_5`, `llama3`, `deepseek_v3`, `deepseek_v4`, `gpt_oss`, and `gemma4`.
- `remote`: optional remote MDS source used with `local` as a node-local cache.
- `pack_strategy`: `HARD_TRUNCATE`, `SENTENCE_BOUNDARY`, `FULL_SAMPLE`, or `NO_PACK`.
- `proportion`, `repeat`, `choose`: mutually exclusive mixing controls. All streams in one run must use the same mode.
- `epoch_size_margin`: safety margin for token-length variance and packing tails.
- `history_policy`: optional per-block-type history `keep`/`drop`/family-supported `summarize` and `keep_last_n`; configured on the stream, not the sample. `insert_summary=True` replaces history with supplied summary blocks and is mutually exclusive with `keep_last_n`.
- `add_bos_eos`: text-only runtime option; do not explicitly set it to `True` for message streams.

Message streams default to `FULL_SAMPLE`; `HARD_TRUNCATE` is rejected because it can split chat-template and loss-mask boundaries.

## PackStrategy

| Value | Behavior |
|---|---|
| `HARD_TRUNCATE` | Fixed-length token windows with maximum utilization; text only. |
| `SENTENCE_BOUNDARY` | Prefer punctuation/sentence boundaries. |
| `FULL_SAMPLE` | Preserve complete samples where possible; the normal SFT choice. |
| `NO_PACK` | One sample per sequence with padding/truncation. |

## Trainer and TrainResult

```python
result = Trainer(
    model=model,
    train=train,
    streams=streams,
    hf_ckpt_path="/models/Qwen3-8B",  # optional
    task_name="my-run",               # optional
)
```

For `TrainConfig + list[StreamConfig]`, `Trainer` validates scheduling and precision compatibility, then calls the TorchTitan training engine. Text-only streams produce Pretrain; message streams produce SFT; mixed text/message streams produce CPT.

`TrainResult` fields are:

- `stage: str`
- `ckpt_dir: str`
- `steps: int`
- `model_name: str`

## Checkpoint conversion

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

`hf_assets_path` is required by `to_hf` so tokenizer/config assets and the HF index mapping are available.

## Asset and diagnostics utilities

- `download_hf_assets(repo_id, local_dir="./assets/hf", asset_types="tokenizer", download_all=False, hf_token=None, additional_patterns=None) -> str`
- `compare_losses(log_dirs, labels=None, metric="loss", output_path=None, show=True) -> dict`
- `diagnose_hf_checkpoint(hf_path, model_name=None, verbose=False) -> DiagnosisResult`

## Templates and monitoring

- `get_template`, `list_templates`, and `register_template` manage registered prompt templates.
- `MoEHealthPlugin` exposes MoE expert-utilization monitoring hooks.
- `MoEBiasSchedule(calibration_steps=0, calibration_multiplier=50.0, freeze_after=None)` configures expert-bias calibration and optional late freezing.

## MDS conversion APIs

After installing `requirements.txt` and the pinned TorchTitan engine, install
MosaicML Streaming from source (`python -m pip install --no-build-isolation --no-deps .`),
then run `python -m pip install --no-deps -e ".[data]"`.

### `DataConvertConfig` and `DataConverter`

```python
from titaninfra.data.converter import DataConvertConfig, DataConverter

config = DataConvertConfig(
    source="/data/raw/pretrain.jsonl",       # required: file or directory
    output="/data/mds/general/train",        # required
    tokenizer_path="/models/Qwen3-8B",
    tokenizer_name="qwen3",                 # must match StreamConfig
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

Required fields are `source` and `output`. `tokenizer_path` and
`tokenizer_name` are strongly recommended for Pretrain so `stream_stats.json`
contains exact token statistics. Important optional fields:

| Group | Fields |
|---|---|
| Input selection | `file_format`, `columns`, `field_mapping`, `text_key`, `row_filter` |
| Output/schema | `output_format="mds"`, `mds_columns`, `compression`, `size_limit`, `hashes`, `keep_local` |
| Tokenization | `add_bos=True`, `add_eos=True`, `store_token_count=False`, `num_tokenizer_workers` |
| Parallelism | `num_shards=16`, `num_workers=4`, `max_workers`, `batch_size=1000`, `retry=2`, `preload` |
| Resume/safety | `checkpoint_interval`, `resume=False`, `overwrite=False`, `validate_after_convert=True`, `validate_sample_ratio=0.01`, `validate_min_samples=2` |
| Stream metadata | `stream_id`, `stream_type="pretrain"`, `family` for message-aware conversion |

`convert() -> dict[str, Any]` returns aggregate statistics and writes balanced
MDS shards, `index.json`, and `stream_stats.json`. Existing output is not
silently overwritten. Remote output paths may be used according to `OutputPath`;
`keep_local` controls whether the local copy is retained.

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

`source`, `output`, `tokenizer_path`, and `family` are required. `family` must
name the native SFT encoder used by training; this is what makes token counts,
protocol markers, and loss masks consistent between conversion and runtime.
`tokenizer_name` is inferred from the tokenizer directory when omitted, but set
it explicitly so it matches `StreamConfig.tokenizer_name`.

Additional parameters are `num_tokenizer_workers`, `batch_size=1000`,
`queue_size=4096`, `row_filter`, `preload`, `compression`,
`message_converter_fn`, `size_limit="256MB"`, `resolution_policy`, and
`spatial_merge_size`. `message_converter_fn(record) -> record` can normalize a
source-specific schema before native tokenization. Resume uses
`checkpoint_interval` plus `resume=True`; do not combine overwrite semantics
with resume accidentally.

The MDS columns are `messages`, `stream_id`, `token_count`, `loss_config`,
`system`, and `tools`. The returned dictionary contains sample/token/shard
statistics; the same values are persisted in `stream_stats.json`.

## Message-format conversion APIs

These functions live in `titaninfra.data.sft.message_converter`. They convert
message structure only; they do not tokenize or write MDS.

### `normalize_to_hf_messages`

```python
from titaninfra.data.sft.message_converter import normalize_to_hf_messages

hf_messages, weights = normalize_to_hf_messages(
    messages=rich_messages,
    system="You are a helpful assistant.",
)
```

Returns `(hf_messages, weights)`. Text/thinking blocks are rendered into HF
`content`; reasoning text may be preserved in `reasoning_content`; `tool_use`
blocks become HF `tool_calls`; tool-result IDs use `tool_call_id`. `weights`
aligns with the returned messages. A supplied system prompt is inserted first
with weight zero. This is a compatibility conversion, not the final per-token
loss mask.

### `to_ir_messages`

```python
from titaninfra.data.sft.message_converter import to_ir_messages

rich_messages = to_ir_messages(hf_messages)
```

Converts HF or loose message dictionaries back to TitanInfra `blocks` IR.
Existing IR messages are preserved. HF `tool_calls`, `tool_call_id`,
`reasoning_content`, and text content are converted into the corresponding rich
blocks. The result can be embedded in the `messages` field of an SFT sample and
passed to `convert_messages_to_mds` or `tokenize_sample`.

For the full schema, block table, loss-weight cascade, history policy, and a
complete tool-loop example, see [Message system and data formats](data.md#message-system-schema).
