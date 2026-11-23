# Elastic Data Pipeline and Formats

[简体中文](../zh-CN/data.md) · [README](../../README.md)

We treat every training input as a declarative stream. Conversion, statistics, packing, mixing, distributed sharding, and resume use the same per-stream contract for Pretrain, SFT, and CPT.

## End-to-end flow

```text
Raw JSONL/Parquet
  → MDS conversion + stream_stats.json
  → per-stream StreamingDataset
  → per-stream packing
  → StreamMixer
  → Trainer and DCP checkpoint
```

`stream_stats.json` stores token counts and length distributions. The runtime uses these statistics for packing-efficiency estimates, epoch sizes, and total-step calculations instead of guessing from file count or byte size.

## Pretrain text input

A JSONL text record contains a `text` field by default:

```json
{"text": "Language models learn statistical structure from data."}
```

Convert it with:

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

`source` may be a file or directory. `output` receives MDS shards, `index.json`, and `stream_stats.json`. `tokenizer_name` must match the corresponding `StreamConfig.tokenizer_name`.

Text streams support `add_bos_eos=True` at runtime. Document boundaries for attention come from `positions == 0`, not from BOS/EOS tokens.

## SFT message input

**Our Message system is built for training.** It carries the information needed
by native family encoders, loss masks, history policy, exact token statistics,
and packing—not only the text displayed in a chat UI.

Messages use rich blocks rather than forcing all content into a single string:

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

### Message system schema

The canonical schema is defined in `titaninfra.data.schema` and validated with
`decode_sample` / `decode_jsonl`. Prefer `blocks` in new datasets. Legacy
`content` and HF-style fields are normalized for compatibility, but rich blocks
are the stable training representation.

| Field | Meaning |
|---|---|
| `messages` | Ordered `Message` objects with `role`, `blocks`, and optional `weight`, `name`, `task`. |
| `system` | Optional top-level system prompt, masked by default. |
| `tools` | Tool declarations using `name`, `description`, and JSON Schema `input_schema`. |
| `loss_config.type_overrides` | Optional per-block-type loss multipliers. |
| `thinking_intent` | Optional sample-level `mode: on/off` and `effort: low/medium/high`. |

Supported block types:

| Block | Required payload | Default loss behavior |
|---|---|---|
| `text` | `text` | Assistant text trains; system/user/tool text is masked. |
| `thinking` | `text` | Reasoning content, subject to the same weight cascade. |
| `tool_use` | `id`, `name`, `input` object | Paired with a later `tool_result.tool_use_id`. |
| `tool_result` | `tool_use_id`, `text`, optional `is_error` | Tool output is masked by default. |
| `image` | `source` object such as `{type: base64, data: ...}` | Image tokens are masked. |
| `history_summary` | `text`, optional `compressed_turn_count` and `style` | Compressed history supplied by upstream; normalize does not invent summaries. |

A complete tool-loop record therefore looks like:

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

### Loss weights and history

The native training path applies loss policy in two stages:

- **`sft_mask_mode="auto"` (default):** assistant messages have weight 1 and
  other roles have weight 0. An explicit message `weight: 0` is always
  respected. Non-zero custom assistant weights are normalized back to 1.
- **`sft_mask_mode="manual"`:** per-message weights pass through unchanged, so
  continuous values such as `0.5` can scale a message's contribution.
- **`loss_config.type_overrides`:** family encoders can scale specific block
  types, such as thinking or tool-call segments. Support is family-specific;
  it is not a universal multiplier for every block.

`block.weight` exists in the schema, but it is not a universal native-training
knob. Use it only with a family that explicitly supports per-block weights;
other families warn and ignore it. Direct `tokenize_sample` calls also accept
`train_on_inputs=True`, which gives every message weight 1; `StreamConfig` does
not expose that switch.

History policy belongs to **`StreamConfig`**, not the on-disk sample. Configure
`per_type` actions (`keep`, `drop`, or family-supported `summarize`) and
`keep_last_n` in the recipe. With no explicit policy, Qwen3, GPT-OSS, and Gemma4
drop historical thinking; other native families keep it by default. Current-turn
blocks are never modified. `insert_summary=True` replaces the whole historical
region with author-supplied `HistorySummaryBlock` content and must not be
combined with `keep_last_n`; `per_type` is not used on that path. If no summary
block exists, the historical region is dropped.

### Validation and direct use

```python
import json
from titaninfra.data.schema import decode_sample
from titaninfra.data.sft.dispatch import tokenize_sample
from transformers import AutoTokenizer

raw = open("/data/raw/one_sft_record.json").read()
decode_sample(raw)  # schema validation
encoded = tokenize_sample(
    json.loads(raw),
    tokenizer=AutoTokenizer.from_pretrained("/models/Qwen3-8B"),
    family="qwen3",
)
```

Invalid tool ID pairing, unsupported block shapes, and empty assistant content
fail loudly instead of silently producing zero-loss samples. `thinking_intent`
is sample-wide: `mode` controls the reasoning branch; `effort` is honored by
GPT-OSS, while other families warn and ignore it.

Convert messages with the same native family encoder used during training:

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

Native families include `qwen3`, `qwen2_5`, `llama3`, `deepseek_v3`, `deepseek_v4`, `gpt_oss`, and `gemma4`. The encoder determines protocol markers, assistant loss, thinking/tool blocks, and family-specific rendering; production training does not render messages with a Jinja fallback.

Tool samples place `tools` at the sample top level. `tool_use.id` pairs with `tool_result.tool_use_id`. Optional `loss_config.type_overrides` can rescale or suppress block-type losses.

## Two-layer packing

1. **Offline shard balance:** conversion distributes records across MDS shards to reduce token-count variance.
2. **Runtime sequence packing:** each stream applies its `PackStrategy` and produces `positions` that reset at document boundaries.

| Strategy | Intended use |
|---|---|
| `HARD_TRUNCATE` | High-utilization text pretraining. |
| `SENTENCE_BOUNDARY` | Text windows that prefer sentence boundaries. |
| `FULL_SAMPLE` | SFT and long reasoning samples; preserves template structure. |
| `NO_PACK` | One sample per sequence for specialized experiments. |

Message streams reject `HARD_TRUNCATE`. `epoch_size_margin` adds capacity for token-length variance and packing tails; increase it when using truncation or highly variable samples.

## Mixing and scheduling

All streams in one run must use the same mode.

| Mode | Configuration | Step calculation |
|---|---|---|
| `proportion` | `StreamConfig(proportion=...)` + explicit `TrainConfig.steps` | Token-weighted sampling for a fixed number of steps. |
| `repeat` | `StreamConfig(repeat=...)` | Derives total steps from per-stream epochs and repeat factors. |
| `choose` | `StreamConfig(choose=...)` | Draws an exact number of samples and derives total steps. |

The estimator uses average tokens, the fine-grained length histogram, packing efficiency, local batch size, gradient accumulation, sequence length, and DP world size. `choose`/`repeat` should not be mixed with `proportion` in one run.

## Distributed data sharding

TorchTitan computes a data-parallel degree from `dp_replicate × dp_shard`. TitanInfra maps MosaicML's view of rank/world size to that DP grid so TP/CP peers in one DP column see the same samples. Every rank computes epoch sizing from the same statistics, avoiding divergent iteration lengths and collective deadlocks.

`remote + local` lets each node cache MDS shards locally. This is useful for shared NFS/S3 sources; generated caches are not training checkpoints and should not be committed.

## Checkpoint and resume

DCP resume can restore model, optimizer, LR scheduler, train state, and dataloader state. By default, the trainer looks below `dump_folder/ckpt_subdir`; `resume_from` can select another checkpoint for migration. Use `exclude_from_loading` only when intentionally excluding supported state categories.

## Test fixtures

Downloadable tokenizer assets are not tracked in the public branch. Bootstrap them with:

```bash
python tests/fixtures/tokenizers/download_all.py
# Or selected families:
python tests/fixtures/tokenizers/download_all.py qwen3 qwen2_5
```

`HF_ENDPOINT` can override the download endpoint. Some upstream repositories require access approval or may be unavailable; those external fixtures are an integration-test prerequisite, not bundled model weights. Small synthetic datasets needed by unit tests remain under `tests/fixtures/`.
