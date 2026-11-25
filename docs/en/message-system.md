# TitanInfra Message System Specification

[简体中文](../zh-CN/message-system.md) · [README](../../README.md) · [API reference](api.md) · [Data pipeline](data.md)

**Our Message system is built for training.** It is the canonical representation
for SFT and mixed Pretrain+SFT data, carrying not only text but also reasoning,
tools, images, loss policy, and history semantics.

This document is the complete public specification for the current Message IR,
native encoding path, and MDS integration. Exact signatures are also listed in
the [API reference](api.md#message-format-conversion-apis).

## 1. Scope and design principles

The system covers:

- canonical `SFTSample` and `Message` schemas;
- text, thinking, tool, image, and history-summary blocks;
- assistant loss masks and supported per-message/per-block weights;
- `thinking_intent`, history policy, and turn truncation;
- native per-family encoding and capability checks;
- MDS conversion, direct tokenization, packing, and validation.

The production training path uses native Python encoders. Jinja is not a
training renderer; it is retained only as a test-side oracle where applicable.
There is no public RL engine or RL Message API in this release.

The central invariant is:

> **Tokens and their loss semantics are produced together.**

Do not render a string and then reconstruct assistant masks by string matching.

## 2. Canonical object model

### `SFTSample`

```python
SFTSample(
    messages=[...],                 # required, ordered Message list
    system="You are helpful.",     # optional top-level system prompt
    tools=[...],                    # optional tool declarations
    loss_config=LossConfig(...),    # optional block-type overrides
    thinking_intent=ThinkingIntent(...),
)
```

| Field | Type | Meaning |
|---|---|---|
| `messages` | `list[Message]` | Ordered conversation turns. Required and non-empty. |
| `system` | `str | None` | Top-level system prompt, masked by default. |
| `tools` | `list[Tool] | None` | Tool declarations rendered by the native family encoder. |
| `loss_config` | `LossConfig` | Per-block-type loss overrides through `type_overrides`. |
| `thinking_intent` | `ThinkingIntent | None` | Sample-level reasoning mode and effort. |

`history_policy` is intentionally configured on **`StreamConfig`**, not stored
on each sample. This keeps the on-disk sample reusable with different recipes.

### `Message`

```python
Message(
    role="assistant",
    blocks=[...],
    weight=1.0,       # optional
    name=None,        # optional compatibility field
    task=None,        # DSv4-specific action/task transition tag
)
```

| Field | Type | Meaning |
|---|---|---|
| `role` | `str` | Usually `system`, `user`, `assistant`, or `tool`. DSv4 additionally models `developer` and `latest_reminder` internally. |
| `blocks` | `list[Block]` | Ordered rich content blocks. |
| `weight` | `float | None` | Optional message-level weight; normalized into the range `[0, 1]` by the canonical normalized message contract. |
| `name` | `str | None` | Optional compatibility name field. |
| `task` | `str | None` | DSv4-specific task tag such as `action`. |

New datasets should use `blocks`. Legacy `content` and HF-style fields are
accepted by normalization and conversion for compatibility, but rich blocks are
the stable training representation.

## 3. Block types

### `TextBlock`

```json
{"type": "text", "text": "Solve 2+2.", "weight": null}
```

Plain text. Assistant text normally trains; system/user/tool text is normally
masked.

### `ThinkingBlock`

```json
{"type": "thinking", "text": "Add the numbers.", "weight": null}
```

Reasoning content. It participates in loss according to the resolved policy and
family support. Do not include `<think>` tags inside `text`; the native encoder
owns the wire envelope.

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

An assistant-authored tool invocation. `id` pairs with the later tool result.
`input` is a dictionary, not a JSON string.

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

The result returned by a tool. Tool output is masked by default. `tool_use_id`
must match a prior `ToolUseBlock.id`.

### `ImageBlock`

```json
{
  "type": "image",
  "source": {"type": "path", "data": "/data/example.jpg"},
  "weight": null
}
```

An image reference. Sources can be path, bytes, base64, or a supported PIL-like
object in internal processing paths. Image tokens are masked and are not treated
as text to be generated. Current public native image support is family-specific,
with Qwen3.5 as the VLM path.

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

An upstream-generated summary of historical turns. `normalize` does not invent
summary text. `style` is `latest_reminder`, `system_prefix`, or
`ephemeral_block`, subject to family support. `compressed_turn_count` records
how many turns the summary replaces.

### Block weight is not universal

`weight` exists on block schemas, but it is not a universal native-training
switch. Use it only when the target family declares per-block loss support;
otherwise dispatch warns and the encoder may ignore it. Use message weights,
`sft_mask_mode`, and supported `type_overrides` for the documented training path.

## 4. Loss and mask semantics

### `sft_mask_mode="auto"`

This is the default:

- assistant messages have weight 1;
- system, user, and tool messages have weight 0;
- an explicit message `weight: 0` is always respected;
- non-zero custom assistant weights are normalized back to 1.

Auto is therefore binary assistant-only loss, with explicit silencing as the
only exception.

### `sft_mask_mode="manual"`

Per-message weights pass through unchanged. Continuous values such as `0.5`
can scale a message's contribution. The canonical normalized message contract
requires weights in `[0, 1]`.

### `train_on_inputs=True`

Direct `tokenize_sample` and the lower-level utility path can set every message
weight to 1, including prompts. `StreamConfig` does not expose this switch.

### `loss_config.type_overrides`

A family encoder can apply block-type multipliers, for example:

```json
{"loss_config": {"type_overrides": {"thinking": 0.5, "tool_use": 0.0}}}
```

Support is family-specific. Do not assume every block type or every family
honors the same override.

### Labels and positions

The final `EncodedSample` contains:

- `input_ids`: token IDs;
- `labels`: token ID where loss contributes, otherwise `IGNORE_INDEX = -100`;
- `positions`: `0` at each document start, increasing within the document;
- optional `loss_weights`: per-token multipliers supplied by the active policy;
- optional `pixel_values` and `grid_thw` for multimodal input.

Padding belongs to the packer/dataloader and uses `-1` positions. The encoder
does not pad samples. FlexAttention document masks are derived downstream from
`positions == 0`, not from BOS/EOS markers.

## 5. Thinking intent

```json
{
  "thinking_intent": {"mode": "on", "effort": "high"},
  "messages": [...]
}
```

| Field | Values | Semantics |
|---|---|---|
| `mode` | `on`, `off`, or omitted | Enable/disable the sample's reasoning branch, where the family supports it. |
| `effort` | `low`, `medium`, `high`, or omitted | Reasoning-depth dial. **Only GPT-OSS currently consumes it.** Other families warn and ignore it. |

The intent is sample-wide, not per-message. This matches the inference model of
choosing the next response's reasoning behavior rather than editing historical
messages individually.

## 6. History policy

Configure history on `StreamConfig(history_policy=...)`:

```python
history_policy = {
    "keep_last_n": 1,
    "per_type": {
        "thinking": {"action": "drop"},
        "tool_result": {"action": "drop"},
    },
}
```

### Fields

| Field | Meaning |
|---|---|
| `per_type` | Maps block type to `HistoryAction(action="keep" | "drop" | "summarize")`. Missing types default to keep. |
| `keep_last_n` | Keep the latest N assistant turns as current; earlier turns are history. |
| `insert_summary` | Replace the whole history region with supplied `HistorySummaryBlock` content. |

`summarize` requires a `summary_template` and family support. Current-turn blocks
are never modified. History turns receive weight zero.

`insert_summary=True` is mutually exclusive with `keep_last_n`, and does not
consume `per_type`. If the source has no summary block, the history region is
dropped. Other history policies require `keep_last_n` and `per_type` together to
avoid hidden inheritance.

With no explicit policy, Qwen3, GPT-OSS, and Gemma4 drop historical thinking by
default; Qwen2.5, Llama3, DeepSeek V3, and DeepSeek V4 keep it by default.
Family-specific defaults are part of the training/inference compatibility
contract, so do not change them silently.

## 7. Tool declarations and conversion

### Tool schema

```python
Tool(
    name="search",
    description="Search an index",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
)
```

The field is `input_schema`, not `parameters`. Native encoders project it into
the family's required declaration location and format.

### Rich → HF messages

```python
from titaninfra.data.sft.message_converter import normalize_to_hf_messages

hf_messages, weights = normalize_to_hf_messages(
    rich_messages,
    system="You are a helpful assistant.",
)
```

Returns `(hf_messages, weights)`. Text and thinking blocks are rendered into HF
content, reasoning can be preserved in `reasoning_content`, tool calls become
HF `tool_calls`, and tool results use `tool_call_id`. The returned weights align
with the messages; they are not the final per-token mask.

### HF messages → rich IR

```python
from titaninfra.data.sft.message_converter import to_ir_messages

rich_messages = to_ir_messages(hf_messages)
```

HF text, `reasoning_content`, `tool_calls`, and `tool_call_id` become rich blocks.
The reverse conversion preserves ordinary `thinking + text` block order and
content instead of collapsing them into one text block.

These functions convert message structure only. They do not tokenize or write
MDS.

## 8. Native encoder families and capability boundaries

Block support is declared by each native encoder. This table is a navigation
summary; feature-level declarations also control tools, parallel calls, history,
and reasoning behavior.

| Family | Text | Thinking | Tool use/result | Image |
|---|---|---|---|---|
| Qwen3 | Yes | Yes | Yes | No |
| Qwen2.5 | Yes | No | No | No |
| Llama3 | Yes | No | No | No |
| DeepSeek V3 | Yes | No | Yes | No |
| DeepSeek V4 | Yes | Yes | Yes | No |
| GPT-OSS | Yes | Yes | Yes | No |
| Gemma4 | Yes | Yes | Yes | No |
| Qwen3.5 | Yes | Yes | Yes | Yes |

Additional feature declarations cover system prompts, multi-turn, parallel tool
calls, thinking history, intent mode/effort, history summary styles, and weight
overrides. If a sample uses an unsupported feature, dispatch emits a warning or
raises the specified error rather than silently treating it as supported.

Adding a family means implementing its native encoder, declaring its supported
blocks/features, and adding oracle/parity coverage. Do not add a generic Jinja
fallback to bypass this work.

## 9. Runtime encoding pipeline

```text
raw sample dict
  → dispatch.tokenize_sample
      → block/feature capability checks
      → normalize to NormalizedSample
      → LossPolicy.resolve_weights
      → native family router/encoder
      → Segment list
      → Assembler
  → EncodedSample
```

### Normalized structures

`NormalizedSample` carries normalized messages, system, tools, thinking intent,
and history policy. `NormalizedMessage` carries role, blocks, resolved weight,
turn role, and optional task. These structures are frozen so invalid encoder
outputs fail at the boundary.

### Segments and token classes

A `Segment` describes one wire-level unit with:

- text or pre-encoded token IDs;
- kind (`MARKER`, `STOP_MARKER`, `CONTENT`, `THINKING`, `TOOL_CALL`,
  `TOOL_RESULT`, `SYSTEM`, `TOOLS_HEADER`, or `IMAGE`);
- role, generation flag, and token class (`PROTOCOL`, `DECODED`, `METADATA`);
- scale factor and derived `loss_scale`.

The structural base mask trains only model-decoded assistant content, thinking,
tool calls, and stop markers. Protocol framing, tool results, system text, and
images are masked. Metadata is distinct from protocol tokens and must not be
silently treated as learnable content.

The Assembler merges compatible adjacent text segments before tokenization to
preserve BPE boundaries, then emits aligned IDs, labels, positions, and optional
weights. Segment truncation fails loudly unless the caller explicitly enables
segment-aligned truncation.

## 10. MDS conversion

`convert_messages_to_mds` writes rich messages with the same native encoder used
by runtime training. This ensures token counts and loss masks share one
implementation.

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

Required arguments are `source`, `output`, `tokenizer_path`, and `family`.
Set `tokenizer_name` explicitly to match `StreamConfig.tokenizer_name`.

MDS columns are:

| Column | Content |
|---|---|
| `messages` | JSON message payload |
| `stream_id` | Stream identifier |
| `token_count` | Native token count |
| `loss_config` | Serialized loss overrides |
| `system` | Serialized system prompt |
| `tools` | Serialized tool declarations |

The returned dictionary contains sample/token/shard statistics, and the same
statistics are persisted to `stream_stats.json` for scheduling.

## 11. Direct tokenization API

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

`sample_json` is a raw dictionary, not an already-decoded `SFTSample` object.
Use `decode_sample` for schema validation, then pass `json.loads(raw)` to
`tokenize_sample` when direct tokenization is needed.

## 12. Validation and failure behavior

The system rejects or reports invalid input instead of silently producing
zero-loss samples:

- malformed schema or block fields fail decoding;
- unsupported blocks are warned and pruned according to capability rules;
- if pruning removes all assistant content, `EmptyAssistantError` is raised;
- invalid tool ID pairing fails;
- overlong segments fail unless segment-aligned truncation is explicitly enabled;
- a first segment exceeding `max_length` raises `TruncatedToEmptyError` for the
  streaming caller to warn and skip.

Do not bypass these checks by rendering a string outside the native pipeline.

## 13. Complete examples

### Reasoning SFT

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

`effort` has an effect only for GPT-OSS in the current release.

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

### Interleaved multimodal message

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

## 14. Compatibility and extension rules

- Append new block fields rather than changing existing meanings.
- Keep family-specific wire behavior inside that family's encoder.
- Declare new capabilities explicitly and add tests for each behavior.
- Preserve `input_ids`, `labels`, `positions`, and optional `loss_weights`
  alignment by token position.
- Keep schema and runtime native encoding consistent with conversion-time
  statistics.
- Use the same pinned TitanInfra and TorchTitan revisions when comparing
  behavior across environments.

For usage, continue with [Data formats](data.md), [API reference](api.md), and
[Training recipes](training.md). For design rationale, see [Research](research.md).
