# Research and Design Rationale

[README](../../README.md) · [Architecture](architecture.md) · [Data formats](data.md) · [Configuration reference](config.md)

This guide collects the research reasoning behind TitanInfra's main design
choices. It is a narrative companion to the API documentation, not a collection
of experiment logs. Where a result depends on a particular model, topology, or
training recipe, we state that boundary rather than generalizing it.

## Native training encoders instead of Jinja on the training path

### Formatting is part of the training distribution

A chat template is not just presentation. Whitespace, role markers, reasoning
envelopes, and tool-call boundaries become tokens that the model learns to
produce. Small changes can therefore change the learned distribution.

Research on prompt sensitivity supports treating this as a first-order concern:

- FORMATSPREAD reports large accuracy spreads from formatting changes
  ([Sclar et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/6c0e99d736da621403018ca7b32b1a4d-Paper-Conference.pdf)).
- Prompt paraphrases can change model rankings, so a single template is not a
  neutral measurement choice
  ([Mizrahi et al., TACL 2024](https://aclanthology.org/2024.tacl-1.52.pdf)).
- Hugging Face's chat-template work explains the need for explicit assistant
  masking support and the `{% generation %}` mechanism
  ([Transformers #30650](https://github.com/huggingface/transformers/pull/30650)).

These results do not imply that one template is universally correct. They imply
that training needs an explicit, testable token/loss contract rather than a
string-rendering convention that happens to work for a simple example.

### Why string-span masking is fragile

A common SFT approach renders a conversation to text and searches for assistant
start/end delimiters. This can break when:

- a BPE merge crosses a delimiter boundary;
- a template version changes whitespace;
- tool messages alter the assistant rendering branch;
- reasoning content is present in some turns but not others;
- a special token appears inside ordinary text.

Even when the rendered text is correct, a one-token mask offset can train the
wrong target. The production path should therefore produce labels and weights
while constructing the token sequence, not recover them afterward by string
matching.

### TitanInfra's canonical builder

TitanInfra uses rich message blocks as the training-side source of truth:

```text
SFTSample / Block schema
  → capability checks
  → canonical normalization
  → native per-family encoder
  → Segment list with role, kind, scale, and token class
  → Assembler tokenization
  → input_ids + labels + loss_weights
```

Jinja remains useful as an inference-side reference and a test oracle. It is not
the production training renderer. This gives each family a small, testable
Python encoder instead of a second, training-specific Jinja implementation.

The key rule is **tokens and their loss semantics are produced together**. A
parity test can then compare the native output with an official rendering where
that comparison is meaningful, while intentional differences remain explicit.

## Token classes and capability boundaries

Every token needs a deliberate decision based on how inference produces it:

| Class | Meaning | Training behavior |
|---|---|---|
| Protocol | Role markers, system/user delimiters, and other framework-injected structure. | Keep in `input_ids`, normally masked from loss. |
| Decoded | Assistant text, reasoning content, tool-call payloads, and autonomous stop tokens. | Train with the resolved message/block weight. |
| Metadata | Dynamic or inference-only additions that do not belong in the training stream. | Remove structurally where the family encoder defines that behavior. |

The distinction between masking and deletion matters. A masked token still
occupies a position and can change KV offsets. A genuinely unwanted dynamic
metadata token should not simply be assigned zero loss if it changes the
sequence layout.

Family capability declarations make unsupported features visible. If a block or
feature is not supported, the encoder warns or fails according to the contract
instead of silently producing a plausible but incorrect sample.

## Tool use, reasoning, and history

### Tool calls are structured training data

Tool declarations use `name`, `description`, and JSON Schema `input_schema`.
Calls and results use explicit IDs:

```json
{
  "tools": [{
    "name": "search",
    "description": "Search an index",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
  }],
  "messages": [
    {"role": "assistant", "blocks": [
      {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "TitanInfra"}}
    ]},
    {"role": "tool", "blocks": [
      {"type": "tool_result", "tool_use_id": "call_1", "text": "Documentation"}
    ]}
  ]
}
```

Native encoders translate this IR into each family's actual tool wire format.
Tool results are normally masked, while assistant-authored calls can be trained
when the family and loss policy allow them. Invalid ID pairing fails loudly.

`normalize_to_hf_messages` and `to_ir_messages` provide compatibility conversion
between rich blocks and HF-style dictionaries. The conversion preserves tool
calls/results and reasoning fields; it is not itself tokenization or MDS
conversion.

### Thinking and effort

Thinking is a block type with explicit loss policy, not a hidden string section.
`thinking_intent.mode` controls whether the reasoning branch is enabled for a
sample. `thinking_intent.effort` supports `low`, `medium`, and `high`, but is
currently consumed by GPT-OSS; other families warn and ignore it.

History policy is configured on `StreamConfig`, not embedded as a mutable sample
implementation detail. It supports per-type `keep`/`drop`/family-supported
`summarize`, `keep_last_n`, and `insert_summary` with author-supplied summary
blocks. Current-turn blocks are not modified by the history policy.

The design goal is to make reasoning/history choices visible in the recipe while
keeping family-specific wire behavior inside the encoder.

## Elastic data and mixed training

### Why a common stream abstraction matters

Separate pretrain and post-training stacks introduce repeated conversion,
different loss normalization, and incompatible optimizer/checkpoint state.
TitanInfra instead treats text and messages as two data types flowing through the
same engine.

This enables mixed Pretrain + SFT (CPT) without an intermediate format rewrite:

```python
streams = [
    StreamConfig(local="/data/mds/general/train", tokenizer_name="qwen3",
                 data_type="text", proportion=0.7),
    StreamConfig(local="/data/mds/sft/train", tokenizer_name="qwen3",
                 data_type="messages", family="qwen3", proportion=0.3),
]
```

The training stage follows the data types, and the same model, distributed
strategy, optimizer, and checkpoint infrastructure can be reused. This is a
programming-model decision, not a claim that every model family supports every
parallel or kernel combination.

### Token-aware scheduling

File count and byte size are poor proxies for training work. For token-packed
streams, TitanInfra uses conversion-time token statistics to estimate epoch
sizes and total steps:

```text
tokens_per_step = global_batch_size × seq_len
samples_per_rank ≈
    steps × local_batch_size × GAS × seq_len × weight
    / (avg_tokens_per_sample × packing_efficiency)
```

Fine-grained token-length histograms inform packing efficiency, while DP-aware
epoch sizing prevents ranks from iterating for different lengths. Offline shard
balancing and runtime sequence packing solve different parts of the utilization
problem and should not be conflated. `NO_PACK` uses sample-based sizing instead:
`steps × local_batch_size × GAS × weight`, without the average-token or
packing-efficiency divisor.

## Muon and MuonAdamW recipe design

### Two LR conventions

MuonAdamW assigns eligible 2D matrices and named 3D expert tensors to Muon and
other parameters to AdamW. The important recipe choice is how the Muon update is
scaled.

TitanInfra uses:

```python
scale = muon_scale_factor * sqrt(max(m, n))
p.add_(update, alpha=-lr * scale)
```

Newton–Schulz orthogonalization gives an update matrix with approximately
`sqrt(min(m, n))` Frobenius norm. Spread across `m × n` entries, the per-element
RMS is approximately `1 / sqrt(max(m, n))`. Multiplying by
`0.2 × sqrt(max(m, n))` yields an approximately shape-independent update RMS of
0.2, close to the AdamW RMS estimate for `beta1=0.9`.

This explains the two practical conventions:

| Convention | `muon_scale_factor` | Muon LR relative to AdamW |
|---|---:|---|
| KellerJordan-style independent Muon LR | `1.0` | Usually independently tuned and substantially larger. |
| Moonlight/Kimi K2-style shared LR | `0.2` | `muon_lr` can equal the AdamW `lr`. |

The 0.2 factor is not a universal theorem. It is an empirical RMS-matching choice
supported by the Muon/Moonlight lineage. Use the current model, batch size, and
training-scale evidence to choose between the conventions.

### Parameter routing and stability

Normal parameter routing is:

| Parameter | Optimizer |
|---|---|
| Eligible 2D linear matrices and named 3D expert tensors | Muon |
| Embeddings, lm_head, norms, biases, and routers by default | AdamW |

Muon uses momentum, Nesterov, a configurable number of Newton–Schulz steps, and
an NS precision (`bf16` by default, `fp32` optional). `muon_min_ndim_size` keeps
small matrices out of Muon. `muon_gather_mode` selects exact gathered NS where
supported rather than per-shard NS.

For large MoE runs, QK-Clip can limit excessive per-head attention logits through
`muon_qk_clip_tau`. This is a recipe-dependent stability mechanism, not a reason
to claim that every scale or model has been validated equally.

Open questions remain for per-shard versus full-matrix NS at very large scale,
bf16 versus fp32 NS stability, and model-specific LR multipliers. Recipes should
record these choices explicitly and monitor loss, gradient norms, expert health,
and singular-value entropy.

## Multimodal data organization

### Caption and interleaved data are complementary

Caption data supports direct image-language alignment. Interleaved image-text
data supports in-context learning and multi-image reasoning. A balanced recipe
normally uses both, plus pure text to preserve language ability.

The research literature supports mixing these data types rather than treating
one as a complete replacement for the other. Resolution and token budgets should
also be stage-aware: coarse visual alignment can use fewer visual tokens, while
fine-grained document/OCR or SFT tasks may need more.

TitanInfra represents interleaved content directly in the block list:

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

### Image-token and packing invariants

An image becomes a contiguous sequence of vision markers plus a computed number
of image-pad tokens. Offline token estimation and runtime expansion must agree on
that count; otherwise packing statistics and actual sequence lengths diverge.

The multimodal collator flattens patches into the model's packed patch contract,
keeps `grid_thw` aligned with placeholder order, and constructs MRoPE positions
where supported. Text-only batches follow the same token/mask path without
injecting image tensors.

Qwen3.5 uses a hybrid GatedDeltaNet/full-attention decoder with a vision tower,
patch merger, and MRoPE. Kimi-VL combines MoonViT, a projector, and a DeepSeek-V3
language tower. These are family-specific paths, not generic multimodal claims
for every model.

## Distributed stability and validation limits

Distributed training introduces both new failure modes and amplified versions of
single-node problems:

- BF16 reduction order and accumulation can affect reproducibility at scale.
- Uneven per-rank token distributions can change gradient variance.
- MoE routing can create dead or overloaded experts.
- EP dispatch/combine can become a communication bottleneck.
- Incorrect parameter sharding or adapter FQNs can silently break resume.

TitanInfra addresses these with DP-aware data sharding, consistent epoch sizing,
family-specific parallel declarations, checkpoint adapters, MoE health metrics,
and explicit validation boundaries. No single test suite proves every model,
kernel, and topology combination.

For reproducible work, pin the TitanInfra commit and its documented TorchTitan
revision, record the data statistics and complete recipe, and run the relevant
CPU/GPU tests for the exact configuration.

Continue with [Architecture](architecture.md), [Training recipes](training.md),
or the [API reference](api.md).
