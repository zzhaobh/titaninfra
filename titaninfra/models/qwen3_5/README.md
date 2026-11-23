# Qwen3.5: Multimodal Model with Hybrid Attention

## Architecture

Qwen3.5 combines:

- **Hybrid decoder:** GatedDeltaNet linear-attention layers interleaved with full-attention layers, output gating, and partial RoPE.
- **Vision encoder:** a Vision Transformer with 2D RoPE and bilinear-interpolated learned position embeddings.
- **Patch merger:** groups `merge_size²` spatial patches into one vision token.
- **MRoPE:** combines temporal, height, and width position IDs in decoder layers.
- **MoE variants:** routed experts plus a shared expert with sigmoid gating.

Text and vision features use a flat packed token stream. Vision features are
scattered into media-placeholder positions after projection. Message streams
use the native Qwen3.5 family encoder rather than a Jinja training fallback.

## Model variants

### Dense

| Variant | LLM dim | Layers | Heads | KV heads | Head dim | ViT dim | ViT layers |
|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | 256 | 8 | 4 | 2 | 64 | 256 | 4 |
| 0.8B | 1024 | 24 | 8 | 2 | 256 | 768 | 12 |
| 2B | 2048 | 24 | 8 | 2 | 256 | 1024 | 24 |
| 4B | 2560 | 32 | 16 | 4 | 256 | 1024 | 24 |
| 9B | 4096 | 32 | 16 | 4 | 256 | 1152 | 27 |
| 27B | 5120 | 64 | 24 | 4 | 256 | 1152 | 27 |

### MoE

| Variant | LLM dim | Layers | Experts | Top-k | Shared expert |
|---|---:|---:|---:|---:|---|
| debugmodel_moe | 256 | 4 | 8 | 2 | Yes |
| 35B-A3B | 2048 | 40 | 256 | 8 | Yes |
| 122B-A10B | 3072 | 48 | 256 | 8 | Yes |
| 397B-A17B | 4096 | 60 | 512 | 10 | Yes |

## Current training scope

- FSDP/HSDP sharding for decoder layers and the vision encoder.
- Tensor/Sequence Parallel declarations for supported projections.
- Expert Parallel declarations for MoE variants.
- MDS text/message streams with document-aware packing and multimodal collation.

Pipeline Parallelism is outside the current supported public scope. Video data
pipelines and CP+VLM combinations are not presented as verified features.

## Usage

Select a flavor through the normal model registry:

```python
from titaninfra import ModelConfig

model = ModelConfig(name="qwen3_5/debugmodel")
```

Use the repository-level [API reference](../../../docs/en/api.md) and
[training guide](../../../docs/en/training.md) for complete configuration and
recipes. For structural and CPU-feasible validation, run the retained tests under
`tests/models/qwen3_5/`.
