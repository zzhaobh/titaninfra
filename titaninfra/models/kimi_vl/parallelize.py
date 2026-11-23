"""Parallelization for the Kimi-VL VLM family.

``KimiVLModel`` subclasses ``DeepSeekV3Model`` (MLA + MoE language tower) and
hangs two extra sibling sub-modules off the top level:
  * ``vision_tower`` — MoonViT (a bare ``nn.Module``, NOT a torchtitan Module).
  * ``projector``    — ``KimiVLMultiModalProjector`` (bare ``nn.Module``).

With the ModelSpec→bare-Config migration, there is no more free-standing
per-family ``parallelize_fn``: ``model.parallelize(...)`` IS the universal
entry point, inherited from ``BaseModel``/``Decoder``/``MTPDecoder``. The
Module-protocol TP/EP walk already handles the decoder sub-tree declaratively
via each submodule's ``sharding_config`` (baked in at construction time by
the ``kimi_vl_a3b`` deepseek_v3 flavor — see ``__init__.py``); bare
``nn.Module`` attributes like ``vision_tower``/``projector`` carry no
``sharding_config`` and are automatically invisible to that walk.

The ONE thing that needs a real override is FSDP: bare ``nn.Module`` subtrees
are not automatically wrapped by ``Decoder._apply_fsdp()`` (which only FSDPs
the decoder's own ``layers``/embeddings), and the vision tower needs its own
FSDP unit, applied the way upstream's own ``MultimodalModel._apply_fsdp``
handles it — one AllGather for the whole (small) tower, inserted BEFORE the
decoder FSDP call (``MultimodalModel._apply_fsdp`` prescribes this ordering:
encoder first, then ``super()._apply_fsdp(...)`` for the decoder).

This mirrors ``MTPDecoder._apply_fsdp`` (mtp.py, upstream) and
``MultimodalModel._apply_fsdp`` (multimodal.py, upstream) — both override
this exact hook rather than reimplementing ``parallelize()`` from scratch.
"""

from torchtitan.config import TORCH_DTYPE_MAP, PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext


def apply_fsdp_to_kimi_vl(
    model,
    *,
    parallelism_context: ParallelismContext,
    precision: PrecisionConfig,
    parallelism: ParallelismConfig,
) -> None:
    """``KimiVLModel._apply_fsdp`` body: vision FSDP before decoder FSDP.

    GPU-VERIFY: ``apply_fsdp_to_multimodal_encoder`` (renamed from the old
    ``apply_fsdp_to_vision_encoder``) and the decoder FSDP chain
    (``MTPDecoder._apply_fsdp``, via ``super()._apply_fsdp(...)``) have not
    been exercised with real multi-GPU TP/EP/FSDP together — confirm the
    vision-before-decoder ordering and the ``dp_mesh_dims``/``shard_on_cp_axis
    =False`` encoder-mesh choice on a GPU node before trusting this path for
    real distributed training.
    """
    from torchtitan.distributed.fsdp import (
        apply_fsdp_to_multimodal_encoder,
        resolve_fsdp_mesh,
    )

    # Encoders are CP-invariant, so FSDP shards them over dp_shard only —
    # same choice MultimodalModel._apply_fsdp makes for its own encoders.
    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context, shard_on_cp_axis=False)
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is not None:
        apply_fsdp_to_multimodal_encoder(
            vision_tower,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
            reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallelism_context.pp_enabled,
            cpu_offload=parallelism.fsdp_enable_cpu_offloading,
            dp_mesh_dims=dp_mesh_dims,
            unshard_in_current_stream=parallelism.fsdp_unshard_in_current_stream,
        )

    # Decoder FSDP via the inherited chain (DeepSeekV3Model -> MTPDecoder ->
    # Decoder._apply_fsdp), which this function's caller invokes via
    # super()._apply_fsdp(...) immediately after calling this helper.
