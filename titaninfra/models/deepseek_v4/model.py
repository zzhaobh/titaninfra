# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DeepSeek V4 model (Phase 1: dense/sliding-window + CSA/HCA compressed attention).

Migrated to the upstream-tip torchtitan API (see
docs/torchtitan_upgrade_plan_2026-11.md §37-39): ``BaseAttention`` -> ``Attention``
(upstream deleted the former), flat ``(T, ...)`` no-batch-dim token-stream
tensors (matching every other already-migrated family and the project-wide
data-pipeline convention), ``attention_masks: AttentionMasksType`` ->
``attention_metadata: AttentionMetadata | None`` (the per-layer single metadata
looked up by ``Decoder``/this model's own ``forward``, not the model-level map).

Compressed layers use upstream's packed-document metadata, ``Compressor`` /
``Indexer`` math, ``InnerIndexer`` selection and Attention Gym inner-attention.
Only the parameter-owning shells remain local: they preserve the production
checkpoint FQNs and the plain compress-RoPE semantics. The sliding path keeps
using ``SlidingWindowFlexInnerAttention``.

Phase 1 has no KV cache. Low-rank Q projection with per-head RMS + RoPE,
single-KV-head projection with RoPE, dense/sliding attention, and grouped
low-rank output projection are ported from the reference without a batch dim.

Faithful to the DeepSeek V4 reference (for HF weight parity): the shared
latent KV feeds attention as BOTH K and V with its rope-dims rotated (value
stream carries positional modulation), and the attention output's rope-dims are
de-rotated (inverse RoPE) before the grouped output projection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import Attention, AttentionMetadata, AttentionMetadataMap
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
)
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.deepseek_v4.sharding import (
    dsa_inner_attention_sharding_config,
)

from titaninfra.models._tt_compat import get_moe_model_nparams_and_flops
from titaninfra.models.deepseek_v4.compressor import (
    CSACompressor,
    HCACompressor,
    _apply_rope_trailing,
    _rope_cos_sin,
    make_compressor_config,
)
from titaninfra.models.deepseek_v4.hyper_connection import (
    HC_MIX_WIDTH,
    hc_head_fn,
    hc_post_fn,
    hc_pre_fn,
)
from titaninfra.models.deepseek_v4.indexer import (
    LightningIndexer,
    make_indexer_config,
)
from titaninfra.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _warn_compress_rope_ignores_yarn() -> None:
    """Warn-once: compressed layers ignore YaRN on ``self.rope``.

    The compressed (CSA/HCA) branch NEVER reads ``self.rope``; it builds its own
    plain (non-YaRN) interleaved RoPE from ``compress_rope_theta`` in
    ``compressor._rope_cos_sin``. For debugmodel-p3 this matches the oracle's
    default compress rope, so this is NOT an error — but a future real-config
    flavor that expects YaRN applied to the compressed KV would be SILENTLY
    diverged. This warn-once flags that gap without hard-failing debugmodel-p3.
    """
    logger.warning(
        "deepseek_v4: model has compressed (CSA/HCA) layers with a YaRN-scaled "
        "self.rope, but the compressed branch uses plain compress_rope_theta "
        "RoPE (YaRN is NOT applied to compressed KV — see compressor._rope_cos_sin). "
        "This matches the HF oracle's default compress rope for debugmodel-p3; if "
        "a real-config flavor expects YaRN on the compress rope, wire it there."
    )


class DeepSeekV4Attention(Attention):
    """DeepSeek V4 attention: dense/sliding-window OR CSA/HCA compressed.

    Low-rank Q projection (``wq_a`` → ``q_norm`` → ``wq_b``) with per-head RMS +
    RoPE, a single KV head (``wkv`` → ``kv_norm`` → RoPE) broadcast across query
    heads via ``enable_gqa=True`` on the sliding path, or routed through
    ``_compressed_forward`` for CSA/HCA layers, and a grouped low-rank output
    projection (``wo_a`` group-wise einsum → ``wo_b``).

    All tensors are flat ``(T, ...)`` — no leading batch dim (see module
    docstring for the document-packing scope boundary this implies).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        dim: int
        head_dim: int = 512
        qk_rope_head_dim: int = 64
        q_lora_rank: int
        o_lora_rank: int
        o_groups: int
        sliding_window: int = 128
        # Compression-layer fields. layer_type ∈ {"sliding_attention",
        # "compressed_sparse_attention","heavily_compressed_attention"}.
        # compress_rate: 0=sliding, 4=CSA, 128=HCA. index_* only used by CSA.
        layer_type: str = "sliding_attention"
        compress_rate: int = 0
        index_topk: int = 512
        index_n_heads: int = 64
        index_head_dim: int = 128
        indexer_loss_coeff: float = 0.0
        compress_rope_theta: float = 160000.0
        max_seq_len: int = 4096
        # Opt-in Stage-B: route the CSA branch through the legacy Triton
        # ``sparse_attn_gather`` kernel instead of upstream inner-attention.
        # The default is upstream only. GPU + triton ONLY; CSA-only (HCA/sliding
        # have no indexer top-k). Default False.
        use_gather_kernel: bool = False
        compressor: HCACompressor.Config | CSACompressor.Config | None = None
        indexer: LightningIndexer.Config | None = None
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: Linear.Config
        wo_b: Linear.Config
        rope: RoPE.Config
        attn_sink_init: float = 0.0
        norm_eps: float = 1e-6

        def flops(self, positions: torch.Tensor) -> int:
            """Count windowed and compressed attention for the new Decoder API."""
            from torchtitan.models.common.flops import quadratic_attention_flops

            # A causal query at document position p attends to p + 1 keys,
            # at most window_size of them uncompressed. Local sliding layers
            # use a generic inner-attention config, so the shared local field
            # is the portable source for every layer type.
            num_keys = positions + 1
            attention_op_flops = quadratic_attention_flops(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                num_keys=num_keys.clamp(max=self.sliding_window),
            )

            if self.compress_rate > 1:
                num_compressed_keys = num_keys // self.compress_rate
                if self.compress_rate == 4:
                    # The indexer scores every compressed key without grad
                    # (forward only); attention reads only the top-k.
                    index_dim = self.index_n_heads * self.index_head_dim
                    attention_op_flops += 2 * index_dim * int(num_compressed_keys.sum())
                    assert self.indexer is not None
                    num_compressed_keys = num_compressed_keys.clamp(
                        max=self.indexer.inner_indexer.topk
                    )
                    # The indexer loss recomputes the top-k scores with grad.
                    # The forward repeats the scoring pass, so only the
                    # backward is new work.
                    if self.inner_attention.aux_loss is not None:
                        attention_op_flops += (
                            4 * index_dim * int(num_compressed_keys.sum())
                        )
                attention_op_flops += quadratic_attention_flops(
                    num_heads=self.n_heads,
                    qk_head_dim=self.head_dim,
                    v_head_dim=self.head_dim,
                    num_keys=num_compressed_keys,
                )
            return attention_op_flops

        def __post_init__(self):
            # dataclass(slots=True) replaces the class, so zero-arg super is invalid.
            Attention.Config.__post_init__(self)
            if self.layer_type == "sliding_attention":
                return
            if self.compressor is None:
                self.compressor = make_compressor_config(
                    dim=self.dim,
                    head_dim=self.head_dim,
                    compress_rate=self.compress_rate,
                    compress_rope_theta=self.compress_rope_theta,
                    max_seq_len=self.max_seq_len,
                    rope_head_dim=self.qk_rope_head_dim,
                    norm_eps=self.norm_eps,
                    overlap=self.layer_type == "compressed_sparse_attention",
                )
            inner_kwargs = {
                "window_size": self.sliding_window,
                "compress_ratio": self.compress_rate,
                "softmax_scale": self.head_dim**-0.5,
                "sharding_config": dsa_inner_attention_sharding_config(
                    compress_ratio=self.compress_rate
                ),
            }
            if self.layer_type == "compressed_sparse_attention":
                if self.indexer is None:
                    self.indexer = make_indexer_config(
                        dim=self.dim,
                        q_lora_rank=self.q_lora_rank,
                        compress_rate=self.compress_rate,
                        index_n_heads=self.index_n_heads,
                        index_head_dim=self.index_head_dim,
                        index_topk=self.index_topk,
                        compress_rope_theta=self.compress_rope_theta,
                        max_seq_len=self.max_seq_len,
                        rope_head_dim=self.qk_rope_head_dim,
                        norm_eps=self.norm_eps,
                    )
                self.inner_attention = CompressedSparseAttention.Config(
                    **inner_kwargs,
                    aux_loss=(
                        None
                        if self.use_gather_kernel or self.indexer_loss_coeff == 0
                        else SparseIndexerLoss.Config(
                            coeff=self.indexer_loss_coeff,
                            reduce_mesh="loss",
                            softmax_scale=self.head_dim**-0.5,
                            num_heads=self.n_heads,
                            compress_ratio=self.compress_rate,
                        )
                    ),
                )
            else:
                self.inner_attention = HeavilyCompressedAttention.Config(
                    **inner_kwargs, aux_loss=None
                )

        def flops_per_token(self, seq_len: int) -> int:
            """Estimate attention FLOPs per token for MFU reporting.

Match upstream DeepSeekV4Attention semantics: every sliding/CSA/HCA layer
uses the sliding-window approximation instead of compress_rate/index_topk-
specific attended-token counts. Trainer requires this estimator during model
initialization.
"""
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            return quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                seq_len=seq_len,
                sliding_window_size=self.sliding_window,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        self.sliding_window = config.sliding_window
        self.eps = config.norm_eps
        self.softmax_scale = self.head_dim**-0.5

        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv = config.wkv.build()
        self.kv_norm = config.kv_norm.build()
        self.wo_a = config.wo_a.build()
        self.wo_b = config.wo_b.build()
        self.rope = config.rope.build()
        self.inner_attention = config.inner_attention.build()

        # Compression (spec §3). Sliding layers keep compressor=None and the
        # unchanged sliding path. CSA/HCA layers build a KV compressor (and,
        # for CSA, a Lightning Indexer for top-k block selection). The
        # compressed KV carries compress_rope_theta rope — NOT the main-theta
        # self.rope — so q/base-kv/output inverse-RoPE in the compressed
        # branch all use compress-theta (see forward()).
        self.layer_type = config.layer_type
        self.compress_rate = config.compress_rate
        self.compress_rope_theta = config.compress_rope_theta
        self.use_gather_kernel = config.use_gather_kernel
        self.compressor = None
        self.indexer = None
        if config.layer_type == "compressed_sparse_attention":
            assert config.compressor is not None and config.indexer is not None
            self.compressor = config.compressor.build()
            self.indexer = config.indexer.build()
        elif config.layer_type == "heavily_compressed_attention":
            assert config.compressor is not None
            self.compressor = config.compressor.build()

        # attn_sink registered for HF-weight parity; not applied under dense
        # FlexInnerAttention (spec §6). It is a *direct* parameter, so the
        # module needs reset_parameters() for torchtitan's init_states (which
        # raises if a module has own params but no param_init/reset_parameters).
        self._attn_sink_init = config.attn_sink_init
        self.attn_sink = nn.Parameter(
            torch.full((self.n_heads,), config.attn_sink_init, dtype=torch.float32)
        )

        # YaRN-gap guard (warn-once, NOT a hard fail — debugmodel-p3's rope IS
        # yarn-scaled and works fine because the compressed branch ignores
        # self.rope entirely). See _warn_compress_rope_ignores_yarn.
        if self.compressor is not None and getattr(config.rope, "scaling", "none") == "yarn":
            _warn_compress_rope_ignores_yarn()

    def reset_parameters(self) -> None:
        # Only attn_sink is a direct param here; sub-modules init themselves.
        with torch.no_grad():
            self.attn_sink.fill_(self._attn_sink_init)

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: AttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        aux_loss_denominator=None,
        padding_mask=None,
    ) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            # Q, KV, compressor, and indexer branches consume the same gathered
            # input at this boundary, matching upstream DeepSeek V4.
            x = spmd.redistribute(
                x,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": x.dtype},
            )
        seqlen = x.size(0)
        rd = self.qk_rope_head_dim

        # Low-rank Q: wq_a -> q_norm -> wq_b, then per-head RMS. The q_residual
        # (post-norm, pre-wq_b) is also the Lightning Indexer's query input
        # (reference `q_a_norm(q_a_proj(x))`), so keep it around for CSA layers.
        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual)
        q = q.view(seqlen, -1, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

        # Single-head KV: wkv -> kv_norm. Shape [T, 1, head_dim].
        kv = self.kv_norm(self.wkv(x))
        kv = kv.view(seqlen, 1, self.head_dim)

        if self.compressor is None:
            # ---- Sliding path (dense/sliding-window FlexInnerAttention) ----
            # Apply RoPE to the last `qk_rope_head_dim` dims of q and the KV head.
            with torch.profiler.record_function("joyai/rope"):
                q_pe, k_pe = self.rope(q[..., -rd:], kv[..., -rd:], positions)
                q = torch.cat([q[..., :-rd], q_pe], dim=-1)
                # DeepSeek V4 shares one latent KV: the SAME rope'd kv feeds
                # attention as BOTH K and V (reference passes a single `kv` to
                # sparse_attn). The value stream therefore carries positional
                # modulation.
                kv = torch.cat([kv[..., :-rd], k_pe], dim=-1)
                k = v = kv

            # The single KV head broadcasts across all query heads via
            # enable_gqa=True. attn_sink is NOT applied here (spec §6).
            o = self.inner_attention(
                q,
                k,
                v,
                attention_metadata=attention_metadata,
                scale=self.softmax_scale,
                enable_gqa=True,
            ).contiguous()

            # Inverse RoPE on the output's rope-dims before the grouped
            # projection (reference: `apply_rotary_emb(o[..., -rd:], freqs_cis,
            # True)`, where `True` = conjugate/inverse rotation). titaninfra's
            # RoPE has no `inverse=` flag, so mirror the reference's complex-
            # conjugate branch: build the ComplexRoPE cache aligned to o's
            # rope-slice, conjugate it, and re-apply. (DeepSeek V4 uses
            # ComplexRoPE; `.conj()` is exact inverse there.)
            with torch.profiler.record_function("joyai/rope_inv"):
                o_pe = o[..., -rd:]
                inv_cache = self.rope._reshape_cache(o_pe, positions).conj()
                o_pe, _ = type(self.rope).apply_rotary_emb(o_pe, o_pe, inv_cache)
                o = torch.cat([o[..., :-rd], o_pe], dim=-1)
        else:
            # ---- Compressed path (CSA / HCA) --------------------------------
            o = self._compressed_forward(
                x,
                q,
                q_residual,
                kv,
                positions,
                attention_metadata,
                aux_loss_denominator=aux_loss_denominator,
                padding_mask=padding_mask,
            )

        # Grouped low-rank output projection: o -> [T, n_groups, -1], group-wise
        # einsum with wo_a's weight, then wo_b. wo_a is applied per-group (not
        # a plain linear), so we use its raw weight reshaped to
        # [n_groups, o_lora_rank, per_group_in], mirroring the reference's
        # `wo_a.weight.view(...)` einsum.
        n_local_heads = o.shape[1]
        n_local_groups = self.n_groups // (self.n_heads // n_local_heads)
        o = o.view(seqlen, n_local_groups, -1)
        wo_a_w = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a_w)
        return self.wo_b(o.flatten(1))

    def _compressed_forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        q_residual: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: AttentionMetadata,
        *,
        aux_loss_denominator=None,
        padding_mask=None,
    ) -> torch.Tensor:
        """CSA/HCA attention over compressed KV.

        Unlike the sliding path, compress layers use the ``compress`` rope
        (theta = ``compress_rope_theta``, interleaved / non-yarn — the SAME
        convention the compressor already commits to), NOT the main-theta
        ``self.rope``. This applies to (a) q + base-kv rope here, and
        (b) the output inverse-rope below (negate sin = conjugate).

        The query attends over ``cat([base_kv, compressed_kv])``: the single
        sliding-window base KV head plus the long-range compressed entries,
        unified under ONE fp32 softmax with an active per-head attention sink.

        Args:
            x: layer input ``[T, dim]`` (compressor / indexer input).
            q: per-head-RMS query ``[T, H, head_dim]`` (pre-rope).
            q_residual: ``[T, q_lora_rank]`` indexer query input.
            kv: single-head base KV ``[T, 1, head_dim]`` (pre-rope).
            positions: absolute query positions ``[T]``.
        """
        seqlen = x.size(0)
        rd = self.qk_rope_head_dim
        device, dtype = x.device, x.dtype

        # (b) compress-theta interleaved rope for q + base-kv. cos/sin are
        # reused for the output inverse-rope (a) below.
        #
        # _rope_cos_sin/_apply_rope_trailing (compressor.py, HF-oracle-
        # verified, see §34.5 -- not touched) require a leading batch axis
        # for their internal broadcasting (`positions[:, None, :]` and
        # `cos.unsqueeze(1)` against a 4D `[B, H, T, head_dim]` x). Wrap with
        # unsqueeze(0)/squeeze(0) at this call boundary, matching the same
        # "flat token stream = batch of 1" adapter used for the compressor/
        # indexer calls below -- a pure shape adapter, not a numerics change.
        cos, sin = _rope_cos_sin(
            rd, positions.unsqueeze(0), self.compress_rope_theta, device, dtype
        )
        q = _apply_rope_trailing(
            q.transpose(0, 1).unsqueeze(0), cos, sin, rd
        ).squeeze(0)  # [H, T, head_dim]
        base_kv = _apply_rope_trailing(
            kv.transpose(0, 1).unsqueeze(0), cos, sin, rd
        ).squeeze(0)  # [1, T, head_dim]

        if attention_metadata is None:
            raise ValueError("Compressed DeepSeek V4 attention requires metadata")
        compressed_kv = self.compressor(x, positions, attention_metadata)
        n_comp = compressed_kv.size(0)

        top_k_indices = topk_scores = None
        if self.indexer is not None:
            return_scores = (
                self.inner_attention.training
                and self.inner_attention.aux_loss is not None
                and not self.use_gather_kernel
            )
            if n_comp == 0:
                top_k_indices = torch.empty(
                    seqlen, 0, dtype=torch.int32, device=x.device
                )
                topk_scores = x.new_empty((seqlen, 0))
            else:
                with torch.set_grad_enabled(
                    torch.is_grad_enabled() and return_scores
                ):
                    top_k_indices, topk_scores = self.indexer(
                        x.detach(),
                        q_residual.detach(),
                        positions=positions,
                        attention_metadata=attention_metadata,
                        return_scores=return_scores,
                    )

        if self.use_gather_kernel and top_k_indices is not None and n_comp > 0:
            # Explicit experiment: retain the legacy Triton gather implementation
            # while the default path below is exclusively upstream inner-attention.
            from titaninfra.models.deepseek_v4.sparse_attn import sparse_attn_gather

            kv_all = torch.cat([base_kv, compressed_kv.unsqueeze(0)], dim=1)
            combined_idx = self._csa_combined_gather_index(top_k_indices, seqlen)
            o = sparse_attn_gather(
                q.unsqueeze(0), kv_all.unsqueeze(0), self.attn_sink,
                combined_idx.unsqueeze(0), self.softmax_scale,
            ).squeeze(0)
            o = o.transpose(0, 1).contiguous()
        else:
            q_THD = q.transpose(0, 1)
            swa_k_TD = base_kv.squeeze(0)
            if self.indexer is not None:
                o = self.inner_attention(
                    q_THD,
                    swa_k_TD,
                    compressed_kv,
                    top_k_indices,
                    topk_scores,
                    self.attn_sink,
                    attention_metadata=attention_metadata,
                    aux_loss_denominator=aux_loss_denominator,
                    padding_mask_T=padding_mask,
                )
            else:
                o = self.inner_attention(
                    q_THD,
                    swa_k_TD,
                    compressed_kv,
                    self.attn_sink,
                    attention_metadata=attention_metadata,
                )

        # (a) inverse rope on the output's rope-dims, compress-theta, conjugate
        # (negate sin) — `apply_rotary_pos_emb(o, cos, -sin)`. Same batch=1
        # wrapper as the forward-rope calls above (cos/sin still carry their
        # leading [1, ...] axis from this method's earlier _rope_cos_sin call).
        o = _apply_rope_trailing(
            o.transpose(0, 1).unsqueeze(0), cos, -sin, rd
        ).squeeze(0).transpose(0, 1)
        return o.contiguous()

    def _csa_combined_gather_index(
        self, top_k_indices: torch.Tensor, seqlen: int
    ) -> torch.Tensor:
        """Fold the dense CSA path's TWO kept-column sets into ONE gather index.

        The dense combined softmax attends over ``cat([base_kv, compressed_kv])``
        (``T + T_c`` columns) where the kept (non-``-inf``) columns per query are:

        * base sliding-window columns: sequence indices ``ki`` with
          ``qi - W < ki <= qi`` (i.e. ``{qi, qi-1, ..., qi-W+1} ∩ [0, T)``), and
        * compressed top-k columns: ``top_k_indices`` (``-1`` = invalid).

        The ``sparse_attn_gather`` kernel instead gathers ``<= k_total`` columns
        of the SINGLE ``kv_all`` tensor per query. To be element-wise equivalent
        we emit exactly those kept columns as one index list into ``kv_all``'s
        ``[0, T + T_c)`` axis:

        * base slot ``w ∈ [0, W)`` → column ``qi - w`` (``-1`` when ``< 0``),
          which lands in ``[0, T)``, and
        * compressed → ``top_k_indices + T`` (``-1`` stays ``-1``), landing in
          ``[T, T + T_c)``.

        The two ranges are disjoint (base ``< T``, compressed ``>= T``) so no
        cross-duplication, and each subset is internally distinct — the kernel's
        set-semantics dedup is a no-op here, matching the dense ``scatter_``
        (each kept column once). Every non-``-1`` index is in ``[0, T + T_c)``:
        base is clamped ``>= 0`` and ``<= qi < T``; compressed is a valid
        ``[0, T_c)`` pick (the indexer clamps OOB/invalid picks to ``-1``)
        shifted by ``T`` into ``[T, T + T_c)`` — so the kernel never reads OOB.

        Args:
            top_k_indices: ``[T, k]`` int, indexer top-k into ``[0, T_c)``;
                ``-1`` = invalid/padding.
            seqlen: base sequence length ``T`` (the shift for the compressed range).

        Returns:
            ``[T, W + k]`` int64 gather index into ``cat([base_kv, compressed_kv])``.
        """
        seq, _ = top_k_indices.shape
        device = top_k_indices.device
        W = self.sliding_window
        qi = torch.arange(seq, device=device).view(seq, 1)
        w = torch.arange(W, device=device).view(1, W)
        base = qi - w  # slot w -> column qi - w, valid iff >= 0
        base = torch.where(base >= 0, base, base.new_full((), -1)).to(torch.int64)
        comp = torch.where(
            top_k_indices >= 0,
            top_k_indices + seqlen,
            top_k_indices.new_full((), -1),
        ).to(torch.int64)
        return torch.cat([base, comp], dim=-1)

class DeepSeekV4TransformerBlock(TransformerBlock):
    """DeepSeek V4 transformer block with Hyper-Connections (mHC).

    Instead of a plain residual, the block maintains ``hc_mult`` copies of the
    hidden state (shape ``[T, hc_mult, dim]``). ``hc_pre`` reduces the copies
    to one via a Sinkhorn-normalized learned mix; the sublayer (attention or
    dense/MoE feed-forward) runs on that single stream; ``hc_post`` expands
    back to ``hc_mult`` copies via learned post-weights and a combination
    matrix.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        dim: int
        hc_mult: int = 4
        hc_sinkhorn_iters: int = 20
        hc_eps: float = 1e-6
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.norm_eps

        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.feed_forward = config.feed_forward.build()

        mix_hc = HC_MIX_WIDTH(self.hc_mult)
        hc_dim = self.hc_mult * self.dim
        # HC mixing params are fp32 (Sinkhorn runs in fp32). These are DIRECT
        # params, so the block defines reset_parameters() for init_states.
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.reset_parameters()

        # Swappable HC impls (optimization seam). Default = eager pure fns;
        # wire_custom_kernels may replace with torch.compile-fused variants.
        self._hc_pre_impl = hc_pre_fn
        self._hc_post_impl = hc_post_fn

    def reset_parameters(self) -> None:
        # fn: small normal (learned mixing projection); base: 0; scale: 1.
        # These give a near-identity HC at init, a stable from-scratch
        # starting point.
        with torch.no_grad():
            nn.init.normal_(self.hc_attn_fn, std=0.02)
            nn.init.normal_(self.hc_ffn_fn, std=0.02)
            self.hc_attn_base.zero_()
            self.hc_ffn_base.zero_()
            self.hc_attn_scale.fill_(1.0)
            self.hc_ffn_scale.fill_(1.0)

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce ``hc_mult`` copies to one. x: [T,hc,d] -> ([T,d], post, comb).

        Delegates to the pure ``hc_pre_fn`` (the optimization seam — the kernel
        registry may swap this for a ``torch.compile``-fused variant).
        ``hc_pre_fn`` itself still assumes a leading batch axis (``[B,L,hc,d]``,
        HF-oracle-independent pure math, not touched by this migration — see
        docs/torchtitan_upgrade_plan_2026-11.md §37.3/§39). Wrap with
        unsqueeze(0)/squeeze(0) here, matching the same call-boundary adapter
        pattern used for compressor.py/indexer.py.
        """
        with torch.profiler.record_function("joyai/hc_pre"):
            y, post, comb = self._hc_pre_impl(
                x.unsqueeze(0), hc_fn, hc_scale, hc_base,
                self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps,
            )
            return y.squeeze(0), post.squeeze(0), comb.squeeze(0)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand one stream back to ``hc_mult`` copies.

        x: [T,d], residual: [T,hc,d], post: [T,hc], comb: [T,hc,hc] -> [T,hc,d].
        Delegates to the pure ``hc_post_fn`` (swappable seam; same batch=1
        wrapper as ``hc_pre`` above).
        """
        with torch.profiler.record_function("joyai/hc_post"):
            return self._hc_post_impl(
                x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
            ).squeeze(0)

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: AttentionMetadata | None,
        positions: torch.Tensor | None = None,
        input_ids_T: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention sublayer with HC mixing (x is [T, hc_mult, dim]).
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attention(
            self.attention_norm(x),
            attention_metadata,
            positions,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask=padding_mask,
        )
        # hc_post reads the attention output with bare ops during remat replay.
        remat.recompute_needs_tensor(x)
        x = self.hc_post(x, residual, post, comb)

        # FFN/MoE sublayer with HC mixing.
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        if isinstance(self.feed_forward, MoE):
            x = self.ffn_norm(x)
            if getattr(self.feed_forward.router, "hash", False):
                x = self.feed_forward(
                    x,
                    padding_mask_T=padding_mask,
                    aux_loss_denominator=aux_loss_denominator,
                    input_ids_T=input_ids_T,
                )
            else:
                x = self.feed_forward(
                    x,
                    padding_mask_T=padding_mask,
                    aux_loss_denominator=aux_loss_denominator,
                )
        else:
            x = self.feed_forward(self.ffn_norm(x))
        # hc_post reads the MoE / feed-forward output with bare ops.
        remat.recompute_needs_tensor(x)
        x = self.hc_post(x, residual, post, comb)
        return x


class DeepSeekV4Model(Decoder):
    """DeepSeek V4 decoder (dense/sliding-window + CSA/HCA compressed attention).

    Extends torchtitan's :class:`Decoder` with Hyper-Connections: the embedding
    is expanded to ``hc_mult`` copies at entry, each block operates on the
    ``[T, hc_mult, dim]`` stream (see :class:`DeepSeekV4TransformerBlock`),
    and a learned ``hc_head`` collapses the copies back to one stream before the
    final norm + lm_head. ``input_ids_T`` (the token ids) are threaded into
    every block so hash-routed MoE layers can look them up.

    MTP (multi-token-prediction) blocks are registered for structural/HF-parity
    completeness but NOT run in forward (``_skip_mtp=True``); real MTP forward
    (upstream's ``mtp_forward``/``roll_mtp_sequence``) is a later phase — same
    scope boundary as before this migration, see module docstring.
    """

    state_dict_adapter_cls = DeepSeekV4StateDictAdapter

    _skip_mtp: bool = True

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    @property
    def mtp_layers(self):
        """Expose MTP blocks to upstream's FSDP helper without registering aliases."""
        return self.mtp

    def _remat_block_container_fqns(self) -> list[str]:
        """Include the structural MTP blocks in upstream activation remat.

        Mirrors upstream DeepSeekV4Model after #6246. The read-only
        ``mtp_layers`` property resolves to ``self.mtp`` without registering a
        second parameter path, so checkpoint FQNs remain unchanged.
        """
        return [*super()._remat_block_container_fqns(), "mtp_layers"]

    def _apply_fsdp(self, *, parallelism_context, precision, parallelism) -> None:
        """Apply per-layer MoE-aware FSDP to the structural-only MTP blocks.

        Upstream #6289 removed ``apply_fsdp_to_mtp_decoder`` and moved its
        behavior into ``MTPDecoder._apply_fsdp``. TitanInfra intentionally
        inherits ``Decoder`` for its structural-only MTP implementation, so it
        mirrors that small lifecycle adapter locally: temporarily append the
        blocks to ``model.layers``, call the real Decoder FSDP path, and
        restore the module tree in ``finally``. The read-only ``mtp_layers``
        property avoids registering a second module path, so checkpoint
        parameter FQNs remain unchanged.
        """
        if not self.mtp_layers:
            return super()._apply_fsdp(
                parallelism_context=parallelism_context,
                precision=precision,
                parallelism=parallelism,
            )
        if parallelism.fsdp_unshard_in_current_stream:
            # Match upstream MTPDecoder: tok_embeddings runs once per MTP
            # depth, which the explicit decoder prefetch chain cannot model.
            raise ValueError(
                "parallelism.fsdp_unshard_in_current_stream is not supported "
                "for models with MTP layers."
            )
        mtp_layer_keys = []
        try:
            first_mtp_layer_id = len(self.layers)
            for i, layer in enumerate(self.mtp_layers):
                key = str(first_mtp_layer_id + i)
                self.layers[key] = layer
                mtp_layer_keys.append(key)
            super()._apply_fsdp(
                parallelism_context=parallelism_context,
                precision=precision,
                parallelism=parallelism,
            )
        finally:
            for key in mtp_layer_keys:
                del self.layers[key]

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 129280
        hc_mult: int = 4
        hc_eps: float = 1e-6
        norm_eps: float = 1e-6
        num_nextn_predict_layers: int = 1

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attn = self.first_attention
            assert isinstance(attn, DeepSeekV4Attention.Config)
            return get_moe_model_nparams_and_flops(
                self, model, attn.n_heads, attn.head_dim, seq_len
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.norm_eps = config.norm_eps

        # Top-level HC head-collapse params (fp32; simple sigmoid mix, no
        # Sinkhorn — reference ParallelHead.hc_head / Transformer.__init__).
        # Direct params → reset_parameters() below for init_states.
        hc_dim = self.hc_mult * config.dim
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.reset_parameters()

        # MTP blocks: structure only (not run in forward, _skip_mtp=True).
        self.mtp = nn.ModuleList(
            [
                config.layers[-1].build()
                for _ in range(config.num_nextn_predict_layers)
            ]
        )

        # Swappable HC-head impl (optimization seam); default = eager pure fn.
        self._hc_head_impl = hc_head_fn

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.hc_head_fn, std=0.02)
            self.hc_head_base.zero_()
            self.hc_head_scale.fill_(1.0)

    # No init_states() override needed: the pre-migration model needed an
    # explicit post-init ``_populate_tid2eid`` call here because the hash
    # router registered an all-zero ``tid2eid`` buffer and relied on this
    # override to fill it in afterward. The new ``DeepSeekV4Router`` (ported
    # from upstream, see router.py) populates ``tid2eid`` with real hash-table
    # values directly in its own ``__init__``/``_init_self_buffers`` — and
    # ``Module.init_states()`` already recursively calls
    # ``_init_self_buffers()`` on every child Module (confirmed by reading
    # ``torchtitan/models/base/module.py``), so the base ``Decoder.
    # init_states()`` now does the right thing with zero overrides.

    def verify_module_protocol(self) -> None:
        """Check every runtime module; only structural-only MTP is exempt."""
        failures: list[tuple[str, str]] = []
        for fqn, mod in self.named_modules():
            if fqn == "mtp" or fqn.startswith("mtp."):
                continue  # Structural-only MTP blocks (not run in forward)
            if not isinstance(mod, Module):
                failures.append((fqn, type(mod).__name__))
        if failures:
            details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
            raise RuntimeError(
                f"The following modules do not satisfy the Module protocol: {details}"
            )

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse ``hc_mult`` copies to one. x: [T,hc,d] -> [T,d].

        Delegates to the pure ``hc_head_fn`` (reference ``ParallelHead.hc_head``:
        RMS-scaled sigmoid-gated weighted sum, no Sinkhorn). Same batch=1
        wrapper as ``DeepSeekV4TransformerBlock.hc_pre``/``hc_post``.
        """
        with torch.profiler.record_function("joyai/hc_head"):
            return self._hc_head_impl(
                x.unsqueeze(0), self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.hc_eps
            ).squeeze(0)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        # Expand to hc_mult copies for Hyper-Connections.
        h = h.unsqueeze(1).repeat(1, self.hc_mult, 1)
        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for layer in self.layers.values():
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention))
            )
            h = layer(
                h,
                layer_attention_metadata,
                positions,
                tokens,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        # Collapse the hc copies back to a single stream, then gather the root
        # norm output for the vocab-parallel LM head and its full-token labels.
        h = self._pre_lm_head_norm_and_gather(self._hc_head(h))
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h
