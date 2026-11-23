"""DeepSeek V4 config registry — flavor factories + ``model_registry``.

Flavors cover dense/sliding-window and CSA/HCA compressed attention; fp4/fp8
remain deferred:

* ``debugmodel`` — tiny CI/CPU-node model.
* ``16B`` — titaninfra-original V4-shaped model sized to Moonlight-16B-A3B scale
  (dim 2048 / 27 layers / 64 experts). NOT a released checkpoint.
* ``Flash`` — DeepSeek-V4-Flash (confirmed config.json values).
* ``Pro`` — DeepSeek-V4-Pro / "1.6T" (confirmed config.json values).

All flavors use ``ComplexRoPE`` — REQUIRED: the attention's output inverse-RoPE
(``model.py``) uses complex-conjugate de-rotation, which is only correct for a
complex RoPE cache (a real cos/sin cache would make ``.conj()`` a no-op → wrong).

Hash routing: the first ``num_hash_layers`` MoE layers route via a per-token-id
``tid2eid`` buffer. That buffer is populated deterministically by
``_populate_tid2eid`` (token_id → experts by modular hash); without population
every token would route to expert 0.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial

import torch.nn as nn
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import (
    ComplexRoPE,
    HiMidLoLinear,
    Linear,
    RMSNorm,
    RowParallelLinear,
    SqrtSoftplus,
    VocabParallelEmbedding,
)
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    colwise_sharding_config,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    replicated_activation_placement,
    replicated_state_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe.sharding import router_gate_sharding_config
from torchtitan.models.common.rope import YaRNScalingConfig

from titaninfra.models.deepseek_v4.model import (
    DeepSeekV4Attention,
    DeepSeekV4Model,
    DeepSeekV4TransformerBlock,
)
from titaninfra.models.deepseek_v4.router import DeepSeekV4Router
from titaninfra.models.deepseek_v4.sharding import (
    attention_sharding_config,
    deepseek_v4_router_sharding_config,
    hc_head_sharding_config,
    transformer_block_sharding_config,
    wo_a_sharding_config,
)

_LINEAR_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02), "bias": nn.init.zeros_}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_EXPERTS_INIT = {
    "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
    "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
    "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


_LAYER_TYPE_TO_RATE = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def _layer_types(n_layers: int) -> list[str]:
    """HF-default schedule: 2× HCA bootstrap + CSA(odd)/HCA(even) interleave."""
    interleave = [
        "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention" for i in range(max(n_layers - 2, 0))
    ]
    return ["heavily_compressed_attention"] * min(n_layers, 2) + interleave


def _rope_config(dim: int, max_seq_len: int) -> ComplexRoPE.Config:
    # YARN scaling per confirmed config.json (factor 16, orig 65536, theta 10000).
    return ComplexRoPE.Config(
        dim=dim,
        max_context_length=max_seq_len,
        theta=10000.0,
        scaling=YaRNScalingConfig(
            factor=16.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=65536,
        ),
    )


def _make_dsv4_attn_config(
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    o_groups: int,
    sliding_window: int,
    max_seq_len: int,
    attn_backend: str,
    layer_type: str = "sliding_attention",
    compress_rate: int = 0,
    index_topk: int = 512,
    index_n_heads: int = 64,
    index_head_dim: int = 128,
    compress_rope_theta: float = 160000.0,
    use_gather_kernel: bool = False,
    enable_sp: bool = False,
) -> DeepSeekV4Attention.Config:
    del attn_backend  # only "flex" makes sense here; sliding layers always
    # build SlidingWindowFlexInnerAttention directly (see docstring above).
    inner_attention = (
        SlidingWindowFlexInnerAttention.Config(sliding_window_size=sliding_window)
        if layer_type == "sliding_attention"
        else FlexInnerAttention.Config()  # never invoked for compressed layers
    )
    inner_attention = dataclasses.replace(
        inner_attention,
        sharding_config=inner_attention_local_spmd_sharding_config(),
    )
    return DeepSeekV4Attention.Config(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        sliding_window=sliding_window,
        layer_type=layer_type,
        compress_rate=compress_rate,
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        compress_rope_theta=compress_rope_theta,
        max_seq_len=max_seq_len,
        use_gather_kernel=use_gather_kernel,
        wq_a=Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank,
            param_init=_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=colwise_sharding_config(input_layout=replicated_activation_placement()),
        ),
        wkv=Linear.Config(
            in_features=dim,
            out_features=head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=head_dim,
            param_init=_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        wo_a=Linear.Config(
            in_features=(n_heads * head_dim) // o_groups,
            out_features=o_groups * o_lora_rank,
            param_init=_LINEAR_INIT,
            sharding_config=wo_a_sharding_config(),
        ),
        wo_b=RowParallelLinear.Config(
            in_features=o_groups * o_lora_rank,
            out_features=dim,
            param_init=_LINEAR_INIT,
            sharding_config=rowwise_sharding_config(output_layout=residual_activation_placement(enable_sp=enable_sp)),
        ),
        rope=dataclasses.replace(
            _rope_config(qk_rope_head_dim, max_seq_len),
            sharding_config=rope_cache_sharding_config(),
        ),
        inner_attention=inner_attention,
        sharding_config=attention_sharding_config(enable_sp=enable_sp),
    )


def _make_dsv4_moe_config(
    *,
    layer_id: int,
    dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    num_hash_layers: int,
    vocab_size: int,
    routed_scaling_factor: float,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    enable_sp: bool = False,
    enable_ep: bool = False,
):
    router = DeepSeekV4Router.Config(
        num_experts=num_experts,
        gate=HiMidLoLinear.Config(
            in_features=dim,
            out_features=num_experts,
            backward_mode="hi_mid_lo",
            bias=False,
            param_init=_LINEAR_INIT,
            sharding_config=router_gate_sharding_config(),
        ),
        top_k=top_k,
        score_func=SqrtSoftplus.Config(),
        route_norm=True,
        route_scale=routed_scaling_factor,
        n_hash_layers=num_hash_layers,
        layer_id=layer_id,
        vocab_size=vocab_size,
        sharding_config=deepseek_v4_router_sharding_config(
            layer_id=layer_id,
            n_hash_layers=num_hash_layers,
        ),
    )
    shared = (
        make_shared_expert_ffn_config(
            dim=dim,
            hidden_dim=moe_hidden_dim * num_shared_experts,
            w13_param_init=_LINEAR_INIT,
            w2_param_init=_LINEAR_INIT,
            enable_sp=enable_sp,
        )
        if num_shared_experts > 0
        else None
    )
    routed_experts_cfg = make_routed_experts_config(
        dim=dim,
        hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        param_init=_EXPERTS_INIT,
        enable_ep=enable_ep,
    )
    if moe_comm_backend != "standard":
        from titaninfra.models._tt_compat import make_token_dispatcher_config

        routed_experts_cfg = dataclasses.replace(
            routed_experts_cfg,
            token_dispatcher=make_token_dispatcher_config(
                num_experts=num_experts,
                top_k=top_k,
                comm_backend=moe_comm_backend,
                hidden_dim=dim,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
            ),
        )
    return make_moe_config(
        num_experts=num_experts,
        router=router,
        routed_experts=routed_experts_cfg,
        shared_experts=shared,
        load_balance_coeff=1e-3,
        enable_sp=enable_sp,
    )


def _build_dsv4_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    o_groups: int,
    sliding_window: int,
    max_seq_len: int,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    num_hash_layers: int,
    vocab_size: int,
    routed_scaling_factor: float,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    hc_eps: float,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    layer_types: list[str] | None = None,
    use_gather_kernel: bool = False,
    enable_sp: bool = False,
    enable_ep: bool = False,
) -> list[DeepSeekV4TransformerBlock.Config]:
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        layer_type = "sliding_attention" if layer_types is None else layer_types[layer_id]
        attn_cfg = _make_dsv4_attn_config(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
            sliding_window=sliding_window,
            max_seq_len=max_seq_len,
            attn_backend=attn_backend,
            layer_type=layer_type,
            compress_rate=_LAYER_TYPE_TO_RATE[layer_type],
            use_gather_kernel=use_gather_kernel,
            enable_sp=enable_sp,
        )
        if layer_id < n_dense_layers:
            feed_forward = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_LINEAR_INIT,
                enable_sp=enable_sp,
            )
        else:
            # Router layer_id is MoE-RELATIVE (layer_id - n_dense_layers) so the
            # first `num_hash_layers` MoE layers are hash-routed. Using the
            # absolute layer_id would let the dense prefix swallow the hash range
            # (e.g. debug: n_dense=1, num_hash=1 → 0 hash-MoE layers). The
            # reference V4 has no dense prefix, so MoE-relative matches its intent.
            feed_forward = _make_dsv4_moe_config(
                layer_id=layer_id - n_dense_layers,
                dim=dim,
                moe_hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                num_hash_layers=num_hash_layers,
                vocab_size=vocab_size,
                routed_scaling_factor=routed_scaling_factor,
                moe_comm_backend=moe_comm_backend,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
                enable_sp=enable_sp,
                enable_ep=enable_ep,
            )
        layers.append(
            DeepSeekV4TransformerBlock.Config(
                dim=dim,
                hc_mult=hc_mult,
                hc_sinkhorn_iters=hc_sinkhorn_iters,
                hc_eps=hc_eps,
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                    sharding_config=norm_sharding,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                    sharding_config=norm_sharding,
                ),
                feed_forward=feed_forward,
                sharding_config=transformer_block_sharding_config(),
            )
        )
    return layers


def _make_model_config(
    *,
    dim: int,
    n_layers: int,
    n_dense_layers: int,
    vocab_size: int,
    n_heads: int,
    q_lora_rank: int,
    o_lora_rank: int,
    o_groups: int,
    moe_hidden_dim: int,
    dense_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    num_hash_layers: int,
    routed_scaling_factor: float,
    max_seq_len: int,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    head_dim: int = 512,
    qk_rope_head_dim: int = 64,
    sliding_window: int = 128,
    num_nextn_predict_layers: int = 1,
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    layer_types: list[str] | None = None,
    use_gather_kernel: bool = False,
    parallelism: ParallelismConfig | None = None,
) -> DeepSeekV4Model.Config:
    enable_sp = parallelism.enable_sequence_parallel if parallelism is not None else False
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    layers = _build_dsv4_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        sliding_window=sliding_window,
        max_seq_len=max_seq_len,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        num_hash_layers=num_hash_layers,
        vocab_size=vocab_size,
        routed_scaling_factor=routed_scaling_factor,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        hc_eps=hc_eps,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        layer_types=layer_types,
        use_gather_kernel=use_gather_kernel,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
    )
    return DeepSeekV4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        max_context_length=max_seq_len,
        hc_mult=hc_mult,
        num_nextn_predict_layers=num_nextn_predict_layers,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        sharding_config=hc_head_sharding_config(),
    )


def _debugmodel(
    attn_backend, moe_comm_backend, non_blocking_capacity_factor=None, parallelism: ParallelismConfig | None = None
):
    return _make_model_config(
        parallelism=parallelism,
        dim=256,
        n_layers=6,
        n_dense_layers=1,
        # vocab matches the qwen3 tokenizer used for GPU test MDS (avoids embed
        # OOB → SIGABRT when the debugmodel trains on real tokenized data; see
        # memory project_debugmodel_vocab_mismatch_sigabrt).
        vocab_size=151936,
        n_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        moe_hidden_dim=256,
        dense_hidden_dim=1024,
        num_experts=8,
        num_shared_experts=1,
        top_k=3,
        num_hash_layers=1,
        routed_scaling_factor=1.5,
        max_seq_len=4096,
        sliding_window=8,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def _debugmodel_p3(
    attn_backend, moe_comm_backend, non_blocking_capacity_factor=None, parallelism: ParallelismConfig | None = None
):
    # Same shape as `_debugmodel`, but with the HF-default compression schedule
    # (`_layer_types`) applied so the 6 layers carry HCA/CSA `layer_type` +
    # `compress_rate` slots for Phase-3 wiring. Only this flavor is scheduled;
    # every Phase-1 flavor stays pure sliding (spec §4).
    return _make_model_config(
        parallelism=parallelism,
        dim=256,
        n_layers=6,
        n_dense_layers=1,
        vocab_size=151936,
        n_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        moe_hidden_dim=256,
        dense_hidden_dim=1024,
        num_experts=8,
        num_shared_experts=1,
        top_k=3,
        num_hash_layers=1,
        routed_scaling_factor=1.5,
        max_seq_len=4096,
        sliding_window=8,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        layer_types=_layer_types(6),
    )


def _debugmodel_p3_32k(
    attn_backend, moe_comm_backend, non_blocking_capacity_factor=None, parallelism: ParallelismConfig | None = None
):
    # Identical to `_debugmodel_p3` but with `max_seq_len=32768` so the Phase-3
    # long-sequence case (case_42) can validate that KV compression (HCA 128x /
    # CSA 4x) keeps a 32K context trainable without OOM. Only RoPE freq-table
    # sizing changes; every other shape matches `_debugmodel_p3`.
    return _make_model_config(
        parallelism=parallelism,
        dim=256,
        n_layers=6,
        n_dense_layers=1,
        vocab_size=151936,
        n_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        moe_hidden_dim=256,
        dense_hidden_dim=1024,
        num_experts=8,
        num_shared_experts=1,
        top_k=3,
        num_hash_layers=1,
        routed_scaling_factor=1.5,
        max_seq_len=32768,
        sliding_window=8,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        layer_types=_layer_types(6),
    )


def _debugmodel_p3_gather(
    attn_backend, moe_comm_backend, non_blocking_capacity_factor=None, parallelism: ParallelismConfig | None = None
):
    # Identical to `_debugmodel_p3` but flips `use_gather_kernel=True` so the CSA
    # layers route through the Stage-B Triton `sparse_attn_gather` kernel
    # (compute-equivalent to the dense scatter-bias path). GPU + triton ONLY.
    return _make_model_config(
        parallelism=parallelism,
        dim=256,
        n_layers=6,
        n_dense_layers=1,
        vocab_size=151936,
        n_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        moe_hidden_dim=256,
        dense_hidden_dim=1024,
        num_experts=8,
        num_shared_experts=1,
        top_k=3,
        num_hash_layers=1,
        routed_scaling_factor=1.5,
        max_seq_len=4096,
        sliding_window=8,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        layer_types=_layer_types(6),
        use_gather_kernel=True,
    )


def _16b(
    attn_backend, moe_comm_backend, non_blocking_capacity_factor=None, parallelism: ParallelismConfig | None = None
):
    # titaninfra-original V4-shaped model at Moonlight-16B-A3B scale (NOT a released checkpoint).
    return _make_model_config(
        parallelism=parallelism,
        dim=2048,
        n_layers=27,
        n_dense_layers=1,
        vocab_size=129280,
        n_heads=16,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=4,
        moe_hidden_dim=1408,
        dense_hidden_dim=11264,
        num_experts=64,
        num_shared_experts=1,
        top_k=6,
        num_hash_layers=3,
        routed_scaling_factor=1.5,
        max_seq_len=8192,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def _flash(
    attn_backend, moe_comm_backend, non_blocking_capacity_factor=None, parallelism: ParallelismConfig | None = None
):
    # DeepSeek-V4-Flash (confirmed config.json).
    return _make_model_config(
        parallelism=parallelism,
        dim=4096,
        n_layers=43,
        n_dense_layers=2,
        vocab_size=129280,
        n_heads=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        moe_hidden_dim=2048,
        dense_hidden_dim=8192,
        num_experts=256,
        num_shared_experts=1,
        top_k=6,
        num_hash_layers=3,
        routed_scaling_factor=1.5,
        max_seq_len=4096,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def _pro(
    attn_backend, moe_comm_backend, non_blocking_capacity_factor=None, parallelism: ParallelismConfig | None = None
):
    # DeepSeek-V4-Pro / "1.6T" (confirmed config.json).
    return _make_model_config(
        parallelism=parallelism,
        dim=7168,
        n_layers=61,
        n_dense_layers=3,
        vocab_size=129280,
        n_heads=128,
        q_lora_rank=1536,
        o_lora_rank=1024,
        o_groups=16,
        moe_hidden_dim=3072,
        dense_hidden_dim=18432,
        num_experts=384,
        num_shared_experts=1,
        top_k=6,
        num_hash_layers=3,
        routed_scaling_factor=2.5,
        max_seq_len=4096,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def _16b_csa(
    attn_backend, moe_comm_backend, non_blocking_capacity_factor=None, parallelism: ParallelismConfig | None = None
):
    # 16B-A3B scale with CSA/HCA sparse attention ENABLED (layer_types) and
    # n_dense_layers=0 so the layer schedule matches HF DeepseekV4 (which has no
    # dense layers) for HF weight loading via the state_dict adapter. Used to
    # validate the full HF-init-weight → titaninfra-train path with the V4 sparse
    # attention (Lightning Indexer + compressor) exercised.
    return _make_model_config(
        parallelism=parallelism,
        dim=2048,
        n_layers=27,
        n_dense_layers=0,
        vocab_size=129280,
        n_heads=16,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=4,
        moe_hidden_dim=1408,
        dense_hidden_dim=11264,
        num_experts=64,
        num_shared_experts=1,
        top_k=6,
        num_hash_layers=3,
        routed_scaling_factor=1.5,
        max_seq_len=8192,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        layer_types=_layer_types(27),
    )


deepseek_v4_configs: dict[str, Callable] = {
    "debugmodel": _debugmodel,
    "debugmodel-p3": _debugmodel_p3,
    "debugmodel-p3-gather": _debugmodel_p3_gather,
    "debugmodel-p3-32k": _debugmodel_p3_32k,
    "16B": _16b,
    "16B-csa": _16b_csa,
    "Flash": _flash,
    "Pro": _pro,
}


# NOTE: the old _populate_tid2eid() post-init hook was removed here -- the
# new DeepSeekV4Router (router.py, ported from upstream, §38.3) builds its
# own real hash-routing table directly in __init__/_init_self_buffers
# (torch.rand(...).topk(...), not the old deterministic-formula stub), and
# Module.init_states() already calls _init_self_buffers() recursively on
# every child -- so this file no longer needs any post-init population
# step, and DeepSeekV4Model.init_states() has no override any more either.


# Helpers exposed for unit tests (kept in sync with the flavor factories).
def _debugmodel_attn_config(layer_id: int = 0, attn_backend: str = "flex"):
    return _make_dsv4_attn_config(
        dim=256,
        n_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        sliding_window=8,
        max_seq_len=4096,
        attn_backend=attn_backend,
    )


def _debugmodel_block_config(layer_id: int = 1, attn_backend: str = "flex"):
    cfg = _debugmodel(attn_backend, "standard")
    return cfg.layers[layer_id]


def _debugmodel_csa_attn_config(
    attn_backend: str = "flex",
    use_gather_kernel: bool = False,
    index_topk: int = 512,
):
    return _make_dsv4_attn_config(
        dim=256,
        n_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        sliding_window=8,
        max_seq_len=4096,
        attn_backend=attn_backend,
        layer_type="compressed_sparse_attention",
        compress_rate=4,
        index_topk=index_topk,
        index_n_heads=4,
        index_head_dim=32,
        use_gather_kernel=use_gather_kernel,
    )


def _debugmodel_hca_attn_config(
    attn_backend: str = "flex", compress_rate: int = 128
):
    return _make_dsv4_attn_config(
        dim=256,
        n_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        sliding_window=8,
        max_seq_len=4096,
        attn_backend=attn_backend,
        layer_type="heavily_compressed_attention",
        compress_rate=compress_rate,
    )


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    parallelism: ParallelismConfig | None = None,
):
    """Build a DeepSeekV4Model.Config with topology-aware declarative sharding.

    ``parallelism=None`` disables SP/EP for legacy single-rank callers. Real
    callers receive the active ParallelismConfig from load_model_spec(), so SP
    and EP placements are fixed correctly before Module._parallelize() runs.
    """
    config = deepseek_v4_configs[flavor](
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        parallelism=parallelism,
    )
    return config
