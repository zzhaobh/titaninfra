"""Gemma4 model configuration and registration.

model_registry returns Gemma4Model.Config; parallelization and checkpoint
adapters are inherited from BaseModel/model classes. Sliding and full attention
use distinct inner_attention config types, and max_context_length is fixed at
construction time.
"""

from collections.abc import Callable
from copy import deepcopy
from functools import partial

import spmd_types as spmd
import torch.nn as nn
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common import CosSinRoPE, Linear, VocabParallelEmbedding
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.decoder_sharding import (
    colwise_sharding_config,
    dense_param_placement,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe.sharding import (
    routed_expert_weight_placement,
    router_gate_sharding_config,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init

from titaninfra.models._tt_compat import make_token_dispatcher_config

from .model import (
    Gemma4Attention,
    Gemma4FeedForward,
    Gemma4Model,
    Gemma4QKVLinear,
    Gemma4TransformerBlock,
)
from .moe import (
    Gemma4Experts,
    Gemma4MoE,
    Gemma4Router,
)

__all__ = [
    "Gemma4Model",
    "Gemma4TransformerBlock",
    "gemma4_configs",
    "model_registry",
]


# ========================================
# Initialization policies.
# ========================================

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}

_EPS = 1e-6

# Gemma4 layer type pattern: [5×sliding, 1×full] × 5
_LAYER_TYPES_26B = (
    ["sliding_attention"] * 5 + ["full_attention"]
) * 5  # 30 layers


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1": partial(nn.init.trunc_normal_, std=0.02),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _gemma4_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT,
        # enable_sp=False: gemma4 has never wired tensor/sequence parallelism
        # (model.py's own upstream-migration docstring already notes the old
        # TP divisibility pre-check was removed as a future-TP placeholder;
        # a reservation for support that was never built). Hardcoded False
        # here follows the exact precedent qwen2_5's equivalent fix uses for
        # the same reason (see qwen2_5/__init__.py::_qwen2_5_norm) -- not a
        # new limitation introduced by this fix.
        sharding_config=norm_sharding_config(enable_sp=False),
    )


def _gemma4_head_norm_sharding(*, has_weight: bool) -> ShardingConfig:
    """Minimal state-only sharding for a Gemma4RMSNorm instance (q_norm/
    k_norm/v_norm/router.norm -- the four direct, Config-less constructions;
    see _components.py's Gemma4RMSNorm docstring for why they take
    sharding_config as a plain kwarg instead of a nested Config field).

    Replicates the weight (if with_scale=True) across TP -- head-dim-sized
    RMSNorm weights are always replicated, independent of sequence
    parallelism, matching every other family's QK-norm treatment (the state
    half of decoder_sharding.head_norm_sharding_config). Deliberately omits
    input/output_shardings: Gemma4RMSNorm.forward's positional arg is named
    ``hidden_states`` (not ``input``), so reusing head_norm_sharding_config
    wholesale would silently no-op its boundary type-check (a dict-key miss
    returns None, not an error) while adding a new, untested code path with
    no bearing on the actual bug this fixes -- FSDP2's "all parameters must
    be DTensors" check reads only state_shardings, never input/output_
    shardings. with_scale=False instances (v_norm, router.norm) have no
    weight parameter at all, so their state_shardings is the empty dict --
    correct and harmless (_distribute_states only iterates parameters that
    actually exist via named_parameters(recurse=False)).
    """
    if not has_weight:
        return ShardingConfig(state_shardings={})
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})


def _gemma4_router_sharding() -> ShardingConfig:
    """Gemma4Router's own direct state: ``scale`` (dim,) and
    ``per_expert_scale`` (num_experts,) -- both raw nn.Parameters set directly
    in Gemma4Router.__init__, not inside a child Linear/Config submodule.

    Both replicated across TP, matching decoder_sharding.
    router_gate_sharding_config's treatment of the common router's own
    .gate Linear: routing decisions can't vary by TP shard, so the router's
    state must be identical on every rank. Under EP, routing happens on the
    full (pre-dispatch) token stream using the full (un-EP-sharded)
    per_expert_scale vector -- selected expert indices range over every
    expert, not just the local EP shard's slice -- so this state is never
    EP-sharded either, consistent with routed_expert_weight_placement only
    applying to the *experts* themselves (Gemma4Experts.w1/w2/w3), never
    the router that decides routing before dispatch.
    """
    return ShardingConfig(
        state_shardings={
            "scale": dense_param_placement(tp=spmd.R),
            "per_expert_scale": dense_param_placement(tp=spmd.R),
        }
    )


def _gemma4_experts_sharding(*, enable_ep: bool) -> ShardingConfig:
    """Gemma4Experts' raw ``w1``/``w2``/``w3`` parameters.

    Gemma4Experts holds plain ``(num_experts, hidden_dim, dim)``-shaped
    nn.Parameters directly (see moe.py) instead of going through the common
    GroupedLinear wrapper that make_routed_experts_config shards via
    ``routed_expert_linear_sharding_config``'s single "weight" field -- same
    underlying placement (``routed_expert_weight_placement``: EP-sparse-
    sharded on dim 0 when EP is enabled, dense-replicated otherwise), just
    applied directly to three named parameters here instead of one "weight"
    inside a child module. Gemma4Experts._experts_forward already defends
    against this parameter being a DTensor (``isinstance(self.w1, DTensor):
    w1 = self.w1.to_local()``) -- that branch was dead code before this fix
    (w1/w2/w3 were never actually distributed), now it is live.
    """
    placement = routed_expert_weight_placement(enable_ep=enable_ep)
    return ShardingConfig(
        state_shardings={"w1": placement, "w2": placement, "w3": placement}
    )


def _gemma4_moe_buffers_sharding() -> ShardingConfig:
    """Gemma4MoE's own direct buffers: ``tokens_per_expert`` (always
    registered) and ``expert_bias`` (only registered when
    ``load_balance_coeff`` is not None -- every current flavor config passes
    ``load_balance_coeff=None``, so this entry is declared for forward
    compatibility and is never looked up today; _distribute_states only
    iterates buffers that actually exist via named_buffers(recurse=False)).

    tokens_per_expert accumulates per-rank partial counts (each DP/CP rank
    processes different data), so DP/CP/TP are all Partial -- reimplemented
    inline rather than importing torchtitan.models.common.moe.sharding.
    _tokens_per_expert_placement, which is module-private (leading
    underscore); the placement itself is the identical reasoning applied to
    gemma4's differently-named buffer (tokens_per_expert, not
    tokens_per_expert_E).
    """
    tokens_per_expert_placement = spmd.SpmdType(
        {
            MeshAxisName.DP: spmd.P,
            MeshAxisName.CP: spmd.P,
            MeshAxisName.TP: spmd.P,
        }
    )
    return ShardingConfig(
        state_shardings={
            "tokens_per_expert": tokens_per_expert_placement,
            "expert_bias": dense_param_placement(tp=spmd.R),
        }
    )


def _gemma4_layer_scalar_sharding() -> ShardingConfig:
    """Gemma4TransformerBlock's own direct ``layer_scalar`` parameter
    (shape ``(1,)``, ``requires_grad=False``) -- replicated across TP.

    This is the exact case named by upstream's own precedent:
    torchtitan/experiments/transformers_modeling_backend/hf_sharding.py's
    "Direct-on-layer state" section names Gemma4's layer_scalar explicitly
    as the motivating example for ``_replicate_sharding_config(layer)``.
    """
    return ShardingConfig(
        state_shardings={"layer_scalar": dense_param_placement(tp=spmd.R)}
    )


def _gemma4_rope_cache_buffers_sharding() -> ShardingConfig:
    """Gemma4Model's direct RoPE caches and ``embed_scale`` buffer.

    Gemma4Model.__init__ copies ``self.rope_sliding.cache`` into a same-
    named direct buffer on ``self`` (``register_buffer("freqs_cis_sliding",
    ...)``) rather than reading through the child ``rope_sliding`` module at
    forward time, and always builds ``freqs_cis_full`` directly with no
    child module backing it at all (``_build_proportional_cos_sin_cache`` --
    the stock ``RoPE.Config`` cannot express gemma4's zero-padded
    proportional layout). Both therefore need their own entry on
    Gemma4Model's own sharding_config, distinct from ``rope_sliding``'s
    ShardingConfig (``rope_cache_sharding_config()``, applied below), which
    covers only its *internal* ``cache`` buffer. Replicated across
    everything, matching decoder_sharding.rope_cache_sharding_config's
    treatment of every other family's RoPE cache: every rank needs the full
    lookup table, there is no sharding benefit.
    """
    return ShardingConfig(
        state_shardings={
            "freqs_cis_sliding": dense_param_placement(tp=spmd.R),
            "freqs_cis_full": dense_param_placement(tp=spmd.R),
            "embed_scale": dense_param_placement(tp=spmd.R),
        }
    )


# ========================================
# Layer builders
# ========================================

def _make_gemma4_attention_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_id: int,
    inner_attention,
    k_eq_v: bool = False,
    rope_dim: int | None = None,
) -> Gemma4Attention.Config:
    """Build Gemma4Attention.Config for a single layer."""
    import dataclasses

    # enable_sp=False: see _gemma4_norm's comment -- gemma4 has never wired
    # tensor/sequence parallelism, hardcoded False follows the qwen2_5
    # precedent for the same reason.
    x_layout = residual_activation_placement(enable_sp=False)

    wq = Linear.Config(
        in_features=dim, out_features=n_heads * head_dim, param_init=_LINEAR_INIT,
        sharding_config=colwise_sharding_config(input_layout=x_layout),
    )
    wk = Linear.Config(
        in_features=dim, out_features=n_kv_heads * head_dim, param_init=_LINEAR_INIT,
        sharding_config=colwise_sharding_config(input_layout=x_layout),
    )
    wv = None if k_eq_v else Linear.Config(
        in_features=dim, out_features=n_kv_heads * head_dim, param_init=_LINEAR_INIT,
        sharding_config=colwise_sharding_config(input_layout=x_layout),
    )
    inner_attention = dataclasses.replace(
        inner_attention, sharding_config=inner_attention_local_spmd_sharding_config()
    )

    return Gemma4Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=Gemma4QKVLinear.Config(head_dim=head_dim, wq=wq, wk=wk, wv=wv),
        wo=Linear.Config(
            in_features=n_heads * head_dim, out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(output_layout=x_layout),
        ),
        inner_attention=inner_attention,
        rope_dim=rope_dim,
        q_norm_sharding_config=_gemma4_head_norm_sharding(has_weight=True),
        k_norm_sharding_config=_gemma4_head_norm_sharding(has_weight=True),
        v_norm_sharding_config=_gemma4_head_norm_sharding(has_weight=False),
    )


def _make_gemma4_moe_config(
    *,
    dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    layer_id: int,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
) -> Gemma4MoE.Config:
    """Build Gemma4MoE.Config."""
    return Gemma4MoE.Config(
        num_experts=num_experts,
        router=Gemma4Router.Config(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init={
                "scale": nn.init.ones_,
                "per_expert_scale": nn.init.ones_,
            },
            proj=Linear.Config(
                in_features=dim, out_features=num_experts,
                param_init=_depth_init(layer_id),
                # Gemma4Router.forward() (moe.py) runs this projection under
                # `torch.autocast(dtype=torch.float32)` for routing-softmax
                # numerical stability (HF Gemma upcasts router logits to
                # fp32); the in-place WGRAD accumulation path does not
                # support running under any active autocast context
                # (common/linear/linear.py's uses_inplace_wgrad_accum check),
                # matching the identical precedent at
                # deepseek_v4/flavors.py:143-150 for its own fp32-autocast
                # compressor projections.
                inplace_wgrad_accum=False,
                sharding_config=router_gate_sharding_config(),
            ),
            norm_sharding_config=_gemma4_head_norm_sharding(has_weight=False),
            sharding_config=_gemma4_router_sharding(),
        ),
        routed_experts=Gemma4Experts.Config(
            dim=dim,
            hidden_dim=moe_hidden_dim,
            num_experts=num_experts,
            param_init=_depth_experts_init(layer_id),
            token_dispatcher=make_token_dispatcher_config(
                num_experts=num_experts,
                top_k=top_k,
                comm_backend=moe_comm_backend,
                hidden_dim=dim,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
            ),
            # The prebuilt config uses the EP-sparse layout; model_registry()
            # replaces it with the dense layout after deepcopy() when ep<=1.
            # Gemma4MoE.Config now follows upstream's MoE.Config contract, so
            # _validate_declared_parallel_axes() recognizes the sparse region.
            # weight_layouts is a config-only child because upstream matches
            # the "routed_experts." prefix (including the trailing dot), while
            # Gemma4Experts owns raw parameters rather than child Linear
            # modules. Gemma4Experts.__init__ applies this same layout directly
            # to itself, preserving runtime moe.experts.w1/w2/w3 FQNs.
            weight_layouts=Module.Config(
                sharding_config=_gemma4_experts_sharding(enable_ep=True)
            ),
        ),
        load_balance_coeff=None,
        sharding_config=_gemma4_moe_buffers_sharding(),
    )


def _make_gemma4_ffn_config(
    *,
    dim: int,
    hidden_dim: int,
    layer_id: int,
) -> Gemma4FeedForward.Config:
    """Build Gemma4FeedForward.Config (GELU)."""
    # enable_sp=False: see _gemma4_norm's comment.
    x_layout = residual_activation_placement(enable_sp=False)
    return Gemma4FeedForward.Config(
        w1=Linear.Config(
            in_features=dim, out_features=hidden_dim, param_init=_LINEAR_INIT,
            sharding_config=colwise_sharding_config(input_layout=x_layout),
        ),
        w2=Linear.Config(
            in_features=hidden_dim, out_features=dim, param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(output_layout=x_layout),
        ),
        w3=Linear.Config(
            in_features=dim, out_features=hidden_dim, param_init=_LINEAR_INIT,
            sharding_config=colwise_sharding_config(input_layout=x_layout),
        ),
    )


def _build_gemma4_layers(
    *,
    layer_types: list[str],
    dim: int,
    n_heads: int,
    sliding_n_kv_heads: int,
    sliding_head_dim: int,
    full_n_kv_heads: int,
    full_head_dim: int,
    full_rope_dim: int,
    mlp_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    sliding_window_size: int,
) -> list[Gemma4TransformerBlock.Config]:
    """Build heterogeneous layers based on layer_types pattern.

    Sliding layers get ``SlidingWindowFlexInnerAttention.Config`` (the window
    bakes causal+sliding masking into the inner_attention type itself); full
    layers get the bare ``FlexInnerAttention.Config``. This replaces the old
    ``mask_type``-string + ``get_attention_masks()`` override dispatch — see
    module docstring and docs/torchtitan_upgrade_plan_2026-11.md §19.2/§23.3
    for why (gpt_oss migration established this pattern first).
    """
    layers = []
    for layer_id, layer_type in enumerate(layer_types):
        is_sliding = layer_type == "sliding_attention"

        if is_sliding:
            inner_attention = SlidingWindowFlexInnerAttention.Config(
                sliding_window_size=sliding_window_size
            )
            attn = _make_gemma4_attention_config(
                dim=dim, n_heads=n_heads, n_kv_heads=sliding_n_kv_heads,
                head_dim=sliding_head_dim, layer_id=layer_id,
                inner_attention=inner_attention,
                k_eq_v=False, rope_dim=None,
            )
        else:
            inner_attention = FlexInnerAttention.Config()
            attn = _make_gemma4_attention_config(
                dim=dim, n_heads=n_heads, n_kv_heads=full_n_kv_heads,
                head_dim=full_head_dim, layer_id=layer_id,
                inner_attention=inner_attention,
                k_eq_v=True, rope_dim=full_rope_dim,
            )

        layers.append(
            Gemma4TransformerBlock.Config(
                attention_norm=_gemma4_norm(dim),
                ffn_norm=_gemma4_norm(dim),
                attention=attn,
                feed_forward=_make_gemma4_ffn_config(
                    dim=dim, hidden_dim=mlp_hidden_dim, layer_id=layer_id,
                ),
                moe=_make_gemma4_moe_config(
                    dim=dim, moe_hidden_dim=moe_hidden_dim,
                    num_experts=num_experts, top_k=top_k, layer_id=layer_id,
                ),
                pre_feedforward_layernorm=_gemma4_norm(dim),
                post_feedforward_layernorm=_gemma4_norm(dim),
                post_feedforward_layernorm_1=_gemma4_norm(dim),
                pre_feedforward_layernorm_2=_gemma4_norm(dim),
                post_feedforward_layernorm_2=_gemma4_norm(dim),
                sharding_config=_gemma4_layer_scalar_sharding(),
            )
        )
    return layers


# ========================================
# Model factory
# ========================================

def _make_gemma4_model(
    *,
    vocab_size: int,
    dim: int,
    layer_types: list[str],
    n_heads: int,
    sliding_n_kv_heads: int,
    sliding_head_dim: int,
    full_n_kv_heads: int,
    full_head_dim: int,
    full_rope_dim: int,
    mlp_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    max_seq_len: int = 4096,
    sliding_theta: float = 10000.0,
    full_theta: float = 1000000.0,
    sliding_window_size: int = 1024,
) -> Gemma4Model.Config:
    import math

    embed_scale = math.sqrt(dim)

    return Gemma4Model.Config(
        max_context_length=max_seq_len,
        vocab_size=vocab_size,
        dim=dim,
        embed_scale=embed_scale,
        final_logit_softcapping=30.0,
        sliding_window_size=sliding_window_size,
        full_head_dim=full_head_dim,
        full_rope_dim=full_rope_dim,
        full_rope_theta=full_theta,
        norm=_gemma4_norm(dim),
        # HF Gemma4Config defaults tie_word_embeddings=True (confirmed for both
        # debugmodel and 26B-A4B via this shared factory). Declaring it here
        # routes tying through Decoder.__init__/init_states and lets
        # apply_fsdp_to_decoder group tok_embeddings/norm/lm_head into one
        # FSDP unit -- the only way FSDP2 tolerates two modules sharing one
        # nn.Parameter (see Gemma4Model.__init__/init_states in model.py).
        enable_weight_tying=True,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
            # enable_sp=False: see _gemma4_norm's comment.
            sharding_config=token_embedding_sharding_config(enable_sp=False),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        rope_sliding=CosSinRoPE.Config(
            dim=sliding_head_dim,
            max_context_length=max_seq_len,
            theta=sliding_theta,
            sharding_config=rope_cache_sharding_config(),
        ),
        layers=_build_gemma4_layers(
            layer_types=layer_types,
            dim=dim,
            n_heads=n_heads,
            sliding_n_kv_heads=sliding_n_kv_heads,
            sliding_head_dim=sliding_head_dim,
            full_n_kv_heads=full_n_kv_heads,
            full_head_dim=full_head_dim,
            full_rope_dim=full_rope_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            moe_hidden_dim=moe_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            sliding_window_size=sliding_window_size,
        ),
        # Gemma4Model's own direct freqs_cis_sliding/freqs_cis_full buffers --
        # see _gemma4_rope_cache_buffers_sharding's docstring for why these
        # are distinct from rope_sliding's own sharding_config above.
        sharding_config=_gemma4_rope_cache_buffers_sharding(),
    )


# ========================================
# Gemma4 configurations maintained by TitanInfra.
# ========================================

gemma4_configs = {
    "debugmodel": _make_gemma4_model(
        vocab_size=151936,
        dim=256,
        layer_types=(["sliding_attention"] * 2 + ["full_attention"]) * 2,  # 6 layers
        n_heads=8,
        sliding_n_kv_heads=4,
        sliding_head_dim=64,
        full_n_kv_heads=2,
        full_head_dim=128,
        full_rope_dim=32,
        mlp_hidden_dim=512,
        moe_hidden_dim=128,
        num_experts=16,
        top_k=4,
    ),
    "26B-A4B": _make_gemma4_model(
        vocab_size=262144,
        dim=2816,
        layer_types=_LAYER_TYPES_26B,
        n_heads=16,
        sliding_n_kv_heads=8,
        sliding_head_dim=256,
        full_n_kv_heads=2,
        full_head_dim=512,
        full_rope_dim=128,
        mlp_hidden_dim=2112,
        moe_hidden_dim=704,
        num_experts=128,
        top_k=8,
        max_seq_len=262144,
        sliding_theta=10000.0,
        full_theta=1000000.0,
        sliding_window_size=1024,
    ),
}


def model_registry(
    flavor: str,
    parallelism=None,
) -> Gemma4Model.Config:
    """Return a Gemma4Model.Config for a bare flavor key.

Config objects are prebuilt and deep-copied before selecting the EP layout.
ep<=1 uses dense layouts; ep>1 preserves sparse layouts without changing
runtime parameter FQNs. Runtime Gemma4MoE remains family-owned because its
dual-input forward, custom router, GELU, and lack of shared experts differ
from the common MoE."""
    import dataclasses

    config = deepcopy(gemma4_configs[flavor])
    enable_ep = (parallelism.expert_parallel_degree > 1) if parallelism is not None else False
    if not enable_ep:
        dense_sharding = _gemma4_experts_sharding(enable_ep=False)
        new_layers = [
            dataclasses.replace(
                layer,
                moe=dataclasses.replace(
                    layer.moe,
                    routed_experts=dataclasses.replace(
                        layer.moe.routed_experts,
                        weight_layouts=dataclasses.replace(
                            layer.moe.routed_experts.weight_layouts,
                            sharding_config=dense_sharding,
                        ),
                    ),
                ),
            )
            for layer in config.layers
        ]
        config = dataclasses.replace(config, layers=new_layers)
    return config
