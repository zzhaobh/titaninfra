"""Gemma4 Decoder with hybrid sliding/full attention, parallel MLP+MoE,
weight tying, logit softcapping, and position-based document masks.

The custom forward carries two RoPE cache choices and uses per-layer
inner_attention types to select metadata."""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch_remat as remat
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import (
    Attention as BaseAttention,
)
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    FlexInnerAttention,
    InnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE

from titaninfra.models.gemma4 import _components as C
from titaninfra.models.gemma4.moe import Gemma4MoE
from titaninfra.models.gemma4.state_dict_adapter import Gemma4StateDictAdapter

# ========================================
# Gemma4 Feed-Forward (GELU instead of SiLU)
# ========================================

class Gemma4FeedForward(Module):
    """Gemma4 GELU feed-forward — gate/up/down with gelu_pytorch_tanh."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config  # gate_proj
        w2: Linear.Config  # down_proj
        w3: Linear.Config  # up_proj

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.w3 = config.w3.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF Gemma4TextMLP: down(gelu_tanh(gate(x)) * up(x))
        return C.gemma4_mlp_forward(x, self.w1, self.w2, self.w3)


# ========================================
# Gemma4 MoE (router + GELU experts) lives in moe.py, following the gpt_oss
# precedent (self-owned MoE, not the common MoE/GroupedExperts). Import it from
# there: `from titaninfra.models.gemma4.moe import Gemma4MoE`.
# ========================================

# Numeric primitives live in _components.py (HF-aligned). This module-level alias
# keeps the historical name used by the full-attention RoPE cache builders below.
_build_proportional_cos_sin_cache = C.build_proportional_cos_sin_cache


# ========================================
# Gemma4 Attention
# ========================================

class Gemma4QKVLinear(Module):
    """Gemma4 QKV projection with optional K==V sharing.

    Sliding layers: separate Q, K, V projections.
    Full layers: separate Q, K projections, V reuses K (wv=None).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build() if config.wv is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (T, dim) flat token stream, no batch dim.
        T = x.shape[0]
        xq = self.wq(x).view(T, -1, self.head_dim)
        xk = self.wk(x).view(T, -1, self.head_dim)
        xv = xk if self.wv is None else self.wv(x).view(T, -1, self.head_dim)
        return xq, xk, xv


class Gemma4Attention(BaseAttention):
    """Gemma4 attention with partial rotary embedding and optional K==V sharing.

    Full attention layers: head_dim=512, partial_rotary_factor=0.25 (rope_dim=128),
    K==V sharing, 2 KV heads.

    Sliding attention layers: head_dim=256, full rotary, separate Q/K/V, 8 KV heads.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int = 16
        n_kv_heads: int = 8
        head_dim: int = 256
        dim: int
        qkv_linear: Gemma4QKVLinear.Config
        wo: Linear.Config
        inner_attention: InnerAttention.Config = field(
            default_factory=FlexInnerAttention.Config
        )
        rope_dim: int | None = None
        # Gemma4RMSNorm (q_norm/k_norm/v_norm) has no .Config of its own (see
        # _components.py docstring), so its sharding_config can't be threaded
        # through a nested Config field the way every other model component
        # is -- these three fields carry it instead, read directly by
        # __init__ below and passed to Gemma4RMSNorm's optional
        # sharding_config kwarg. None is a valid value (no sharding applied,
        # the pre-2026-11-20 behavior); titaninfra.models.gemma4.__init__
        # always supplies one so FSDP2 sees a DTensor for q_norm/k_norm's
        # weight and v_norm's (parameter-free) state.
        q_norm_sharding_config: ShardingConfig | None = None
        k_norm_sharding_config: ShardingConfig | None = None
        v_norm_sharding_config: ShardingConfig | None = None

        def flops_per_token(self, seq_len: int) -> int:
            """Estimate attention FLOPs per token for MFU reporting.

Sliding layers use their configured sliding_window_size; full-attention
layers use global attention. Trainer initializes this estimator for every
run, so the method is required even when metrics are not inspected."""
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            return quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                seq_len=seq_len,
                sliding_window_size=getattr(
                    self.inner_attention, "sliding_window_size", None
                ),
            )

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads
        self.rope_dim = config.rope_dim

        self.qkv_linear = config.qkv_linear.build()
        self.wo = config.wo.build()
        self.inner_attention = config.inner_attention.build()

        # q/k/v norms — gemma4-owned HF-aligned RMSNorm (HF Gemma4TextAttention
        # @1210-1215). q_norm/k_norm carry a learnable weight; v_norm does NOT
        # (with_scale=False). Same FQNs as before: attention.q_norm.weight /
        # k_norm.weight (no v_norm.weight) — DCP-compatible.
        self.q_norm = C.Gemma4RMSNorm(
            config.head_dim, eps=1e-6, with_scale=True,
            sharding_config=config.q_norm_sharding_config,
        )
        self.k_norm = C.Gemma4RMSNorm(
            config.head_dim, eps=1e-6, with_scale=True,
            sharding_config=config.k_norm_sharding_config,
        )
        self.v_norm = C.Gemma4RMSNorm(
            config.head_dim, eps=1e-6, with_scale=False,
            sharding_config=config.v_norm_sharding_config,
        )

        self.scaling = 1.0

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_metadata,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (T, dim) flat token stream, no batch dim.
        T = x.shape[0]
        xq, xk, xv = self.qkv_linear(x)

        # q_norm/k_norm/v_norm read the qkv_linear projection outputs (each a
        # remat.region output -- Gemma4QKVLinear.forward calls Linear.forward
        # for wq/wk/wv) with bare ops. All three must be marked here: this is
        # not just the q_norm instance the GPU regression first surfaced, but
        # the same gap on k_norm/v_norm's reads of wk/wv's outputs too (when
        # self.wv is None, xv is xk itself -- the common case needs no extra
        # handling since recompute_needs_tensor on the same tensor twice is
        # harmless).
        remat.recompute_needs_tensor(xq, xk, xv)
        xq = self.q_norm(xq)
        xv = self.v_norm(xv)
        xk = self.k_norm(xk)

        # RoPE is applied to the FULL head for both sliding and full layers
        # (gemma4-owned, HF-aligned). For full-attention layers the cache is the
        # HF "proportional" layout: only the first `full_rope_dim` channels carry
        # nonzero frequencies, the rest are zero (cos=1, sin=0 → identity).
        # Slicing the head manually would pair channel k with k+rope_dim/2 instead
        # of HF's k+head_dim/2 — that was the loss=441 bug.
        #
        # _components.apply_rotary_emb_from_cache (and the HF-parity tests that
        # exercise it directly) keep the batched (B, S, H, D) convention on
        # purpose, for byte-for-byte auditability against HF's own calling
        # convention. This flat-token-stream forward wraps a size-1 batch axis
        # around the call and unwraps it after — the _components module itself
        # is untouched. Positions are supplied explicitly (never None) at this
        # boundary: _select_cos_sin's None-branch produces a 1D index -> 2D
        # cos/sin, which breaks unsqueeze_dim=2's broadcast against a 4D
        # (B,S,H,D) tensor unless head_dim == n_heads by coincidence. The
        # pre-migration code only worked because production always supplied
        # explicit (B,S)-shaped positions (confirmed in
        # tests/test_gemma4_hf_parity.py, which always passes
        # positions.unsqueeze(0)) — never actually relying on the None branch
        # with real multi-dim tensors. Preserve that invariant explicitly
        # rather than re-deriving it.
        positions_1T = (
            positions if positions is not None else torch.arange(T, device=x.device)
        ).unsqueeze(0)
        xq_b, xk_b = C.apply_rotary_emb_from_cache(
            xq.unsqueeze(0), xk.unsqueeze(0), rope_cache, positions_1T
        )
        xq, xk = xq_b.squeeze(0), xk_b.squeeze(0)

        # Attention math: production uses inner_attention (flex/BlockMask-capable
        # dispatch required by MoE+packing), standard scaled-dot-product attention
        # with scale=self.scaling (1.0 for gemma4 — q_norm/k_norm supply the
        # normalization). _components.gemma4_sdpa is the numerically equivalent
        # gemma4-owned reference exercised by the CPU parity tests.
        output = self.inner_attention(
            xq, xk, xv,
            attention_metadata=attention_metadata,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        output = output.reshape(output.shape[0], -1).contiguous()
        return self.wo(output)


# ========================================
# Gemma4 Transformer Block
# ========================================

class Gemma4TransformerBlock(TransformerBlock):
    """Gemma4 Transformer Block — parallel MLP + MoE with per-layer scalar.

    Forward:
      residual = x
      x = attention(input_layernorm(x)) → post_attention_layernorm → + residual
      mlp_out = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(residual)))
      moe_out = post_feedforward_layernorm_2(moe(pre_feedforward_layernorm_2(residual)))
      x = (residual + post_feedforward_layernorm(mlp_out + moe_out)) × layer_scalar
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        moe: Gemma4MoE.Config  # pyrefly: ignore [bad-override] (base TransformerBlock.Config no longer declares `moe`; this family needs both feed_forward AND moe simultaneously, not either/or)
        pre_feedforward_layernorm: RMSNorm.Config | None = None
        post_feedforward_layernorm: RMSNorm.Config | None = None
        post_feedforward_layernorm_1: RMSNorm.Config | None = None
        pre_feedforward_layernorm_2: RMSNorm.Config | None = None
        post_feedforward_layernorm_2: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()

        assert config.moe is not None
        self.moe = config.moe.build()

        assert config.pre_feedforward_layernorm is not None
        self.pre_feedforward_layernorm = config.pre_feedforward_layernorm.build()
        assert config.post_feedforward_layernorm is not None
        self.post_feedforward_layernorm = config.post_feedforward_layernorm.build()
        assert config.post_feedforward_layernorm_1 is not None
        self.post_feedforward_layernorm_1 = config.post_feedforward_layernorm_1.build()
        assert config.pre_feedforward_layernorm_2 is not None
        self.pre_feedforward_layernorm_2 = config.pre_feedforward_layernorm_2.build()
        assert config.post_feedforward_layernorm_2 is not None
        self.post_feedforward_layernorm_2 = config.post_feedforward_layernorm_2.build()

        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)
        self._param_init = {"layer_scalar": nn.init.ones_}

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: dict[str, torch.Tensor],
        attention_metadata_map: AttentionMetadataMap | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        # Per-layer attention-metadata dispatch, inlined (mirrors what
        # Decoder.forward_embeddings does generically for a homogeneous layer
        # loop — gemma4 can't delegate to it because its own forward() needs
        # an extra freqs_cis argument that generic signature has no room for).
        if isinstance(self.attention.inner_attention, SlidingWindowFlexInnerAttention):
            rope_cache = freqs_cis["sliding"]
        else:
            rope_cache = freqs_cis["full"]
        layer_attention_metadata = (
            None
            if attention_metadata_map is None
            else attention_metadata_map.get(type(self.attention.inner_attention))
        )

        residual = x
        # padding_mask is deliberately NOT threaded into attention, matching
        # every upstream reference family: llama3/qwen3/deepseek_v3's
        # TransformerBlock.forward() (torchtitan/models/{llama3,qwen3,
        # deepseek_v3}/model.py) all call
        # `self.attention(norm(x), attention_metadata, positions)` with no
        # padding_mask -- none of the 3 reference families pass it at this
        # layer either. For gemma4's own inner_attention types (Flex /
        # SlidingWindowFlex, built in gemma4/__init__.py::_build_gemma4_layers)
        # this is also a safe no-op at the mask level, not just "upstream
        # doesn't do it either": both
        # FlexInnerAttention.Config.build_attention_metadata and
        # SlidingWindowFlexInnerAttention.Config.build_attention_metadata
        # (torchtitan/models/common/attention/attention.py:687-696,
        # 869-878) receive padding_mask and immediately `del` it -- the
        # packed-document BlockMask they build uses only `positions`
        # (document boundaries at positions==0), never padding_mask. (Only
        # the Varlen backend's build_attention_metadata, which gemma4 does
        # not use, actually consumes padding_mask -- attention.py:1086-1127.)
        # This matches tests/test_p2_not_pad_deadcode.py::
        # test_not_pad_is_deadcode_gemma4, which independently proves real-
        # token query rows are bit-identical whether or not a not-pad term
        # is added to gemma4's own causal+document mask.
        x = self.attention(self.attention_norm(x), rope_cache, layer_attention_metadata, positions)
        # ffn_norm reads the attention output (self.wo's remat.region output,
        # the return value of Gemma4Attention.forward()) with bare ops. The
        # marker lives here, at the call site, rather than inside
        # Gemma4Attention.forward() itself, because nothing inside that
        # method's own body reads self.wo(...)'s result -- it is returned
        # directly and consumed by this bare norm call one level up.
        remat.recompute_needs_tensor(x)
        x = self.ffn_norm(x)
        residual = residual + x

        # padding_mask is also NOT threaded into the dense feed_forward,
        # matching upstream: llama3/model.py:47-56 accepts then
        # `del aux_loss_denominator, padding_mask` outright (dense FFN, no
        # MoE); qwen3/deepseek_v3's own dense branch
        # (`self.feed_forward(self.ffn_norm(x))` when feed_forward is not an
        # MoE) never passes it either. Dense FFN compute is per-token
        # regardless, so this is a no-op either way.
        ffn_out = self.feed_forward(self.pre_feedforward_layernorm(residual))
        # post_feedforward_layernorm_1 reads the feed_forward output (w2's
        # remat.region output, via Gemma4FeedForward -> _components.
        # gemma4_mlp_forward) with bare ops.
        remat.recompute_needs_tensor(ffn_out)
        mlp_out = self.post_feedforward_layernorm_1(ffn_out)
        # padding_mask_T IS threaded into the MoE branch, copied verbatim
        # from upstream's MoE-branch call (qwen3/model.py:63-68,
        # deepseek_v3/model.py:55-60):
        #   ffn_out = self.feed_forward(norm(x), padding_mask_T=padding_mask,
        #                                aux_loss_denominator=aux_loss_denominator)
        # Gemma4MoE/Gemma4Router declare no aux_loss field at all (moe.py),
        # so only padding_mask_T carries over -- see Gemma4MoE.forward()
        # (moe.py) for why it is accepted-and-unused there, same as
        # upstream's own RoutedExperts.forward() (common/moe/moe.py:128-148).
        moe_raw = self.moe(
            residual,
            self.pre_feedforward_layernorm_2(residual),
            padding_mask_T=padding_mask,
        )
        # post_feedforward_layernorm_2 reads the moe output with bare ops.
        # Whether that output is actually a remat.region output is topology-
        # dependent: Gemma4MoE.forward() ultimately returns whatever
        # token_dispatcher.combine() returns. At ep=1 (LocalTokenDispatcher,
        # e.g. this family's GPU verification script) combine() never wraps
        # its result in a remat.region, so the marker below is a no-op; at
        # ep>1 (AllToAllTokenDispatcher), combine() returns the output of a
        # genuine "<fqn>.combine" remat.region, so the marker is required
        # there to avoid a placeholder crash. Marking unconditionally is
        # correct and cheap (a no-op when the producer isn't a droppable
        # region) and keeps this model code correct regardless of the
        # runtime EP degree, which model code cannot assume.
        remat.recompute_needs_tensor(moe_raw)
        moe_out = self.post_feedforward_layernorm_2(moe_raw)
        # mlp_out/moe_out are each the bare output of a norm call (not
        # remat.region outputs themselves -- a norm's own forward is an
        # ordinary nn.Module call, never a remat.region), so this add and
        # the trailing "(residual + x) * layer_scalar" below need no markers
        # or remat.region wrapping. This differs from qwen3/llama3/
        # deepseek_v3's trailing residual add (`x + ffn_out`), where
        # `ffn_out` is consumed immediately after `self.feed_forward(...)`
        # with no norm in between and is therefore still a live
        # remat.region output at the point of the add. Gemma4 always
        # interposes a bare norm (post_feedforward_layernorm[_1/_2]) between
        # the last remat.region-wrapped projection and the residual
        # arithmetic below, which already forces real data by the time it
        # is read here -- converting these adds/multiply to
        # remat.region(torch.add/torch.mul, ..., recompute=False) would be
        # unnecessary over-annotation, not a correctness fix.
        x = self.post_feedforward_layernorm(mlp_out + moe_out)
        x = (residual + x) * self.layer_scalar
        return x


# ========================================
# Gemma4 Model
# ========================================

class Gemma4Model(Decoder):
    """Gemma4 model with hybrid attention, parallel MLP+MoE, and softcapping."""

    state_dict_adapter_cls = Gemma4StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2816
        vocab_size: int = 262144
        embed_scale: float = 1.0
        final_logit_softcapping: float = 30.0
        sliding_window_size: int = 1024
        rope_sliding: RoPE.Config | None = None
        # Full-attention "proportional" RoPE geometry (HF parity). The full head is
        # `full_head_dim` wide but only the first `full_rope_dim` channels rotate, with
        # inv_freq computed over `full_head_dim` as denominator (see
        # _build_proportional_cos_sin_cache). Defaults match gemma-4-26B-A4B.
        full_head_dim: int = 512
        full_rope_dim: int = 128
        full_rope_theta: float = 1_000_000.0
        # 2026-11-21: gemma4 never declared this field (it inherited the
        # empty-list default from Decoder.Config/BaseModel.Config), so
        # BaseModel's own `config.local_compile.apply()` call at trainer
        # construction time left every @local_compile-registered function
        # running eager for this family -- a correctness-neutral but real
        # performance gap relative to qwen3/llama3/deepseek_v3/qwen3_5,
        # which all declare this list. Checked gemma4's actual call graph
        # against every @local_compile-registered name in torchtitan (loss,
        # fused_binary_activation, rope, fp32_to_bf16_split, gated_rmsnorm,
        # moe_aux_loss, moe_masked_output_postprocess, mla_qk, mla_kv,
        # gdn_split_in_proj, offset_rmsnorm, partial_rope,
        # shared_expert_gate) rather than copying another family's list:
        # - "loss": torchtitan/components/loss.py's cross_entropy_loss /
        #   cross_entropy_and_logsumexp are the generic, model-agnostic loss
        #   functions every family's training loop reaches regardless of
        #   architecture -- applicable, included below.
        # - All the others are tied to specific shared-module code paths
        #   gemma4 never calls: fused_binary_activation/BinaryActivationFn
        #   (gemma4's dense FFN and experts call raw F.gelu directly, not
        #   common/activation.py's gated-activation classes);
        #   rope/CosSinRoPE.forward (gemma4 only reads
        #   rope_sliding.cache as a buffer at init time -- its actual RoPE
        #   application is the gemma4-owned
        #   _components.apply_rotary_emb_from_cache/apply_rotary_pos_emb,
        #   which never calls CosSinRoPE.forward/__call__);
        #   fp32_to_bf16_split/HiMidLoLinear, gated_rmsnorm (gemma4 uses the
        #   common RMSNorm, which is not @local_compile-decorated at all,
        #   and its own Gemma4RMSNorm, which isn't either), moe_aux_loss/
        #   moe_masked_output_postprocess (common MoE class, not Gemma4MoE),
        #   mla_qk/mla_kv (DeepSeek MLA only), gdn_split_in_proj/
        #   offset_rmsnorm/partial_rope/shared_expert_gate (qwen3_5's hybrid
        #   attention only) -- none reachable from this family's forward
        #   pass, so none belong here. Not applicable; not declared.
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(regions=["loss"])
        )

    def __init__(self, config: Config):
        super().__init__(config)
        # Keep the Python float's float64 value so the forward-time cast has
        # exactly the same input and rounding mode as torch.tensor(float, dtype=...).
        # A non-persistent buffer follows model device migration without adding a
        # checkpoint key; _init_self_buffers recreates it after meta/to_empty().
        self.register_buffer(
            "embed_scale",
            torch.tensor(config.embed_scale, dtype=torch.float64),
            persistent=False,
        )
        self.final_logit_softcapping = config.final_logit_softcapping

        # Dual RoPE caches.
        # Sliding layers: standard cos/sin RoPE over the full sliding head_dim.
        # Full layers: HF "proportional" RoPE — only the first `full_rope_dim`
        # channels rotate, but inv_freq uses `full_head_dim` as denominator and the
        # cache is `full_head_dim`-wide so rotate_half pairs k with k+full_head_dim/2.
        # Built directly (not via RoPE.Config) because the stock cos/sin builder
        # cannot produce the zero-padded proportional layout.
        self.rope_sliding = config.rope_sliding.build() if config.rope_sliding else None
        if self.rope_sliding is not None:
            self.register_buffer("freqs_cis_sliding", self.rope_sliding.cache, persistent=False)
        self.register_buffer(
            "freqs_cis_full",
            _build_proportional_cos_sin_cache(
                head_dim=config.full_head_dim,
                rope_dim=config.full_rope_dim,
                theta=config.full_rope_theta,
                max_seq_len=config.max_context_length,
            ),
            persistent=False,
        )

    def _apply_fsdp(self, *, parallelism_context, precision, parallelism) -> None:
        """EP-aware experts FSDP before decoder FSDP -- see parallelize.py's
        ``apply_fsdp_to_gemma4_experts`` docstring for the full rationale
        (mirrors upstream's ``MultimodalModel._apply_fsdp``/
        ``MTPDecoder._apply_fsdp`` override pattern and this repo's own
        ``KimiVLModel._apply_fsdp``: this is the sanctioned hook, not a
        free-standing ``parallelize_fn`` -- that concept no longer exists
        post-ModelSpec removal)."""
        from titaninfra.models.gemma4.parallelize import apply_fsdp_to_gemma4_experts

        apply_fsdp_to_gemma4_experts(
            self,
            parallelism_context=parallelism_context,
            precision=precision,
            parallelism=parallelism,
        )
        super()._apply_fsdp(
            parallelism_context=parallelism_context,
            precision=precision,
            parallelism=parallelism,
        )

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if buffer_device is None:
            buffer_device = self.freqs_cis_full.device
        super().init_states(buffer_device=buffer_device)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        if self.rope_sliding is not None:
            self.freqs_cis_sliding = self.rope_sliding.cache
        elif self.config.rope_sliding is not None:
            rope_s = self.config.rope_sliding.build()
            rope_s._init_self_buffers(buffer_device=buffer_device)
            self.freqs_cis_sliding = rope_s.cache
        full_cache = _build_proportional_cos_sin_cache(
            head_dim=self.config.full_head_dim,
            rope_dim=self.config.full_rope_dim,
            theta=self.config.full_rope_theta,
            max_seq_len=self.config.max_context_length,
        )
        if buffer_device is not None:
            full_cache = full_cache.to(buffer_device)
        self.freqs_cis_full = full_cache
        if buffer_device is None:
            buffer_device = self.embed_scale.device
        self.embed_scale = torch.tensor(
            self.config.embed_scale,
            dtype=torch.float64,
            device=buffer_device,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        # tokens: (T,) flat token stream, no batch dim.
        #
        # padding_mask: keyword-only, mirrors Decoder.forward()'s own
        # signature (torchtitan/models/common/decoder.py:316-324) exactly.
        # Required because Decoder.preprocess_inputs() -- inherited
        # unchanged by Gemma4Model, which has no override -- reads
        # input_dict.get("padding_mask") but never pops it
        # (decoder.py:411-421), so the key stays in the dict that becomes
        # model_kwargs and is always passed as a keyword argument to
        # model_parts[0](inputs, **model_kwargs)
        # (torchtitan/training_engine.py:865), even when its value is None.
        # Without this parameter every real training step (not only ones
        # with actual padding) raises "Gemma4Model.forward() got an
        # unexpected keyword argument 'padding_mask'".
        if positions is not None:
            positions = positions.clamp(min=0)

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        # HF casts the sqrt(dim) normalizer to the embedding dtype BEFORE multiply,
        # so sqrt(2816)=53.0667 rounds to bf16 (~53.0). Match that to stay bit-aligned.
        # The float64 buffer is already on h's device; only a same-device dtype
        # cast remains in forward, which is safe during accelerator graph capture.
        h = h * self.embed_scale.to(dtype=h.dtype)

        freqs = {
            "sliding": self.freqs_cis_sliding,
            "full": self.freqs_cis_full,
        }

        for layer in self.layers.values():
            h = layer(h, freqs, attention_metadata, positions, padding_mask=padding_mask)

        h = self.norm(h) if self.norm is not None else h
        output = self.lm_head(h) if self.lm_head is not None else h

        if self.final_logit_softcapping > 0:
            # The divide reads the lm_head output (a remat.region output)
            # with bare ops. Unlike the token-embedding scale multiply and
            # the inter-layer `h` above, this one genuinely is a live region
            # output: self.lm_head(h) is a plain Linear call with nothing
            # bare interposed before this line reads it.
            remat.recompute_needs_tensor(output)
            output = torch.tanh(output / self.final_logit_softcapping) * self.final_logit_softcapping

        return output

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
