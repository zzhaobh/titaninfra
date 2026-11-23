"""Gemma4 self-owned MoE (router + GELU experts + aux-loss-free balancing).

Following the torchtitan ``gpt_oss`` precedent, gemma4 maintains its own MoE
classes rather than reusing the common ``MoE``/``GroupedExperts``. Gemma4's MoE
diverges structurally from the common one in ways that make the common code a
poor fit:

  - **Dual input**: ``Gemma4MoE.forward(x_router, x_experts)`` — the router sees
    the raw post-attention residual while the experts see a separately-normed
    tensor (HF Gemma4TextDecoderLayer parallel MLP+MoE branch). The common
    ``MoE.forward(x)`` is single-input.
  - **Custom router**: ``Gemma4Router`` has ``.proj`` / ``.scale`` /
    ``.per_expert_scale`` (HF Gemma4TextRouter), NOT the common router's
    ``.gate``.
  - **GELU experts**: gelu-tanh activation (vs the common SwiGLU).
  - **No shared_experts**.

The runtime classes remain gemma4-owned for these architectural reasons. Only
``Gemma4MoE.Config`` inherits upstream ``MoE.Config`` so upstream validation
can identify the sparse expert region; its ``routed_experts`` config field is
still built as runtime ``self.experts`` and does not change checkpoint FQNs.

Expert numerics delegate to ``_components`` (HF-aligned). Parameter FQNs are
unchanged from the original in-``model.py`` definitions — ``moe.experts.w1/w2/w3``,
``moe.router.scale`` / ``per_expert_scale`` / ``proj.weight`` — so DCP checkpoint
loading is unaffected.

EP/TP sharding for these classes lives in __init__.py and parallelize.py and
requires multi-GPU validation.

Migration notes: inputs are flat (T, dim) token streams. dispatch() also
requires routing_map_TE, a boolean one-hot top-k selection matrix returned by
Gemma4Router. combine() accepts routed_output_RD, metadata, and x_TD; Gemma4
has no shared_experts argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.tensor import DTensor, Partial
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE

from titaninfra.models.gemma4 import _components as C

# ========================================
# Gemma4 Experts (GELU instead of SiLU)
# ========================================


class Gemma4Experts(Module):
    """Gemma4 experts with GELU activation.

    Same parameter layout as the common GroupedExperts (w1, w2, w3) but uses
    gelu-tanh. Reuses a token_dispatcher for EP compatibility. Expert matmuls
    delegate to ``_components`` (HF-aligned, parity-tested).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        use_grouped_mm: bool = True
        token_dispatcher: Module.Config
        # Config-only child: upstream's sparse-prefix check requires a path
        # segment below routed_experts. Runtime parameters intentionally stay
        # directly on Gemma4Experts to preserve moe.experts.w1/w2/w3 FQNs.
        weight_layouts: Module.Config

    def __init__(self, config: Config):
        super().__init__()
        self._sharding_config = config.weight_layouts.sharding_config
        self.num_experts = config.num_experts
        self.w1 = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim, config.dim))
        self.w2 = nn.Parameter(torch.empty(config.num_experts, config.dim, config.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim, config.dim))
        self.use_grouped_mm = config.use_grouped_mm
        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
        self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3
        if self.use_grouped_mm:
            return C.gemma4_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
        return C.gemma4_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

    def forward(
        self, x: torch.Tensor, top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        routed_input, num_tokens_local, metadata = self.token_dispatcher.dispatch(
            x, top_scores, selected_experts_indices, routing_map
        )
        routed_output = self._experts_forward(routed_input, num_tokens_local)
        return self.token_dispatcher.combine(routed_output, metadata, x)


# ========================================
# Gemma4 Router — HF Gemma4TextRouter @ modeling_gemma4.py:1334
# ========================================


class Gemma4Router(Module):
    """Gemma4 custom router.

    HF op order: norm(no scale) → x * scale * dim^-0.5 → proj → softmax(dim=-1)
    → topk → renormalize by sum → × per_expert_scale[selected]. Router math is
    exercised by ``tests/test_gemma4_hf_parity.py::test_router_selection_matches_hf``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_experts: int
        top_k: int = 8
        proj: Linear.Config
        _debug_force_load_balance: bool = False
        # Gemma4RMSNorm (router pre-norm) has no .Config of its own (see
        # _components.py docstring) -- same reasoning as Gemma4Attention's
        # q/k/v_norm_sharding_config fields (model.py). None is a valid value
        # (no sharding applied); titaninfra.models.gemma4.__init__ always
        # supplies one so FSDP2 sees this module's (parameter-free, with_scale
        # =False) state correctly declared.
        norm_sharding_config: ShardingConfig | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self._debug_force_load_balance = config._debug_force_load_balance

        # Router pre-norm — HF Gemma4TextRouter uses an RMSNorm without scale.
        # No learnable weight → no FQN (DCP-neutral).
        self.norm = C.Gemma4RMSNorm(
            config.dim, with_scale=False,
            sharding_config=config.norm_sharding_config,
        )
        self.scale = nn.Parameter(torch.ones(config.dim))
        self.proj = config.proj.build()
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            h = self.norm(x)
            h = h * (self.scale * (x.shape[-1] ** -0.5))
            scores = self.proj(h)

        # softmax reads the router projection output (self.proj's
        # remat.region output) with bare ops. This is the same bug class as
        # the reported attention q_norm crash, one level deeper in the MoE
        # branch: self.proj is a plain Linear call with nothing bare
        # interposed before this line reads its result.
        remat.recompute_needs_tensor(scores)
        scores = F.softmax(scores, dim=1)

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        if self._debug_force_load_balance:
            n_tokens = scores.size(0)
            selected = (
                torch.arange(n_tokens * self.top_k, device=scores.device, dtype=torch.int64)
                .reshape(n_tokens, self.top_k) % self.num_experts
            )
            top_scores = scores.gather(dim=1, index=selected)
        else:
            _, selected = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
            top_scores = scores.gather(dim=1, index=selected)

        denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denominator

        top_scores = top_scores * self.per_expert_scale[selected]

        num_tokens_per_expert = torch.histc(
            # torch.histc's CPU kernel only supports float dtypes (confirmed:
            # "histogram_cpu" not implemented for 'Long' — a pre-existing gap
            # in gemma4's own CPU compatibility, not introduced by this
            # migration; gemma4's forward pass was never actually exercised
            # on CPU before this round's verification). GPU's kernel accepts
            # integer dtypes directly. Cast through float32 here — lossless
            # for any realistic token count (well under float32's 2^24 exact-
            # integer range) and numerically identical to the integer-dtype
            # result; the CPU/GPU dtype difference doesn't change the counts.
            selected.view(-1).float(), bins=self.num_experts, min=0, max=self.num_experts,
        )

        # One-hot boolean routing map (T, E) — required by the new
        # AllToAllTokenDispatcher.dispatch() signature. Construction mirrors
        # torchtitan's own TokenChoiceTopKRouter._select_experts byte-for-byte
        # (see module docstring).
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)

        return top_scores, selected, num_tokens_per_expert, routing_map


# ========================================
# Gemma4 MoE — HF Gemma4TextDecoderLayer MoE branch @ modeling_gemma4.py:1429
# ========================================


class Gemma4MoE(Module):
    """Gemma4 MoE: custom router + GELU experts + auxiliary-loss-free load balancing.

    Dual-input: ``forward(x_router, x_experts)`` — the router scores the raw
    post-attention residual (``x_router``) while the experts process a separately
    pre-normed tensor (``x_experts``). This mirrors HF where the router takes
    ``residual`` and the experts take ``pre_feedforward_layernorm_2(residual)``.

    ``x_router``/``x_experts`` are ``(T, D)`` flat token streams (no batch
    dim), matching the post-migration convention of the rest of this family.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        """Use upstream's sparse-region config contract without renaming state.

        ``routed_experts`` is a config-tree name only. ``Gemma4MoE.__init__``
        still builds it as ``self.experts`` so checkpoint FQNs remain
        ``moe.experts.w1/w2/w3``. The runtime module deliberately does not
        inherit ``MoE``: its dual-input forward and custom router are not
        compatible with the common implementation's module structure.
        """

        num_experts: int = 128
        routed_experts: Gemma4Experts.Config
        router: Gemma4Router.Config
        load_balance_coeff: float | None = 1e-3

        def __post_init__(self) -> None:
            expert_counts = {
                "moe": self.num_experts,
                "router": self.router.num_experts,
                "routed_experts": self.routed_experts.num_experts,
            }
            if len(set(expert_counts.values())) != 1:
                raise ValueError(
                    "MoE expert counts must match: "
                    + ", ".join(f"{owner}={count}" for owner, count in expert_counts.items())
                )
            if self.shared_experts is not None:
                raise ValueError("Gemma4MoE does not support common shared_experts")
            if self.freeze_expert_bias:
                raise ValueError("Gemma4MoE does not implement freeze_expert_bias")

        @property
        def experts(self) -> Gemma4Experts.Config:
            """Preserve the historical read-only config accessor."""
            return self.routed_experts

    def __init__(self, config: Config):
        super().__init__()
        self.experts = config.routed_experts.build()
        self.router = config.router.build()
        self.load_balance_coeff = config.load_balance_coeff
        num_experts = config.num_experts

        if self.load_balance_coeff is not None:
            self.register_buffer(
                "expert_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True,
            )
        else:
            self.expert_bias = None
        self.register_buffer(
            "tokens_per_expert", torch.zeros(num_experts, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        x_router: torch.Tensor,
        x_experts: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route ``x_router`` and compute GELU experts on ``x_experts``.

        ``padding_mask_T`` mirrors the keyword-only param upstream's common
        ``MoE.forward()`` accepts and threads to its router/routed_experts
        (torchtitan/models/common/moe/moe.py:827-834) -- gemma4's own
        TransformerBlock.forward() (model.py) passes it here the exact same
        way qwen3/deepseek_v3's TransformerBlock.forward() pass it to their
        MoE call. It is accepted-and-unused here for the same reason
        upstream's own RoutedExperts.forward() accepts-and-deletes it
        (common/moe/moe.py:128-148): "padding rows are dispatched and
        computed like real tokens" -- expert FFN compute is per-token
        (dispatch/combine never mix rows), so a padding row's output lands
        only in its own row and is discarded later by the caller's loss
        mask; it never corrupts a real token's output. (Independently
        confirmed for gemma4's own causal + position-document attention mask
        by tests/test_p2_not_pad_deadcode.py::test_not_pad_is_deadcode_gemma4,
        which proves real-token query rows are bit-identical with or without
        a not-pad term.)

        Gemma4 does not even reach the one place upstream's router *does*
        use this mask -- excluding padding from the tokens_per_expert
        load-balancing stats before an expert_bias update (common/moe/
        moe.py:355-363): every gemma4 flavor sets load_balance_coeff=None
        (gemma4/__init__.py::_make_gemma4_moe_config), so self.expert_bias
        below is always None; the common register_moe_load_balancing_hook
        only recognizes the common ``MoE`` class via isinstance and never
        matches this self-owned Gemma4MoE; and moe_health.py's MoE
        monitoring does not see gemma4 layers today either
        (_find_moe_layers() gates on a ``moe_enabled`` attribute
        Gemma4TransformerBlock never sets). So unlike upstream,
        self.tokens_per_expert's accumulation below has no consumer at all
        right now -- masking it would be new routing-logic behavior with no
        existing precedent at this layer to copy, not a signature-only
        change, so it is deliberately left untouched here.
        """
        del padding_mask_T
        if isinstance(x_router, DTensor):
            assert x_router.device_mesh.ndim == 1
            x_router = x_router.to_local(grad_placements=(Partial(),))
        if isinstance(x_experts, DTensor):
            assert x_experts.device_mesh.ndim == 1
            x_experts = x_experts.to_local(grad_placements=(Partial(),))

        top_scores, selected, num_tokens_per_expert, routing_map = self.router(
            x_router, self.expert_bias
        )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        return self.experts(x_experts, top_scores, selected, routing_map)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert isinstance(buffer_device, torch.device)
        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
