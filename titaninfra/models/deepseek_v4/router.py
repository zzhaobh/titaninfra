# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DeepSeek V4 MoE router: hash-based routing for early layers.

Ported from upstream torchtitan's ``torchtitan/models/deepseek_v4/moe.py::
DeepSeekV4Router`` (see docs/torchtitan_upgrade_plan_2026-11.md §37.2):
upstream's new ``TokenChoiceTopKRouter.Config.score_func`` is a real
``UnaryActivationFn.Config`` field (not a string), and ``SqrtSoftplus`` is a
registered activation class in ``torchtitan.models.common.activation`` — so
DeepSeek V4's sqrtsoftplus scoring no longer needs a custom string branch
(the old titaninfra router's ``__init__`` had to pass a throwaway
``Sigmoid.Config()`` to its parent just to get past a ``.build()`` call that
expected a real Config, then swap ``self.score_func`` back to a string
afterward — that workaround is gone). ``route_norm``/``route_scale``/
``aux_loss``/``padding_mask_T`` are now base-class ``TokenChoiceTopKRouter``
features (see that class's ``Config``/``forward``), so this subclass only
needs to add the hash-routing override, exactly like upstream.

Hash routing: the first ``n_hash_layers`` MoE layers route via a per-token-id
``tid2eid`` lookup buffer (deterministic function of the token id) rather than
the gate scores.

**Correctness improvement over the old titaninfra router (not just a
refactor)**: the old ``DeepSeekV4Router._forward_hash`` used UNIFORM combine
weights (``1/top_k``) and never called ``self.gate`` for hash layers, with an
explicit code comment documenting this as a "Phase-1 decision... deferred to
the Phase-2 HF-parity pass" divergence from the real DeepSeek reference
(which gathers sqrtsoftplus gate scores at the hashed experts). This new
class, by only overriding ``_select_experts`` (upstream's own design), lets
the inherited ``TokenChoiceTopKRouter.forward`` compute ``scores_TE`` via
``self.gate``/``self.score_func`` UNCONDITIONALLY (hash or not) and gather
those real scores at the hash-selected expert positions for the combine
weight -- exactly the reference behavior the old comment deferred, and
``self.gate`` now receives a gradient on hash layers too.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group, spmd_sparse_mesh
from torchtitan.models.common.moe import TokenChoiceTopKRouter


def _build_hash_routing_table(
    vocab_size: int,
    num_experts: int,
    top_k: int,
    device=None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Deterministic per-token-id -> top_k expert ids lookup table.

    Built in chunks (default 8192 rows) to bound peak memory for large
    vocabularies; each chunk's experts are drawn independently via a
    per-chunk random permutation, so the table is reproducible given the
    same (unseeded-but-fixed) call sequence -- it is a structural routing
    assignment, not a learned/trained parameter.
    """
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    tid2eid = torch.empty((vocab_size, top_k), dtype=torch.long, device=device)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        tid2eid[start:end] = (
            torch.rand((end - start, num_experts), device=device)
            .topk(top_k, dim=-1)
            .indices
        )
    return tid2eid


class DeepSeekV4Router(TokenChoiceTopKRouter):
    """Token-choice top-k router with optional hash-based expert selection.

    For layers where ``layer_id < n_hash_layers``, routing is fully
    deterministic (hash of the token id -> expert ids, uniform combine
    weights); otherwise the inherited score-based path runs unchanged
    (``score_func`` carries the sqrtsoftplus/softmax/sigmoid choice as a
    real activation module, same as every other already-migrated family's
    router).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        vocab_size: int
        n_hash_layers: int = 0
        layer_id: int = 0

    def __init__(self, config: Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.n_hash_layers = config.n_hash_layers
        self.layer_id = config.layer_id
        self.hash = config.layer_id < config.n_hash_layers
        if self.hash:
            self.register_buffer(
                "tid2eid",
                _build_hash_routing_table(
                    self.vocab_size,
                    self.num_experts,
                    self.top_k,
                ),
                persistent=True,
            )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None):
        super()._init_self_buffers(buffer_device=buffer_device)
        if self.hash:
            if buffer_device is None:
                buffer_device = self.tid2eid.device
            # The build below takes the device explicitly; no default-device
            # context switch is needed here.
            self.tid2eid = _build_hash_routing_table(
                self.vocab_size,
                self.num_experts,
                self.top_k,
                device=buffer_device,
            )

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        input_ids_T: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        if self.hash:
            if input_ids_T is None:
                raise ValueError(
                    "input_ids_T is required for DeepSeek V4 hash routing."
                )
            # Hash routing must index the same TP token shard as scores_TE.
            # GPU/TP-only path -- not exercised on CPU (TP=1 everywhere in
            # this sandbox's test matrix), ported verbatim from upstream so
            # TP>1 correctness isn't silently dropped; see
            # docs/torchtitan_upgrade_plan_2026-11.md §37.2.
            tp_group = spmd_mesh_group(MeshAxisName.TP)
            if spmd_sparse_mesh() is not None and tp_group is not None:
                input_ids_T = spmd.redistribute(
                    input_ids_T,
                    tp_group,
                    src=spmd.R,
                    dst=spmd.S(0),
                    backward_options={"op_dtype": input_ids_T.dtype},
                )
            return self.tid2eid.to(input_ids_T.device)[input_ids_T]
        return super()._select_experts(
            scores_TE,
            expert_bias_E,
            **router_kwargs,
        )
