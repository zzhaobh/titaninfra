"""
titaninfra-owned Trainer/TrainingEngine extensions.

TitanInfra adds only two overrides to the upstream Trainer/TrainingEngine
split: a batch-validator hook in microbatch_generator, and MoE auxiliary-loss
step scaling plus sequence-position injection in TrainingEngine. The helpers
for process-group timeout compatibility and optional all-reduce toggling remain
independent of this override surface.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torchtitan.components.data.types import TrainingMicrobatch
    from torchtitan.trainer import Trainer as _TorchTitanTrainer
    from torchtitan.training_engine import TrainingEngine as _TorchTitanTrainingEngine

logger = logging.getLogger(__name__)


def install_torch_dist_set_timeout_compat() -> None:
    """Shim ``torch.distributed.set_timeout`` when the installed torch lacks it.

    upstream torchtitan (`distributed/utils.py:set_pg_timeouts`) calls the
    module-level ``torch.distributed.set_timeout(timeout, group)`` — added in a
    torch newer than the node's 2.13. Missing it crashes after step 1
    (`AttributeError: module 'torch.distributed' has no attribute 'set_timeout'`).

    Do not patch TorchTitan. Delegate to ProcessGroup.set_timeout on older
    Torch versions and use the default process group when group is None. If the
    module-level function already exists, leave it untouched.
    """
    import torch.distributed as dist

    if hasattr(dist, "set_timeout"):
        return

    # Prefer ProcessGroup.set_timeout, then _set_default_timeout. Fake backends
    # do not support timeout configuration and are intentionally ignored.
    def _set_timeout_via_method(timeout, group=None):
        pg = group if group is not None else dist.distributed_c10d._get_default_group()
        try:
            if hasattr(pg, "set_timeout"):
                pg.set_timeout(timeout)
            else:
                pg._set_default_timeout(timeout)
        except RuntimeError as e:
            # Unsupported fake-backend timeout errors are harmless.
            if "does not support setting timeout" in str(e):
                return
            raise

    dist.set_timeout = _set_timeout_via_method


def _set_microbatch_all_reduce(model_parts, *, is_last_microbatch, enabled):
    """Toggle HSDP cross-node all-reduce for FSDP modules during GAS.

    Disabled mode is a no-op. Enabled mode suppresses reduction for intermediate
    microbatches and enables it for the last one. Objects without the FSDP
    method are ignored. This optional optimization is not currently wired into
    the new TrainingEngine and has no effect on numerical semantics.
    """
    if not enabled:
        return
    want = is_last_microbatch  # Reduce only for the last microbatch.
    for part in model_parts:
        fn = getattr(part, "set_requires_all_reduce", None)
        if fn is not None:
            fn(want, recurse=True)


def _microbatch_to_legacy_2d(
    microbatch: TrainingMicrobatch, *, local_batch_size: int, seq_len: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reshape a flat TokenizedTrainingMicrobatch to [B, seq_len].

    _FlatTokenCollator guarantees exactly local_batch_size * seq_len tokens,
    including padded samples, so this reshape is safe for TitanInfra's own
    dataloader. loss_weights comes from model_kwargs["loss_kwargs"].
    """
    input_dict: dict[str, torch.Tensor] = {
        "input": microbatch.input.reshape(local_batch_size, seq_len),
        "positions": microbatch.positions.reshape(local_batch_size, seq_len),
    }
    loss_kwargs = microbatch.model_kwargs.get("loss_kwargs", {})
    loss_weights = loss_kwargs.get("loss_weights")
    if loss_weights is not None:
        input_dict["loss_weights"] = loss_weights.reshape(local_batch_size, seq_len)
    labels = microbatch.labels.reshape(local_batch_size, seq_len)
    return input_dict, labels


try:
    from torchtitan.trainer import Trainer as _TorchTitanTrainer
    from torchtitan.training_engine import TrainingEngine as _TorchTitanTrainingEngine
except ImportError:
    _TorchTitanTrainer = None
    _TorchTitanTrainingEngine = None


if _TorchTitanTrainingEngine is not None:

    class TitanInfraTrainingEngine(_TorchTitanTrainingEngine):
        """Add MoE auxiliary-loss support with microbatch-safe timing.

        Sequence positions are set immediately before each microbatch forward,
        not during prefetch. Step scale is set once per optimizer step when
        global valid-token count and GAS are known. A missing plugin makes both
        overrides transparent pass-throughs.
        """

        _moe_aux_plugin: Any = None

        def _non_pp_forward_backward_microbatch(
            self,
            *,
            inputs: torch.Tensor | tuple[torch.Tensor, ...],
            labels: torch.Tensor | tuple[torch.Tensor, ...],
            model_kwargs: dict[str, Any],
            loss_kwargs: dict[str, Any],
        ) -> torch.Tensor:
            aux_plugin = self._moe_aux_plugin
            if aux_plugin is not None and getattr(aux_plugin, "_aux_scope", "global") == "sequence":
                aux_plugin.set_positions(model_kwargs.get("positions"))
            return super()._non_pp_forward_backward_microbatch(
                inputs=inputs, labels=labels, model_kwargs=model_kwargs, loss_kwargs=loss_kwargs,
            )

        def forward_backward(
            self,
            *,
            microbatch_groups: list[list[TrainingMicrobatch]],
            global_loss_token_counts: int | torch.Tensor,
            global_routing_token_counts: torch.Tensor,
        ):
            aux_plugin = self._moe_aux_plugin
            if aux_plugin is not None:
                gvt = torch.as_tensor(global_loss_token_counts)
                main_gvt = gvt if gvt.ndim == 0 else gvt[0]
                aux_plugin.set_step_scale(gvt=float(main_gvt), gas=len(microbatch_groups))
            result = super().forward_backward(
                microbatch_groups=microbatch_groups,
                global_loss_token_counts=global_loss_token_counts,
                global_routing_token_counts=global_routing_token_counts,
            )
            if aux_plugin is not None:
                loss_mesh = self.parallelism_context.get_optional_mesh("loss")
                loss_pg = loss_mesh.get_group() if loss_mesh is not None else None
                aux_plugin.reduce_and_cache_counts(loss_pg=loss_pg)
            return result


if _TorchTitanTrainer is not None:

    class TitanInfraTrainer(_TorchTitanTrainer):
        """Run the optional batch validator before yielding microbatches.

        The validator belongs to Trainer because it inspects raw training
        batches rather than model execution. engine_cls also enables the MoE
        auxiliary-loss overrides. A missing validator is a pass-through.
        """

        engine_cls = TitanInfraTrainingEngine
        _batch_validator: Any = None

        def microbatch_generator(
            self, data_iterable: Iterable[TrainingMicrobatch]
        ) -> Iterator[TrainingMicrobatch]:
            local_batch_size = (
                self.config.training.num_tokens_per_microbatch_per_dp_rank
                // self.config.model.max_context_length
            )
            seq_len = self.config.model.max_context_length
            for microbatch in super().microbatch_generator(data_iterable):
                validator = self._batch_validator
                if validator is not None:
                    step = self.engine.num_completed_steps + 1
                    input_dict, labels = _microbatch_to_legacy_2d(
                        microbatch, local_batch_size=local_batch_size, seq_len=seq_len,
                    )
                    validator(step, input_dict, labels)
                yield microbatch
