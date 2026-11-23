"""Low-intrusion metrics plugins for training observability.

MetricsPlugin defines the collection protocol. inject_metrics_plugins wraps
metrics_processor.log once and merges plugin metrics into each log call.
"""

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def _try_wandb_bar(stream_tokens: dict[str, float]):
    """Return a WandB bar chart, or None when unavailable."""
    try:
        import wandb

        table = wandb.Table(columns=["stream", "tokens"], data=[[s, t] for s, t in stream_tokens.items()])
        return wandb.plot.bar(table, label="stream", value="tokens", title="Stream Token Distribution")
    except Exception:
        return None


@runtime_checkable
class MetricsPlugin(Protocol):
    """Protocol implemented by metrics plugins through collect()."""

    def collect(self, step: int, trainer: Any) -> dict[str, Any]:
        """Return custom metrics for the current training step."""
        ...


class DataMetricsPlugin:
    """Track per-stream token and epoch-progress metrics by stream_id.

    Metrics are derived from the main-process step count rather than dataset
    internals, so they remain valid when iteration happens in workers.
    """

    def __init__(
        self,
        tokens_per_step: int,
        stream_total_tokens: dict[str, int] | None = None,
        stream_proportions: dict[str, float] | None = None,
    ) -> None:
        """Configure tokens per step, stream totals, and stream proportions."""
        self._tokens_per_step = tokens_per_step
        self._stream_total_tokens = stream_total_tokens or {}
        self._stream_proportions = stream_proportions or {}

    def collect(self, step: int, trainer: Any) -> dict[str, Any]:  # noqa: ARG002
        """Collect per-stream metrics without depending on dataset workers."""
        metrics: dict[str, Any] = {}
        global_tokens = step * self._tokens_per_step

        # Split global tokens across streams by proportion.
        per_stream_tokens: dict[str, float] = {}
        for sid, prop in self._stream_proportions.items():
            stream_tokens = global_tokens * prop
            per_stream_tokens[sid] = stream_tokens
            metrics[f"data/{sid}/tokens"] = float(stream_tokens)

            # Compute epoch progress from stream_stats totals.
            total = self._stream_total_tokens.get(sid, 0)
            if total > 0:
                metrics[f"data/{sid}/epoch_progress"] = stream_tokens / total

        # Add a bar chart alongside per-stream line metrics.
        if per_stream_tokens:
            bar_chart = _try_wandb_bar(per_stream_tokens)
            if bar_chart is not None:
                metrics["data/stream_tokens_dist"] = bar_chart

        # Keep total_tokens_seen consistent with per-stream values.
        total_tokens = (
            sum(per_stream_tokens.values()) if per_stream_tokens
            else float(global_tokens)
        )
        metrics["data/total_tokens_seen"] = total_tokens

        # Compute global epoch progress.
        if self._stream_total_tokens and per_stream_tokens:
            all_total = sum(self._stream_total_tokens.values())
            if all_total > 0:
                metrics["data/epoch_progress"] = total_tokens / all_total

        return metrics


class QKClipMetricsPlugin:
    """Report the number of heads clipped by QK-Clip at each step."""

    def __init__(self, optimizer: Any) -> None:
        self._optimizer = optimizer

    def collect(self, step: int, trainer: Any) -> dict[str, Any]:  # noqa: ARG002
        state = getattr(self._optimizer, "_qk_clip_state", None)
        if state is None:
            return {}
        return {"optimizer/qk_clip_heads_clipped": state["n_clipped"]}


def inject_metrics_plugins(trainer: Any, plugins: list[MetricsPlugin]) -> None:
    """Wrap metrics_processor.log once to merge all plugin metrics."""
    if not plugins:
        return

    if not hasattr(trainer, "metrics_processor") or not hasattr(
        trainer.metrics_processor, "log"
    ):
        logger.warning("trainer 不支持 metrics_processor.log，跳过插件注入")
        return

    original_log = trainer.metrics_processor.log

    def log_with_plugins(
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        # Forward keyword arguments, including num_flops, without coupling this
        # wrapper to upstream's exact signature.
        merged_metrics: dict[str, float] = extra_metrics or {}

        # Collect metrics from every plugin.
        for plugin in plugins:
            try:
                plugin_metrics = plugin.collect(step, trainer)
                merged_metrics.update(plugin_metrics)
            except Exception as e:
                logger.warning(f"Metrics 插件 {type(plugin).__name__} 失败：{e}")

        # TensorBoard accepts scalars; rich objects go through wandb.log.
        scalar_metrics = {}
        rich_metrics = {}
        for k, v in merged_metrics.items():
            if isinstance(v, (int, float)):
                scalar_metrics[k] = v
            else:
                rich_metrics[k] = v

        original_log(step, global_avg_loss, global_max_loss, grad_norm, scalar_metrics, **kwargs)

        if rich_metrics:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(rich_metrics, step=step)
            except Exception:
                pass

    trainer.metrics_processor.log = log_with_plugins

    plugin_names = ", ".join(type(p).__name__ for p in plugins)
    logger.info(f"已注入 {len(plugins)} 个 metrics 插件: {plugin_names}")
