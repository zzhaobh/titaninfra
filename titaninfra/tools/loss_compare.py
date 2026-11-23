"""Compare training loss curves from TensorBoard or CSV logs."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compare_losses(
    log_dirs: list[str],
    labels: list[str] | None = None,
    metric: str = "loss",
    output_path: str | None = None,
    show: bool = True,
) -> dict:
    """Compare one scalar metric across training output directories.

    Args:
        log_dirs: TorchTitan dump_folder paths.
        labels: Optional curve labels; directory names are used by default.
        metric: Scalar name to read, default "loss".
        output_path: Optional output image path.
        show: Whether to display a matplotlib window when available.

    Returns:
        {label: {"steps": list[int], "values": list[float]} for each run.
    """
    if labels is None:
        labels = [Path(d).name for d in log_dirs]

    if len(labels) != len(log_dirs):
        raise ValueError("labels 和 log_dirs 长度不匹配")

    results: dict[str, dict] = {}

    for log_dir, label in zip(log_dirs, labels, strict=False):
        steps, values = _read_loss_from_dir(log_dir, metric)
        results[label] = {"steps": steps, "values": values}
        logger.info(f"读取 '{label}'：{len(steps)} 步")

    if output_path or show:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            for label, data in results.items():
                if data["steps"]:
                    ax.plot(data["steps"], data["values"], label=label, linewidth=1.5)

            ax.set_xlabel("Steps")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Training {metric.capitalize()} Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"图表已保存：{output_path}")

            if show:
                plt.show()

            plt.close(fig)

        except ImportError:
            logger.warning("matplotlib 未安装，跳过绘图。pip install matplotlib")

    return results


def _read_loss_from_dir(log_dir: str, metric: str) -> tuple[list[int], list[float]]:
    """Read steps and values for metric from TensorBoard or CSV logs."""
    log_path = Path(log_dir)

    tb_dir = log_path / "tb"
    if tb_dir.exists():
        try:
            return _read_tensorboard(str(tb_dir), metric)
        except Exception as e:
            logger.debug(f"TensorBoard 读取失败：{e}")

    csv_files = list(log_path.glob("*.csv"))
    if csv_files:
        try:
            return _read_csv_log(str(csv_files[0]), metric)
        except Exception as e:
            logger.debug(f"CSV 读取失败：{e}")

    logger.warning(f"在 {log_dir} 中未找到 '{metric}' 数据")
    return [], []


def _read_tensorboard(tb_dir: str, metric: str) -> tuple[list[int], list[float]]:
    """Read a scalar series from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as e:
        raise ImportError("读取 TensorBoard 日志需要 tensorboard：pip install tensorboard") from e

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    matching_tag = None
    for tag in available_tags:
        if metric.lower() in tag.lower():
            matching_tag = tag
            break

    if matching_tag is None:
        raise KeyError(f"在 TensorBoard 日志中未找到 '{metric}'，可用标量：{available_tags}")

    events = ea.Scalars(matching_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def _read_csv_log(csv_path: str, metric: str) -> tuple[list[int], list[float]]:
    """Read steps and the selected metric column from a CSV log."""
    import csv

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return [], []

    headers = list(rows[0].keys())
    step_col = next((h for h in headers if "step" in h.lower()), None)
    metric_col = next((h for h in headers if metric.lower() in h.lower()), None)

    if step_col is None or metric_col is None:
        raise KeyError(f"CSV 中未找到 step 或 '{metric}' 列，可用列：{headers}")

    steps = []
    values = []
    for row in rows:
        try:
            steps.append(int(row[step_col]))
            values.append(float(row[metric_col]))
        except (ValueError, KeyError):
            continue

    return steps, values
