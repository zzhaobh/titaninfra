"""WandB environment and MetricsProcessor configuration helpers.

A wandb_state.json file lets DCP resume continue the same WandB run instead of
creating a new run.
"""

import importlib.util
import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from titaninfra.config.train import TrainConfig

logger = logging.getLogger(__name__)

WANDB_STATE_FILE = "wandb_state.json"


def setup_wandb_env(
    config: "TrainConfig",
    dump_folder: str = "",
    is_resume: bool = False,
) -> None:
    """Configure WandB environment variables for a training run.

    During DCP resume, restore run_id from wandb_state.json and set
    WANDB_RESUME=allow so logging continues the existing run.
    """
    if not config.enable_wandb:
        return

    # Fail clearly during configuration when WandB is enabled but unavailable,
    # rather than failing later inside the metrics processor.
    if importlib.util.find_spec("wandb") is None:
        logger.warning(
            "enable_wandb=True 但未安装 wandb，WandB 日志将不可用。"
            "请 `pip install wandb`，或设置 enable_wandb=False。"
        )

    run_id = config.wandb_run_id

    # Restore the previous run ID during resume.
    if is_resume and run_id is None and dump_folder:
        run_id = _load_wandb_run_id(dump_folder)
        if run_id is not None:
            logger.info(f"从 {WANDB_STATE_FILE} 恢复 WandB run_id={run_id}")

    # Continue an existing run when an ID is available.
    if run_id is not None:
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "allow"

    # Propagate remaining optional environment variables.
    env_mappings = {
        "WANDB_PROJECT": config.wandb_project,
        "WANDB_TEAM": config.wandb_entity,
        "WANDB_RUN_NAME": config.wandb_run_name,
        "WANDB_RUN_TAGS": config.wandb_tags,
        "WANDB_RUN_NOTES": config.wandb_notes,
        "WANDB_MODE": config.wandb_mode,
    }
    for env_key, value in env_mappings.items():
        if value is not None:
            os.environ[env_key] = value

    resume_hint = f"，续写 run_id={run_id}" if run_id else ""
    logger.info(
        f"WandB 已启用，项目: {config.wandb_project or os.getenv('WANDB_PROJECT', 'torchtitan')}，"
        f"模式: {config.wandb_mode or 'offline'}{resume_hint}"
    )


def save_wandb_run_id(dump_folder: str) -> None:
    """Persist the current WandB run ID for the next DCP resume."""
    try:
        import wandb

        if wandb.run is not None and wandb.run.id:
            path = os.path.join(dump_folder, WANDB_STATE_FILE)
            os.makedirs(dump_folder, exist_ok=True)
            with open(path, "w") as f:
                json.dump({"run_id": wandb.run.id}, f)
            logger.info(f"WandB run_id={wandb.run.id} 已保存到 {path}")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"保存 WandB run_id 失败: {e}")


def _load_wandb_run_id(dump_folder: str) -> str | None:
    """Read a previously saved run ID, returning None when unavailable."""
    path = os.path.join(dump_folder, WANDB_STATE_FILE)
    try:
        with open(path) as f:
            return json.load(f).get("run_id")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def build_metrics_config(train_config: "TrainConfig"):
    """Build MetricsProcessor.Config from TrainConfig."""
    from torchtitan.observability.metrics import (
        MetricsProcessor,
        TensorBoardLogger,
        WandBLogger,
    )

    # Upstream now accepts explicit logger configs. TensorBoard stays enabled
    # by default, while WandB remains controlled by enable_wandb.
    loggers: list = [TensorBoardLogger.Config()]
    if train_config.enable_wandb:
        loggers.append(WandBLogger.Config())

    return MetricsProcessor.Config(
        log_freq=train_config.log_interval,
        loggers=loggers,
        disable_color_printing=False,
    )
