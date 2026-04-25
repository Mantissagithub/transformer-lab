import os
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir: Path, neptune_run: Optional[Any] = None) -> None:
        self.tb = SummaryWriter(log_dir=str(log_dir))
        self.neptune = neptune_run

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.tb.add_scalar(tag, value, step)
        if self.neptune is not None:
            self.neptune[tag].append(value, step=step)

    def flush(self) -> None:
        self.tb.flush()

    def close(self) -> None:
        self.tb.close()
        if self.neptune is not None:
            self.neptune.stop()


def build_logger(cfg: DictConfig, experiment_name: str) -> TrainingLogger:
    log_dir = Path("runs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    backend = cfg.logging.get("backend", "tensorboard")
    neptune_run = None
    if backend == "neptune":
        token = os.environ.get("NEPTUNE_API_TOKEN")
        if not token:
            raise RuntimeError(
                "logging.backend=neptune requires NEPTUNE_API_TOKEN environment variable"
            )
        import neptune  # type: ignore
        from neptune_tensorboard import enable_tensorboard_logging  # type: ignore

        neptune_run = neptune.init_run(project=cfg.logging.project, api_token=token)
        enable_tensorboard_logging(neptune_run, log_dir=str(log_dir))
    return TrainingLogger(log_dir=log_dir, neptune_run=neptune_run)
