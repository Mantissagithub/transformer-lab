import os
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.prompt import Confirm, Prompt
from torch.utils.tensorboard import SummaryWriter

from .hf_credentials import PROJECT_ROOT, _append_env


class TrainingLogger:
    def __init__(
        self,
        log_dir: Path,
        neptune_run: Optional[Any] = None,
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.tb = SummaryWriter(log_dir=str(log_dir))
        self.neptune = neptune_run
        self.wandb = wandb_run

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.tb.add_scalar(tag, value, step)
        if self.neptune is not None:
            self.neptune[tag].append(value, step=step)
        if self.wandb is not None:
            self.wandb.log({tag: value}, step=step)

    def flush(self) -> None:
        self.tb.flush()

    def close(self) -> None:
        self.tb.close()
        if self.neptune is not None:
            self.neptune.stop()
        if self.wandb is not None:
            self.wandb.finish()


def ensure_logging_backend(cfg: DictConfig) -> None:
    """Interactive backend picker. Default is tensorboard; offer wandb/neptune at startup.

    If cfg.logging.backend is already explicitly wandb/neptune (Hydra override),
    skip the picker and only ensure the API key.
    """
    backend = cfg.logging.get("backend", "tensorboard")
    console = Console()

    if backend in ("wandb", "neptune"):
        _ensure_api_key(backend, console)
        return

    console.print("[bold]Experiment tracking backend:[/]")
    console.print("  1) tensorboard only [default]")
    console.print("  2) wandb")
    console.print("  3) neptune")
    choice = Prompt.ask("Select", choices=["1", "2", "3"], default="1")
    if choice == "1":
        return

    new_backend = "wandb" if choice == "2" else "neptune"
    _ensure_api_key(new_backend, console)

    OmegaConf.set_struct(cfg, False)
    cfg.logging.backend = new_backend
    if cfg.logging.get("project") is None:
        if new_backend == "wandb":
            cfg.logging.project = Prompt.ask("W&B project", default="hyperconnections").strip()
        else:
            project = Prompt.ask("Neptune project (e.g. user/project)").strip()
            if not project:
                raise RuntimeError("Neptune project is required")
            cfg.logging.project = project
    OmegaConf.set_struct(cfg, True)


def _ensure_api_key(backend: str, console: Console) -> None:
    env_var = "WANDB_API_KEY" if backend == "wandb" else "NEPTUNE_API_TOKEN"
    if os.environ.get(env_var):
        return
    console.print(f"[yellow]{env_var} not found in env or .env.[/]")
    key = Prompt.ask(f"Enter {env_var}", password=True).strip()
    if not key:
        raise RuntimeError(f"{env_var} is required for logging.backend={backend}")
    os.environ[env_var] = key
    if Confirm.ask(f"Save {env_var} to .env for next run?", default=True):
        _append_env(PROJECT_ROOT / ".env", env_var, key)


def build_logger(cfg: DictConfig, experiment_name: str) -> TrainingLogger:
    log_dir = Path("runs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    backend = cfg.logging.get("backend", "tensorboard")
    neptune_run = None
    wandb_run = None
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
    elif backend == "wandb":
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "logging.backend=wandb requires WANDB_API_KEY environment variable"
            )
        import wandb  # type: ignore

        wandb_run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.get("entity", None),
            name=experiment_name,
            dir=str(log_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    return TrainingLogger(log_dir=log_dir, neptune_run=neptune_run, wandb_run=wandb_run)
