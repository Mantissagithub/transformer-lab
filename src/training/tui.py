from __future__ import annotations

import time
from typing import Any, Dict

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


LOG_EVERY_N_STEPS = 50


class TrainingTUI:
    def __init__(
        self,
        *,
        experiment_name: str,
        model_summary: Dict[str, Any],
        total_steps: int,
        steps_per_epoch: int,
        num_epochs: int,
        device: torch.device,
    ) -> None:
        self.experiment_name = experiment_name
        self.model_summary = model_summary
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        self.device = device
        self.console = Console()

        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        self.epoch_task = self.progress.add_task(
            f"Epoch 1/{num_epochs}", total=steps_per_epoch
        )
        self.total_task = self.progress.add_task("Training", total=total_steps)

        m = model_summary
        self._header = (
            f"{experiment_name}  device={device}  "
            f"params={m['n_params'] / 1e6:.2f}M  layers={m['n_layers']}  "
            f"d_model={m['d_model']}  attn={m['attention']}"
        )

    def __enter__(self) -> "TrainingTUI":
        self.console.print(self._header, highlight=False)
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.stop()

    def update_step(
        self,
        *,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        loss: float,
        lr: float,
    ) -> None:
        self.progress.update(
            self.epoch_task,
            completed=step_in_epoch + 1,
            description=f"Epoch {epoch + 1}/{self.num_epochs}",
        )
        self.progress.update(self.total_task, completed=global_step + 1)
        gs = global_step + 1
        if gs % LOG_EVERY_N_STEPS == 0:
            self.event(f"step {gs}  loss={loss:.4f}  lr={lr:.2e}")

    def reset_epoch(self, epoch: int) -> None:
        self.progress.reset(self.epoch_task, total=self.steps_per_epoch)
        self.progress.update(
            self.epoch_task,
            description=f"Epoch {epoch + 1}/{self.num_epochs}",
        )

    def event(self, msg: str, style: str | None = None) -> None:
        ts = time.strftime("%H:%M:%S")
        self.console.print(f"{ts} {msg}", highlight=False)
