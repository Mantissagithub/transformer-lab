from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict

import torch
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


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

        self.epoch_progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green3"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        )
        self.epoch_task = self.epoch_progress.add_task(
            f"Epoch 1/{num_epochs}", total=steps_per_epoch
        )

        self.total_progress = Progress(
            SpinnerColumn(style="magenta"),
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(complete_style="magenta", finished_style="green3"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        )
        self.total_task = self.total_progress.add_task("Training", total=total_steps)

        self.logs: Deque[str] = deque(maxlen=10)

    def _header(self) -> Panel:
        m = self.model_summary
        line = Text()
        line.append("⚡ ", style="yellow")
        line.append(self.experiment_name, style="bold yellow")
        line.append("   ")
        line.append(f"device={self.device}", style="dim")
        line.append("  ·  ")
        line.append(f"params={m['n_params'] / 1e6:.2f}M", style="dim")
        line.append("  ·  ")
        line.append(f"layers={m['n_layers']}", style="dim")
        line.append("  ·  ")
        line.append(f"d_model={m['d_model']}", style="dim")
        line.append("  ·  ")
        line.append(f"attn={m['attention']}", style="dim")
        return Panel(line, border_style="bright_blue", padding=(0, 1))

    def _progress_panel(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_row(self.epoch_progress)
        grid.add_row(self.total_progress)
        return Panel(grid, border_style="magenta", title="[bold]progress[/]", title_align="left")

    def _logs_panel(self) -> Panel:
        if not self.logs:
            body: Any = Text("waiting…", style="dim italic")
        else:
            body = Text.from_markup("\n".join(self.logs))
        return Panel(body, border_style="bright_black", title="[bold]logs[/]", title_align="left")

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=3),
            Layout(self._progress_panel(), name="progress", size=6),
            Layout(self._logs_panel(), name="logs"),
        )
        return layout

    def update_step(
        self,
        *,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        loss: float,
        lr: float,
    ) -> None:
        self.epoch_progress.update(
            self.epoch_task,
            completed=step_in_epoch + 1,
            description=f"Epoch {epoch + 1}/{self.num_epochs}",
        )
        self.total_progress.update(self.total_task, completed=global_step + 1)
        gs = global_step + 1
        if gs % LOG_EVERY_N_STEPS == 0:
            self.event(f"step {gs}  loss={loss:.4f}  lr={lr:.2e}", "white")

    def reset_epoch(self, epoch: int) -> None:
        self.epoch_progress.reset(self.epoch_task, total=self.steps_per_epoch)
        self.epoch_progress.update(
            self.epoch_task,
            description=f"Epoch {epoch + 1}/{self.num_epochs}",
        )

    def event(self, msg: str, style: str = "white") -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[dim]{ts}[/] [{style}]{msg}[/]")
