from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizers: List[Optimizer],
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizers: List[Optimizer]) -> dict:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    for opt, opt_state in zip(optimizers, state["optimizers"]):
        opt.load_state_dict(opt_state)
    return state


def checkpoint_path(ckpt_dir: str, basename: str, epoch: int) -> Path:
    return Path(ckpt_dir) / f"{basename}{epoch:02d}.pt"
