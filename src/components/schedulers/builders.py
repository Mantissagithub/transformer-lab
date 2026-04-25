import math
from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from src.registry import SCHEDULER


@SCHEDULER.register("none")
def build_none(optimizers: List[Optimizer], **kwargs) -> List[Optional[_LRScheduler]]:
    return [None for _ in optimizers]


@SCHEDULER.register("linear_warmup")
def build_linear_warmup(
    optimizers: List[Optimizer],
    warmup_steps: int = 1000,
    total_steps: int = 100_000,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> List[_LRScheduler]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 1.0 - progress)

    return [LambdaLR(opt, lr_lambda=fn) for opt in optimizers]


@SCHEDULER.register("cosine_warmup")
def build_cosine_warmup(
    optimizers: List[Optimizer],
    warmup_steps: int = 1000,
    total_steps: int = 100_000,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> List[_LRScheduler]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return [LambdaLR(opt, lr_lambda=fn) for opt in optimizers]
