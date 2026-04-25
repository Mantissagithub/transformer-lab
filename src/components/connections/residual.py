from typing import Callable

import torch
import torch.nn as nn

from src.registry import CONNECTION, NORM

from .base import Connection


@CONNECTION.register("residual")
class ResidualConnection(Connection):
    is_stateful = False

    def __init__(self, d_model: int, dropout: float = 0.0, norm: str = "layernorm") -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = NORM.build(norm, d_model=d_model)

    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to_output(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def apply(self, state: torch.Tensor, sublayer_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return state + self.dropout(sublayer_fn(self.norm(state)))
