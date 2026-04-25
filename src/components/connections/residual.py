from typing import Callable

import torch
import torch.nn as nn

from src.registry import CONNECTION, NORM

from .base import Connection


@CONNECTION.register("residual")
class ResidualConnection(Connection):
    is_stateful = False

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        norm: str = "layernorm",
        post_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = NORM.build(norm, d_model=d_model)
        self.post_norm_module = NORM.build(norm, d_model=d_model) if post_norm else None

    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to_output(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def apply(self, state: torch.Tensor, sublayer_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        sub = sublayer_fn(self.norm(state))
        if self.post_norm_module is not None:
            sub = self.post_norm_module(sub)
        return state + self.dropout(sub)
