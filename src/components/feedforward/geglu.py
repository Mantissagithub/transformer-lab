import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import FFN

from .base import FeedForwardBase


@FFN.register("geglu")
class GeGLUFeedForward(FeedForwardBase):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.gelu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))
