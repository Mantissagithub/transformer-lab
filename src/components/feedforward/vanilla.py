import torch
import torch.nn as nn

from src.registry import FFN

from .base import FeedForwardBase


@FFN.register("relu_ffn")
class ReluFeedForward(FeedForwardBase):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
