import torch
import torch.nn as nn

from src.registry import PROJECTION


@PROJECTION.register("linear")
class LinearProjection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, log_softmax: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        self.log_softmax = log_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.proj(x)
        if self.log_softmax:
            return torch.log_softmax(logits, dim=-1)
        return logits
