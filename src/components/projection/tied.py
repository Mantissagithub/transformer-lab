import torch
import torch.nn as nn

from src.registry import PROJECTION


@PROJECTION.register("tied")
class TiedProjection(nn.Module):
    def __init__(self, embedding: nn.Module, log_softmax: bool = True) -> None:
        super().__init__()
        self.embedding = embedding
        self.log_softmax = log_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.embedding.embeddings.weight
        logits = x @ weight.t()
        if self.log_softmax:
            return torch.log_softmax(logits, dim=-1)
        return logits
