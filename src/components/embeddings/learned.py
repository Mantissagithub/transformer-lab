import math

import torch
import torch.nn as nn

from src.registry import EMBEDDING


@EMBEDDING.register("learned")
class LearnedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, scale_by_sqrt_d: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model) if scale_by_sqrt_d else 1.0
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x) * self.scale
