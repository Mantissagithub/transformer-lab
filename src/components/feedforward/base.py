import torch
import torch.nn as nn


class FeedForwardBase(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError
