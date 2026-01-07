import torch
import torch.nn as nn
class LayerNormalization(nn.Module):
    def __init__(self, eps : float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (torch.sqrt(std) + self.eps) + self.bias
