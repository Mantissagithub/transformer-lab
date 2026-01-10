import torch
import torch.nn as nn
import math
from .layer_normalization import LayerNormalization

class ResidualCOnnection(nn.Module):
    def __init__(self, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

