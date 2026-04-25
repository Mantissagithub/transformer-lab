import math

import torch
import torch.nn as nn

from src.registry import POS


def _get_slopes(n_heads: int) -> torch.Tensor:
    def slopes_pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(slopes_pow2(n_heads))
    closest_pow2 = 2 ** math.floor(math.log2(n_heads))
    base = slopes_pow2(closest_pow2)
    extra = slopes_pow2(2 * closest_pow2)[0::2][: n_heads - closest_pow2]
    return torch.tensor(base + extra)


@POS.register("alibi")
class ALiBi(nn.Module):
    """Identity on embeddings — bias tensor is applied inside attention via `alibi_bias`."""

    def __init__(self, d_model: int, max_len: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        slopes = _get_slopes(n_heads)
        positions = torch.arange(max_len)
        rel = positions.unsqueeze(0) - positions.unsqueeze(1)
        bias = slopes.view(n_heads, 1, 1) * rel.unsqueeze(0)
        self.register_buffer("bias", bias, persistent=False)
        self.is_alibi = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)
