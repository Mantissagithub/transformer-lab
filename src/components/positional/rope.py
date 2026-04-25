import torch
import torch.nn as nn

from src.registry import POS


@POS.register("rope")
class RotaryPositionalEncoding(nn.Module):
    """Identity on token embeddings — RoPE is applied inside attention via `apply_rope`.

    Stored on the model so attention modules can fetch the precomputed cos/sin.
    """

    def __init__(self, d_model: int, max_len: int, base: float = 10000.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.is_rotary = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq = x.shape[-2]
    cos = cos[:seq].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.stack([rx1, rx2], dim=-1).flatten(-2)
    return out
