from typing import Optional

import math

import torch
import torch.nn as nn


class AttentionBase(nn.Module):
    """All attention implementations expose forward(q, k, v, mask, past_kv=None, return_kv=False).

    With return_kv=False (the default and the only path used by the encoder-decoder
    trainer) the return is a tensor of shape (b, s, d). With return_kv=True (used by
    CausalLM.generate) the return is (out, (k, v)) where k, v are the post-projection
    head-split tensors that should be passed back as past_kv on the next step.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) not divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):  # pragma: no cover - interface
        raise NotImplementedError


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    dropout: Optional[nn.Dropout],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if bias is not None:
        scores = scores + bias
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = scores.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v
