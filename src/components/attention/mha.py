import torch
import torch.nn as nn

from src.registry import ATTENTION

from .base import AttentionBase, scaled_dot_product


@ATTENTION.register("mha")
class MultiHeadAttention(AttentionBase):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__(d_model, n_heads, dropout)
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        query = self._split(self.wq(q))
        key = self._split(self.wk(k))
        value = self._split(self.wv(v))
        if past_kv is not None:
            key, value = past_kv.update(key, value)
        out = scaled_dot_product(query, key, value, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, self.d_model)
        out = self.wo(out)
        if return_kv:
            return out, past_kv
        return out
