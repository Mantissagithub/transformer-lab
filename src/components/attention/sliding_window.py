import torch
import torch.nn as nn

from src.registry import ATTENTION

from .base import AttentionBase, scaled_dot_product


@ATTENTION.register("sliding_window")
class SlidingWindowAttention(AttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        self.window_size = window_size
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    def _window_mask(self, sq: int, sk: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(sq, device=device).unsqueeze(1)
        j = torch.arange(sk, device=device).unsqueeze(0)
        return ((j - i).abs() <= self.window_size).int().unsqueeze(0).unsqueeze(0)

    def forward(self, q, k, v, mask=None, kv_cache=None):
        query = self._split(self.wq(q))
        key = self._split(self.wk(k))
        value = self._split(self.wv(v))
        window = self._window_mask(query.shape[-2], key.shape[-2], query.device)
        combined = window if mask is None else (mask & window)
        out = scaled_dot_product(query, key, value, combined, self.dropout)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, self.d_model)
        return self.wo(out)
