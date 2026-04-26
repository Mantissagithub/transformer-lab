from typing import Optional

import torch
import torch.nn as nn

from src.components.positional.rope import apply_rope
from src.registry import ATTENTION

from .base import AttentionBase, scaled_dot_product


@ATTENTION.register("sliding_gqa")
class SlidingGroupedQueryAttention(AttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        window_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_kv_heads = n_kv_heads
        self.group = n_heads // n_kv_heads
        self.kv_dim = self.d_head * n_kv_heads
        self.window_size = window_size
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.wv = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.rope = rope

    def _window_mask(self, sq: int, sk: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(sq, device=device).unsqueeze(1)
        j = torch.arange(sk, device=device).unsqueeze(0)
        return ((j - i).abs() <= self.window_size).int().unsqueeze(0).unsqueeze(0)

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        if past_kv is not None or return_kv:
            raise NotImplementedError("sliding_gqa does not support KV cache")
        b, sq, _ = q.shape
        sk = k.shape[1]
        query = self.wq(q).view(b, sq, self.n_heads, self.d_head).transpose(1, 2)
        key = self.wk(k).view(b, sk, self.n_kv_heads, self.d_head).transpose(1, 2)
        value = self.wv(v).view(b, sk, self.n_kv_heads, self.d_head).transpose(1, 2)
        if self.rope is not None:
            query = apply_rope(query, self.rope.cos, self.rope.sin)
            key = apply_rope(key, self.rope.cos, self.rope.sin)
        key = key.repeat_interleave(self.group, dim=1)
        value = value.repeat_interleave(self.group, dim=1)
        window = self._window_mask(sq, sk, query.device)
        combined = window if mask is None else (mask & window)
        out = scaled_dot_product(query, key, value, combined, self.dropout)
        out = out.transpose(1, 2).contiguous().view(b, sq, self.d_model)
        return self.wo(out)
