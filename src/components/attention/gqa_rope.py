from typing import Optional

import torch
import torch.nn as nn

from src.components.positional.rope import apply_rope
from src.registry import ATTENTION

from .base import AttentionBase, scaled_dot_product


@ATTENTION.register("gqa_rope")
class GroupedQueryAttentionRoPE(AttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
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
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.wv = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.rope = rope

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        b, sq, _ = q.shape
        sk = k.shape[1]
        query = self.wq(q).view(b, sq, self.n_heads, self.d_head).transpose(1, 2)
        key = self.wk(k).view(b, sk, self.n_kv_heads, self.d_head).transpose(1, 2)
        value = self.wv(v).view(b, sk, self.n_kv_heads, self.d_head).transpose(1, 2)
        past_len = 0 if past_kv is None else past_kv[0].shape[-2]
        if self.rope is not None:
            query = apply_rope(query, self.rope.cos, self.rope.sin, position_offset=past_len)
            key = apply_rope(key, self.rope.cos, self.rope.sin, position_offset=past_len)
        if past_kv is not None:
            past_k, past_v = past_kv
            key = torch.cat([past_k, key], dim=-2)
            value = torch.cat([past_v, value], dim=-2)
        new_kv = (key, value) if return_kv else None
        key_x = key.repeat_interleave(self.group, dim=1)
        value_x = value.repeat_interleave(self.group, dim=1)
        out = scaled_dot_product(query, key_x, value_x, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(b, sq, self.d_model)
        out = self.wo(out)
        if return_kv:
            return out, new_kv
        return out
