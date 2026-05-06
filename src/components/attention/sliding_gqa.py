from typing import Optional

import torch
import torch.nn as nn

from src.components.positional.rope import apply_rope
from src.registry import ATTENTION

from .base import AttentionBase, scaled_dot_product
from .kv_cache import SlidingKVCache


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

    def init_cache(self) -> SlidingKVCache:
        return SlidingKVCache(self.window_size)

    def _window_mask(self, sq: int, sk: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(sq, device=device).unsqueeze(1)
        j = torch.arange(sk, device=device).unsqueeze(0)
        return ((j - i).abs() <= self.window_size).int().unsqueeze(0).unsqueeze(0)

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        b, sq, _ = q.shape
        sk = k.shape[1]
        query = self.wq(q).view(b, sq, self.n_heads, self.d_head).transpose(1, 2)
        key = self.wk(k).view(b, sk, self.n_kv_heads, self.d_head).transpose(1, 2)
        value = self.wv(v).view(b, sk, self.n_kv_heads, self.d_head).transpose(1, 2)

        offset = past_kv.position() if past_kv is not None else 0
        if self.rope is not None:
            query = apply_rope(query, self.rope.cos, self.rope.sin, position_offset=offset)
            key = apply_rope(key, self.rope.cos, self.rope.sin, position_offset=offset)

        # build the attention K/V as (cached + new), untrimmed; cache.update is
        # called AFTER attention so its trim doesn't drop entries this query
        # still needs to attend to.
        if past_kv is not None and past_kv.k is not None:
            attn_key = torch.cat([past_kv.k, key], dim=-2)
            attn_value = torch.cat([past_kv.v, value], dim=-2)
        else:
            attn_key, attn_value = key, value

        attn_key_x = attn_key.repeat_interleave(self.group, dim=1)
        attn_value_x = attn_value.repeat_interleave(self.group, dim=1)

        # prefill: band-diagonal window mask. decode (sq==1 with cache): attn_key
        # has ≤ window+1 entries, all in the new query's window → no mask. user
        # `mask` is dropped on the decode branch (fine for causal_lm; revisit
        # for padding-mask callers).
        if past_kv is None or sq > 1:
            window = self._window_mask(sq, attn_key.shape[-2], q.device)
            combined = window if mask is None else (mask & window)
        else:
            combined = None

        out = scaled_dot_product(query, attn_key_x, attn_value_x, combined, self.dropout)
        out = out.transpose(1, 2).contiguous().view(b, sq, self.d_model)
        out = self.wo(out)
        if past_kv is not None:
            past_kv.update(key, value)
        if return_kv:
            return out, past_kv
        return out
