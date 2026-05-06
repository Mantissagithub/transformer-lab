import torch
import torch.nn as nn

from src.registry import ATTENTION

from .base import AttentionBase, scaled_dot_product
from .kv_cache import SlidingKVCache


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

    def init_cache(self) -> SlidingKVCache:
        return SlidingKVCache(self.window_size)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    def _window_mask(self, sq: int, sk: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(sq, device=device).unsqueeze(1)
        j = torch.arange(sk, device=device).unsqueeze(0)
        return ((j - i).abs() <= self.window_size).int().unsqueeze(0).unsqueeze(0)

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        b, sq, _ = q.shape
        query = self._split(self.wq(q))
        key = self._split(self.wk(k))
        value = self._split(self.wv(v))

        # attention K/V is the (cached + new) view, untrimmed. cache.update is
        # called AFTER attention so its trim doesn't drop entries the current
        # query still needs.
        if past_kv is not None and past_kv.k is not None:
            attn_key = torch.cat([past_kv.k, key], dim=-2)
            attn_value = torch.cat([past_kv.v, value], dim=-2)
        else:
            attn_key, attn_value = key, value

        # prefill or multi-token: explicit band-diagonal window mask. on a
        # single-token decode against a non-empty cache, attn_key holds at most
        # window+1 entries (window cached + the new one) which are all visible
        # to the new query — no mask needed.
        # note: a user-passed `mask` is dropped on the decode branch — fine for
        # the only current caller (causal_lm), but a future padding-mask caller
        # would need to revisit this.
        if past_kv is None or sq > 1:
            window = self._window_mask(sq, attn_key.shape[-2], q.device)
            combined = window if mask is None else (mask & window)
        else:
            combined = None

        out = scaled_dot_product(query, attn_key, attn_value, combined, self.dropout)
        out = out.transpose(1, 2).contiguous().view(b, sq, self.d_model)
        out = self.wo(out)
        if past_kv is not None:
            past_kv.update(key, value)
        if return_kv:
            return out, past_kv
        return out
