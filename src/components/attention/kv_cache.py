"""Per-layer KV-cache primitives for incremental decode.

Three cache shapes, one per attention family:

- ``KVCache``         -- append-only k/v buffer for full attention
                         (mha, gqa, mqa, gqa_rope).
- ``SlidingKVCache``  -- trailing-window buffer for sliding_{window,gqa};
                         tracks absolute position so RoPE rotates at the
                         true token index even after the buffer is trimmed.
- ``CSACache``        -- per-block compressed cache for CSA, with the
                         scratch state needed to incrementally compress
                         each new completed block during decode.
- ``HCACache``        -- per-block compressed cache for HCA. No prev_block,
                         no SWA, no indexer keys -- HCA blocks are
                         self-contained (single-stream, no overlap).

``CausalLM`` builds one cache per layer via ``layer.attn.init_cache()`` on
the first decode call and threads the same instances through every step.
"""

from __future__ import annotations

from typing import Optional

import torch


class KVCache:
    """Append-only k/v buffer. Stores head-split tensors of shape
    [b, n_kv_heads, s, d_head]. ``update`` mutates and returns the new (k, v)."""

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor):
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=-2)
            self.v = torch.cat([self.v, new_v], dim=-2)
        return self.k, self.v

    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]

    def position(self) -> int:
        # absolute position of the next token to append; equals seq_len for
        # full attention since nothing is dropped.
        return self.seq_len()


class SlidingKVCache(KVCache):
    """Trailing-window cache. Trims to ``window`` entries after each update
    and tracks absolute position separately so RoPE-aware callers rotate at
    the true token index."""

    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = window
        self._abs = 0

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor):
        super().update(new_k, new_v)
        self._abs += new_k.shape[-2]
        if self.k.shape[-2] > self.window:
            self.k = self.k[..., -self.window :, :].contiguous()
            self.v = self.v[..., -self.window :, :].contiguous()
        return self.k, self.v

    def position(self) -> int:
        return self._abs


class CSACache:
    """Incremental decode state for compressed sparse attention.

    Fields::

        C_comp:     [b, n_blocks, c]      compressed kv blocks (eq 9)
        K_IComp:    [b, n_blocks, c_I]    compressed indexer keys
        prev_block: [b, m, d] | None      raw of the most-recent compressed
                                          block; becomes the stream-b input
                                          when the next block is built.
        raw_tail:   [b, t, d]             partial next block (t < m).
        swa:        [b, swa_len, c]       last n_win projections through
                                          W_KV_swa.
        total_seen: int                   absolute token count.
    """

    def __init__(self, m: int, n_win: int) -> None:
        self.m = m
        self.n_win = n_win
        self.C_comp: Optional[torch.Tensor] = None
        self.K_IComp: Optional[torch.Tensor] = None
        self.prev_block: Optional[torch.Tensor] = None
        self.raw_tail: Optional[torch.Tensor] = None
        self.swa: Optional[torch.Tensor] = None
        self.total_seen = 0

    def position(self) -> int:
        return self.total_seen


class HCACache:
    """Incremental decode state for heavily compressed attention.

    Fields::

        C_comp:     [b, n_blocks, c]   compressed kv blocks (eq 23)
        raw_tail:   [b, t, d]          partial next block (t < m)
        total_seen: int                absolute token count
    """

    def __init__(self, m: int) -> None:
        self.m = m
        self.C_comp: Optional[torch.Tensor] = None
        self.raw_tail: Optional[torch.Tensor] = None
        self.total_seen = 0

    def position(self) -> int:
        return self.total_seen
