"""Multi-head Latent Attention (DeepSeek-V2 §2.1).

Cache memory is the only optimization: instead of storing per-head K/V we
store a single shared low-rank latent ``c_kv`` (rank ``kv_lora_rank``) plus a
small head-shared decoupled-RoPE key ``k_pe`` (size ``qk_rope_head_dim``).
Per-head K and V are reconstructed each forward by up-projecting through
``wkv_b``.

This v1 does NOT absorb ``W_UQ``/``W_UK`` into the query path, so decode-time
FLOPs are the same as recomputing dense K/V every step. KV-cache *memory* is
the only win.

Self-attention only: ``q``, ``k``, ``v`` must be the same tensor (warns
otherwise).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.registry import ATTENTION

from .base import AttentionBase, scaled_dot_product
from .kv_cache import MLACache


def _build_rope_tables(rope_dim: int, max_seq_len: int, base: float = 10000.0):
    """Half-split rope cos/sin tables. Same convention as csa.py — do not
    swap in the interleaved variant from positional/rope.py."""
    half = rope_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(max_seq_len).float()
    angles = pos[:, None] * freqs[None, :]
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(
    x: torch.Tensor,           # [..., s, rope_dim]
    cos_table: torch.Tensor,   # [max_seq_len, rope_dim/2]
    sin_table: torch.Tensor,
    position_offset: int,
) -> torch.Tensor:
    s = x.shape[-2]
    cos = cos_table[position_offset : position_offset + s]
    sin = sin_table[position_offset : position_offset + s]
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


@ATTENTION.register("mla")
class MultiHeadLatentAttention(AttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        rope_base: float = 10000.0,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        if qk_rope_head_dim % 2 != 0:
            raise ValueError(f"qk_rope_head_dim ({qk_rope_head_dim}) must be even")
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.max_seq_len = max_seq_len

        # Q path: down-project, RMSNorm, up-project to per-head [nope, pe].
        self.wq_a = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim, bias=False)

        # KV down-projection: produces shared latent c_kv and head-shared k_pe.
        self.wkv_a = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        # KV up-projection: c_kv -> per-head [k_nope, value].
        self.wkv_b = nn.Linear(
            kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )

        self.wo = nn.Linear(n_heads * v_head_dim, d_model, bias=bias)

        cos, sin = _build_rope_tables(qk_rope_head_dim, max_seq_len, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def init_cache(self) -> MLACache:
        return MLACache()

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        if k is not q or v is not q:
            warnings.warn(
                "mla ignores k/v; treating q as the hidden state stream",
                stacklevel=2,
            )

        b, s, _ = q.shape
        pos_offset = past_kv.position() if past_kv is not None else 0
        if pos_offset + s > self.max_seq_len:
            raise ValueError(
                f"mla: pos_offset ({pos_offset}) + s ({s}) exceeds "
                f"max_seq_len ({self.max_seq_len})"
            )

        # Q: down -> norm -> up -> reshape to [b, n_heads, s, qk_head_dim],
        # then split nope/pe along the last dim.
        c_q = self.q_norm(self.wq_a(q))
        q_proj = self.wq_b(c_q).view(b, s, self.n_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = q_proj.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV down-projection: c_kv stays pre-norm (norm is applied AFTER cache
        # concat so that a parameter update doesn't stale the cached values),
        # k_pe gets a broadcast head dim.
        kv_a = self.wkv_a(q)
        c_kv, k_pe = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(1)  # [b, 1, s, qk_rope_head_dim]

        # Apply rope at the absolute position of the new tokens.
        q_pe = _apply_rope(q_pe, self.rope_cos, self.rope_sin, pos_offset)
        k_pe = _apply_rope(k_pe, self.rope_cos, self.rope_sin, pos_offset)

        # Cache: store pre-norm c_kv and post-rope k_pe.
        if past_kv is not None:
            c_kv, k_pe = past_kv.update(c_kv, k_pe)

        # Up-project the (now full) latent into per-head k_nope and value.
        s_total = c_kv.shape[-2]
        kv = self.wkv_b(self.kv_norm(c_kv))
        kv = kv.view(b, s_total, self.n_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        key = torch.cat([k_nope, k_pe.expand(-1, self.n_heads, -1, -1)], dim=-1)
        query = torch.cat([q_nope, q_pe], dim=-1)

        # F.scaled_dot_product_attention reads its scale from key.shape[-1],
        # so it correctly uses 1/sqrt(qk_head_dim) even though v_head_dim differs.
        out = scaled_dot_product(query, key, value, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.v_head_dim)
        out = self.wo(out)

        if return_kv:
            return out, past_kv
        return out
