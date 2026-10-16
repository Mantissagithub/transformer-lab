# mla (multi-head latent attention) is deepseek-v2's trick for shrinking the
# kv-cache without giving up much quality. the main idea is: instead of caching
# per-head k and v, cache a single shared low-rank latent c_kv (rank
# kv_lora_rank << d_model) plus a small head-shared decoupled-rope key k_pe.
# weight_absorption optionally moves the key projection to the query and
# the value projection after attention, avoiding per-head k/v reconstruction
#
# also: self-attention only. q, k, v must be the same tensor (warns otherwise);
# cross-attn would silently throw away the encoder output, so the encoder-
# decoder builder rejects mla up front.

"""multi-head latent attention (deepseek-v2 §2.1).

instead of storing per-head k/v we
store a single shared low-rank latent ``c_kv`` (rank ``kv_lora_rank``) plus a
small head-shared decoupled-rope key ``k_pe`` (size ``qk_rope_head_dim``).
by default, per-head k and v are reconstructed through ``wkv_b``.
with ``weight_absorption=True``, attention operates on the latent directly,
while the position-dependent RoPE path stays separate. both paths share
parameters and support training and cached decoding.

self-attention only: ``q``, ``k``, ``v`` must be the same tensor (warns
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
    """half-split rope cos/sin tables. same convention as csa.py — do not
    swap in the interleaved variant from positional/rope.py; both rotate the
    same pairs but in different memory layouts, so mixing silently produces
    wrong attention scores."""
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
        weight_absorption: bool = False,
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
        self.weight_absorption = weight_absorption

        # q path: down-project, rmsnorm, up-project to per-head [nope, pe].
        self.wq_a = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim, bias=False)

        # kv down-projection: produces the shared latent c_kv and the head-shared k_pe.
        self.wkv_a = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        # kv up-projection: c_kv -> per-head [k_nope, value].
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

        # q: down -> norm -> up -> reshape to [b, n_heads, s, qk_head_dim], then
        # split nope/pe along the last dim.
        c_q = self.q_norm(self.wq_a(q))
        q_proj = self.wq_b(c_q).view(b, s, self.n_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = q_proj.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # kv down-projection: c_kv stays pre-norm — kv_norm is applied after the
        # cache concat so a parameter update doesn't stale the cached values
        # (rmsnorm has a learnable affine; caching post-norm freezes the scale at
        # the snapshot it was written under). k_pe gets a broadcast head dim.
        kv_a = self.wkv_a(q)
        c_kv, k_pe = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(1)  # [b, 1, s, qk_rope_head_dim]

        # rope at the absolute position of the new tokens.
        q_pe = _apply_rope(q_pe, self.rope_cos, self.rope_sin, pos_offset)
        k_pe = _apply_rope(k_pe, self.rope_cos, self.rope_sin, pos_offset)

        # cache stores pre-norm c_kv and post-rope k_pe.
        if past_kv is not None:
            c_kv, k_pe = past_kv.update(c_kv, k_pe)

        c_kv = self.kv_norm(c_kv)
        if self.weight_absorption:
            # apply key up-projection weights to the query and value up-projection
            # weights after attention, **avoiding per-head cached k/v reconstruction**
            # wkv_b: [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
            wkv_b = self.wkv_b.weight.view(
                self.n_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )

            wk_b, wv_b = wkv_b.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=1
            )

            # [b, n_heads, s, qk_nope_head_dim] @ [n_heads, qk_nope_head_dim, kv_lora_rank]
            # -> [b, n_heads, s, kv_lora_rank]
            q_absorbed = torch.matmul(q_nope, wk_b)

            # concatenate nope (content) and rope (position) query / key vectors
            query = torch.cat([q_absorbed, q_pe], dim=-1)
            key = torch.cat([c_kv.unsqueeze(1), k_pe], dim=-1).expand(
                -1, self.n_heads, -1, -1
            )
            value = c_kv.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

            # out_latent: [b, n_heads, s, kv_lora_rank]
            out_latent = scaled_dot_product(
                query, key, value, mask, self.dropout,
                # sdpa uses the query's last dimension to compute its default scale
                # explicitly preserve the original attention scale after absorption
                scale=self.qk_head_dim ** -0.5,
            )

            # [b, n_heads, s, kv_lora_rank] @ [n_heads, kv_lora_rank, v_head_dim]
            # -> [b, n_heads, s, v_head_dim]
            out = torch.matmul(out_latent, wv_b.transpose(-1, -2))
        else:
            # up-project the (now full) latent into per-head k_nope and value.
            s_total = c_kv.shape[-2]
            kv = self.wkv_b(c_kv)
            kv = kv.view(b, s_total, self.n_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
            k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            key = torch.cat([k_nope, k_pe.expand(-1, self.n_heads, -1, -1)], dim=-1)
            query = torch.cat([q_nope, q_pe], dim=-1)
            # f.scaled_dot_product_attention reads its scale from key.shape[-1], so
            # it correctly uses 1/sqrt(qk_head_dim) even though v_head_dim differs.
            out = scaled_dot_product(query, key, value, mask, self.dropout)
        # merge heads: [b, s, n_heads * v_head_dim]
        out = out.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.v_head_dim)
        # output: [b, s, d_model]
        out = self.wo(out)

        if return_kv:
            return out, past_kv
        return out
