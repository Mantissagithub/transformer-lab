# mla with weight absorption (deepseek-v2 §21)
# cache a shared kv latent c_kv and a head-shared decoupled-rope key k_pe
# apply key up-projection weights to the query and value up-projection
# weights after attention, **avoiding per-head cached k/v reconstruction**

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.registry import ATTENTION

from .base import AttentionBase, scaled_dot_product
from .kv_cache import MLACache
from .mla import _build_rope_tables, _apply_rope


@ATTENTION.register("mla_absorbed")
class MultiHeadLatentAttentionAbsorbed(AttentionBase):
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

        # q path: down-project, rmsnorm, up-project to per-head [nope, pe].
        self.wq_a = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim, bias=False)

        # kv down-projection: produces the shared latent c_kv and the head-shared k_pe.
        # cache concat so a parameter update doesn't stale the cached values
        # (rmsnorm has a learnable affine; caching post-norm freezes the scale at
        # the snapshot it was written under). k_pe gets a broadcast head dim.
        self.wkv_a = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
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
        # this is okay because q is not saved unlike k,v
        c_q = self.q_norm(self.wq_a(q))
        q_proj = self.wq_b(c_q).view(b, s, self.n_heads, self.qk_head_dim).transpose(1, 2)
        # split qk_head_dim into the configured nope and rope dimensions.
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
        # after update: c_kv [b, s_total, kv_lora_rank];
        # k_pe [b, 1, s_total, qk_rope_head_dim]
        if past_kv is not None:
            c_kv, k_pe = past_kv.update(c_kv, k_pe)
        c_kv = self.kv_norm(c_kv)
        
        # packed kv up projection weights
        # [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        # -> [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]
        wkv_b = self.wkv_b.weight.view(
             self.n_heads, 
             self.qk_nope_head_dim + self.v_head_dim,
             self.kv_lora_rank,
             )

        # wk_b: [n_heads, qk_nope_head_dim, kv_lora_rank]
        # wv_b: [n_heads, v_head_dim, kv_lora_rank]
        wk_b, wv_b = wkv_b.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
            )

        # [b, n_heads, s, qk_nope_head_dim] @ [n_heads, qk_nope_head_dim, kv_lora_rank]
        # -> [b, n_heads, s, kv_lora_rank]
        q_absorbed = torch.matmul(q_nope, wk_b)

        # concatenate nope (content) and rope (position) query / key vectors
        # query: [b, n_heads, s, kv_lora_rank + qk_rope_head_dim]
        query = torch.cat([q_absorbed, q_pe], dim=-1)
        # key: [b, n_heads, s_total, kv_lora_rank + qk_rope_head_dim]
        key = torch.cat([c_kv.unsqueeze(1), k_pe], dim=-1).expand(
            -1, self.n_heads, -1, -1,
        )
        # value: [b, n_heads, s_total, kv_lora_rank]
        value = c_kv.unsqueeze(1).expand(
            -1, self.n_heads, -1, -1,
        )

        # out_latent: [b, n_heads, s, kv_lora_rank]
        # preserve the original scale: 1 / sqrt(qk_head_dim)
        out_latent = scaled_dot_product(
            query, key, value, mask=mask, dropout=self.dropout,
            scale=self.qk_head_dim ** -0.5
        )

        # [b, n_heads, s, kv_lora_rank] @ [n_heads, kv_lora_rank, v_head_dim]
        # -> [b, n_heads, s, v_head_dim]
        out = torch.matmul(out_latent, wv_b.transpose(-1,-2))
        # merge heads: [b, s, n_heads * v_head_dim]
        out = out.transpose(1,2).contiguous().view(b, s, -1)
        # output: [b, s, d_model]
        out = self.wo(out)

        if return_kv:
            return out, past_kv
        return out
