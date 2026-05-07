# hca (heavily compressed attention) is the deepseek-v4 §2.3.2 sibling of csa.
# the idea is the same skeleton — compress kv along the sequence, run shared-kv
# mqa over the compressed entries, then grouped output projection — but with
# two big simplifications:
#   1) compression is a single stream with NO overlap (no a/b duals). every m'
#      tokens collapse into one entry via softmax(Z + B) over the m' rows.
#   2) NO sparse selection, NO lightning indexer, NO sliding window. every
#      compressed entry is attended for every query (modulo causal masking at
#      the block boundary).
#
# m' is meant to be much larger than csa's m (the paper writes m' >> m=4), so
# the kv cache is heavily compressed — hence the name. there's no fine-grained
# local branch to fill in intra-block detail; if you want that, use csa.
#
# this implementation tracks paper §2.3.2 strictly. no rmsnorm, no rope, no
# learnable sink — those are §2.3.3 add-ons that csa adopts but the user asked
# for §2.3.2 only. position info enters only through the per-position bias B
# inside each block.

"""
shapes:
  n      = sequence length
  d      = hidden size
  c      = head dim / compressed kv dim
  n_h    = number of query heads
  d_c    = query latent dim
  m      = compression rate m' (heavy; >> csa's m=4)
  g      = number of output projection groups
  d_g    = intermediate output dim per group

implemented:
  compression (eq 20–23), low-rank query (eq 24–25), shared-kv mqa over
  all compressed blocks (eq 26), grouped output projection. vectorized
  per-t compute on a sequence; kv-cache for incremental decode.

not implemented (future):
  batched-over-bi vectorization (forward still loops batch elements).
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.registry import ATTENTION

from .base import AttentionBase
from .csa import build_main_queries, build_query_latent, grouped_output_projection
from .kv_cache import HCACache


def compress_kv_entries_hca(
    H: Tensor,     # [n, d]
    W_KV: Tensor,  # [d, c]
    W_Z:  Tensor,  # [d, c]
    B:    Tensor,  # [m, c]
    m: int,
) -> Tensor:       # [n_blocks, c]
    """single-stream, non-overlapping compression (paper eq 20–23)."""
    n, _ = H.shape
    c = W_KV.shape[1]
    n_full = (n // m) * m
    if n_full == 0:
        return torch.zeros(0, c, dtype=H.dtype, device=H.device)
    H_full = H[:n_full]
    n_blocks = n_full // m

    C = (H_full @ W_KV).view(n_blocks, m, c)        # eq 20
    Z = (H_full @ W_Z ).view(n_blocks, m, c)        # eq 21
    weights = torch.softmax(Z + B, dim=1)           # eq 22 — softmax over m
    return (weights * C).sum(dim=1)                 # eq 23


def compress_one_block_hca(
    block: Tensor,  # [b, m, d]
    W_KV: Tensor,
    W_Z:  Tensor,
    B:    Tensor,
) -> Tensor:        # [b, c]
    """compress a single completed block. equivalent to one row of
    compress_kv_entries_hca, but stays on the [b, m, d] layout the decode
    path uses."""
    C = block @ W_KV                                # [b, m, c]
    Z = block @ W_Z                                 # [b, m, c]
    weights = torch.softmax(Z + B, dim=-2)          # softmax over m
    return (weights * C).sum(dim=-2)                # [b, c]


def shared_kv_mqa_hca(
    main_q: Tensor,         # [n_h, c]
    kv_entries: Tensor,     # [n_kv, c]
) -> Tensor:                # [n_h, c]
    """vanilla shared-kv mqa, no sink. matches paper eq 26."""
    n_h, c = main_q.shape
    if kv_entries.shape[0] == 0:
        return main_q.new_zeros(n_h, c)
    logits = (main_q @ kv_entries.T) / math.sqrt(c)
    weights = torch.softmax(logits, dim=-1)
    return weights @ kv_entries


def heavily_compressed_attention(
    hidden_states: Tensor,    # [n, d]
    t: int,
    weights: dict,
    m: int,
    g: int,
    n_h: int,
    c: int,
) -> Tensor:                  # [d]
    """reference single-token forward, used for tests and as documentation:

        # compression (eq 20–23)
        C^Comp   = compress(H ; W_KV, W_Z, B, m')

        # query latent (eq 24–25)
        c^Q_t    = h_t · W^DQ
        q_t      = c^Q_t · W^UQ

        # core attention over all visible compressed blocks (eq 26)
        valid    = { b : b < floor(t/m') }
        o_{t,i}  = softmax( q_{t,i} · C^Comp[valid]^T / sqrt(c) ) · C^Comp[valid]

        # grouped output projection
        ô_t      = grouped_proj( [o_{t,1}; ...; o_{t,n_h}] ; W_group, W_final, g )
    """
    C_comp = compress_kv_entries_hca(hidden_states, **weights["compress"], m=m)
    n_blocks = C_comp.shape[0]
    n_valid = min(t // m, n_blocks)
    visible = C_comp[:n_valid]
    c_Q_t  = build_query_latent(hidden_states[t], weights["W_DQ"])
    main_q = build_main_queries(c_Q_t, weights["W_UQ"], n_h, c)
    heads  = shared_kv_mqa_hca(main_q, visible)
    return grouped_output_projection(heads, weights["W_group"], weights["W_final"], g)


@ATTENTION.register("hca")
class HeavilyCompressedAttention(AttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        m:       int,
        g:       int,
        c:       int,
        d_c:     int,
        d_g:     int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        if n_heads % g != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by g ({g})")

        self.c, self.d_c, self.d_g = c, d_c, d_g
        self.m, self.g = m, g

        # compression (eq 20–23)
        self.W_KV = nn.Parameter(torch.empty(d_model, c))
        self.W_Z  = nn.Parameter(torch.empty(d_model, c))
        self.B    = nn.Parameter(torch.empty(m, c))

        # low-rank query (eq 24–25)
        self.W_DQ = nn.Parameter(torch.empty(d_model, d_c))
        self.W_UQ = nn.Parameter(torch.empty(d_c, n_heads * c))

        # grouped output projection
        hpg = n_heads // g
        self.W_group = nn.Parameter(torch.empty(g, hpg * c, d_g))
        self.W_final = nn.Parameter(torch.empty(g * d_g, d_model))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.W_KV, self.W_Z, self.W_DQ, self.W_UQ, self.W_final):
            nn.init.xavier_uniform_(p)
        for gi in range(self.g):
            nn.init.xavier_uniform_(self.W_group[gi])
        nn.init.zeros_(self.B)

    def init_cache(self) -> HCACache:
        return HCACache(self.m)

    def _weights(self) -> dict:
        return {
            "compress": dict(W_KV=self.W_KV, W_Z=self.W_Z, B=self.B),
            "W_DQ":    self.W_DQ,
            "W_UQ":    self.W_UQ,
            "W_group": self.W_group,
            "W_final": self.W_final,
        }

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        """hca is intrinsically causal self-attention. k/v/mask are ignored;
        causality lives in the per-query block-validity mask `b < floor(t/m)`."""
        if k is not q or v is not q:
            warnings.warn(
                "hca ignores k/v; treating q as the hidden state stream",
                stacklevel=2,
            )

        if past_kv is not None and past_kv.total_seen > 0 and q.shape[1] == 1:
            out = self._decode_step(q, past_kv)
            return (out, past_kv) if return_kv else out

        b, _, _ = q.shape
        outs = [self._forward_seq(q[bi]) for bi in range(b)]
        out = self.dropout(torch.stack(outs, dim=0))
        if return_kv:
            if past_kv is None:
                past_kv = self.init_cache()
            self._populate_cache_from_prefill(q, past_kv)
            return out, past_kv
        return out

    def _forward_seq(self, H: Tensor) -> Tensor:
        """vectorized per-t compute for one sequence. [s, d] -> [s, d]."""
        s, _ = H.shape
        c, n_h, m, g = self.c, self.n_heads, self.m, self.g

        C_comp = compress_kv_entries_hca(H, self.W_KV, self.W_Z, self.B, m)
        n_blocks = C_comp.shape[0]

        c_Q    = H @ self.W_DQ                                                 # [s, d_c]
        main_q = (c_Q @ self.W_UQ).view(s, n_h, c)                             # [s, n_h, c]

        if n_blocks == 0:
            heads = main_q.new_zeros(s, n_h, c)
        else:
            ts        = torch.arange(s, device=H.device)
            bs        = torch.arange(n_blocks, device=H.device)
            valid     = bs[None, :] < (ts[:, None] // m)                       # [s, n_blocks]
            any_valid = valid.any(dim=-1)                                      # [s]

            logits = torch.einsum("shc,bc->shb", main_q, C_comp) / math.sqrt(c)
            mask   = valid[:, None, :].expand(s, n_h, n_blocks)
            logits = logits.masked_fill(~mask, float("-inf"))
            # rows with no visible blocks would softmax over all-(-inf) and NaN.
            # substitute zeros there; we'll zero the head outputs out below.
            no_kv  = (~any_valid)[:, None, None].expand(s, n_h, n_blocks)
            logits = logits.masked_fill(no_kv, 0.0)
            weights = torch.softmax(logits, dim=-1)
            heads   = torch.einsum("shb,bc->shc", weights, C_comp)             # [s, n_h, c]
            heads   = heads * any_valid[:, None, None]

        hpg   = n_h // g
        flat  = heads.reshape(s, g, hpg * c)
        inter = torch.einsum("sgi,gid->sgd", flat, self.W_group)               # [s, g, d_g]
        return inter.reshape(s, g * self.d_g) @ self.W_final                   # [s, d]

    def _populate_cache_from_prefill(self, H: Tensor, cache: HCACache) -> None:
        """build the decode-side state from the just-processed prefill batch.
        H is [b, s, d]; cache fields land with leading b dim."""
        b, s, _ = H.shape
        m = self.m
        n_blocks = s // m

        if n_blocks > 0:
            full = H[:, : n_blocks * m, :]
            C_list = [
                compress_kv_entries_hca(full[bi], self.W_KV, self.W_Z, self.B, m)
                for bi in range(b)
            ]
            cache.C_comp = torch.stack(C_list, dim=0)              # [b, n_blocks, c]
        else:
            cache.C_comp = H.new_zeros(b, 0, self.c)

        cache.raw_tail   = H[:, n_blocks * m :, :].clone()         # [b, s%m, d]
        cache.total_seen = s

    def _decode_step(self, q: Tensor, cache: HCACache) -> Tensor:
        """incremental decode for a single new token. q is [b, 1, d]; returns
        [b, 1, d]. mutates `cache` in place."""
        b, _, d = q.shape
        m, n_h, c, g = self.m, self.n_heads, self.c, self.g
        h_new = q[:, 0, :]                                          # [b, d]

        new_tok = h_new[:, None, :]                                 # [b, 1, d]
        if cache.raw_tail is None or cache.raw_tail.shape[1] == 0:
            cache.raw_tail = new_tok
        else:
            cache.raw_tail = torch.cat([cache.raw_tail, new_tok], dim=1)

        if cache.raw_tail.shape[1] == m:
            new_C = compress_one_block_hca(
                cache.raw_tail, self.W_KV, self.W_Z, self.B,
            )                                                        # [b, c]
            cache.C_comp = torch.cat([cache.C_comp, new_C[:, None, :]], dim=1)
            cache.raw_tail = h_new.new_zeros(b, 0, d)

        t_abs = cache.total_seen
        cache.total_seen += 1

        outs = []
        for bi in range(b):
            c_Q_t   = h_new[bi] @ self.W_DQ                         # [d_c]
            main_q  = (c_Q_t @ self.W_UQ).view(n_h, c)              # [n_h, c]

            n_blocks = cache.C_comp.shape[1]
            n_valid  = min(t_abs // m, n_blocks)
            if n_valid > 0:
                kv    = cache.C_comp[bi, :n_valid]                  # [n_valid, c]
                heads = shared_kv_mqa_hca(main_q, kv)               # [n_h, c]
            else:
                heads = main_q.new_zeros(n_h, c)

            outs.append(grouped_output_projection(
                heads, self.W_group, self.W_final, g,
            ))                                                       # [d]

        return self.dropout(torch.stack(outs, dim=0)[:, None, :])    # [b, 1, d]

    def _forward_seq_serial(self, H: Tensor) -> Tensor:
        """per-t reference implementation, used to verify the vectorized path."""
        s, _ = H.shape
        out = H.new_zeros(s, self.d_model)

        C_comp   = compress_kv_entries_hca(H, self.W_KV, self.W_Z, self.B, self.m)
        n_blocks = C_comp.shape[0]

        for t in range(s):
            c_Q_t   = build_query_latent(H[t], self.W_DQ)
            main_q  = build_main_queries(c_Q_t, self.W_UQ, self.n_heads, self.c)
            n_valid = min(t // self.m, n_blocks)
            if n_valid > 0:
                heads = shared_kv_mqa_hca(main_q, C_comp[:n_valid])
            else:
                heads = main_q.new_zeros(self.n_heads, self.c)
            out[t] = grouped_output_projection(
                heads, self.W_group, self.W_final, self.g,
            )
        return out


if __name__ == "__main__":
    # smoke test for the module — not a unit suite.
    torch.manual_seed(0)
    mod = HeavilyCompressedAttention(
        d_model=128, n_heads=8, m=8, g=2,
        c=32, d_c=64, d_g=32,
    )
    x = torch.randn(2, 64, 128)

    out = mod(x, x, x)
    assert out.shape == (2, 64, 128), out.shape
    assert not torch.isnan(out).any(), "hca produced nans"

    with torch.no_grad():
        ref = torch.stack([mod._forward_seq_serial(x[bi]) for bi in range(x.shape[0])], dim=0)
    diff = (out - ref).abs().max().item()
    assert diff < 1e-4, f"vectorized != serial, max abs diff {diff}"

    loss = out.sum()
    loss.backward()
    for name, p in mod.named_parameters():
        assert p.grad is not None, f"{name} did not receive a gradient"
    print(f"hca smoke test ok (vec vs serial max diff: {diff:.2e})")
