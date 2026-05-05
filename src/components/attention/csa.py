# csa (compressed sparse attention) is the new attention mechanism deepseek introduced in v4 to make long-context attention way cheaper. the main idea is: don't keep a full kv entry for every token, and don't even attend to all the entries you do keep. compress first, then sparsify on top of that.

# here's the actual flow:

# first, the compression. csa takes the hidden states and projects them into two parallel streams of latent kv entries (c^a and c^b) along with their own compression weights (z^a and z^b). then every m tokens get crunched into a single compressed kv entry — they do a softmax over 2m elements (because the two streams overlap across block boundaries) and a weighted sum to produce one compressed entry per m tokens. so the sequence dimension shrinks by 1/m. for v4-pro, m = 4.

# next, the sparse selection part — and this is where the lightning indexer comes in. once you have the compressed kv entries, you don't want to attend to all of them either. so the indexer generates its own queries in a low-rank way: the query token's hidden state is down-projected into a small latent vector c^Q_t, then up-projected into indexer queries. it scores each compressed block using a relu + weighted sum across indexer heads, and a top-k selector picks the k best compressed blocks. the indexer runs in fp4 to keep it fast — that's the "lightning" bit.

# then comes the core attention. on the selected top-k compressed entries, csa does multi-query attention where the compressed entry serves as both the key and the value (shared kv mqa). the attention queries are also produced by up-projecting the same latent c^Q_t that the indexer reused — so no extra down-projection cost.

# and here's the catch — they cleverly implemented grouped output projection. in v4-pro, head dim c = 512 and n_h = 128 heads, so c·n_h = 65,536. naively projecting that all the way back to the d-dim hidden state in one shot would be a massive matmul. so instead, they split the n_h heads into g groups, project each group down to a smaller intermediate dim d_g (with d_g < c·n_h/g), then concatenate the g intermediate outputs and do one final projection to d. two smaller projections instead of one huge one — way cheaper, same expressivity.

# on top of all this there's a small sliding-window branch (n_win = 128 recent uncompressed kv entries) tacked on, since strict block-level compression cuts off intra-block tokens and you still want fine-grained local dependencies.

# so in one line: csa = compress kv along the sequence (1/m), pick top-k blocks via a fp4 lightning indexer over latent queries, do shared-kv mqa on the selected blocks, and project the output back through grouped projection — giving you both kv-cache and flop savings at the same time.

"""
shapes follow the paper:
  n      = sequence length
  d      = hidden size (4096 for flash, 7168 for pro)
  c      = head dim (512)
  n_h    = number of query heads (64 for flash, 128 for pro)
  d_c    = query compression dim (1024 for flash, 1536 for pro)
  m      = compression rate (4)
  k      = top-k for sparse selection (512 for flash, 1024 for pro)
  n_win  = sliding window size (128)
  g      = number of output projection groups (8 for flash, 16 for pro)
  d_g    = intermediate output dim per group (1024)

implemented:
  compression (eq 9), lightning indexer, top-k selection, sliding window,
  shared-kv mqa with learnable per-head sink (eq 27), rmsnorm + partial rope
  with relative-position trick on q and kv (§2.3.3), grouped output projection.

not implemented (todo):
  batched-vectorized per-t compute, kv-cache for incremental decoding,
  fp4 indexer precision.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.registry import ATTENTION

from .base import AttentionBase


def compress_kv_entries(
    hidden_input_state: Tensor,    # [n, d]
    W_aKV: Tensor, W_bKV: Tensor,  # [d, c] each
    W_aZ:  Tensor, W_bZ:  Tensor,  # [d, c] each
    B_a:   Tensor, B_b:   Tensor,  # [m, c] each — learnable positional biases
    m: int,
) -> Tensor:                       # [n_blocks, c]
    n, _ = hidden_input_state.shape
    c = W_aKV.shape[1]
    n_full = (n // m) * m
    if n_full == 0:
        return torch.zeros(0, c, dtype=hidden_input_state.dtype, device=hidden_input_state.device)
    H = hidden_input_state[:n_full]
    n_blocks = n_full // m

    C_a = (H @ W_aKV).view(n_blocks, m, c)
    C_b = (H @ W_bKV).view(n_blocks, m, c)
    Z_a = (H @ W_aZ ).view(n_blocks, m, c)
    Z_b = (H @ W_bZ ).view(n_blocks, m, c)

    pad_C = torch.zeros(1, m, c, dtype=C_b.dtype, device=C_b.device)
    pad_Z = torch.full((1, m, c), float("-inf"), dtype=Z_b.dtype, device=Z_b.device)
    C_b = torch.cat([pad_C, C_b[:-1]], dim=0)
    Z_b = torch.cat([pad_Z, Z_b[:-1]], dim=0)

    logits = torch.cat([Z_a + B_a, Z_b + B_b], dim=1)   # [n_blocks, 2m, c]
    weights = torch.softmax(logits, dim=1)
    values = torch.cat([C_a, C_b], dim=1)               # [n_blocks, 2m, c]
    return (weights * values).sum(dim=1)                # [n_blocks, c]


def build_compressed_indexer_keys(
    hidden_states: Tensor,  # [n, d]
    w: dict,                # same key set as the main compression weights
    m: int,
) -> Tensor:                # [n_blocks, c_I]
    return compress_kv_entries(hidden_states, **w, m=m)


def build_query_latent(h_t: Tensor, W_DQ: Tensor) -> Tensor:
    return h_t @ W_DQ                                   # [d_c]


def build_indexer_queries(c_Q_t: Tensor, W_IUQ: Tensor, n_I_h: int, c_I: int) -> Tensor:
    return (c_Q_t @ W_IUQ).view(n_I_h, c_I)             # [n_I_h, c_I]


def build_main_queries(c_Q_t: Tensor, W_UQ: Tensor, n_h: int, c: int) -> Tensor:
    return (c_Q_t @ W_UQ).view(n_h, c)                  # [n_h, c]


def lightning_indexer(
    h_t: Tensor,         # [d]
    indexer_q: Tensor,   # [n_I_h, c_I]
    K_IComp: Tensor,     # [n_blocks, c_I]
    W_w: Tensor,         # [d, n_I_h]
) -> Tensor:             # [n_blocks]
    # paper runs this in fp4 ("lightning"); we run in the model dtype.
    if K_IComp.shape[0] == 0:
        return K_IComp.new_zeros(0)
    w = h_t @ W_w                                       # [n_I_h]
    inner = F.relu(indexer_q @ K_IComp.T)               # [n_I_h, n_blocks]
    return (w[:, None] * inner).sum(dim=0)              # [n_blocks]


def top_k_selector(
    scores: Tensor,   # [n_blocks]
    C_comp: Tensor,   # [n_blocks, c]
    t: int,
    m: int,
    k: int,
) -> Tensor:          # [k_actual, c]
    n_valid = min(t // m, scores.shape[0])
    if n_valid == 0:
        return C_comp.new_zeros(0, C_comp.shape[1])
    k_actual = min(k, n_valid)
    idx = scores[:n_valid].topk(k_actual).indices
    return C_comp[idx]


def sliding_window_kv(
    hidden_states: Tensor,  # [n, d]
    t: int,
    n_win: int,
    W_KV_swa: Tensor,       # [d, c]
) -> Tensor:                # [<=n_win, c]
    start = max(0, t - n_win + 1)
    return hidden_states[start : t + 1] @ W_KV_swa


def shared_kv_mqa(
    main_q: Tensor,      # [n_h, c]
    kv_entries: Tensor,  # [n_kv, c]   shared across all heads (k = v = kv)
    z_sink: Tensor,      # [n_h]       learnable per-head sink logit (eq 27)
) -> Tensor:             # [n_h, c]
    """
    shared-kv multi-query attention with a learnable per-head sink (paper eq 27).
    each head can dump prob mass into "nothing" via z_sink[i] instead of being
    forced to attend to whatever kv it's seen — useful when the indexer
    happens to surface low-relevance blocks.
    """
    n_h, c = main_q.shape
    if kv_entries.shape[0] == 0:
        # defensive: empty kv early in seq, sink would handle it but skip the matmul
        return main_q.new_zeros(n_h, c)
    logits = (main_q @ kv_entries.T) / math.sqrt(c)     # [n_h, n_kv]
    sink = z_sink[:, None]                              # [n_h, 1]
    ext = torch.cat([logits, sink], dim=-1)
    weights = torch.softmax(ext, dim=-1)[:, :-1]        # drop sink column
    return weights @ kv_entries                         # [n_h, c]


def grouped_output_projection(
    head_outputs: Tensor,  # [n_h, c]
    W_group: Tensor,       # [g, hpg*c, d_g]
    W_final: Tensor,       # [g*d_g, d]
    g: int,
) -> Tensor:               # [d]
    n_h, c = head_outputs.shape
    hpg = n_h // g
    flat = head_outputs.view(g, hpg * c)
    inter = torch.bmm(flat.unsqueeze(1), W_group).squeeze(1)   # [g, d_g]
    return inter.reshape(-1) @ W_final                          # [d]


def _build_rope_tables(rope_dim: int, max_seq_len: int, base: float = 10000.0):
    """precompute cos/sin tables for half-split rope (paper §2.3.3)."""
    half = rope_dim // 2
    if half == 0:
        empty = torch.zeros(max_seq_len, 0)
        return empty, empty.clone()
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))   # [half]
    pos = torch.arange(max_seq_len).float()                          # [max_seq_len]
    angles = pos[:, None] * freqs[None, :]                           # [max_seq_len, half]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: Tensor,              # [..., rope_dim]   rope_dim must be even
    positions,              # int (broadcast) or 1d long tensor over the last-1 dim
    cos_table: Tensor,      # [max_seq_len, rope_dim/2]
    sin_table: Tensor,      # [max_seq_len, rope_dim/2]
) -> Tensor:
    """
    half-split rope rotation. supports negative scalar positions for the
    post-attention de-rotate trick: cos(-θ) = cos(θ), sin(-θ) = -sin(θ).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    if isinstance(positions, int):
        idx = abs(positions)
        sign = 1.0 if positions >= 0 else -1.0
        cos = cos_table[idx]
        sin = sign * sin_table[idx]
    else:
        cos = cos_table[positions]
        sin = sin_table[positions]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# reference impl, single-token, no batching. used for tests; not called by the module.
def compressed_sparse_attention(
    hidden_states: Tensor,    # [n, d]
    t: int,                   # current query token
    weights: dict,
    m: int = 4,
    k: int = 1024,
    n_win: int = 128,
    g: int = 16,
    n_h: int = 128,
    n_I_h: int = 64,
    c: int = 512,
    c_I: int = 128,
    d_c: int = 1536,
    d: int = 7168,
) -> Tensor:                  # [d]
    """
    full csa for one query token, math summary end-to-end:

        # compression (eq 9)
        C^Comp   = compress(H ; W^aKV, W^bKV, W^aZ, W^bZ, B^a, B^b, m)
        K^IComp  = compress(H ; W^aIK, W^bIK, W^aIZ, W^bIZ, B^aI, B^bI, m)

        # query latent (down-project ONCE, reuse twice)
        c^Q_t    = h_t · W^DQ
        q^I_t    = c^Q_t · W^IUQ                  # indexer queries
        q_t      = c^Q_t · W^UQ                   # main queries

        # sparse selection
        w^I_t        = h_t · W^w
        I[t, s]      = sum_h  w^I_{t,h} · ReLU( q^I_{t,h} · K^IComp[s] )
        SprsComp_t   = { C^Comp[s] | s ∈ argTopk_s I[t, :], s < floor(t/m) }

        # local context
        SWA_t    = (last n_win hidden states) · W^KV_swa
        KV_t     = concat(SprsComp_t, SWA_t)       # [k + n_win, c]

        # core attention (shared-kv mqa with learnable sink, eq 27)
        o_{t, i} = softmax_with_sink( q_{t,i} · KV_t^T / sqrt(c) ; z_sink_i ) · KV_t

        # grouped output projection
        ô_t      = grouped_proj( [o_{t,1}; ...; o_{t,n_h}] ; W_group, W_final, g )

        return ô_t

    note: this reference omits rmsnorm and partial rope (paper §2.3.3). those
    live in the module wrapper since they require positional context.
    """
    C_comp       = compress_kv_entries(hidden_states, **weights["compress"], m=m)
    K_IComp      = build_compressed_indexer_keys(hidden_states, weights["indexer_keys"], m=m)
    c_Q_t        = build_query_latent(hidden_states[t], weights["W_DQ"])
    indexer_q    = build_indexer_queries(c_Q_t, weights["W_IUQ"], n_I_h, c_I)
    scores       = lightning_indexer(hidden_states[t], indexer_q, K_IComp, weights["W_w"])
    selected_kv  = top_k_selector(scores, C_comp, t, m, k)
    swa_kv       = sliding_window_kv(hidden_states, t, n_win, weights["W_KV_swa"])
    kv_entries   = torch.cat([selected_kv, swa_kv], dim=0)
    main_q       = build_main_queries(c_Q_t, weights["W_UQ"], n_h, c)
    head_outputs = shared_kv_mqa(main_q, kv_entries, weights["z_sink"])
    return grouped_output_projection(head_outputs, weights["W_group"], weights["W_final"], g)


@ATTENTION.register("csa")
class CompressedSparseAttention(AttentionBase):
    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        n_I_h:       int,
        m:           int,
        k:           int,
        n_win:       int,
        g:           int,
        c:           int,
        c_I:         int,
        d_c:         int,
        d_g:         int,
        rope_dim:    int = 64,
        max_seq_len: int = 32768,
        rope_base:   float = 10000.0,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        if n_heads % g != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by g ({g})")
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim ({rope_dim}) must be even")
        if rope_dim > c:
            raise ValueError(f"rope_dim ({rope_dim}) must be <= c ({c})")

        self.c, self.c_I, self.d_c, self.d_g = c, c_I, d_c, d_g
        self.n_I_h, self.m, self.k, self.n_win, self.g = n_I_h, m, k, n_win, g
        self.rope_dim = rope_dim
        self.max_seq_len = max_seq_len

        # compression weights — kv
        self.W_aKV = nn.Parameter(torch.empty(d_model, c))
        self.W_bKV = nn.Parameter(torch.empty(d_model, c))
        self.W_aZ  = nn.Parameter(torch.empty(d_model, c))
        self.W_bZ  = nn.Parameter(torch.empty(d_model, c))
        self.B_a   = nn.Parameter(torch.empty(m, c))
        self.B_b   = nn.Parameter(torch.empty(m, c))

        # compression weights — indexer keys
        self.W_aIK = nn.Parameter(torch.empty(d_model, c_I))
        self.W_bIK = nn.Parameter(torch.empty(d_model, c_I))
        self.W_aIZ = nn.Parameter(torch.empty(d_model, c_I))
        self.W_bIZ = nn.Parameter(torch.empty(d_model, c_I))
        self.B_aI  = nn.Parameter(torch.empty(m, c_I))
        self.B_bI  = nn.Parameter(torch.empty(m, c_I))

        # query down/up + indexer score weights
        self.W_DQ  = nn.Parameter(torch.empty(d_model, d_c))
        self.W_UQ  = nn.Parameter(torch.empty(d_c, n_heads * c))
        self.W_IUQ = nn.Parameter(torch.empty(d_c, n_I_h * c_I))
        self.W_w   = nn.Parameter(torch.empty(d_model, n_I_h))

        # sliding-window kv projection
        self.W_KV_swa = nn.Parameter(torch.empty(d_model, c))

        # grouped output projection
        hpg = n_heads // g
        self.W_group = nn.Parameter(torch.empty(g, hpg * c, d_g))
        self.W_final = nn.Parameter(torch.empty(g * d_g, d_model))

        # learnable per-head attention sink (paper eq 27)
        self.z_sink = nn.Parameter(torch.zeros(n_heads))

        # rmsnorm on q and kv (paper §2.3.3)
        self.q_norm  = nn.RMSNorm(c)
        self.kv_norm = nn.RMSNorm(c)

        # rope tables — buffers, not learned
        cos, sin = _build_rope_tables(rope_dim, max_seq_len, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 2d projection weights — xavier-uniform
        for p in (
            self.W_aKV, self.W_bKV, self.W_aZ, self.W_bZ,
            self.W_aIK, self.W_bIK, self.W_aIZ, self.W_bIZ,
            self.W_DQ,  self.W_UQ,  self.W_IUQ, self.W_w,
            self.W_KV_swa, self.W_final,
        ):
            nn.init.xavier_uniform_(p)
        # 3d grouped-projection: xavier per group slice
        for gi in range(self.g):
            nn.init.xavier_uniform_(self.W_group[gi])
        # learnable biases and sink — zero
        for p in (self.B_a, self.B_b, self.B_aI, self.B_bI, self.z_sink):
            nn.init.zeros_(p)

    def _weights(self) -> dict:
        return {
            "compress": dict(
                W_aKV=self.W_aKV, W_bKV=self.W_bKV,
                W_aZ=self.W_aZ,   W_bZ=self.W_bZ,
                B_a=self.B_a,     B_b=self.B_b,
            ),
            "indexer_keys": dict(
                W_aKV=self.W_aIK, W_bKV=self.W_bIK,
                W_aZ=self.W_aIZ,  W_bZ=self.W_bIZ,
                B_a=self.B_aI,    B_b=self.B_bI,
            ),
            "W_DQ":     self.W_DQ,
            "W_UQ":     self.W_UQ,
            "W_IUQ":    self.W_IUQ,
            "W_w":      self.W_w,
            "W_KV_swa": self.W_KV_swa,
            "W_group":  self.W_group,
            "W_final":  self.W_final,
            "z_sink":   self.z_sink,
        }

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        """
        csa is intrinsically causal self-attention. `q` carries the hidden
        states; `k`, `v`, and `mask` are ignored — causality is enforced by
        the top-k selector's `s < floor(t/m)` constraint and by the sliding
        window's `: t+1` slice.

        positional scheme (paper §2.3.3): partial rope is applied to the last
        `rope_dim` dims of the main queries and the kv entries. queries get
        rope at position t. sliding-window kv entries get rope at their actual
        token positions [start, t]. top-k compressed kv entries get rope at
        their block index. after attention, the rope half of the per-head
        output is re-rotated at position -t to make the result
        translation-invariant for the layer above (relative-position trick).
        """
        if past_kv is not None or return_kv:
            raise NotImplementedError("csa does not support KV cache")
        if k is not q or v is not q:
            warnings.warn(
                "csa ignores k/v; treating q as the hidden state stream",
                stacklevel=2,
            )

        b, s, _ = q.shape
        rope_dim = self.rope_dim
        main_dim = self.c - rope_dim
        out = q.new_zeros(b, s, self.d_model)

        # todo: vectorize the per-batch and per-t loops for training throughput
        for bi in range(b):
            H = q[bi]   # [s, d]

            # compression depends only on the full sequence — hoist out of t-loop
            C_comp = compress_kv_entries(
                H, self.W_aKV, self.W_bKV, self.W_aZ, self.W_bZ,
                self.B_a, self.B_b, self.m,
            )                                               # [n_blocks, c]
            K_IComp = compress_kv_entries(
                H, self.W_aIK, self.W_bIK, self.W_aIZ, self.W_bIZ,
                self.B_aI, self.B_bI, self.m,
            )                                               # [n_blocks, c_I]

            for t in range(s):
                c_Q_t     = build_query_latent(H[t], self.W_DQ)
                indexer_q = build_indexer_queries(c_Q_t, self.W_IUQ, self.n_I_h, self.c_I)
                scores    = lightning_indexer(H[t], indexer_q, K_IComp, self.W_w)

                # inline top-k so we keep block indices for rope positions
                n_valid = min(t // self.m, scores.shape[0])
                if n_valid > 0:
                    k_actual = min(self.k, n_valid)
                    sel_idx  = scores[:n_valid].topk(k_actual).indices
                    selected = C_comp[sel_idx]
                else:
                    sel_idx  = scores.new_zeros(0, dtype=torch.long)
                    selected = C_comp.new_zeros(0, self.c)

                start  = max(0, t - self.n_win + 1)
                swa_kv = H[start : t + 1] @ self.W_KV_swa   # [<=n_win, c]

                kv     = torch.cat([selected, swa_kv], dim=0)
                main_q = build_main_queries(c_Q_t, self.W_UQ, self.n_heads, self.c)

                # rmsnorm on q and kv before rope (§2.3.3)
                main_q = self.q_norm(main_q)
                kv     = self.kv_norm(kv)

                # partial rope on the last rope_dim of q and kv
                if rope_dim > 0:
                    q_main, q_rope   = main_q[..., :main_dim], main_q[..., main_dim:]
                    kv_main, kv_rope = kv[..., :main_dim],     kv[..., main_dim:]

                    q_rope = apply_rope(q_rope, t, self.rope_cos, self.rope_sin)

                    swa_pos = torch.arange(start, t + 1, device=H.device)
                    if sel_idx.numel() > 0:
                        kv_pos = torch.cat([sel_idx, swa_pos], dim=0)
                    else:
                        kv_pos = swa_pos
                    kv_rope = apply_rope(kv_rope, kv_pos, self.rope_cos, self.rope_sin)

                    main_q = torch.cat([q_main, q_rope], dim=-1)
                    kv     = torch.cat([kv_main, kv_rope], dim=-1)

                heads = shared_kv_mqa(main_q, kv, self.z_sink)

                # de-rotate the rope half at -t (relative-position trick)
                if rope_dim > 0:
                    o_main, o_rope = heads[..., :main_dim], heads[..., main_dim:]
                    o_rope = apply_rope(o_rope, -t, self.rope_cos, self.rope_sin)
                    heads  = torch.cat([o_main, o_rope], dim=-1)

                out[bi, t] = grouped_output_projection(
                    heads, self.W_group, self.W_final, self.g,
                )

        return self.dropout(out)


if __name__ == "__main__":
    # smoke test for the module — not a unit suite.
    torch.manual_seed(0)
    m = CompressedSparseAttention(
        d_model=128, n_heads=8, n_I_h=4,
        m=4, k=8, n_win=16, g=2,
        c=32, c_I=16, d_c=64, d_g=32,
        rope_dim=16, max_seq_len=128,
    )
    q = torch.randn(2, 64, 128)
    out = m(q, q, q)
    assert out.shape == (2, 64, 128), out.shape
    assert not torch.isnan(out).any(), "csa produced nans"
    loss = out.sum()
    loss.backward()
    assert m.z_sink.grad is not None, "z_sink did not receive a gradient"
    print("csa smoke test ok")
