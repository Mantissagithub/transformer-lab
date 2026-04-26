from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.components.attention.base import AttentionBase
from src.components.connections.base import Connection
from src.components.feedforward.base import FeedForwardBase


class EncoderBlock(nn.Module):
    def __init__(
        self,
        attn: AttentionBase,
        ffn: FeedForwardBase,
        connections: List[Connection],
    ) -> None:
        super().__init__()
        if len(connections) != 2:
            raise ValueError(f"EncoderBlock needs 2 connections, got {len(connections)}")
        self.attn = attn
        self.ffn = ffn
        self.connections = nn.ModuleList(connections)

    def forward(self, state: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        state = self.connections[0].apply(state, lambda x: self.attn(x, x, x, src_mask))
        state = self.connections[1].apply(state, self.ffn)
        return state


class CausalBlock(nn.Module):
    """Decoder-only block: self-attention + FFN, with optional KV-cache pass-through.

    The training path uses ``forward(state, mask)`` and behaves exactly like an
    EncoderBlock with a causal mask. The generation path uses
    ``forward_with_cache(state, mask, past_kv)`` which (1) requires a non-stateful
    Connection (e.g. ResidualConnection) so the state shape is plain (b, s, d),
    and (2) returns the new (k, v) tuple for that layer.
    """

    def __init__(
        self,
        attn: AttentionBase,
        ffn: FeedForwardBase,
        connections: List[Connection],
    ) -> None:
        super().__init__()
        if len(connections) != 2:
            raise ValueError(f"CausalBlock needs 2 connections, got {len(connections)}")
        self.attn = attn
        self.ffn = ffn
        self.connections = nn.ModuleList(connections)

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        state = self.connections[0].apply(state, lambda x: self.attn(x, x, x, mask))
        state = self.connections[1].apply(state, self.ffn)
        return state

    def forward_with_cache(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_conn = self.connections[0]
        if attn_conn.is_stateful:
            raise NotImplementedError(
                "KV-cache generation requires a non-stateful Connection (e.g. residual)"
            )
        normed = attn_conn.norm(x)
        attn_out, new_kv = self.attn(normed, normed, normed, mask, past_kv=past_kv, return_kv=True)
        if attn_conn.post_norm_module is not None:
            attn_out = attn_conn.post_norm_module(attn_out)
        x = x + attn_conn.dropout(attn_out)
        x = self.connections[1].apply(x, self.ffn)
        return x, new_kv


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attn: AttentionBase,
        cross_attn: AttentionBase,
        ffn: FeedForwardBase,
        connections: List[Connection],
    ) -> None:
        super().__init__()
        if len(connections) != 3:
            raise ValueError(f"DecoderBlock needs 3 connections, got {len(connections)}")
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn = ffn
        self.connections = nn.ModuleList(connections)

    def forward(
        self,
        state: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = self.connections[0].apply(
            state, lambda x: self.self_attn(x, x, x, tgt_mask)
        )
        state = self.connections[1].apply(
            state, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask)
        )
        state = self.connections[2].apply(state, self.ffn)
        return state
