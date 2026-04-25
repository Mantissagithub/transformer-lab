from typing import List

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
