import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        src_pos: nn.Module,
        tgt_pos: nn.Module,
        projection: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_pos(self.src_embed(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_pos(self.tgt_embed(tgt))
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
