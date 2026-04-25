import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, norm: nn.Module) -> None:
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        head_conn = self.layers[0].connections[0]
        state = head_conn.init_state(x)
        for layer in self.layers:
            state = layer(state, encoder_output, src_mask, tgt_mask)
        out = head_conn.to_output(state)
        return self.norm(out)
