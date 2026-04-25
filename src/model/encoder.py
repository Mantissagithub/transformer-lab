import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, norm: nn.Module) -> None:
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Layer-0's first connection owns state init for the stack.
        head_conn = self.layers[0].connections[0]
        state = head_conn.init_state(x)
        for layer in self.layers:
            state = layer(state, src_mask)
        out = head_conn.to_output(state)
        return self.norm(out)
