from typing import Callable

import torch
import torch.nn as nn

from src.registry import CONNECTION

from .base import Connection


@CONNECTION.register("hyperconnection")
class HyperConnection(Connection):
    """Hyper-connection from arxiv 2409.19606 — width n parallel streams."""

    is_stateful = True

    def __init__(
        self,
        d_model: int,
        hyper_n: int = 4,
        layer_idx: int = 0,
        dropout: float = 0.1,
        dynamic: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = hyper_n
        self.dropout = nn.Dropout(dropout)
        self.dynamic = dynamic

        self.static_beta = nn.Parameter(torch.ones(self.n))
        init_alpha0 = torch.zeros((self.n, 1))
        init_alpha0[layer_idx % self.n, 0] = 1.0
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(self.n)], dim=1))

        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(d_model, self.n + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(d_model, self.n))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = nn.LayerNorm(d_model)

    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(2).expand(-1, -1, self.n, -1).contiguous()

    def to_output(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :, 0, :]

    def _width_connection(self, h: torch.Tensor):
        if self.dynamic:
            norm_h = self.layer_norm(h)
            norm_h_mean = norm_h.mean(dim=2)
            wc = torch.tanh(norm_h_mean @ self.dynamic_alpha_fn) * self.dynamic_alpha_scale
            alpha = wc.unsqueeze(2) + self.static_alpha
            dc = torch.tanh(norm_h_mean @ self.dynamic_beta_fn) * self.dynamic_beta_scale
            beta = dc + self.static_beta
        else:
            alpha = self.static_alpha.unsqueeze(0).unsqueeze(0)
            beta = self.static_beta.unsqueeze(0).unsqueeze(0)
        mix_h = torch.einsum("bsni,bsnd->bsid", alpha, h)
        return mix_h, beta

    def _depth_connection(self, mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bsd,bsn->bsnd", h_o, beta) + mix_h[:, :, 1:, :]

    def apply(self, state: torch.Tensor, sublayer_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        mix_h, beta = self._width_connection(state)
        branch_input = mix_h[:, :, 0, :]
        h_o = self.dropout(sublayer_fn(branch_input))
        return self._depth_connection(mix_h, h_o, beta)
