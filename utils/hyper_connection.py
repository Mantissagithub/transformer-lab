# hyper_connection.py
# implementation based on algorithm 2 from appendix J of https://arxiv.org/pdf/2409.19606
# this maintains a hyper hidden matrix (batch, seq, n, dim) and mixes across width/depth

import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperConnection(nn.Module):
    def __init__(self, input_dim, n, layer_idx=0, dropout=0.1, dynamic=True, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.n = n  # rate in paper - number of parallel streams
        self.layer_idx = layer_idx
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.dynamic = dynamic

        # static beta: weights for depth connection, initialized to ones
        self.static_beta = nn.Parameter(torch.ones(n, device=device))

        # static alpha: weights for width connection
        # shape (n, n+1) = [e_k | I_n] where e_k is one-hot at layer_idx % n
        init_alpha0 = torch.zeros((n, 1), device=device)
        init_alpha0[layer_idx % n, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye(n, device=device)], dim=1)
        )

        if dynamic:
            # dynamic modulation parameters - start small to not break initialization
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(input_dim, n+1, device=device))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(input_dim, n, device=device))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.layer_norm = nn.LayerNorm(input_dim)

    def width_connection(self, h):
        batch, seq, n, dim = h.shape

        # compute alpha for width mixing
        if self.dynamic:
            # normalize and average across width dimension to get global context
            norm_h = self.layer_norm(h)  # (b, s, n, d)
            norm_h_mean = norm_h.mean(dim=2)  # (b, s, d)

            # compute dynamic alpha adjustment
            wc_weight = norm_h_mean @ self.dynamic_alpha_fn  # (b, s, n+1)
            wc_weight = torch.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha.unsqueeze(2) + self.static_alpha  # (b, s, n, n+1)
        else:
            alpha = self.static_alpha.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n+1)

        # compute beta for depth mixing
        if self.dynamic:
            dc_weight = norm_h_mean @ self.dynamic_beta_fn  # (b, s, n)
            dc_weight = torch.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta  # (b, s, n)
        else:
            beta = self.static_beta.unsqueeze(0).unsqueeze(0)  # (1, 1, n)

        # width connection: mix_h = alpha^T @ h
        # alpha is (b, s, n, n+1), h is (b, s, n, d) -> want (b, s, n+1, d)
        mix_h = torch.einsum('bsni,bsnd->bsid', alpha, h)  # (b, s, n+1, d)

        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        # h = h_o * beta + mix_h[:, :, 1:, :]
        # einsum: h_o is (b, s, d), beta is (b, s, n) -> (b, s, n, d)
        h = torch.einsum("bsd,bsn->bsnd", h_o, beta) + mix_h[:, :, 1:, :]
        return h

    def forward(self, h, sublayer_fn):
        # width connection extracts branch input and computes mixing weights
        mix_h, beta = self.width_connection(h)  # mix_h: (b, s, n+1, d), beta: (b, s, n)

        # first slice of mix_h is the input to current layer
        branch_input = mix_h[:, :, 0, :]  # (b, s, d)

        # apply sublayer (attention or ffn) to branch input
        h_o = self.dropout(sublayer_fn(branch_input))  # (b, s, d)

        # depth connection: add back output and advance hyper hidden streams
        h = self.depth_connection(mix_h, h_o, beta)  # (b, s, n, d)

        return h
