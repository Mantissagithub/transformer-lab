# this file is gonna contain the implementation of manifold constrained hyper connections
# this is the paper link: [https://www.arxiv.org/pdf/2512.24880](https://www.arxiv.org/pdf/2512.24880)

import torch
import torch.nn as nn
import torch.nn.functional as F

def sinkhorn_knopp(m: torch.tensor):
  m = torch.exp(m)
  for _ in range(20):
    m = m / m.sum(dim=0, keepdim=True)
    m = m / m.sum(dim=1, keepdim=True)
  return m

class ManifoldConstrainedHyperConnection(nn.Module):
    def __init__(self, input_dim, n, layer_idx=0, dropout=0.1, dynamic=True, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.n = n
        self.layer_idx = layer_idx
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.dynamic = dynamic

        self.static_beta = nn.Parameter(torch.ones(n, device=device))

        init_alpha0 = torch.zeros((n, 1), device=device)
        init_alpha0[layer_idx % n, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye(n, device=device)], dim=1)
        )

        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(input_dim, n+1, device=device))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(input_dim, n, device=device))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.layer_norm = nn.LayerNorm(input_dim)

    def width_connection(self, h):
        batch, seq, n, dim = h.shape

        if self.dynamic:
            norm_h = self.layer_norm(h)
            norm_h_mean = norm_h.mean(dim=2)

            wc_weight = norm_h_mean @ self.dynamic_alpha_fn
            wc_weight = torch.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha.unsqueeze(2) + self.static_alpha
        else:
            alpha = self.static_alpha.unsqueeze(0).unsqueeze(0)

        if self.dynamic:
            dc_weight = norm_h_mean @ self.dynamic_beta_fn
            dc_weight = torch.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta
        else:
            beta = self.static_beta

        alpha_mc = sinkhorn_knopp(alpha.view(-1, self.n, self.n+1)).view(batch, seq, n, n+1)

        h_expanded = torch.cat([
            torch.zeros(batch, seq, 1, dim, device=h.device),
            h
        ], dim=2)

        mix_h = torch.einsum("bsnm,bsmd->bsnd", alpha_mc, h_expanded)

        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):   
        h_o_expanded = h_o.unsqueeze(2).expand(-1, -1, self.n, -1)
        h = beta.unsqueeze(3) * h_o_expanded + mix_h
        return h

    def forward(self, h, sublayer_fn):
        mix_h, beta = self.width_connection(h)

        branch_input = mix_h[:, :, 0, :]

        h_o = self.dropout(sublayer_fn(branch_input))

        h = self.depth_connection(mix_h, h_o, beta)

        return h
