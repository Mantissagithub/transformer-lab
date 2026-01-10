# this is referred from this paper: https://arxiv.org/pdf/2409.19606
# and i have a good summarized writing on this one too: https://www.pradheep.dev/hc

# so there are 3 weight matrices over here: w(beta), w(m), and w(r)
# and also matrices like this
# Matrix Equation:
#  ⎛  0_{1×1} B^k ⎞     ⎛  0_{1×1} 1_{1×n}  ⎞
#  ⎜              |  =  |                   ⎟
#  ⎝  A_m^k A_r^k ⎠     ⎝  e_kmodn e_{n×n}  ⎠

# this is to basically replace residual connections, so need to decide the n too, like the widht and height of the connection, so that i can replicate the incoming features axccoridngly ad multiply with these and get the result
# H = norm(H) (10)
# B(H) = sβ ◦ tanh(HWβ)⊺ + B ∈ R1×n
# Am(H) = sα ◦ tanh(HWm) + Am ∈ R n×1
# Ar(H) = sα ◦ tanh(HWr) + Ar ∈ R n×n

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HyperConnection(nn.Module):
    def __init__(self, input_dim, n, layer_idx=0, dropout=0.1, device='cuda'):
        super(HyperConnection, self).__init__()
        self.input_dim = input_dim
        self.n = n
        self.layer_idx = layer_idx
        self.device = device
        self.dropout = nn.Dropout(dropout)

        # weight matrices
        self.W_beta = nn.Parameter(torch.Tensor(input_dim, n))
        self.W_m = nn.Parameter(torch.Tensor(input_dim, n))
        self.W_r = nn.Parameter(torch.Tensor(input_dim, n))

        # scaling parameters
        self.s_beta = nn.Parameter(torch.Tensor(1, n))
        self.s_alpha = nn.Parameter(torch.Tensor(n, 1))

        # bias terms
        self.B = nn.Parameter(torch.Tensor(1, n))
        self.A_m = nn.Parameter(torch.Tensor(n, 1))
        self.A_r = nn.Parameter(torch.Tensor(n, n))

        self.reset_parameters()

    def initialize_weights(self):
        # according to the paper, the weights are initialized as 0
        nn.init.zeros_(self.W_beta)
        nn.init.zeros_(self.W_m)
        nn.init.zeros_(self.W_r)

        # and the other a, b things are intilized according to the above matrix in comments
        # B^k -> 1_{1×n}
        nn.init.ones_(self.B)

        # A_m^k -> e_kmodn
        with torch.no_grad():
            self.A_m.zero_()
            k_idx = self.layer_idx % self.n
            self.A_m[k_idx, 0] = 1.0

        # A_r^k -> e_{n×n}
        with torch.no_grad():
            nn.init.eye_(self.A_r) # the eye is the identity matrix

        nn.init.ones_(self.s_beta)
        nn.init.ones_(self.s_alpha)

    def reset_parameters(self):
        self.initialize_weights()

    def prepare_hyper_hiddens(self, hyper_hiddens, batch_size, seq_len):
        selected = hyper_hiddens[-self.n:] if len(hyper_hiddens) >= self.n else list(hyper_hiddens)

        while len(selected) < self.n:
            # basically need to pad them with zeroes
            selected.insert(0, torch.zeros(batch_size, seq_len, self.input_dim, device=self.device))

        stacked = torch.stack(selected, dim=0)

        stacked = stacked.permute(1, 2, 0, 3).contiguous()
        stacked = stacked.view(batch_size * seq_len, self.n, self.input_dim)

        return stacked

    def forward(self, x, sublayer_fn, hyper_hiddens=None): # here the sublayer function can be anything like attetion or ffn
        batch_size, seq_len, dim = x.size()

        sublayer_output = self.dropout(sublayer_fn(x))

        if hyper_hiddens is None or len(hyper_hiddens) == 0:
            return x + sublayer_output # normal residual connection

        hyper_stack = self.prepare_hyper_hiddens(hyper_hiddens, batch_size, seq_len)

        H_bar = hyper_stack.mean(dim=1)
        H_bar = F.layer_norm(H_bar, [dim])

        B_H = self.s_beta * torch.tanh(H_bar @ self.W_beta) + self.B

        A_m_H = self.s_alpha.T * torch.tanh(H_bar @ self.W_m) + self.A_m.T

        H_global = H_bar.mean(dim=0, keepdim=True)
        A_r_H = self.s_alpha * torch.tanh(H_global @ self.W_r).T + self.A_r

        hyper_contribution = torch.einsum('bn,bnd->bd', B_H, hyper_stack)

        depth_contribution = torch.einsum('nm,bmd->bnd', A_r_H, hyper_stack)
        depth_contribution = depth_contribution.sum(dim=1)

        hyper_contribution = hyper_contribution.view(batch_size, seq_len, dim)
        depth_contribution = depth_contribution.view(batch_size, seq_len, dim)

        return x + sublayer_output + hyper_contribution + depth_contribution
