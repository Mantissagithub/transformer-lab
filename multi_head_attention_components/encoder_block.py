import torch
import torch.nn as nn
import math

from utils.feed_forward import FeedForwardNetwork
from utils.layer_normalization import LayerNormalization
from utils.residual_connection import ResidualCOnnection
from multi_head_attention_components.multihead_attention import MultiHeadAttentionNetwork
from utils.hyper_connection import HyperConnection

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionNetwork, feed_forward_block : FeedForwardNetwork, dropout : float, d_model:int, hyper_n:int=4, use_hyper_connection:bool=False, layer_idx:int=0, device:str='cuda'):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.use_hyper_connection = use_hyper_connection
        self.d_model = d_model
        self.hyper_n = hyper_n

        if use_hyper_connection:
            self.hc_attention = HyperConnection(d_model, hyper_n, layer_idx*2, dropout, dynamic=True, device=device)
            self.hc_ffn = HyperConnection(d_model, hyper_n, layer_idx*2+1, dropout, dynamic=True, device=device)
        else:
            self.residual_connection = nn.ModuleList([ResidualCOnnection(dropout) for _ in range(2)])
            self.norm_1 = LayerNormalization()
            self.norm_2 = LayerNormalization()

    def forward(self, x, src_mask, h=None):
        if self.use_hyper_connection:
            if h is None:
                batch, seq, d = x.shape
                h = x.unsqueeze(2).expand(-1, -1, self.hyper_n, -1)

            h = self.hc_attention(h, lambda branch_in: self.self_attention_block(branch_in, branch_in, branch_in, src_mask))

            h = self.hc_ffn(h, self.feed_forward_block)

            x = h[:, :, 0, :]
            return x, h
        else:
            x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
            x = self.residual_connection[1](x, self.feed_forward_block)
            return x, None


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        h = None
        for layer in self.layers:
            x, h = layer(x, src_mask, h)
        return self.norm(x)