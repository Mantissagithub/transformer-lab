import torch
import torch.nn as nn
import math

from utils.feed_forward import FeedForwardNetwork
from utils.layer_normalization import LayerNormalization
from utils.residual_connection import ResidualCOnnection
from multi_head_attention_components.multihead_attention import MultiHeadAttentionNetwork
from utils.hyper_connection import HyperConnection

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionNetwork, cross_attention_block : MultiHeadAttentionNetwork, feed_forward_block : FeedForwardNetwork, dropout : float, d_model:int, hyper_n:int =4, use_hyper_connection:bool=False, layer_idx:int=0, device:str='cuda'):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_network = feed_forward_block
        self.use_hyper_connection = use_hyper_connection
        self.d_model = d_model
        self.hyper_n = hyper_n

        if use_hyper_connection:
            self.hc_self_attn = HyperConnection(d_model, hyper_n, layer_idx*3, dropout, dynamic=True, device=device)
            self.hc_cross_attn = HyperConnection(d_model, hyper_n, layer_idx*3+1, dropout, dynamic=True, device=device)
            self.hc_ffn = HyperConnection(d_model, hyper_n, layer_idx*3+2, dropout, dynamic=True, device=device)
        else:
            self.residual_connection = nn.ModuleList([ResidualCOnnection(dropout) for _ in range(3)])
            self.norm_1 = LayerNormalization()
            self.norm_2 = LayerNormalization()
            self.norm_3 = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask, h=None):
        if self.use_hyper_connection:
            if h is None:
                batch, seq, d = x.shape
                h = x.unsqueeze(2).expand(-1, -1, self.hyper_n, -1)

            h = self.hc_self_attn(h, lambda branch_in: self.self_attention_block(branch_in, branch_in, branch_in, target_mask))

            h = self.hc_cross_attn(h, lambda branch_in: self.cross_attention_block(encoder_output, branch_in, encoder_output, src_mask))

            h = self.hc_ffn(h, self.feed_forward_network)

            x = h[:, :, 0, :]
            return x, h
        else:
            x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
            x = self.residual_connection[1](x, lambda x: self.cross_attention_block(encoder_output, x, encoder_output, src_mask))
            x = self.residual_connection[2](x, self.feed_forward_network)
            return x, None


class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        h = None
        for layer in self.layers:
            x, h = layer(x, encoder_output, src_mask, target_mask, h)
        return self.norm(x)
