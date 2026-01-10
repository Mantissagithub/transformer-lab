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
        self.residual_connection = nn.ModuleList([ResidualCOnnection(dropout) for _ in range(3)])
        self.use_hyper_connection = use_hyper_connection

        if use_hyper_connection:
            self.connection_1 = HyperConnection(d_model, hyper_n, layer_idx*3, dropout, device)
            self.connection_2 = HyperConnection(d_model, hyper_n, layer_idx*3+1, dropout, device)
            self.connection_3 = HyperConnection(d_model, hyper_n, layer_idx*3+2, dropout, device)
        else:
            self.connection_1 = self.residual_connection[0]
            self.connection_2 = self.residual_connection[1]
            self.connection_3 = self.residual_connection[2]

    def forward(self, x, encoder_output, src_mask, target_mask, hyper_hiddens=None):
        if self.use_hyper_connection:
            if hyper_hiddens is None:
                hyper_hiddens = []

            x = self.connection_1(x, lambda x: self.self_attention_block(x, x, x, target_mask), hyper_hiddens)
            hyper_hiddens.append(x.detach())

            x = self.connection_2(x, lambda x: self.cross_attention_block(encoder_output, x, encoder_output, src_mask), hyper_hiddens)
            hyper_hiddens.append(x.detach())

            x = self.connection_3(x, self.feed_forward_network, hyper_hiddens)
            hyper_hiddens.append(x.detach())

        else:
            x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
            x = self.residual_connection[1](x, lambda x: self.cross_attention_block(encoder_output, x, encoder_output, src_mask))
            x = self.residual_connection[2](x, self.feed_forward_network)

        return x, hyper_hiddens

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        hyper_hiddens = []

        for layer in self.layers:
            x, hyper_hiddens = layer(x, encoder_output, src_mask, target_mask, hyper_hiddens)
        return self.norm(x)
