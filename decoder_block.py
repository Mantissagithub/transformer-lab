import torch 
import torch.nn as nn
import math
from feed_forward import FeedForwardNetwork
from layer_normalization import LayerNormalization
from residual_connection import ResidualCOnnection
from multihead_attention import MultiHeadAttentionNetwork

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionNetwork, cross_attention_block : MultiHeadAttentionNetwork, feed_forward_block : FeedForwardNetwork, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_network = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualCOnnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_network)

        return x
    
class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    