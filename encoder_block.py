import torch
import torch.nn as nn
import math
from feed_forward import FeedForwardNetwork
from layer_normalization import LayerNormalization
from residual_connection import ResidualCOnnection
from multihead_attention import MultiHeadAttentionNetwork

class EnocoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionNetwork, feed_forward_block : FeedForwardNetwork, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualCOnnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)