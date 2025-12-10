import torch
import torch.nn as nn
import math
from multi_head_attention_components.encoder_block import Encoder, EncoderBlock
from multi_head_attention_components.decoder_block import Decoder, DecoderBlock
from utils.input_embedding import InputEmbeddings
from utils.positional_encoding import PositionalEncoding
from utils.projection_layer import ProjectionLayer
from multi_head_attention_components.multihead_attention import MultiHeadAttentionNetwork
from utils.feed_forward import FeedForwardNetwork

class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, src_embed : InputEmbeddings, target_embed : InputEmbeddings, src_pos : PositionalEncoding, target_pos : PositionalEncoding, projection_layer : ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_embed
        self.target_emb = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_emb(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size : int, target_vocab_size : int, src_seq_len : int, target_seq_len : int, d_model_size : int = 512, d_ff : int = 2048, h : int = 8, dropout : float = 0.1, n : int = 6) -> Transformer:
    src_embed = InputEmbeddings(d_model_size, src_vocab_size)
    target_embed = InputEmbeddings(d_model_size, target_vocab_size)

    src_pos = PositionalEncoding(d_model_size, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model_size, target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(n):
        encoder_self_attention_block = MultiHeadAttentionNetwork(d_model_size, h, dropout)
        encoder_ff = FeedForwardNetwork(d_model_size, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_ff, dropout)
        encoder_blocks.append(encoder_block)


    decoder_blocks = []
    for _ in range(n):
        decoder_self_attention_block = MultiHeadAttentionNetwork(d_model_size, h, dropout)
        decoder_cross_attetntion_block = MultiHeadAttentionNetwork(d_model_size, h, dropout)
        decoder_ff = FeedForwardNetwork(d_model_size, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attetntion_block, decoder_ff, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model_size, target_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer