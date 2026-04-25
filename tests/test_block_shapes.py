"""Encoder/Decoder blocks produce correct output shapes for every connection type."""
import pytest
import torch

from src.components.attention.mha import MultiHeadAttention
from src.components.feedforward.vanilla import ReluFeedForward
from src.model.blocks import DecoderBlock, EncoderBlock
from src.registry import CONNECTION

B, S, D, H = 2, 7, 32, 4


def _build_conns(name: str, count: int):
    if name == "residual":
        return [CONNECTION.build(name, d_model=D, dropout=0.0) for _ in range(count)]
    return [
        CONNECTION.build(name, d_model=D, hyper_n=4, dropout=0.0, layer_idx=i)
        for i in range(count)
    ]


@pytest.mark.parametrize("name", CONNECTION.names())
def test_encoder_block(name: str) -> None:
    attn = MultiHeadAttention(d_model=D, n_heads=H, dropout=0.0)
    ffn = ReluFeedForward(d_model=D, d_ff=64, dropout=0.0)
    conns = _build_conns(name, 2)
    block = EncoderBlock(attn, ffn, conns)
    x = torch.randn(B, S, D)
    state = block.connections[0].init_state(x)
    state = block(state, src_mask=None)
    out = block.connections[0].to_output(state)
    assert out.shape == (B, S, D)


def test_encoder_block_residual_sandwich() -> None:
    attn = MultiHeadAttention(d_model=D, n_heads=H, dropout=0.0)
    ffn = ReluFeedForward(d_model=D, d_ff=64, dropout=0.0)
    conns = [
        CONNECTION.build("residual", d_model=D, dropout=0.0, norm="rmsnorm", post_norm=True)
        for _ in range(2)
    ]
    block = EncoderBlock(attn, ffn, conns)
    x = torch.randn(B, S, D)
    state = block.connections[0].init_state(x)
    state = block(state, src_mask=None)
    out = block.connections[0].to_output(state)
    assert out.shape == (B, S, D)
    assert all(c.post_norm_module is not None for c in block.connections)


@pytest.mark.parametrize("name", CONNECTION.names())
def test_decoder_block(name: str) -> None:
    self_attn = MultiHeadAttention(d_model=D, n_heads=H, dropout=0.0)
    cross_attn = MultiHeadAttention(d_model=D, n_heads=H, dropout=0.0)
    ffn = ReluFeedForward(d_model=D, d_ff=64, dropout=0.0)
    conns = _build_conns(name, 3)
    block = DecoderBlock(self_attn, cross_attn, ffn, conns)
    x = torch.randn(B, S, D)
    enc = torch.randn(B, S, D)
    state = block.connections[0].init_state(x)
    state = block(state, encoder_output=enc, src_mask=None, tgt_mask=None)
    out = block.connections[0].to_output(state)
    assert out.shape == (B, S, D)
