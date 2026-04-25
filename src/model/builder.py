from typing import List

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.registry import (
    ATTENTION,
    CONNECTION,
    EMBEDDING,
    FFN,
    NORM,
    POS,
    PROJECTION,
)

from .blocks import DecoderBlock, EncoderBlock
from .decoder import Decoder
from .encoder import Encoder
from .transformer import Transformer


def _strip_name(cfg) -> dict:
    out = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    out.pop("name", None)
    return out


def _build_attention(cfg, d_model: int, dropout: float):
    kwargs = _strip_name(cfg)
    name = cfg.name
    kwargs.setdefault("d_model", d_model)
    kwargs.setdefault("dropout", dropout)
    return ATTENTION.build(name, **kwargs)


def _build_ffn(cfg, d_model: int, dropout: float):
    kwargs = _strip_name(cfg)
    name = cfg.name
    kwargs.setdefault("d_model", d_model)
    kwargs.setdefault("dropout", dropout)
    return FFN.build(name, **kwargs)


def _build_connections(cfg, d_model: int, count: int, layer_idx: int) -> List[nn.Module]:
    kwargs = _strip_name(cfg)
    name = cfg.name
    out = []
    for i in range(count):
        local_kwargs = dict(kwargs)
        local_kwargs.setdefault("d_model", d_model)
        if name in ("hyperconnection", "mhc"):
            local_kwargs["layer_idx"] = layer_idx * count + i
        out.append(CONNECTION.build(name, **local_kwargs))
    return out


def _build_pos(cfg, d_model: int, max_len: int, n_heads: int):
    kwargs = _strip_name(cfg)
    name = cfg.name
    kwargs.setdefault("d_model", d_model)
    kwargs.setdefault("max_len", max_len)
    if name == "alibi":
        kwargs.setdefault("n_heads", n_heads)
    return POS.build(name, **kwargs)


def _build_norm(cfg, d_model: int):
    name = cfg.name if isinstance(cfg, (DictConfig, dict)) else cfg
    return NORM.build(name, d_model=d_model)


def build_transformer(cfg: DictConfig) -> Transformer:
    model_cfg = cfg.model
    d_model = model_cfg.d_model
    dropout = model_cfg.dropout
    n_layers = model_cfg.n_layers
    n_heads = cfg.attention.get("n_heads", 8)
    src_vocab = model_cfg.src_vocab_size
    tgt_vocab = model_cfg.tgt_vocab_size
    max_src_len = model_cfg.max_src_len
    max_tgt_len = model_cfg.max_tgt_len

    src_embed = EMBEDDING.build("learned", vocab_size=src_vocab, d_model=d_model)
    tgt_embed = EMBEDDING.build("learned", vocab_size=tgt_vocab, d_model=d_model)
    src_pos = _build_pos(cfg.positional, d_model, max_src_len, n_heads)
    tgt_pos = _build_pos(cfg.positional, d_model, max_tgt_len, n_heads)

    encoder_blocks = []
    for i in range(n_layers):
        attn = _build_attention(cfg.attention, d_model, dropout)
        ffn = _build_ffn(cfg.feedforward, d_model, dropout)
        conns = _build_connections(cfg.connection, d_model, count=2, layer_idx=i)
        encoder_blocks.append(EncoderBlock(attn, ffn, conns))

    decoder_blocks = []
    for i in range(n_layers):
        self_attn = _build_attention(cfg.attention, d_model, dropout)
        cross_attn = _build_attention(cfg.attention, d_model, dropout)
        ffn = _build_ffn(cfg.feedforward, d_model, dropout)
        conns = _build_connections(cfg.connection, d_model, count=3, layer_idx=i)
        decoder_blocks.append(DecoderBlock(self_attn, cross_attn, ffn, conns))

    encoder = Encoder(nn.ModuleList(encoder_blocks), _build_norm(cfg.normalization, d_model))
    decoder = Decoder(nn.ModuleList(decoder_blocks), _build_norm(cfg.normalization, d_model))
    projection = PROJECTION.build("linear", d_model=d_model, vocab_size=tgt_vocab)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)

    _init_parameters(model)
    return model


_HC_PARAM_TOKENS = ("static_alpha", "static_beta", "dynamic_alpha", "dynamic_beta")


def _init_parameters(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if p.dim() <= 1:
            continue
        if any(tok in name for tok in _HC_PARAM_TOKENS):
            continue
        nn.init.xavier_uniform_(p)
