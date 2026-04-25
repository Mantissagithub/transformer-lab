from typing import List, Optional

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


_ROPE_AWARE = {"gqa_rope", "sliding_gqa"}


def _build_attention_for_layer(
    cfg,
    layer_idx: int,
    d_model: int,
    dropout: float,
    rope: Optional[nn.Module],
):
    if "pattern" in cfg:
        pattern = list(cfg.pattern)
        name = pattern[layer_idx % len(pattern)]
        layer_cfg = cfg.layers[name]
        kwargs = (
            OmegaConf.to_container(layer_cfg, resolve=True)
            if isinstance(layer_cfg, DictConfig)
            else dict(layer_cfg)
        )
    else:
        kwargs = _strip_name(cfg)
        name = cfg.name
    kwargs.setdefault("d_model", d_model)
    kwargs.setdefault("dropout", dropout)
    if name in _ROPE_AWARE:
        kwargs["rope"] = rope
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
    kwargs.pop("skip_every_n", None)
    if name == "rope":
        kwargs["d_model"] = d_model // n_heads
    else:
        kwargs.setdefault("d_model", d_model)
    kwargs.setdefault("max_len", max_len)
    if name == "alibi":
        kwargs.setdefault("n_heads", n_heads)
    return POS.build(name, **kwargs)


def _build_norm(cfg, d_model: int):
    name = cfg.name if isinstance(cfg, (DictConfig, dict)) else cfg
    return NORM.build(name, d_model=d_model)


def _rope_for_layer(positional_cfg, base_rope: Optional[nn.Module], layer_idx: int) -> Optional[nn.Module]:
    if base_rope is None or not getattr(base_rope, "is_rotary", False):
        return None
    skip = positional_cfg.get("skip_every_n", None)
    if skip and (layer_idx + 1) % skip == 0:
        return None
    return base_rope


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
    tie_embeddings = model_cfg.get("tie_embeddings", False)

    if tie_embeddings:
        if src_vocab != tgt_vocab:
            raise ValueError(
                f"tie_embeddings requires src_vocab_size == tgt_vocab_size, got {src_vocab} != {tgt_vocab}"
            )
        shared_embed = EMBEDDING.build("learned", vocab_size=src_vocab, d_model=d_model)
        src_embed = shared_embed
        tgt_embed = shared_embed
    else:
        src_embed = EMBEDDING.build("learned", vocab_size=src_vocab, d_model=d_model)
        tgt_embed = EMBEDDING.build("learned", vocab_size=tgt_vocab, d_model=d_model)

    src_pos = _build_pos(cfg.positional, d_model, max_src_len, n_heads)
    tgt_pos = _build_pos(cfg.positional, d_model, max_tgt_len, n_heads)

    encoder_blocks = []
    for i in range(n_layers):
        rope = _rope_for_layer(cfg.positional, src_pos, i)
        attn = _build_attention_for_layer(cfg.attention, i, d_model, dropout, rope)
        ffn = _build_ffn(cfg.feedforward, d_model, dropout)
        conns = _build_connections(cfg.connection, d_model, count=2, layer_idx=i)
        encoder_blocks.append(EncoderBlock(attn, ffn, conns))

    decoder_blocks = []
    for i in range(n_layers):
        rope = _rope_for_layer(cfg.positional, tgt_pos, i)
        self_attn = _build_attention_for_layer(cfg.attention, i, d_model, dropout, rope)
        cross_attn = _build_attention_for_layer(cfg.attention, i, d_model, dropout, None)
        ffn = _build_ffn(cfg.feedforward, d_model, dropout)
        conns = _build_connections(cfg.connection, d_model, count=3, layer_idx=i)
        decoder_blocks.append(DecoderBlock(self_attn, cross_attn, ffn, conns))

    encoder = Encoder(nn.ModuleList(encoder_blocks), _build_norm(cfg.normalization, d_model))
    decoder = Decoder(nn.ModuleList(decoder_blocks), _build_norm(cfg.normalization, d_model))

    if tie_embeddings:
        projection = PROJECTION.build("tied", embedding=shared_embed)
    else:
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
