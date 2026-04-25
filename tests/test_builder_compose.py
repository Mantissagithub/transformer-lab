"""Builder composes a working transformer for arbitrary swaps without code edits."""
import itertools

import pytest
import torch
from omegaconf import OmegaConf

from src.model.builder import build_transformer

D, H, NL, S = 32, 4, 2, 6
SRC_VOCAB, TGT_VOCAB = 50, 60


def _cfg(**overrides) -> OmegaConf:
    cfg = OmegaConf.create({
        "model": {
            "d_model": D, "d_ff": 64, "n_layers": NL, "dropout": 0.0,
            "max_src_len": S, "max_tgt_len": S,
            "src_vocab_size": SRC_VOCAB, "tgt_vocab_size": TGT_VOCAB,
        },
        "attention": {"name": "mha", "n_heads": H, "bias": True},
        "feedforward": {"name": "relu_ffn", "d_ff": 64},
        "normalization": {"name": "layernorm"},
        "positional": {"name": "sinusoidal", "dropout": 0.0},
        "connection": {"name": "residual", "dropout": 0.0, "norm": "layernorm"},
    })
    for k, v in overrides.items():
        OmegaConf.set_struct(cfg, False)
        cfg[k] = OmegaConf.create(v)
    return cfg


CASES = list(
    itertools.product(
        [{"name": "mha", "n_heads": H, "bias": True},
         {"name": "gqa", "n_heads": H, "n_kv_heads": 2, "bias": True}],
        [{"name": "relu_ffn", "d_ff": 64},
         {"name": "swiglu", "d_ff": 64}],
        [{"name": "layernorm"}, {"name": "rmsnorm"}],
        [{"name": "residual", "dropout": 0.0, "norm": "layernorm"},
         {"name": "hyperconnection", "hyper_n": 4, "dropout": 0.0, "dynamic": True},
         {"name": "mhc", "hyper_n": 4, "dropout": 0.0, "dynamic": True}],
    )
)


@pytest.mark.parametrize("attn,ffn,norm,conn", CASES)
def test_compose_forward(attn, ffn, norm, conn) -> None:
    cfg = _cfg(attention=attn, feedforward=ffn, normalization=norm, connection=conn)
    model = build_transformer(cfg)
    src = torch.randint(0, SRC_VOCAB, (1, S))
    tgt = torch.randint(0, TGT_VOCAB, (1, S))
    src_mask = torch.ones(1, 1, 1, S, dtype=torch.int)
    tgt_mask = torch.ones(1, 1, S, S, dtype=torch.int)
    enc = model.encode(src, src_mask)
    dec = model.decode(enc, src_mask, tgt, tgt_mask)
    logits = model.project(dec)
    assert logits.shape == (1, S, TGT_VOCAB)
