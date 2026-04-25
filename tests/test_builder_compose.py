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
         {"name": "gqa", "n_heads": H, "n_kv_heads": 2, "bias": True},
         {"name": "gqa_rope", "n_heads": H, "n_kv_heads": 2, "bias": False},
         {"name": "sliding_gqa", "n_heads": H, "n_kv_heads": 2, "window_size": 4, "bias": False}],
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


def test_compose_gemma3_hybrid() -> None:
    n_layers = 6
    vocab = 50
    cfg = OmegaConf.create({
        "model": {
            "d_model": D, "d_ff": 64, "n_layers": n_layers, "dropout": 0.0,
            "max_src_len": S, "max_tgt_len": S,
            "src_vocab_size": vocab, "tgt_vocab_size": vocab,
            "tie_embeddings": True,
        },
        "attention": {
            "name": "gemma3_hybrid",
            "n_heads": H,
            "pattern": ["sliding_gqa"] * 5 + ["gqa_rope"],
            "layers": {
                "sliding_gqa": {"n_heads": H, "n_kv_heads": 2, "window_size": 4, "bias": False},
                "gqa_rope": {"n_heads": H, "n_kv_heads": 2, "bias": False},
            },
        },
        "feedforward": {"name": "swiglu", "d_ff": 64},
        "normalization": {"name": "rmsnorm"},
        "positional": {"name": "rope", "base": 10000.0, "dropout": 0.0, "skip_every_n": 4},
        "connection": {"name": "residual", "dropout": 0.0, "norm": "rmsnorm", "post_norm": True},
    })
    model = build_transformer(cfg)

    expected = ["SlidingGroupedQueryAttention"] * 5 + ["GroupedQueryAttentionRoPE"]
    actual = [type(blk.attn).__name__ for blk in model.encoder.layers]
    assert actual == expected

    assert model.src_embed is model.tgt_embed
    assert model.projection.embedding is model.src_embed

    rope_present = [model.encoder.layers[i].attn.rope is not None for i in range(n_layers)]
    assert rope_present == [True, True, True, False, True, True]

    src = torch.randint(0, vocab, (1, S))
    tgt = torch.randint(0, vocab, (1, S))
    src_mask = torch.ones(1, 1, 1, S, dtype=torch.int)
    tgt_mask = torch.ones(1, 1, S, S, dtype=torch.int)
    enc = model.encode(src, src_mask)
    dec = model.decode(enc, src_mask, tgt, tgt_mask)
    logits = model.project(dec)
    assert logits.shape == (1, S, vocab)


def test_tied_embeddings_param_count() -> None:
    base = _cfg()
    base.model.src_vocab_size = 50
    base.model.tgt_vocab_size = 50
    OmegaConf.set_struct(base, False)

    untied = OmegaConf.create(OmegaConf.to_container(base))
    untied.model.tie_embeddings = False
    tied = OmegaConf.create(OmegaConf.to_container(base))
    tied.model.tie_embeddings = True

    n_untied = sum(p.numel() for p in build_transformer(untied).parameters())
    n_tied = sum(p.numel() for p in build_transformer(tied).parameters())
    # Tying merges src/tgt embeddings (saves V*D) and replaces Linear(D, V) projection
    # (V*D weight + V bias) with a weight-sharing reference, saving 2*V*D + V.
    assert n_untied - n_tied == 2 * 50 * D + 50
