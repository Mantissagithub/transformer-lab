"""capability matrix + encoder-decoder validation guards.

these tests pin the contract that self-attention-only / intrinsically-causal
attention variants must not silently train inside an encoder-decoder model —
the cross-attn path would throw away the encoder output. the trainer relies
on this guard to fail fast and steer the user toward causal_lm mode.
"""
import pytest
from omegaconf import OmegaConf

from src.model.builder import (
    ATTENTION_CAPABILITIES,
    build_transformer,
    default_model_kind,
    supports_encoder_decoder,
)
from src.registry import ATTENTION


def test_capability_matrix_covers_every_registered_variant():
    registered = set(ATTENTION.names())
    documented = set(ATTENTION_CAPABILITIES)
    assert registered <= documented, (
        f"missing capability entries for {registered - documented}: "
        "every registered attention must declare cross_attn / bidirectional."
    )


@pytest.mark.parametrize("name", ["mha", "gqa", "gqa_rope", "mqa", "sliding_window", "sliding_gqa"])
def test_encoder_decoder_capable_variants(name):
    cfg = OmegaConf.create({"name": name})
    assert supports_encoder_decoder(cfg)
    assert default_model_kind(cfg) == "encoder_decoder"


@pytest.mark.parametrize("name", ["csa", "hca", "mla"])
def test_self_attention_only_variants(name):
    cfg = OmegaConf.create({"name": name})
    assert not supports_encoder_decoder(cfg)
    assert default_model_kind(cfg) == "causal_lm"


def test_hybrid_pattern_inherits_member_capabilities():
    # gemma3_hybrid composes sliding_gqa + gqa_rope, both encoder-decoder ok.
    cfg = OmegaConf.create({
        "name": "gemma3_hybrid",
        "pattern": ["sliding_gqa", "gqa_rope"],
        "layers": {
            "sliding_gqa": {"n_heads": 4, "n_kv_heads": 2, "window_size": 4, "bias": False},
            "gqa_rope":    {"n_heads": 4, "n_kv_heads": 2, "bias": False},
        },
    })
    assert supports_encoder_decoder(cfg)


def test_hybrid_with_self_attn_only_member_is_rejected():
    cfg = OmegaConf.create({
        "name": "frankenstein",
        "pattern": ["mha", "csa"],
        "layers": {"mha": {"n_heads": 4}, "csa": {"n_heads": 4}},
    })
    assert not supports_encoder_decoder(cfg)


def _ed_cfg(attn) -> OmegaConf:
    return OmegaConf.create({
        "model": {
            "d_model": 32, "d_ff": 64, "n_layers": 2, "dropout": 0.0,
            "max_src_len": 8, "max_tgt_len": 8,
            "src_vocab_size": 50, "tgt_vocab_size": 60,
        },
        "attention": attn,
        "feedforward": {"name": "relu_ffn", "d_ff": 64},
        "normalization": {"name": "layernorm"},
        "positional": {"name": "sinusoidal", "dropout": 0.0},
        "connection": {"name": "residual", "dropout": 0.0, "norm": "layernorm"},
    })


@pytest.mark.parametrize("name,extra", [
    ("csa", {"n_I_h": 2, "m": 2, "k": 4, "n_win": 4, "g": 2,
             "c": 8, "c_I": 4, "d_c": 16, "d_g": 8, "rope_dim": 4, "max_seq_len": 16}),
    ("hca", {"m": 2, "g": 2, "c": 8, "d_c": 16, "d_g": 8}),
    ("mla", {"kv_lora_rank": 8, "q_lora_rank": 16, "qk_nope_head_dim": 4,
             "qk_rope_head_dim": 4, "v_head_dim": 8, "max_seq_len": 16}),
])
def test_build_transformer_rejects_self_attn_only(name, extra):
    cfg = _ed_cfg({"name": name, "n_heads": 4, **extra})
    with pytest.raises(ValueError, match="encoder-decoder"):
        build_transformer(cfg)
