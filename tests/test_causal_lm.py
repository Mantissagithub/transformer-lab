"""Causal-LM path: shape, backward, mask correctness, KV-cache parity, and the
real 500M config's parameter count."""
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

from src.model.builder import build_causal_lm
from src.model.causal_lm import CausalLM


CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def _small_cfg(**overrides) -> OmegaConf:
    base = {
        "model": {
            "kind": "causal_lm",
            "d_model": 32,
            "d_ff": 64,
            "n_layers": 2,
            "dropout": 0.0,
            "max_seq_len": 16,
            "src_vocab_size": 64,
            "tgt_vocab_size": 64,
            "tie_embeddings": True,
        },
        "attention": {"name": "gqa_rope", "n_heads": 4, "n_kv_heads": 2, "bias": False},
        "feedforward": {"name": "swiglu", "d_ff": 64},
        "normalization": {"name": "rmsnorm"},
        "positional": {"name": "rope", "base": 10000.0, "dropout": 0.0},
        "connection": {"name": "residual", "dropout": 0.0, "norm": "rmsnorm"},
    }
    cfg = OmegaConf.create(base)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=True)
    return cfg


def test_forward_shape_and_backward():
    cfg = _small_cfg()
    model = build_causal_lm(cfg)
    assert isinstance(model, CausalLM)
    ids = torch.randint(0, 64, (2, 8))
    out = model(ids)
    assert out.shape == (2, 8, 64)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_causal_mask_blocks_future():
    cfg = _small_cfg()
    model = build_causal_lm(cfg).eval()
    ids_a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    ids_b = torch.tensor([[1, 2, 3, 4, 9, 9]])
    with torch.no_grad():
        out_a = model(ids_a)
        out_b = model(ids_b)
    # Positions <= 3 must match; later positions should differ.
    assert torch.allclose(out_a[:, :4, :], out_b[:, :4, :], atol=1e-6)
    assert not torch.allclose(out_a[:, 4:, :], out_b[:, 4:, :], atol=1e-6)


def test_kv_cache_parity():
    cfg = _small_cfg()
    model = build_causal_lm(cfg).eval()
    prompt = torch.randint(0, 64, (1, 5))

    with torch.no_grad():
        full = model(prompt)
        prefill_logits, past = model(prompt, past_kvs=None, return_kvs=True)
        # Greedy continuation, one token at a time, vs. recompute-from-scratch.
        next_tok = full[:, -1, :].argmax(dim=-1, keepdim=True)
        cached_logits, past = model(next_tok, past_kvs=past, return_kvs=True)
        recomputed = model(torch.cat([prompt, next_tok], dim=-1))

    assert torch.allclose(prefill_logits, full, atol=1e-5)
    assert torch.allclose(cached_logits[:, -1, :], recomputed[:, -1, :], atol=1e-4)


def _parity_check(cfg, prompt_len: int = 5, decode_steps: int = 3, atol: float = 1e-4) -> None:
    """Prefill on a random prompt, decode one token at a time with the cache,
    and at each step assert the cached last-token logits match a from-scratch
    recompute over (prompt + tokens-so-far)."""
    torch.manual_seed(0)
    model = build_causal_lm(cfg).eval()
    prompt = torch.randint(0, cfg.model.src_vocab_size, (1, prompt_len))

    with torch.no_grad():
        full = model(prompt)
        prefill_logits, past = model(prompt, past_kvs=None, return_kvs=True)
        assert torch.allclose(prefill_logits, full, atol=1e-5)
        ids = prompt
        for _ in range(decode_steps):
            next_tok = full[:, -1, :].argmax(dim=-1, keepdim=True)
            cached_logits, past = model(next_tok, past_kvs=past, return_kvs=True)
            ids = torch.cat([ids, next_tok], dim=-1)
            recomputed = model(ids)
            assert torch.allclose(
                cached_logits[:, -1, :], recomputed[:, -1, :], atol=atol
            ), f"divergence at len={ids.shape[1]}, max diff {(cached_logits[:, -1, :] - recomputed[:, -1, :]).abs().max().item():.3e}"
            full = recomputed


def test_kv_cache_parity_sliding_gqa():
    cfg = _small_cfg()
    cfg.attention = OmegaConf.create({
        "name": "sliding_gqa", "n_heads": 4, "n_kv_heads": 2,
        "window_size": 4, "bias": False,
    })
    _parity_check(cfg)


def test_kv_cache_parity_sliding_window():
    # sliding_window isn't rope-aware; the rope positional in _small_cfg falls
    # through to a no-op forward, so the test exercises the cache mechanism
    # with position-blind embeddings (decode and recompute must still match).
    cfg = _small_cfg()
    cfg.attention = OmegaConf.create({
        "name": "sliding_window", "n_heads": 4, "window_size": 4, "bias": True,
    })
    _parity_check(cfg)


def test_kv_cache_parity_csa():
    # csa carries its own internal rope tables; the model-level rope positional
    # is a no-op forward on the embedding for non-rope-aware attention.
    cfg = _small_cfg()
    cfg.attention = OmegaConf.create({
        "name": "csa",
        "n_heads": 4, "n_I_h": 2, "m": 2, "k": 4, "n_win": 4, "g": 2,
        "c": 8, "c_I": 4, "d_c": 16, "d_g": 8,
        "rope_dim": 4, "max_seq_len": 32,
    })
    _parity_check(cfg)


def test_kv_cache_parity_hca():
    # hca has no rope/sink/rmsnorm — strict §2.3.2. m=2 so a 5-token prompt
    # produces 2 compressed blocks, exercising both the empty-kv early-token
    # path and the post-block-completion path during decode.
    cfg = _small_cfg()
    cfg.attention = OmegaConf.create({
        "name": "hca",
        "n_heads": 4, "m": 2, "g": 2,
        "c": 8, "d_c": 16, "d_g": 8,
    })
    _parity_check(cfg)


def test_kv_cache_parity_mla():
    # mla carries its own internal half-split rope tables; the model-level
    # rope positional is a no-op forward on the embedding for non-rope-aware
    # attention. Caching pre-norm c_kv (and applying kv_norm after concat)
    # is what keeps prefill+decode == from-scratch under live parameters.
    cfg = _small_cfg()
    cfg.attention = OmegaConf.create({
        "name": "mla",
        "n_heads": 4,
        "kv_lora_rank": 8, "q_lora_rank": 16,
        "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "bias": False, "rope_base": 10000.0, "max_seq_len": 16,
    })
    _parity_check(cfg)


def test_generate_with_eos_and_no_temperature():
    cfg = _small_cfg()
    model = build_causal_lm(cfg).eval()
    prompt = torch.tensor([[1, 2, 3]])
    out = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert out.shape[0] == 1
    assert out.shape[1] >= prompt.shape[1] + 1
    assert torch.equal(out[:, : prompt.shape[1]], prompt)


def test_real_500m_param_count():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["+experiment=pretrain_500m"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.src_vocab_size = cfg.data.vocab_size
    cfg.model.tgt_vocab_size = cfg.data.vocab_size
    OmegaConf.set_struct(cfg, True)
    model = build_causal_lm(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 5e8 <= n <= 6e8, f"expected ~500M params, got {n / 1e6:.2f}M"
