import torch

from src.components.attention.mla import MultiHeadLatentAttention
from src.components.attention.mla_absorbed import MultiHeadLatentAttentionAbsorbed


def _config():
    return dict(
        d_model=32,
        n_heads=2,
        kv_lora_rank=12,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=6,
        dropout=0.0,
        max_seq_len=16,
    )


def test_absorbed_matches_standard():
    torch.manual_seed(0)
    mod_config = _config()
    standard = MultiHeadLatentAttention(**mod_config).eval()
    absorbed = MultiHeadLatentAttentionAbsorbed(**mod_config).eval()
    absorbed.load_state_dict(standard.state_dict(), strict=True)

    x = torch.randn(2, 5, mod_config["d_model"])
    mask = torch.ones(5, 5, dtype=torch.bool).tril()

    with torch.no_grad():
        expected = standard(x, x, x, mask=mask)
        actual = absorbed(x, x, x, mask=mask)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_absorbed_backward():
    torch.manual_seed(0)
    mod = MultiHeadLatentAttentionAbsorbed(**_config())
    x = torch.randn(2, 5, mod.d_model, requires_grad=True)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()

    out = mod(x, x, x, mask=mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()

    for tensor in [x, *mod.parameters()]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


@torch.no_grad()
def test_absorbed_causal_mask():
    torch.manual_seed(0)
    mod = MultiHeadLatentAttentionAbsorbed(**_config()).eval()
    x = torch.randn(2, 5, mod.d_model)
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])
    mask = torch.ones(5, 5, dtype=torch.bool).tril()

    expected = mod(x, x, x, mask=mask)
    actual = mod(changed, changed, changed, mask=mask)

    torch.testing.assert_close(actual[:, :3], expected[:, :3])


@torch.no_grad()
def test_absorbed_cached_decode_matches_full_forward():
    torch.manual_seed(0)
    mod = MultiHeadLatentAttentionAbsorbed(**_config()).eval()
    x = torch.randn(2, 5, mod.d_model)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    expected = mod(x, x, x, mask=mask)

    cache = mod.init_cache()
    prefix = x[:, :3]
    outputs = [mod(prefix, prefix, prefix, mask=mask[:3, :3], past_kv=cache)]
    for i in range(3, 5):
        token = x[:, i:i + 1]
        outputs.append(mod(token, token, token, past_kv=cache))

    assert cache.position() == x.shape[1]
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=1e-4, atol=1e-5)
