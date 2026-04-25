import torch

from src.components.attention.gqa_rope import GroupedQueryAttentionRoPE
from src.components.positional.rope import RotaryPositionalEncoding


D, H, KV, S = 32, 4, 2, 8
DH = D // H


def _swap_first_two(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, [0, 1], :] = out[:, [1, 0], :]
    return out


def test_rope_breaks_position_equivariance() -> None:
    torch.manual_seed(0)
    rope = RotaryPositionalEncoding(d_model=DH, max_len=S)
    attn = GroupedQueryAttentionRoPE(d_model=D, n_heads=H, n_kv_heads=KV, rope=rope, bias=False)
    attn.eval()

    x = torch.randn(1, S, D)
    out = attn(x, x, x)
    out_swapped = attn(_swap_first_two(x), _swap_first_two(x), _swap_first_two(x))

    out_swapped_back = _swap_first_two(out_swapped)
    assert not torch.allclose(out, out_swapped_back, atol=1e-5), (
        "Outputs collapse to a permutation — RoPE is not actually being applied."
    )


def test_nope_is_position_equivariant() -> None:
    torch.manual_seed(0)
    attn = GroupedQueryAttentionRoPE(d_model=D, n_heads=H, n_kv_heads=KV, rope=None, bias=False)
    attn.eval()

    x = torch.randn(1, S, D)
    out = attn(x, x, x)
    out_swapped = attn(_swap_first_two(x), _swap_first_two(x), _swap_first_two(x))

    out_swapped_back = _swap_first_two(out_swapped)
    assert torch.allclose(out, out_swapped_back, atol=1e-5)
