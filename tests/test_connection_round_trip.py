"""Connection.init_state -> apply -> to_output preserves (b, s, d) shape."""
import pytest
import torch

from src.registry import CONNECTION

B, S, D = 2, 7, 16


def _identity_sublayer(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


@pytest.mark.parametrize("name", CONNECTION.names())
def test_round_trip(name: str) -> None:
    if name == "residual":
        conn = CONNECTION.build(name, d_model=D, dropout=0.0)
    else:
        conn = CONNECTION.build(name, d_model=D, hyper_n=4, dropout=0.0)
    x = torch.randn(B, S, D)
    state = conn.init_state(x)
    state = conn.apply(state, _identity_sublayer)
    out = conn.to_output(state)
    assert out.shape == (B, S, D)
