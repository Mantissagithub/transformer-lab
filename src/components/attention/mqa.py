from src.registry import ATTENTION

from .gqa import GroupedQueryAttention


@ATTENTION.register("mqa")
class MultiQueryAttention(GroupedQueryAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__(d_model=d_model, n_heads=n_heads, n_kv_heads=1, dropout=dropout, bias=bias)
