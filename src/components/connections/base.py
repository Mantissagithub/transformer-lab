from typing import Callable

import torch
import torch.nn as nn


class Connection(nn.Module):
    """Connection contract: state-shape is implementation-defined; blocks stay shape-agnostic.

    is_stateful=False  -> state is (b, s, d)            (residual)
    is_stateful=True   -> state is (b, s, n, d)         (HC, MHC)
    """

    is_stateful: bool = False

    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_output(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, state: torch.Tensor, sublayer_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
