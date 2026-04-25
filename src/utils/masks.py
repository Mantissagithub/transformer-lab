import torch


def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
