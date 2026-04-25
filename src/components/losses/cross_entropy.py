import torch.nn as nn

from src.registry import LOSS


@LOSS.register("cross_entropy")
def build_cross_entropy(label_smoothing: float = 0.0, ignore_index: int = -100, **_) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
