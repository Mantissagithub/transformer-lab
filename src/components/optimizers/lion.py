import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer (https://arxiv.org/abs/2302.06675). Self-contained — no third-party dep."""

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = (exp_avg * beta1 + grad * (1 - beta1)).sign_()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
