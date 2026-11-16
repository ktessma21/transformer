from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
            raise ValueError(f"Invalid betas: {betas}")

        if weight_decay < 0:
            raise ValueError(f"Invalid lambda: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        self.t = 0

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        self.t += 1

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, dtype=torch.float32)
                        state["v"] = torch.zeros_like(p, dtype=torch.float32)


                    grad = p.grad

                    alpha = lr * (
                        math.sqrt(1 - betas[1] ** self.t)
                        / (1 - betas[0] ** self.t)
                    )

                    p.mul_(1 - lr * weight_decay)

                    state["m"] = betas[0] * state["m"] + (1 - betas[0]) * grad
                    state["v"] = betas[1] * state["v"] + (1 - betas[1]) * (grad ** 2)

                    p -= alpha * (
                        state["m"] / (torch.sqrt(state["v"]) + eps)
                    )
