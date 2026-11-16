from typing import Iterable
import torch
import math

from typing import Iterable
import torch
import math

eps = 1e-6

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return
    
    total_sq = torch.zeros((), device=grads[0].device)

    for g in grads:
        total_sq += torch.sum(g ** 2)

    total_norm = torch.sqrt(total_sq)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)

        for grad in grads:
            grad.mul_(scale)