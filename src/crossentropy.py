from jaxtyping import Float, Int
from torch import Tensor
import torch

def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    maximum = torch.max(inputs, dim=-1).values
    z_y = inputs[torch.arange(targets.shape[0], device=inputs.device), targets]


    exp = torch.exp(inputs - maximum[:, None])
    summed = torch.sum(exp, dim=-1)
    residual = -z_y + maximum + torch.log(summed)

    return torch.mean(residual)