from jaxtyping import Float, Bool
from torch import Tensor
import torch

from src.softmax import softmax

import math


# currently the number of heads are the same so queries and keys are going to be similar. but we can maybe have them different. 
def attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    sqrt_d_k = math.sqrt(Q.shape[-1])

    # even though it is transpose i guess we don't 
    mul = Q @ K.transpose(-2, -1) # 2 BS(D_k^2) * 2

    if mask is not None:
        mul = torch.where(mask, mul, float("-inf"))

    attent = softmax(mul/sqrt_d_k, -1) @ V


    return attent