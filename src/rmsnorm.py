import torch 
from torch import nn, Tensor 

from jaxtyping import Float


class RMSNorm(nn.Module): # 4 BSD
        def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
            super().__init__()

            self.W = nn.Parameter(
                torch.ones(d_model, device=device, dtype=dtype)
            ) # all 1. 

            self.d_model = d_model 
            self.eps = eps 


        def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> torch.Tensor:
            in_dtype = x.dtype
            x = x.to(torch.float32)
            x_squared = x ** 2 # BSD
            x_mean = torch.mean(x_squared, dim=-1, keepdim=True) + self.eps # BSD + BS 
            x_sqrt = torch.sqrt(x_mean) # BS
            
            return ((x/x_sqrt) * self.W).to(in_dtype) # 2BSD