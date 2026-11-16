

from typing import Any

import torch 

from torch import nn



class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()

        denominator = theta ** (torch.arange(0, d_k, 2) / d_k)
        numerator = torch.arange(0, max_seq_len) 
        angles = numerator[:, None]/denominator

        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    # angles = token_positions[..., None] * self.freq

        cos = getattr(self, "cos")[token_positions]
        sin = getattr(self, "sin")[token_positions]
        
        x_even = x[..., 0::2]  # (..., seq_len, d_k // 2)
        x_odd  = x[..., 1::2]  # (..., seq_len, d_k // 2)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos


        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        return torch.flatten(rotated, start_dim=-2)
 