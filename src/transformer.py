from jaxtyping import Float, Bool

from torch import nn, Tensor
from src.rmsnorm import RMSNorm
from src.multihead import MultiHeadAttention
from src.SwiGLU import SwiGLU
from src.rope import Rope

import torch

class Transformer_Block(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float ) -> None:
        
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff) # 6BSDF+2BSF

        self.ln1 = RMSNorm(d_model) # 4BSD
        self.ln2 = RMSNorm(d_model) # 4BSD

        self.rope = Rope(theta=theta, d_k=self.attn.head_dim, max_seq_len=max_seq_len)

    def forward(self, 
                in_features: Float[Tensor, " batch sequence_length d_model"]
                )-> Float[Tensor, " batch sequence_length d_model"]: # Total = 2*B*S*D + FLOPs(attn, 4BSD + 6BSDF + 2BSF + 4BSD)

        token_positions = torch.arange(
                in_features.shape[1],
                device=in_features.device
            )
        
        y = in_features + self.attn(self.ln1(in_features), token_positions, self.rope) # addition [seq_len * d_model] flops
        y = y + self.ffn(self.ln2(y)) # addition [seq_len * d_model] flops 
        
        return y
    