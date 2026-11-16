from jaxtyping import Float, Bool
from torch import nn, Tensor
import torch

from src.attention import attention


class MultiHeadAttention(nn.Module):
    def __init__(self,  
        d_model: int,
        num_heads: int):
        
        super().__init__()

        assert d_model % num_heads == 0

        self.WQ = nn.Parameter(torch.empty(d_model, d_model))
        self.WK = nn.Parameter(torch.empty(d_model, d_model))
        self.WV = nn.Parameter(torch.empty(d_model, d_model))
        self.WO = nn.Parameter(torch.empty(d_model, d_model))

        std = (1.0 / d_model) ** 0.5

        for W in [self.WQ, self.WK, self.WV, self.WO]:
            nn.init.trunc_normal_(
                W,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # self.rope = Rope(self.head_dim, )
        
    def forward(self, 
                in_features: Float[Tensor, " ... sequence_length d_model"], 
                token_positions: torch.Tensor | None = None,
                rope: nn.Module | None = None):

        # first get Q, K, V 
        Q = in_features @ self.WQ.T # 2 BS(D^2)
        K = in_features @ self.WK.T # 2 BS(D^2)
        V = in_features @ self.WV.T # 2 BS(D^2)

        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.head_dim)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.head_dim)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.head_dim)

        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)
        # then create a mask to turn off some part 

        mask = torch.ones(
            (Q.shape[-2], Q.shape[-2]),
            dtype=torch.bool,
            device=Q.device
        )

        mask = ~torch.triu(mask, diagonal=1)

        if rope is not None:
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        multihead = attention(Q, K, V, mask)
        multihead = multihead.transpose(-3, -2)
        multihead = multihead.reshape(*multihead.shape[:-2], self.d_model)

        # return by multiplying it with self.WO
        return multihead @ self.WO.T # 2BS(D^2) flops 


