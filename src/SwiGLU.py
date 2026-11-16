import torch
from jaxtyping import Float
from torch import Tensor, nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # convention is (out_features, in_features) , we can change it for avoiding the transpose but we won't improve perforance

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]: # summation of each line. 
        assert x.shape[-1] == self.d_model
        gate = x @ self.w1.T           # 2*B*S*D*F FLOPs
        silu = gate * torch.sigmoid(gate)  # B*S*F + sigmoid cost
        value = x @ self.w3.T          # 2*B*S*D*F FLOPs

        return (silu * value) @ self.w2.T  # B*S*F + 2*B*S*F*D FLOPs
