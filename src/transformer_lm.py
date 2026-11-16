from jaxtyping import Float, Int

from torch import nn, Tensor
from src.transformer import Transformer_Block
from src.Embeddings import Embedding
from src.rmsnorm import RMSNorm
from src.Linear import Linear


class Transformer_LM(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                theta: float, 
                vocab_size: int, 
                context_length: int,
                num_layers: int) -> None:
        
        super().__init__()

        self.layers = nn.ModuleList([
            Transformer_Block(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)
        ])

        self.embedding = Embedding(vocab_size, d_model)

        self.linear = Linear(d_model, vocab_size)

        self.norm = RMSNorm(d_model)
        

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, "batch sequence_length vocab_size"]:
        
        e = self.embedding(in_indices)
        for transformer in self.layers:
            e = transformer(e)

        e = self.norm(e)
        e = self.linear(e)

        return e