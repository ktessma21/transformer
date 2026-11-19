"""Load the model, data, and training components together."""

import torch
import pickle

from pathlib import Path

from src.bpe import Tokenizer
from src import Transformer_LM
from src.softmax import softmax

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_pkl(file_src: Path):
    if not file_src.is_file():
        raise FileNotFoundError()

    with file_src.open(mode="rb") as f:
        return pickle.load(f)

def top_p(x: torch.Tensor, p, dim = -1):
    # kept_ids = torch.Tensor()
    sorted_probs, sorted_ids = x.sort(dim=dim, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = torch.cat([
        torch.zeros_like(cumulative[..., :1], dtype=torch.bool),
        cumulative[..., :-1] >= p,
    ], dim=-1)

    kept_probs = sorted_probs.masked_fill(remove, 0.0)
    # kept_probs = kept_probs / kept_probs.sum(dim = -1, keepdim = True)
    index = torch.multinomial(kept_probs, num_samples=1)

    return sorted_ids.gather(dim=1, index=index).squeeze(-1)


def decode(
    args,
    *,
    log_src: Path | None = None,
    max_token: int,
    temp: float,
    p_val: float,
    input_ids: torch.Tensor,  # [batch_size, sequence_length]
):
    model = Transformer_LM(
        d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff,
        theta=args.theta, vocab_size=args.vocab_size,
        context_length=args.context_length, num_layers=args.num_layers,
    ).to(args.device)

    vocab = load_pkl(DATA_DIR / "vocab.pkl")
    merges = load_pkl(DATA_DIR / "merges.pkl")
    
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    stop_token_id = tokenizer.encode("<|endoftext|>")[0]
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)   

    prompt_len = input_ids.shape[1] 

    for i in range(max_token):
        pred = model(input_ids) # [batch_size, sequence_length, vocab_size]
        # here we need to choose which exact vocab_size is our choice. 

        pred = pred[:, -1, :] # [batch_size, vocab_size]
        pred = softmax(pred, dim = -1, T = temp)
        pred = top_p(pred, p_val)

        pred = pred.masked_fill(finished, stop_token)
        finished |= pred.eq(stop_token)
        input_ids = torch.cat([input_ids, pred[:, None]], dim=1)

        if finished.all():
            break
    
    results = []

    for row in input_ids[:, prompt_len:].tolist():
        if stop_token_id in row: # there might not be there if max_token encountered
            row = row[:row.index(stop_token_id)]
        results.append(tokenizer.decode(row))

    return results
        

