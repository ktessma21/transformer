"""Load the model, data, and training components together."""

import argparse
import json
from time import perf_counter

import numpy as np
import torch

from src import Adam, Transformer_LM, cross_entropy, load_data, load_checkpoint, save_checkpoint
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-src", "--log_src", dest="log_src", type=Path)
    parser.add_argument(
        "--checkpoint-src", "--checkpointsrc", dest="checkpointsrc", type=Path
    )

    parser.add_argument(
        "--num-iterations", "--num_iterations", dest="num_iterations",
        type=int, required=True, help="Number of training steps to run.",
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--data-src", type=Path,
        default=Path(__file__).resolve().parents[1] / "data/train.bin",
    )
    parser.add_argument(
        "--valid-src", type=Path,
        default=Path(__file__).resolve().parents[1] / "data/dev.bin",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))

    model = parser.add_argument_group("Transformer model")
    for name in (
        "d_model", "num_heads", "d_ff", "vocab_size", "context_length", "num_layers"
    ):
        model.add_argument(
            "--" + name.replace("_", "-"), "--" + name,
            dest=name, type=int, required=True,
        )
    model.add_argument("--theta", type=float, default=10000.0)

    optimizer = parser.add_argument_group("AdamW optimizer")
    optimizer.add_argument("--lr", type=float, default=3e-4)
    optimizer.add_argument(
        "--betas", type=float, nargs=2, metavar=("BETA1", "BETA2"),
        default=(0.9, 0.999),
    )
    optimizer.add_argument(
        "--weight-decay", "--weight_decay", dest="weight_decay",
        type=float, default=0.01,
    )
    optimizer.add_argument("--eps", type=float, default=1e-8)

    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch_size must be positive")
    if args.num_iterations <= 0:
        parser.error("num_iterations must be positive")
    for name in (
        "d_model", "num_heads", "d_ff", "vocab_size", "context_length", "num_layers"
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.d_model % args.num_heads:
        parser.error("d_model must be divisible by num_heads")
    if args.theta <= 0:
        parser.error("theta must be positive")
    if args.lr < 0 or args.weight_decay < 0:
        parser.error("lr and weight_decay must be nonnegative")
    if args.eps <= 0:
        parser.error("eps must be positive")
    if not all(0 <= beta < 1 for beta in args.betas):
        parser.error("betas must each be in [0, 1)")
    args.betas = tuple(args.betas)
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if (args.d_model // args.num_heads) % 2:
        raise ValueError("Each attention head needs an even dimension for RoPE.")

    # Map each split's little-endian uint16 token IDs read-only so batches can be read on demand.
    training_tokens = np.memmap(args.data_src, dtype="<u2", mode="r")
    validation_tokens = np.memmap(args.valid_src, dtype="<u2", mode="r")

    if len(training_tokens) <= args.context_length:
        raise ValueError("Dataset needs at least context_length + 1 tokens.")

    model = Transformer_LM(
        d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff,
        theta=args.theta, vocab_size=args.vocab_size,
        context_length=args.context_length, num_layers=args.num_layers,
    ).to(args.device)

    optimizer = Adam(
        model.parameters(), lr=args.lr, betas=args.betas,
        weight_decay=args.weight_decay, eps=args.eps,
    )

    iteration = 0
    if args.checkpointsrc is not None:
        iteration = load_checkpoint(args.checkpointsrc, model, optimizer)
        optimizer.t = iteration

    model.train()

    validation_loss = []

    if args.log_src is not None:
        args.log_src.parent.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    for _ in range(args.num_iterations):
        inputs, targets = load_data(
            training_tokens, args.batch_size, args.context_length, args.device
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(
            logits.reshape(-1, args.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        iteration += 1

        if iteration % 100 == 0:
            with torch.no_grad():
                valid_inputs, valid_targets = load_data(
                    validation_tokens, args.batch_size, args.context_length, args.device
                )

                model.eval()
                valid_logits = model(valid_inputs)
                valid_loss = cross_entropy(
                    valid_logits.reshape(-1, args.vocab_size), valid_targets.reshape(-1)
                    )
                print(f"step {iteration}: loss={valid_loss.item():.4f}", flush=True)

                if args.log_src is not None:
                    with args.log_src.open("a", encoding="utf-8") as log:
                        print(json.dumps({
                            "step": iteration,
                            "validation_loss": valid_loss.item(),
                            "elapsed_seconds": perf_counter() - start_time,
                        }), file=log, flush=True)
                model.train()

        if args.log_src is not None:
            with args.log_src.open("a", encoding="utf-8") as log:
                print(json.dumps({"step": iteration, "loss": loss.item()}), file=log, flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"step_{iteration}.pt"
    save_checkpoint(model, optimizer, iteration, checkpoint_path)
    print(f"Saved {checkpoint_path}")


if __name__ == "__main__":
    main()
