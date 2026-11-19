"""Train the TinyStories transformer with the assignment's starting settings."""

import sys

from scripts import loading_together


TOKEN_BUDGET = 327_680_000
BATCH_SIZE = 8
CONTEXT_LENGTH = 256
TRAINING_STEPS = TOKEN_BUDGET // (BATCH_SIZE * CONTEXT_LENGTH)


def main():
    loading_together.main([
        "--batch-size", str(BATCH_SIZE),
        "--context-length", str(CONTEXT_LENGTH),
        "--num-iterations", str(TRAINING_STEPS),
        "--vocab-size", "10000",
        "--d-model", "512",
        "--d-ff", "1344",
        "--theta", "10000",
        "--num-layers", "4",
        "--num-heads", "16",
        "--device", "mps",
        "--log-src", "logs/tinystories.jsonl",
        *sys.argv[1:],
    ])


if __name__ == "__main__":
    main()
