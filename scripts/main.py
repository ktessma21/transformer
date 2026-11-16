import pickle
from pathlib import Path

from src.bpe import train_bpe


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VOCAB_SIZE = 10_000


def main():
    vocab, merges = train_bpe(
        DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
    )
    for name, value in (("vocab.pkl", vocab), ("merges.pkl", merges)):
        path = DATA_DIR / name
        with path.open("wb") as output:
            pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {path}")
    print(f"Vocabulary size: {len(vocab):,}")


if __name__ == "__main__":
    main()
