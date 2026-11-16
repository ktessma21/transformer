"""Evaluate the saved tokenizer on ten TinyStories validation documents."""

import json
import pickle
from pathlib import Path
import struct

from src.bpe import Tokenizer


DATA = Path(__file__).resolve().parents[1] / "data"
SEPARATOR = "<|endoftext|>"


def sample_stories(count=10):
    stories = []
    lines = []
    # The validation file begins with a truncated story; skip that first entry.
    skip_first = True
    with (DATA / "TinyStoriesV2-GPT4-valid.txt").open(encoding="utf-8") as source:
        for line in source:
            if line.strip() == SEPARATOR:
                text = "".join(lines).strip()
                lines = []
                if skip_first:
                    skip_first = False
                    continue
                if text:
                    stories.append(text)
                    if len(stories) == count:
                        return stories
            else:
                lines.append(line)
    raise ValueError(f"Could not find {count} complete stories")


def main():
    with (DATA / "vocab.pkl").open("rb") as source:
        vocab = pickle.load(source)
    with (DATA / "merges.pkl").open("rb") as source:
        merges = pickle.load(source)
    tokenizer = Tokenizer(vocab, merges, [SEPARATOR])
    stories = sample_stories()
    output = DATA / "tinystories_sample_10"
    output.mkdir(exist_ok=True)
    total_bytes = total_tokens = 0
    for index, text in enumerate(stories, 1):
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids) != text:
            raise ValueError(f"Round-trip failed for story {index}")
        raw = text.encode("utf-8")
        (output / f"story_{index:02}.txt").write_bytes(raw)
        (output / f"story_{index:02}.uint32").write_bytes(
            struct.pack(f"<{len(ids)}I", *ids)
        )
        (output / f"story_{index:02}.tokens.json").write_text(
            json.dumps(ids), encoding="utf-8"
        )
        total_bytes += len(raw)
        total_tokens += len(ids)
        print(f"Story {index:02}: {len(raw):,} bytes, {len(ids):,} tokens")
    print(f"Total: {total_bytes:,} UTF-8 bytes, {total_tokens:,} tokens")
    print(f"Bytes per token: {total_bytes / total_tokens:.3f}")
    print(f"Packed uint32 bytes: {total_tokens * 4:,}")
    print(f"Compression ratio (raw / uint32): {total_bytes / (total_tokens * 4):.3f}x")
    print("All 10 encode/decode round-trips passed.")
    print(f"Samples and token IDs: {output}")


if __name__ == "__main__":
    main()
