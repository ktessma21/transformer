"""Encode TinyStories into ordered, memory-mappable uint16 token sequences."""

from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from time import perf_counter

from src.bpe import Tokenizer


DATA = Path(__file__).resolve().parents[1] / "data"
SEPARATOR = "<|endoftext|>"
TOKENIZER = None


def initialize_worker():
    global TOKENIZER
    with (DATA / "vocab.pkl").open("rb") as source:
        vocab = pickle.load(source)
    with (DATA / "merges.pkl").open("rb") as source:
        merges = pickle.load(source)
    if not vocab or min(vocab) < 0 or max(vocab) > 65535:
        raise ValueError("Vocabulary IDs must fit in uint16")
    if SEPARATOR.encode() not in vocab.values():
        raise ValueError("The saved vocabulary must contain the story separator")
    TOKENIZER = Tokenizer(vocab, merges, [SEPARATOR])
    TOKENIZER.encode_pretoken = lru_cache(maxsize=65536)(TOKENIZER.encode_pretoken)


def encode_batch(text):
    ids = TOKENIZER.encode(text)
    if TOKENIZER.decode(ids) != text:
        raise ValueError("Encoded batch did not decode to the original text")
    tokens = array("H", ids)
    if tokens.itemsize != 2:
        raise RuntimeError("Expected two-byte unsigned shorts")
    if sys.byteorder != "little":
        tokens.byteswap()
    return tokens.tobytes(), len(text.encode("utf-8")), text.count(SEPARATOR)


def batches(path, target_chars=262144):
    lines = []
    size = 0
    # End batches at document boundaries; preserve all whitespace and separators.
    with path.open(encoding="utf-8", newline="") as source:
        for line in source:
            lines.append(line)
            size += len(line)
            if size >= target_chars and line.strip() == SEPARATOR:
                text = "".join(lines)
                boundary = text.rfind(SEPARATOR) + len(SEPARATOR)
                yield text[:boundary]
                # Keep the newline with the next document so whitespace
                # pre-tokens are identical to encoding the entire file.
                remainder = text[boundary:]
                lines = [remainder]
                size = len(remainder)
    if lines:
        yield "".join(lines)


def tokenize_split(pool, source_name, split, workers):
    source = DATA / source_name
    destination = DATA / f"{split}.bin"
    temporary = destination.with_suffix(".bin.partial")
    source_bytes = source.stat().st_size
    pending = deque()
    completed_bytes = token_count = separators = 0
    start = last_report = perf_counter()
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()

    def consume(output):
        nonlocal completed_bytes, token_count, separators, last_report
        encoded, byte_count, separator_count = pending.popleft().result()
        output.write(encoded)
        output_hash.update(encoded)
        completed_bytes += byte_count
        token_count += len(encoded) // 2
        separators += separator_count
        now = perf_counter()
        if now - last_report >= 10:
            print(
                f"{split}: {100 * completed_bytes / source_bytes:.1f}% | "
                f"{token_count:,} tokens | {now - start:.0f}s", flush=True
            )
            last_report = now

    print(f"Encoding {source.name} -> {destination.name}", flush=True)
    with temporary.open("wb") as output:
        for text in batches(source):
            source_hash.update(text.encode("utf-8"))
            pending.append(pool.submit(encode_batch, text))
            if len(pending) >= workers * 2:
                consume(output)
        while pending:
            consume(output)
    if completed_bytes != source_bytes or temporary.stat().st_size != token_count * 2:
        raise ValueError("Input or output size mismatch")
    temporary.replace(destination)
    metadata = {
        "source": source_name,
        "source_bytes": source_bytes,
        "source_sha256": source_hash.hexdigest(),
        "output": destination.name,
        "output_sha256": output_hash.hexdigest(),
        "dtype": "<u2",
        "byte_order": "little",
        "bytes_per_token": 2,
        "token_count": token_count,
        "separator": SEPARATOR,
        "separator_count": separators,
        "round_trip_verified": True,
        "elapsed_seconds": round(perf_counter() - start, 3),
        "tokenizer_sha256": {
            name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            for name in ("vocab.pkl", "merges.pkl")
        },
    }
    (DATA / f"{split}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {destination}: {token_count:,} tokens ({token_count * 2:,} bytes)", flush=True)


def main():
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        tokenize_split(pool, "TinyStoriesV2-GPT4-valid.txt", "dev", workers)
        tokenize_split(pool, "TinyStoriesV2-GPT4-train.txt", "train", workers)


if __name__ == "__main__":
    main()
