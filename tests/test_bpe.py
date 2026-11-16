from collections import Counter
from pathlib import Path
import tempfile
import unittest

from src.bpe import build_new, train_bpe


class BpeTests(unittest.TestCase):
    def test_build_new_combines_matching_replacements(self):
        affected = Counter({(b"a", b"b", b"ab"): 2, (b"ab", b"a", b"b"): 3})
        sequences = Counter()
        result = build_new((b"a", b"b"), affected, sequences)
        expected = Counter({(b"ab", b"ab"): 5})
        self.assertEqual(result, expected)
        self.assertEqual(sequences, expected)

    def test_train_and_pickle_round_trip(self):
        import pickle

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("hello hello world\n\nhello<|endoftext|>world", encoding="utf-8")
            vocab, merges = train_bpe(path, 260, ["<|endoftext|>"])
        self.assertEqual(len(vocab), 260)
        self.assertEqual(set(vocab), set(range(260)))
        self.assertIn(b"<|endoftext|>", vocab.values())
        self.assertTrue(merges)
        self.assertEqual(pickle.loads(pickle.dumps(vocab)), vocab)


if __name__ == "__main__":
    unittest.main()
