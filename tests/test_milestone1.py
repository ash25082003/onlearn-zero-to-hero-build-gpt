"""Milestone 1: the dataset - vocabulary, split, batches."""
import unittest

import torch

from data import load_text, build_vocab, split_data, get_batch

text = None


def setUpModule():
    global text
    text = load_text()


class TestVocab(unittest.TestCase):

    def test_vocab_size_is_65(self):
        stoi, itos, encode, decode = build_vocab(text)
        self.assertEqual(len(stoi), 65,
                         "Shakespeare uses exactly 65 characters: newline, "
                         "space, punctuation, digits, both cases")
        self.assertEqual(len(itos), 65)

    def test_stoi_itos_are_inverses(self):
        stoi, itos, encode, decode = build_vocab(text)
        for ch, i in stoi.items():
            self.assertEqual(itos[i], ch)

    def test_encode_decode_roundtrip(self):
        stoi, itos, encode, decode = build_vocab(text)
        s = "First Citizen:\nSpeak, speak."
        self.assertEqual(decode(encode(s)), s,
                         "decode(encode(s)) must give back exactly s")
        self.assertIsInstance(encode(s), list)
        self.assertTrue(all(isinstance(i, int) for i in encode(s)))


class TestSplit(unittest.TestCase):

    def test_ninety_ten(self):
        stoi, itos, encode, decode = build_vocab(text)
        train_data, val_data = split_data(text, encode)
        self.assertEqual(len(train_data), 1003854,
                         "first 90% of 1,115,394 characters")
        self.assertEqual(len(val_data), 111540)
        self.assertEqual(train_data.dtype, torch.long)

    def test_no_shuffling(self):
        stoi, itos, encode, decode = build_vocab(text)
        train_data, val_data = split_data(text, encode)
        self.assertTrue(
            torch.equal(train_data[:20], torch.tensor(encode(text[:20]))),
            "the text stays in order - train is the first 90%, val is the "
            "last 10% (plays the model never saw)")


class TestGetBatch(unittest.TestCase):

    def setUp(self):
        stoi, itos, encode, decode = build_vocab(text)
        self.train_data, _ = split_data(text, encode)

    def test_shapes(self):
        x, y = get_batch(self.train_data, block_size=8, batch_size=4,
                         generator=torch.Generator().manual_seed(1337))
        self.assertEqual(tuple(x.shape), (4, 8))
        self.assertEqual(tuple(y.shape), (4, 8))

    def test_targets_are_one_ahead(self):
        x, y = get_batch(self.train_data, block_size=8, batch_size=16,
                         generator=torch.Generator().manual_seed(1337))
        self.assertTrue(
            torch.equal(x[:, 1:], y[:, :-1]),
            "y must be x shifted one character right: x[b, t]'s target is "
            "the NEXT character. Check the +1 in your y slices")

    def test_rows_are_real_text(self):
        x, y = get_batch(self.train_data, block_size=8, batch_size=4,
                         generator=torch.Generator().manual_seed(1337))
        # every row must be a contiguous slice of the data tensor
        n = len(self.train_data)
        windows = self.train_data.unfold(0, 9, 1)  # all (x row + next char)
        for b in range(4):
            row = torch.cat([x[b], y[b, -1:]])
            self.assertTrue((windows == row).all(1).any(),
                            "each batch row must be a contiguous chunk of "
                            "the text, not shuffled characters")

    def test_deterministic_with_generator(self):
        x1, y1 = get_batch(self.train_data, 8, 4,
                           generator=torch.Generator().manual_seed(99))
        x2, y2 = get_batch(self.train_data, 8, 4,
                           generator=torch.Generator().manual_seed(99))
        self.assertTrue(torch.equal(x1, x2) and torch.equal(y1, y2),
                        "draw the offsets with the generator you were handed")


if __name__ == "__main__":
    unittest.main()
