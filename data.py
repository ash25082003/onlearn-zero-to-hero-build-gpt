"""
data.py - a million characters of Shakespeare, served in batches.

Your file (Milestone 1). New dataset, new scale: no more names - the full
text of Shakespeare's plays, modeled character by character. You build the
vocabulary, the 90/10 split, and the batcher that serves random chunks of
text with their one-character-ahead targets.

Check your progress any time with:  python check.py
"""
import torch


def load_text(path="input.txt"):
    """PROVIDED - read the whole play collection as one string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    # TODO (Milestone 1): character-level tokenizer, Module 2 style but on
    # a richer alphabet (65 characters: newline, space, punctuation, digits,
    # upper and lower case).
    #   1. chars = sorted(set(text))
    #   2. stoi maps char -> integer 0..64; itos maps back
    #   3. encode = lambda s: [stoi[c] for c in s]     (string -> list of ints)
    #      decode = lambda l: "".join(itos[i] for i in l)  (back to a string)
    # Return (stoi, itos, encode, decode).
    raise NotImplementedError("Milestone 1: build_vocab")


def split_data(text, encode, train_frac=0.9):
    # TODO (Milestone 1): encode the WHOLE text into one long
    # torch.tensor(..., dtype=torch.long), then cut it: the first 90% is
    # training data, the last 10% is validation. (No shuffling - it's one
    # continuous text, so the val set is simply "plays the model never saw".)
    # Return (train_data, val_data).
    raise NotImplementedError("Milestone 1: split_data")


def get_batch(data, block_size, batch_size, generator=None):
    # TODO (Milestone 1): serve batch_size random chunks of the text.
    #   1. ix = torch.randint(len(data) - block_size, (batch_size,),
    #      generator=generator) - random starting offsets
    #   2. x = torch.stack([data[i:i + block_size] for i in ix])
    #   3. y = the same chunks shifted ONE character right:
    #      data[i + 1 : i + block_size + 1]
    # x[b, t]'s target is y[b, t] - so one (x, y) pair packs block_size
    # separate prediction problems per row, from 1 character of context up
    # to block_size. Return (x, y).
    raise NotImplementedError("Milestone 1: get_batch")
