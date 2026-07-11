"""Milestone 8: the training harness - estimate_loss and train_gpt."""
import unittest

import torch

from data import load_text, build_vocab, split_data
from model import GPT
from train import estimate_loss, train_gpt

train_data = val_data = None


def setUpModule():
    global train_data, val_data
    text = load_text()
    stoi, itos, encode, decode = build_vocab(text)
    train_data, val_data = split_data(text, encode)


def tiny_gpt(dropout=0.0, seed=1337):
    torch.manual_seed(seed)
    return GPT(vocab_size=65, block_size=16, n_embd=32, n_head=4, n_layer=2,
               dropout=dropout)


class TestEstimateLoss(unittest.TestCase):

    def test_returns_both_splits(self):
        m = tiny_gpt()
        out = estimate_loss(m, train_data, val_data, block_size=16,
                            batch_size=8, eval_iters=5,
                            generator=torch.Generator().manual_seed(7))
        self.assertIn("train", out)
        self.assertIn("val", out)
        self.assertIsInstance(out["train"], float)
        self.assertTrue(3.5 < out["val"] < 6.0,
                        "an untrained GPT should read ~4.2 on both splits")

    def test_flips_dropout_off_while_measuring(self):
        m = tiny_gpt(dropout=0.5)
        o1 = estimate_loss(m, train_data, val_data, 16, 8, eval_iters=3,
                           generator=torch.Generator().manual_seed(7))
        o2 = estimate_loss(m, train_data, val_data, 16, 8, eval_iters=3,
                           generator=torch.Generator().manual_seed(7))
        self.assertEqual(o1["val"], o2["val"],
                         "same batches, different answers - dropout is "
                         "still firing during evaluation. model.eval() "
                         "first, model.train() after")
        self.assertTrue(m.training,
                        "leave the model back in training mode when done")


class TestTrainGpt(unittest.TestCase):

    def test_loss_falls(self):
        m = tiny_gpt()
        losses = train_gpt(m, train_data, block_size=16, batch_size=8,
                           steps=300, lr=1e-3,
                           generator=torch.Generator().manual_seed(1337))
        self.assertEqual(len(losses), 300, "one loss per step")
        self.assertIsInstance(losses[0], float)
        first = sum(losses[:20]) / 20
        last = sum(losses[-20:]) / 20
        self.assertLess(last, first * 0.8,
                        f"loss went {first:.3f} -> {last:.3f} over 300 "
                        "steps; with AdamW it should fall fast from ~4.2 - "
                        "check zero_grad, backward, step, in that rhythm")
        self.assertLess(last, 3.2,
                        f"after 300 steps the batch loss should be under "
                        f"3.2 (got {last:.3f})")

    def test_arguments_are_respected(self):
        m = tiny_gpt(seed=9)
        losses = train_gpt(m, train_data, block_size=16, batch_size=4,
                           steps=3, generator=torch.Generator().manual_seed(9))
        self.assertEqual(len(losses), 3,
                         "steps must control the loop - no hard-coded counts")

    def test_parameters_actually_move(self):
        m = tiny_gpt()
        before = m.lm_head.weight.detach().clone()
        train_gpt(m, train_data, block_size=16, batch_size=8, steps=5,
                  generator=torch.Generator().manual_seed(11))
        self.assertFalse(torch.equal(m.lm_head.weight, before),
                         "five steps and the weights didn't move - is "
                         "optimizer.step() in the loop?")


if __name__ == "__main__":
    unittest.main()
