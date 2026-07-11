"""Milestone 2: the bigram language model, reborn as an nn.Module."""
import unittest

import torch
import torch.nn.functional as F

from model import BigramLanguageModel


class TestBigram(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1337)
        self.model = BigramLanguageModel(65)
        g = torch.Generator().manual_seed(1)
        self.xb = torch.randint(0, 65, (4, 8), generator=g)
        self.yb = torch.randint(0, 65, (4, 8), generator=g)

    def test_logits_shape(self):
        logits, loss = self.model(self.xb, self.yb)
        self.assertEqual(tuple(logits.shape), (4, 8, 65),
                         "one row of the table per input token: (B, T, "
                         "vocab_size)")

    def test_loss_is_none_without_targets(self):
        logits, loss = self.model(self.xb)
        self.assertIsNone(loss, "no targets means generation mode - no loss")

    def test_loss_matches_cross_entropy(self):
        logits, loss = self.model(self.xb, self.yb)
        expected = F.cross_entropy(logits.reshape(32, 65),
                                   self.yb.reshape(32))
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6),
                        "reshape (B, T, C) -> (B*T, C) and targets to "
                        "(B*T,), then F.cross_entropy")

    def test_untrained_loss_is_clueless(self):
        _, loss = self.model(self.xb, self.yb)
        self.assertGreater(loss.item(), 3.5)
        self.assertLess(loss.item(), 6.0,
                        "an untrained model should sit near -ln(1/65) = "
                        "4.17 (default init lands ~4.7)")

    def test_generate_grows_the_context(self):
        ctx = torch.zeros((2, 3), dtype=torch.long)
        out = self.model.generate(ctx, max_new_tokens=10,
                                  generator=torch.Generator().manual_seed(2))
        self.assertEqual(tuple(out.shape), (2, 13),
                         "generate returns context + max_new_tokens")
        self.assertTrue(torch.equal(out[:, :3], ctx),
                        "the original context stays untouched at the front")
        self.assertTrue(((out >= 0) & (out < 65)).all(),
                        "every sampled index must be a valid character")

    def test_generate_deterministic_with_generator(self):
        ctx = torch.zeros((1, 1), dtype=torch.long)
        o1 = self.model.generate(ctx, 20, generator=torch.Generator().manual_seed(3))
        o2 = self.model.generate(ctx, 20, generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(o1, o2),
                        "pass the generator into torch.multinomial")


if __name__ == "__main__":
    unittest.main()
