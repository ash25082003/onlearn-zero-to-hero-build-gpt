"""Milestone 7: the assembled GPT."""
import unittest

import torch
import torch.nn.functional as F

from model import GPT


def make_gpt(seed=1337):
    torch.manual_seed(seed)
    m = GPT(vocab_size=65, block_size=16, n_embd=32, n_head=4, n_layer=2,
            dropout=0.0)
    m.eval()
    return m


class TestGPT(unittest.TestCase):

    def test_parameter_count(self):
        m = make_gpt()
        n = sum(p.numel() for p in m.parameters())
        self.assertEqual(n, 30017,
                         f"got {n:,}; the (block 16, embd 32, 4 heads, 2 "
                         "layers) test config has exactly 30,017 parameters "
                         "- token AND position embeddings, 2 blocks, final "
                         "LayerNorm, lm_head with bias")

    def test_forward_shapes_and_loss(self):
        m = make_gpt()
        g = torch.Generator().manual_seed(1)
        xb = torch.randint(0, 65, (4, 16), generator=g)
        yb = torch.randint(0, 65, (4, 16), generator=g)
        logits, loss = m(xb, yb)
        self.assertEqual(tuple(logits.shape), (4, 16, 65))
        expected = F.cross_entropy(logits.reshape(64, 65), yb.reshape(64))
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        logits, loss = m(xb)
        self.assertIsNone(loss)

    def test_init_is_unsure(self):
        m = make_gpt()
        g = torch.Generator().manual_seed(2)
        xb = torch.randint(0, 65, (8, 16), generator=g)
        yb = torch.randint(0, 65, (8, 16), generator=g)
        with torch.no_grad():
            _, loss = m(xb, yb)
        self.assertLess(loss.item(), 4.5,
                        f"first loss {loss.item():.2f}; with the std-0.02 "
                        "init it should sit near -ln(1/65) = 4.17 - did you "
                        "call self.apply(self._init_weights)?")

    def test_causality(self):
        m = make_gpt()
        g = torch.Generator().manual_seed(3)
        i1 = torch.randint(0, 65, (2, 12), generator=g)
        i2 = i1.clone()
        i2[:, 9:] = torch.randint(0, 65, (2, 3), generator=g)
        with torch.no_grad():
            l1, _ = m(i1)
            l2, _ = m(i2)
        self.assertTrue(torch.allclose(l1[:, :9], l2[:, :9], atol=1e-5),
                        "changing future tokens changed a past prediction - "
                        "the mask must hold through the whole stack")

    def test_position_matters(self):
        m = make_gpt()
        idx = torch.full((1, 6), 7, dtype=torch.long)  # same token, 6 times
        with torch.no_grad():
            logits, _ = m(idx)
        self.assertFalse(
            torch.allclose(logits[0, 0], logits[0, 3], atol=1e-4),
            "six copies of the same token got identical predictions at "
            "every position - the position embedding isn't reaching the "
            "stream. tok_emb + pos_emb")

    def test_generate_and_crop(self):
        m = make_gpt()
        ctx = torch.zeros((1, 1), dtype=torch.long)
        out = m.generate(ctx, 12, generator=torch.Generator().manual_seed(4))
        self.assertEqual(tuple(out.shape), (1, 13))
        self.assertTrue(torch.equal(out[:, :1], ctx))
        # context LONGER than block_size: the crop must handle it
        long_ctx = torch.zeros((1, 40), dtype=torch.long)
        out = m.generate(long_ctx, 3, generator=torch.Generator().manual_seed(5))
        self.assertEqual(tuple(out.shape), (1, 43),
                         "crop idx to the last block_size tokens before the "
                         "forward - the position table only knows "
                         "block_size positions")

    def test_generate_deterministic(self):
        m = make_gpt()
        ctx = torch.zeros((1, 1), dtype=torch.long)
        o1 = m.generate(ctx, 15, generator=torch.Generator().manual_seed(6))
        o2 = m.generate(ctx, 15, generator=torch.Generator().manual_seed(6))
        self.assertTrue(torch.equal(o1, o2))

    def test_gpt2_style_init(self):
        m = make_gpt()
        std = m.lm_head.weight.std().item()
        self.assertTrue(0.015 < std < 0.025,
                        f"lm_head weight std is {std:.4f}; _init_weights "
                        "should set every Linear to std 0.02")


if __name__ == "__main__":
    unittest.main()
