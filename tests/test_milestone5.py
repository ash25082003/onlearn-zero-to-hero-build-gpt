"""Milestone 5: multi-head attention and the feed-forward layer."""
import unittest

import torch

from attention import FeedForward, MultiHeadAttention


class TestMultiHead(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1337)
        self.mha = MultiHeadAttention(n_embd=32, num_heads=4, head_size=8,
                                      block_size=64, dropout=0.0)
        self.mha.eval()

    def test_output_shape(self):
        x = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(1))
        self.assertEqual(tuple(self.mha(x).shape), (2, 10, 32),
                         "heads concatenate back to head_size * num_heads, "
                         "then project to n_embd")

    def test_equals_concat_of_own_heads(self):
        x = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(2))
        cat = torch.cat([h(x) for h in self.mha.heads], dim=-1)
        expected = cat @ self.mha.proj.weight.T + self.mha.proj.bias
        self.assertTrue(torch.allclose(self.mha(x), expected, atol=1e-5),
                        "run every head, cat on the last dim, then proj")

    def test_parameter_count(self):
        n = sum(p.numel() for p in self.mha.parameters())
        self.assertEqual(n, 4 * 3 * 32 * 8 + 32 * 32 + 32,
                         f"got {n}; 4 heads of 3 bias-free (32x8) "
                         "projections, plus the (32x32)+bias output proj")

    def test_causality_survives(self):
        g = torch.Generator().manual_seed(3)
        x1 = torch.randn(2, 8, 32, generator=g)
        x2 = x1.clone()
        x2[:, 6:, :] = torch.randn(2, 2, 32, generator=g)
        self.assertTrue(torch.allclose(self.mha(x1)[:, :6],
                                       self.mha(x2)[:, :6], atol=1e-6))


class TestFeedForward(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1337)
        self.ff = FeedForward(n_embd=32, dropout=0.0)
        self.ff.eval()

    def test_output_shape(self):
        x = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(1))
        self.assertEqual(tuple(self.ff(x).shape), (2, 10, 32))

    def test_parameter_count(self):
        n = sum(p.numel() for p in self.ff.parameters())
        self.assertEqual(n, 32 * 128 + 128 + 128 * 32 + 32,
                         f"got {n}; expand to 4 * n_embd (with bias), "
                         "ReLU, project back (with bias)")

    def test_positions_do_not_mix(self):
        g = torch.Generator().manual_seed(2)
        x1 = torch.randn(1, 6, 32, generator=g)
        x2 = x1.clone()
        x2[:, 3, :] = torch.randn(32, generator=g)
        o1, o2 = self.ff(x1), self.ff(x2)
        mask = torch.ones(6, dtype=torch.bool)
        mask[3] = False
        self.assertTrue(
            torch.allclose(o1[:, mask], o2[:, mask], atol=1e-6),
            "feed-forward is per-token: changing position 3's input must "
            "not change any OTHER position's output. Attention gathers, "
            "feed-forward thinks alone")
        self.assertFalse(torch.allclose(o1[:, 3], o2[:, 3], atol=1e-3),
                         "...but position 3's own output should change")


if __name__ == "__main__":
    unittest.main()
