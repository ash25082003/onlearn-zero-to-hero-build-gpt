"""Milestone 6: the Transformer block - communicate, then compute."""
import unittest

import torch
import torch.nn as nn

from model import Block


def make_block(seed=1337):
    torch.manual_seed(seed)
    b = Block(n_embd=32, n_head=4, block_size=64, dropout=0.0)
    b.eval()
    return b


class TestBlock(unittest.TestCase):

    def test_output_shape(self):
        b = make_block()
        x = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(1))
        self.assertEqual(tuple(b(x).shape), (2, 10, 32),
                         "a block maps (B, T, n_embd) to (B, T, n_embd) - "
                         "that's what lets you stack them")

    def test_exact_residual_structure(self):
        b = make_block()
        x = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(2))
        mid = x + b.sa(b.ln1(x))
        expected = mid + b.ffwd(b.ln2(mid))
        self.assertTrue(
            torch.allclose(b(x), expected, atol=1e-5),
            "x = x + sa(ln1(x)), then x = x + ffwd(ln2(x)) - norm BEFORE "
            "the sublayer, and the residual '+' both times")

    def test_has_two_layernorms(self):
        b = make_block()
        self.assertIsInstance(b.ln1, nn.LayerNorm)
        self.assertIsInstance(b.ln2, nn.LayerNorm)

    def test_head_size_split(self):
        b = make_block()
        n = sum(p.numel() for p in b.sa.parameters())
        self.assertEqual(n, 4 * 3 * 32 * 8 + 32 * 32 + 32,
                         "head_size = n_embd // n_head, so 4 heads of "
                         "size 8 for n_embd 32")

    def test_causality_survives_the_block(self):
        b = make_block()
        g = torch.Generator().manual_seed(3)
        x1 = torch.randn(2, 8, 32, generator=g)
        x2 = x1.clone()
        x2[:, 6:, :] = torch.randn(2, 2, 32, generator=g)
        self.assertTrue(
            torch.allclose(b(x1)[:, :6], b(x2)[:, :6], atol=1e-5),
            "the future must stay invisible through the whole block - "
            "residuals and LayerNorm are per-position, so only attention "
            "could leak it")


if __name__ == "__main__":
    unittest.main()
