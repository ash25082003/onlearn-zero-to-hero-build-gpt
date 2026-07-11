"""Milestone 4: one head of self-attention."""
import unittest

import torch
import torch.nn.functional as F

from attention import Head


def make_head(seed=1337, n_embd=32, head_size=16, block_size=64):
    torch.manual_seed(seed)
    return Head(n_embd, head_size, block_size, dropout=0.0)


class TestHead(unittest.TestCase):

    def test_output_shape(self):
        h = make_head()
        x = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(1))
        self.assertEqual(tuple(h(x).shape), (2, 10, 16),
                         "(B, T, n_embd) in, (B, T, head_size) out")

    def test_exact_attention_formula(self):
        h = make_head()
        h.eval()
        x = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(2))
        k = x @ h.key.weight.T
        q = x @ h.query.weight.T
        v = x @ h.value.weight.T
        wei = q @ k.transpose(-2, -1) * 16 ** -0.5
        wei = wei.masked_fill(torch.tril(torch.ones(10, 10)) == 0,
                              float("-inf"))
        expected = F.softmax(wei, dim=-1) @ v
        self.assertTrue(
            torch.allclose(h(x), expected, atol=1e-5),
            "scores = q @ k^T scaled by head_size**-0.5, mask the future "
            "with -inf, softmax, then weights @ v - check each step")

    def test_causality(self):
        h = make_head()
        g = torch.Generator().manual_seed(3)
        x1 = torch.randn(2, 10, 32, generator=g)
        x2 = x1.clone()
        x2[:, 7:, :] = torch.randn(2, 3, 32, generator=g)
        o1, o2 = h(x1), h(x2)
        self.assertTrue(
            torch.allclose(o1[:, :7], o2[:, :7], atol=1e-6),
            "changing FUTURE tokens changed a PAST output - the mask isn't "
            "doing its job. This is the property that makes a decoder a "
            "decoder")

    def test_tril_is_a_buffer_not_a_parameter(self):
        h = make_head()
        self.assertTrue(hasattr(h, "tril"), "register_buffer('tril', ...)")
        self.assertTrue(all("tril" not in name
                            for name, _ in h.named_parameters()),
                        "the mask is structure, not something to train - "
                        "register_buffer, not nn.Parameter")

    def test_projections_have_no_bias(self):
        h = make_head()
        n = sum(p.numel() for p in h.parameters())
        self.assertEqual(n, 3 * 32 * 16,
                         f"got {n} parameters; a head is exactly three "
                         "bias-free (n_embd x head_size) projections")

    def test_works_shorter_than_block_size(self):
        h = make_head(block_size=64)
        x = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(4))
        self.assertEqual(tuple(h(x).shape), (1, 5, 16),
                         "crop the mask to [:T, :T] - contexts shorter than "
                         "block_size happen constantly during generation")


if __name__ == "__main__":
    unittest.main()
