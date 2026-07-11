"""Milestone 3: the mathematical trick - causal averaging as a matmul."""
import unittest

import torch

from attention import causal_weights, xbow_loop


def cumulative_mean(x):
    """Independent ground truth: running mean along the time dimension."""
    B, T, C = x.shape
    return torch.cumsum(x, dim=1) / torch.arange(1, T + 1).view(1, T, 1)


class TestCausalWeights(unittest.TestCase):

    def test_rows_are_uniform_averages(self):
        wei = causal_weights(5)
        self.assertEqual(tuple(wei.shape), (5, 5))
        for t in range(5):
            expected = torch.zeros(5)
            expected[:t + 1] = 1.0 / (t + 1)
            self.assertTrue(torch.allclose(wei[t], expected, atol=1e-6),
                            f"row {t} must average positions 0..{t} equally: "
                            "zeros, mask the future with -inf, softmax")

    def test_no_peeking_at_the_future(self):
        wei = causal_weights(8)
        self.assertTrue(torch.equal(wei.triu(1), torch.zeros(8, 8)),
                        "everything above the diagonal must be exactly 0 - "
                        "softmax of -inf")

    def test_rows_sum_to_one(self):
        wei = causal_weights(8)
        self.assertTrue(torch.allclose(wei.sum(1), torch.ones(8), atol=1e-6))


class TestXbowLoop(unittest.TestCase):

    def test_matches_independent_ground_truth(self):
        g = torch.Generator().manual_seed(1337)
        x = torch.randn(2, 6, 3, generator=g)
        self.assertTrue(torch.allclose(xbow_loop(x), cumulative_mean(x),
                                       atol=1e-5),
                        "out[b, t] must be the mean of x[b, 0..t] inclusive")


class TestTheTrick(unittest.TestCase):

    def test_matmul_equals_loop(self):
        g = torch.Generator().manual_seed(7)
        x = torch.randn(4, 8, 2, generator=g)
        self.assertTrue(
            torch.allclose(causal_weights(8) @ x, xbow_loop(x), atol=1e-5),
            "the whole point: causal_weights(T) @ x must equal your loop - "
            "the O(T^2) intent, done as one matrix multiply")


if __name__ == "__main__":
    unittest.main()
