"""
attention.py - the idea of the decade, built in three steps.

Your file (Milestones 3-5). First the trick that makes attention cheap
(Milestone 3), then a real head of self-attention (Milestone 4), then many
heads plus the feed-forward layer that lets tokens think about what they
gathered (Milestone 5).

A note on imports: after six modules of building everything by hand, you
get to use torch.nn - you've EARNED these. nn.Linear is your Module 4
class, done professionally; nn.Dropout you met in spirit in Module 4's
regularization talk; F.softmax you've written with exp and sum since
Module 2.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


def xbow_loop(x):
    # TODO (Milestone 3): the O(T^2) honest version ("bow" = bag of words).
    # x is (B, T, C). For every batch b and position t, the output at
    # (b, t) is the MEAN of x[b, 0], x[b, 1], ..., x[b, t] - each position
    # summarizes everything up to and including itself, with no peeking
    # ahead. Two nested loops and x[b, :t+1].mean(0). Slow is fine; this is
    # the specification.
    raise NotImplementedError("Milestone 3: xbow_loop")


def causal_weights(T):
    # TODO (Milestone 3): the same averaging as ONE matrix. Start from
    # torch.zeros(T, T), use torch.tril(torch.ones(T, T)) to find the
    # forbidden (future) entries, masked_fill them with float("-inf"),
    # then F.softmax(..., dim=-1). Rows come out as [1, 0, 0...],
    # [1/2, 1/2, 0...], [1/3, 1/3, 1/3...] - so causal_weights(T) @ x
    # equals your loop, for any x. The -inf-then-softmax pattern is the
    # exact mechanism a real attention head uses; here the scores are all
    # zero (every visible token equally interesting) - a head's job is to
    # make them UNEQUAL.
    raise NotImplementedError("Milestone 3: causal_weights")


class Head(nn.Module):

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        # TODO (Milestone 4): every token emits three vectors, so three
        # nn.Linear(n_embd, head_size, bias=False) projections:
        #   self.key   - "what I contain"
        #   self.query - "what I'm looking for"
        #   self.value - "what I'll hand over if you attend to me"
        # Plus the mask: self.register_buffer("tril",
        # torch.tril(torch.ones(block_size, block_size))) - a buffer, not a
        # parameter: it's structure, nothing trains it. And
        # self.dropout = nn.Dropout(dropout).
        raise NotImplementedError("Milestone 4: Head.__init__")

    def forward(self, x):
        # TODO (Milestone 4): x is (B, T, n_embd).
        #   1. k, q = self.key(x), self.query(x)          # (B, T, head_size)
        #   2. scores: wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        #      - every query dotted with every key, (B, T, T). The scaling
        #      keeps softmax from saturating (Module 4's lesson wearing a
        #      new coat: keep the pre-activations tame).
        #   3. the past-only mask, exactly Milestone 3's pattern:
        #      masked_fill(self.tril[:T, :T] == 0, float("-inf")), softmax
        #      over dim=-1, then self.dropout(wei).
        #   4. v = self.value(x); return wei @ v          # (B, T, head_size)
        raise NotImplementedError("Milestone 4: Head.forward")


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.0):
        super().__init__()
        # TODO (Milestone 5): several heads looking for different things at
        # once. self.heads = nn.ModuleList of num_heads Heads (each gets
        # n_embd, head_size, block_size, dropout); self.proj =
        # nn.Linear(head_size * num_heads, n_embd) to mix what they found
        # back into one stream; self.dropout = nn.Dropout(dropout).
        raise NotImplementedError("Milestone 5: MultiHeadAttention.__init__")

    def forward(self, x):
        # TODO (Milestone 5): run every head on x, torch.cat their outputs
        # on the last dim (back to (B, T, head_size * num_heads)), then
        # self.dropout(self.proj(...)).
        raise NotImplementedError("Milestone 5: MultiHeadAttention.forward")


class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        # TODO (Milestone 5): attention gathered information; this is where
        # each token thinks about it - alone. One nn.Sequential:
        # Linear(n_embd, 4 * n_embd), ReLU(), Linear(4 * n_embd, n_embd),
        # Dropout(dropout). (The 4x expansion is straight from the
        # "Attention Is All You Need" paper.) Store it as self.net.
        raise NotImplementedError("Milestone 5: FeedForward.__init__")

    def forward(self, x):
        # TODO (Milestone 5): return self.net(x). Note there's no mixing
        # across positions here - it's per-token, which the tests check.
        raise NotImplementedError("Milestone 5: FeedForward.forward")
