"""
model.py - from a lookup table to a GPT.

Your file (Milestones 2, 6, 7). It starts where Module 2 started - a bigram
model, one lookup table - except now written as a proper nn.Module with a
generate() method. It ends with the real thing: token and position
embeddings, a stack of Transformer blocks, and the same generate() loop,
untouched, suddenly producing Shakespeare.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import FeedForward, MultiHeadAttention


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # TODO (Milestone 2): Module 2's counting table, reborn:
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size).
        # Row i holds the logits for "what follows character i" - the whole
        # model is one lookup.
        raise NotImplementedError("Milestone 2: BigramLanguageModel.__init__")

    def forward(self, idx, targets=None):
        # TODO (Milestone 2): idx is (B, T) integers.
        #   logits = self.token_embedding_table(idx)      # (B, T, vocab)
        # If targets is None, loss is None (that's generation mode).
        # Otherwise F.cross_entropy - but it wants (N, C) vs (N,), so
        # reshape: logits.view(B*T, C) against targets.view(B*T). Every
        # position in every row is its own little prediction problem.
        # Return (logits, loss).
        raise NotImplementedError("Milestone 2: BigramLanguageModel.forward")

    def generate(self, idx, max_new_tokens, generator=None):
        # TODO (Milestone 2): the loop you wrote in Modules 2-6, as a
        # method. max_new_tokens times:
        #   1. logits, _ = self(idx)
        #   2. keep only the LAST position: logits[:, -1, :]     # (B, vocab)
        #   3. probs = F.softmax(..., dim=-1)
        #   4. idx_next = torch.multinomial(probs, num_samples=1,
        #      generator=generator)
        #   5. idx = torch.cat((idx, idx_next), dim=1)
        # Return idx - the original context with max_new_tokens new
        # characters grown on the end.
        raise NotImplementedError("Milestone 2: BigramLanguageModel.generate")


class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        # TODO (Milestone 6): one full Transformer block = communicate,
        # then compute:
        #   head_size = n_embd // n_head
        #   self.sa = MultiHeadAttention(n_embd, n_head, head_size,
        #                                block_size, dropout)
        #   self.ffwd = FeedForward(n_embd, dropout)
        #   self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        # LayerNorm is your Module 4 BatchNorm question answered a different
        # way: normalize each token's OWN vector, so no batch statistics,
        # no running buffers, no train/eval personalities - and none of the
        # 3D bug you hunted in Module 6.
        raise NotImplementedError("Milestone 6: Block.__init__")

    def forward(self, x):
        # TODO (Milestone 6): two residual updates ("x +" is the highway;
        # the sublayers are exits that merge back on):
        #   x = x + self.sa(self.ln1(x))
        #   x = x + self.ffwd(self.ln2(x))
        # Return x. Norm BEFORE the sublayer (pre-norm) - deep stacks of
        # these train stably, which is the whole reason GPTs can be deep.
        raise NotImplementedError("Milestone 6: Block.forward")


class GPT(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer,
                 dropout=0.0):
        super().__init__()
        self.block_size = block_size
        # TODO (Milestone 7): the full assembly:
        #   self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #   self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #     (attention is permutation-blind - this table is how the model
        #      knows WHERE each token sits)
        #   self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size,
        #                                 dropout) for _ in range(n_layer)])
        #   self.ln_f = nn.LayerNorm(n_embd)
        #   self.lm_head = nn.Linear(n_embd, vocab_size)
        # Then self.apply(self._init_weights) - the Module 4 reflex, GPT
        # edition: small gaussian weights so the network starts unsure.
        raise NotImplementedError("Milestone 7: GPT.__init__")

    def _init_weights(self, module):
        """PROVIDED - std 0.02 gaussian init, the GPT-2 convention. Your
        Module 4 instincts, now industry standard."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # TODO (Milestone 7): idx is (B, T) with T <= self.block_size.
        #   tok_emb = self.token_embedding_table(idx)            # (B, T, C)
        #   pos_emb = self.position_embedding_table(torch.arange(T))  # (T, C)
        #   x = tok_emb + pos_emb        (broadcasting adds position to all)
        #   x = self.blocks(x); x = self.ln_f(x)
        #   logits = self.lm_head(x)                         # (B, T, vocab)
        # Loss exactly like the bigram: None without targets, else reshape
        # and F.cross_entropy. Return (logits, loss).
        raise NotImplementedError("Milestone 7: GPT.forward")

    def generate(self, idx, max_new_tokens, generator=None):
        # TODO (Milestone 7): the bigram's generate loop with ONE new line:
        # the position table only knows block_size positions, so before each
        # forward, crop the context to the last block_size tokens:
        #   idx_cond = idx[:, -self.block_size:]
        # Then exactly as before: forward idx_cond, last position, softmax,
        # multinomial (pass the generator), cat. Return idx.
        raise NotImplementedError("Milestone 7: GPT.generate")
