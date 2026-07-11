# Curriculum — Let's build GPT

Goal: the student builds a working GPT - a decoder-only Transformer - from empty files,
and trains it to write Shakespeare-shaped verse. Three acts: a new dataset at a new
scale with the bigram model as the honest baseline (Milestones 1-2), attention built
from its central trick outward - averaging, one head, many heads, the block (Milestones
3-6), and the full assembly plus training (Milestones 7-8). This is also the module
where six modules of hand-building pay off as *imports*: nn.Linear is their Module 4
class, nn.Embedding is Module 6's, LayerNorm answers their BatchNorm questions, AdamW
replaces the hand-rolled SGD they've written five times. The finale: `python train.py`
trains a 209,729-parameter GPT in about five minutes to val ~2.0 (the bigram's ceiling
is ~2.5) and generates 500 characters of text with speakers, line breaks, and iambic
ambitions.

Reveal these milestones **one at a time**. Do not show the whole list to the student.

## Milestone 1 — A million characters of Shakespeare
New dataset: the complete plays as one string, 1,115,394 characters, vocabulary of 65.
Character-level tokenizer (encode/decode), a 90/10 split with no shuffling (val = plays
the model never saw), and `get_batch`: random offsets, x is a chunk, y is the same chunk
shifted one right - so one (x, y) pair packs block_size prediction problems per row.
*Concept:* language modeling is next-token prediction, served in parallel. (Real GPTs
tokenize subwords, not characters - that's the NEXT lecture.)

## Milestone 2 — The bigram, reborn in torch.nn
Module 2's model, rewritten as a proper nn.Module: one nn.Embedding(vocab, vocab) whose
rows ARE the logits, cross-entropy via the (B*T, C) reshape, and a `generate` method -
forward, last position, softmax, sample, append, repeat.
*Concept:* the nn.Module contract (forward returning (logits, loss), parameters handled
for you) and the autoregressive loop. Also the baseline: this model scores ~2.5 and
writes gibberish. Everything after this milestone is about beating it.

## Milestone 3 — The trick at the heart of attention
Tokens need to talk to their past. Version 1: a double loop averaging x[b, :t+1] - the
honest specification. Version 3 (the lecture's arc): a (T, T) matrix - zeros, future
masked to -inf, softmax - whose rows are [1, 0...], [1/2, 1/2, 0...], so `wei @ x` IS
the loop.
*Concept:* masked-softmax-matmul is the entire mechanism of attention. Here the scores
are all zeros (every visible token equally interesting); an attention head's only job
will be to make those scores data-dependent.

## Milestone 4 — One head of self-attention
Every token emits a query ("what I'm looking for"), a key ("what I contain"), and a
value ("what I'll hand over"). Affinities = q @ k^T, scaled by 1/sqrt(head_size), masked
with tril, softmaxed, then used to mix the values. The tril lives in a register_buffer -
structure, not parameters.
*Concept:* data-dependent communication. The scaling is Module 4's init lesson wearing a
new coat - keep the softmax's inputs tame or it saturates into one-hot attention. The
decisive test: changing FUTURE tokens must not change past outputs.

## Milestone 5 — Many heads, then a moment to think
MultiHeadAttention: several heads in parallel (different conversations - "who's the
subject?", "any vowels lately?"), concatenated and projected back. FeedForward: Linear
4x up, ReLU, back down - per-token, no cross-position mixing.
*Concept:* attention gathers, feed-forward thinks. The tests literally check that
FeedForward can't mix positions - communication and computation are separate organs.

## Milestone 6 — The block: communicate, then compute
One Transformer block: `x = x + sa(ln1(x)); x = x + ffwd(ln2(x))`. Residual connections
(the '+' highway that lets gradients flow straight through - Module 5 gave them the
tools to see why) and LayerNorm before each sublayer.
*Concept:* LayerNorm is their Module 4 BatchNorm question answered differently -
normalize each token's own vector, so no batch statistics, no running buffers, no
train/eval personalities, and none of Module 6's 3D bug. Residual + pre-norm is what
makes DEEP stacks trainable; this block is why GPTs can be 96 layers.

## Milestone 7 — Assemble the GPT
Token embeddings + position embeddings (attention is permutation-blind; the position
table is how the model knows where things are), n_layer blocks, final LayerNorm,
lm_head. GPT-2-style std-0.02 init (provided - it's their Module 4 reflex, now industry
standard). `generate` is the bigram's loop plus one line: crop the context to the last
block_size tokens.
*Concept:* the full decoder-only Transformer - this IS the GPT architecture, differing
from OpenAI's GPT-2/3 only in size. Pinned test: the (16, 32, 4, 2) config has exactly
30,017 parameters.

## Milestone 8 — Train your GPT
`estimate_loss` (average over many batches, model.eval()/train() around it - Dropout
must sleep while you measure) and `train_gpt` - the loop they know by heart, except
after five modules of `p.data += -lr * p.grad` they finally get
`torch.optim.AdamW`.
*Concept:* the professional training harness. Nothing conceptually new - which is the
point; they've earned every line of it.

Each milestone has tests in `tests/`. A milestone is **done when its tests pass**.

When all tests pass, run `python train.py` for the payoff, in three acts: the bigram
trains first and posts its ~2.5 (the bar), then a 209,729-parameter GPT (4 blocks, 4
heads, 32 characters of context) trains for about five minutes to val ~2.0, then the
model speaks - 500 characters with speaker names, line breaks, and almost-words.
`python train.py --long` trains a bigger one (~half an hour) for noticeably better
verse. Honest note: the lecture's 10M-parameter GPU model reaches val 1.48; ours is the
same architecture exactly, sized for a laptop CPU. Then `explore.ipynb`: the attention
matrix as a picture, watching generations improve across training, bigram vs GPT loss
curves. Next lecture: the GPT tokenizer.
