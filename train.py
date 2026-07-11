"""
train.py - the training harness, and the payoff.

You write two functions (Milestone 8): the batched loss estimator (the loss
on ONE minibatch is too noisy to trust - you saw that in Module 6's
notebook) and the training loop, which after five modules of hand-rolled
`p.data += -lr * p.grad` finally gets a professional optimizer.
The block at the bottom is the reward. Run it with:  python train.py
(Or `python train.py --long` for a bigger model and a longer cook.)
"""
import sys
import time

import torch

from data import load_text, build_vocab, split_data, get_batch
from model import BigramLanguageModel, GPT


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size,
                  eval_iters=100, generator=None):
    # TODO (Milestone 8): the honest loss readout. For each of the two
    # splits, average model loss over eval_iters fresh batches; return
    # {"train": ..., "val": ...} as plain floats.
    # Two professional touches:
    #   - model.eval() first, model.train() again before returning - the
    #     Dropout layers must switch off while you measure (your Module 4
    #     train/eval lesson; the @torch.no_grad() above handles gradients)
    #   - use get_batch(data, block_size, batch_size, generator=generator)
    raise NotImplementedError("Milestone 8: estimate_loss")


def train_gpt(model, train_data, block_size, batch_size, steps, lr=1e-3,
              generator=None):
    # TODO (Milestone 8): the loop you know by heart, with one upgrade.
    # Five modules of `p.data += -lr * p.grad` have earned you AdamW:
    #   optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Then per step: get_batch -> forward (model returns (logits, loss)) ->
    # optimizer.zero_grad(set_to_none=True) -> loss.backward() ->
    # optimizer.step() -> record loss.item() in lossi. Also call
    # model.train() once before the loop. Return lossi.
    raise NotImplementedError("Milestone 8: train_gpt")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # The payoff. Finish all 8 milestones, then run:  python train.py
    # A few minutes on a laptop CPU. `--long` trains a bigger model for
    # roughly half an hour and the text gets noticeably more Shakespearean.
    # ------------------------------------------------------------------
    long_run = "--long" in sys.argv

    text = load_text()
    stoi, itos, encode, decode = build_vocab(text)
    vocab_size = len(stoi)
    train_data, val_data = split_data(text, encode)
    print(f"{len(text):,} characters of Shakespeare, vocab of {vocab_size}")

    # Act 1: where this course started - a bigram model, trained properly.
    torch.manual_seed(1337)
    bigram = BigramLanguageModel(vocab_size)
    train_gpt(bigram, train_data, block_size=8, batch_size=32, steps=2000,
              lr=1e-2, generator=torch.Generator().manual_seed(1337))
    b_losses = estimate_loss(bigram, train_data, val_data, 8, 32,
                             generator=torch.Generator().manual_seed(7))
    print(f"\nact 1 - the bigram (Module 2's model): val loss "
          f"{b_losses['val']:.4f}. One character of context. "
          f"That's the bar.")

    # Act 2: the GPT.
    if long_run:
        cfg = dict(block_size=64, n_embd=96, n_head=4, n_layer=4)
        batch_size, steps, lr, dropout = 16, 5000, 1e-3, 0.1
    else:
        cfg = dict(block_size=32, n_embd=64, n_head=4, n_layer=4)
        batch_size, steps, lr, dropout = 16, 1600, 1e-3, 0.0
    torch.manual_seed(1337)
    model = GPT(vocab_size, **cfg, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nact 2 - your GPT: {n_params:,} parameters, "
          f"{cfg['n_layer']} blocks of {cfg['n_head']}-head attention, "
          f"{cfg['block_size']} characters of context")

    g = torch.Generator().manual_seed(1337)
    t0 = time.time()
    done = 0
    while done < steps:
        chunk = min(400, steps - done)
        lossi = train_gpt(model, train_data, cfg["block_size"], batch_size,
                          steps=chunk, lr=lr, generator=g)
        done += chunk
        print(f"  {done:5d}/{steps} steps:  batch loss "
              f"~{sum(lossi[-100:]) / min(100, len(lossi)):.4f}   "
              f"({time.time() - t0:.0f}s)")

    losses = estimate_loss(model, train_data, val_data, cfg["block_size"],
                           batch_size, generator=torch.Generator().manual_seed(7))
    print(f"\n  train loss: {losses['train']:.4f}")
    print(f"  val loss:   {losses['val']:.4f}   (the bigram managed "
          f"{b_losses['val']:.4f})")
    if not long_run:
        print("  (try `python train.py --long` sometime: a bigger model, "
              "~half an hour, noticeably better verse)")

    print("\nact 3 - the model speaks. 500 characters, from one newline:")
    print("-" * 60)
    model.eval()
    ctx = torch.tensor([[stoi["\n"]]], dtype=torch.long)
    out = model.generate(ctx, max_new_tokens=500,
                         generator=torch.Generator().manual_seed(1337))
    print(decode(out[0].tolist()))
    print("-" * 60)
