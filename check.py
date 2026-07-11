#!/usr/bin/env python3
"""
check.py - your progress scoreboard for building a GPT.

Run it any time from the project root:

    python check.py

It runs your milestones in order and stops at the first one that isn't finished,
so you only ever see what you're working on now - not a wall of red for
milestones you haven't reached yet. Completed milestones stay green; everything
ahead shows as locked until you get there.

The only dependency is PyTorch (see SETUP.md); the tests themselves run on
Python's built-in test tools.
"""
import sys
import unittest
import warnings

# torch prints a harmless warning if numpy isn't installed - not our problem here
warnings.filterwarnings("ignore", message=".*NumPy.*")

try:
    import torch  # noqa: F401
except ImportError:
    print()
    print("  PyTorch isn't installed (or your virtual environment isn't active).")
    print("  This project needs it - it's the one thing to install:")
    print()
    print("      pip install torch")
    print()
    print("  See SETUP.md for the full setup, then run `python check.py` again.")
    print()
    sys.exit(1)

# Ordered milestones: number, title, test module, and a warm hint shown only
# when this is the milestone you're currently on.
MILESTONES = [
    (1, "A million characters of Shakespeare", "tests.test_milestone1",
     "build_vocab: chars = sorted(set(text)), stoi/itos both ways, encode/"
     "decode as one-liners. split_data: encode everything into one long "
     "tensor, cut at 90%. get_batch: random offsets via torch.randint(len "
     "- block_size, ...), x = stacked chunks, y = the same chunks shifted "
     "one character right."),
    (2, "The bigram, reborn in torch.nn", "tests.test_milestone2",
     "One nn.Embedding(vocab_size, vocab_size) IS the model - logits "
     "straight from the table. For the loss, F.cross_entropy wants (N, C) "
     "vs (N,): view the logits to (B*T, C), targets to (B*T). generate: "
     "forward, last position only, softmax, multinomial (pass the "
     "generator), cat, repeat."),
    (3, "The trick at the heart of attention", "tests.test_milestone3",
     "xbow_loop: two loops, x[b, :t+1].mean(0). causal_weights: zeros, "
     "masked_fill the upper triangle (tril == 0) with float('-inf'), "
     "softmax dim=-1. Row t comes out [1/(t+1)] * (t+1) - so wei @ x IS "
     "your loop."),
    (4, "One head of self-attention", "tests.test_milestone4",
     "Three bias-free Linears (key/query/value), tril as register_buffer. "
     "Forward: wei = q @ k.transpose(-2, -1) * head_size**-0.5, mask with "
     "tril[:T, :T], softmax, dropout, then wei @ v. The scale and the :T "
     "crop are the two classic misses."),
    (5, "Many heads, then a moment to think", "tests.test_milestone5",
     "MultiHeadAttention: nn.ModuleList of Heads, torch.cat outputs on the "
     "last dim, then proj and dropout. FeedForward: Sequential(Linear(n, "
     "4n), ReLU(), Linear(4n, n), Dropout) - per-token, no mixing across "
     "positions."),
    (6, "The block: communicate, then compute", "tests.test_milestone6",
     "head_size = n_embd // n_head. Forward is two residual updates: "
     "x = x + self.sa(self.ln1(x)), then x = x + self.ffwd(self.ln2(x)). "
     "Norm goes BEFORE the sublayer, and don't drop the 'x +'."),
    (7, "Assemble the GPT", "tests.test_milestone7",
     "Token table (vocab, n_embd) + position table (block_size, n_embd); "
     "x = tok_emb + pos_emb; nn.Sequential of Blocks; ln_f; lm_head. "
     "self.apply(self._init_weights) at the end of __init__. In generate, "
     "crop the context: idx[:, -self.block_size:]."),
    (8, "Train your GPT", "tests.test_milestone8",
     "estimate_loss: model.eval(), average the loss over eval_iters fresh "
     "batches per split, model.train() again, return plain floats. "
     "train_gpt: AdamW, then the rhythm - zero_grad(set_to_none=True), "
     "backward, step - recording loss.item() each step."),
]

GLYPH = {"done": "✅", "current": "▶ ", "locked": "\U0001f512"}  # ✅ ▶ 🔒


def run_module(module_name):
    """Run one milestone's tests. Returns (passed, total, first_problem_detail)."""
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(module_name)
    except Exception as exc:  # tests couldn't even import (e.g. broken data.py)
        return 0, 1, f"could not load tests ({exc})"
    result = unittest.TestResult()
    suite.run(result)
    total = result.testsRun
    problems = result.failures + result.errors
    passed = total - len(problems)
    detail = None
    if problems:
        lines = problems[0][1].strip().splitlines()
        detail = lines[-1] if lines else "a check failed"
    return passed, total, detail


def main():
    statuses = []  # each: (num, title, state, passed, total, hint, detail)
    blocked = False
    for num, title, module, hint in MILESTONES:
        if blocked:
            statuses.append((num, title, "locked", 0, 0, hint, None))
            continue
        passed, total, detail = run_module(module)
        if total > 0 and passed == total:
            statuses.append((num, title, "done", passed, total, hint, None))
        else:
            statuses.append((num, title, "current", passed, total, hint, detail))
            blocked = True

    done_count = sum(1 for s in statuses if s[2] == "done")
    total_count = len(statuses)

    print()
    print("=" * 62)
    print("  Let's build GPT - your progress")
    print("=" * 62)
    for num, title, state, passed, total, hint, detail in statuses:
        line = f"  {GLYPH[state]}  Milestone {num}  {title}"
        if state == "current":
            extra = f"{passed} of {total} checks passing" if total else "in progress"
            line += f"   <- you are here ({extra})"
        print(line)
    print("-" * 62)

    if not blocked:
        print(f"  \U0001f389  All {total_count} milestones complete. You built a GPT.")
        print("      Run `python train.py` and listen to it speak - then")
        print("      open explore.ipynb.")
        print("=" * 62)
        print()
        return 0

    current = next(s for s in statuses if s[2] == "current")
    print(f"  {done_count} of {total_count} milestones complete.")
    print()
    print(f"  ▶  Milestone {current[0]} - {current[1]}: what to do next")
    print(f"     {current[5]}")
    if current[6]:
        print(f"     Failing check: {current[6]}")
    print("=" * 62)
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
