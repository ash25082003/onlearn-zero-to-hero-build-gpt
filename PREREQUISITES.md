# Module 0 — Prerequisites (taught inline)

Before the milestones begin, the learner needs a few things checked. Run a quick
**diagnostic** at the very first session, then teach only the warm-ups they actually need.
A confident learner can skip straight to Milestone 1.

## Step 0 — Is PyTorch installed?

Run `python check.py`. If it complains about torch, walk them through `SETUP.md`
(`pip install torch`, CPU is fine) before anything else. This is the only required
install. (`explore.ipynb` also wants `jupyter` and `matplotlib` — optional dessert; it
can wait until the milestones are done.)

## The diagnostic (ask conversationally, don't make it feel like a test)

Find out, in a friendly back-and-forth:

1. **How far through the course are they?** This module leans on Module 2 (bigram +
   sampling loop, reborn here as Milestone 2), Module 4 (init scale, train/eval modes —
   the attention scaling and LayerNorm land on those hooks), and Module 6 (containers,
   3D tensors, and the BatchNorm bug that makes LayerNorm feel earned). Modules 1-6 done
   → ideal. Missing some → fine if they can hold (B, T, C) tensors in their head, but
   say honestly that the "you built this by hand already" moments will land softer.
2. **The (B, T, C) reflex** — if a batch is (4, 8, 32), what are the three axes? (Batch,
   time/position, channels.) Attention lives entirely in this shape; someone who has to
   re-derive it each time will struggle at Milestone 4.
3. **What does softmax do** — and what happens to a slot holding -inf? (It gets
   probability exactly 0. That one fact IS the causal mask.)
4. **Matrix multiply as mixing** — in `wei @ x`, what does row t of `wei` mean? (The
   recipe for position t's output: how much of every other position to blend in. If
   they get this, Milestone 3 will feel like a magic trick they're in on.)

Based on their answers, route them:
- Solid on all → go straight to Milestone 1.
- Shaky on one → do just that warm-up, then continue.
- New to this → warm-ups first, and set expectations warmly (this is the biggest module
  of the course, and the one everything else was building toward - two or three
  sittings, and the ending talks).

## Warm-up A — The course so far, as an import list (only if needed)
Goal: frame the module. They've BUILT: backprop (Modules 1, 5), embeddings + sampling
(Modules 2-3), Linear/BatchNorm/init/train-eval (Module 4), containers and hierarchy
(Module 6). This module hands them the professional versions: `nn.Linear`,
`nn.Embedding`, `nn.Sequential`, `nn.LayerNorm`, `F.cross_entropy`, `AdamW`. Go down
the list and let them say which module each one comes from. Ten minutes, huge payoff:
torch.nn stops being magic and becomes *their* code, typed faster.

## Warm-up B — Softmax and the -inf mask (only if needed)
Goal: the mechanism of Milestone 3 in a REPL. `t = torch.tensor([2.0, 1.0, 0.0])`,
softmax it. Now set `t[2] = float("-inf")` and softmax again - the last slot is exactly
zero, and the others renormalized. Masking isn't deleting; it's making something
impossible and letting softmax redistribute.

## Warm-up C — Batched matmul shapes (only if needed)
Goal: the one shape rule Milestone 4 leans on. If `q` is (B, T, 16) and `k` is
(B, T, 16), what is `q @ k.transpose(-2, -1)`? ((B, T, T) - every query dotted with
every key, per batch element.) And (B, T, T) @ (B, T, 16)? (Back to (B, T, 16).) Three
lines in a REPL with torch.randn; the whole head is these two multiplies plus a mask.

When the needed warm-ups are done (or skipped), continue into the main CURRICULUM.md.
