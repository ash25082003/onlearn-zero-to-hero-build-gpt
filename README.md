# Let's build GPT

**Neural Networks: Zero to Hero - Module 7**
Companion lecture: [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY) by Andrej Karpathy.

This is the module the whole course was building toward. You write a GPT - a
decoder-only Transformer, the same architecture as GPT-2 and GPT-3, smaller numbers -
from empty files, and train it on the complete works of Shakespeare until it speaks.
Attention gets built from its central trick outward: a triangular matrix that lets every
character attend to its past (one masked softmax), then a real head with queries, keys,
and values, then many heads, then the Transformer block with residuals and LayerNorm.
And after six modules of building everything by hand, this is where it pays off as
*imports*: `nn.Linear` is your Module 4 class, `nn.Embedding` is Module 6's, LayerNorm
answers your BatchNorm questions, and `AdamW` retires the `p.data += -lr * p.grad`
you've hand-rolled five times. You earned every one of them.

You'll build every piece yourself, with an AI tutor guiding you one step at a time.

## What you need
- Python 3.9+ and **PyTorch** - the one required install: `pip install torch` (CPU is
  plenty). Progress checks run on Python's built-in `unittest` via `python check.py`.
- Any coding agent or AI assistant that can read the project files and run shell commands
  (Claude, Codex, Cursor, or another tool)
- **The course so far** - Modules 2, 4, and 6 especially (the sampling loop, the init
  and train/eval lessons, and the container/3D-tensor fluency all come back here). The
  further you've come, the more this module feels like a victory lap that happens to
  build a GPT.

## How to start
1. Download or clone this project and `cd` into it.
2. Open it with your preferred coding agent.
3. Just say hi. Your tutor takes it from there.

For tool-specific setup notes, see [SETUP.md](./SETUP.md). It covers Claude Code, Codex,
Cursor, and generic coding assistants.

Your code goes in `data.py` (the character pipeline and batcher), `attention.py` (the
trick, the head, many heads), `model.py` (bigram baseline, the block, the GPT), and
`train.py` (the harness). The dataset is `input.txt` - 1,115,394 characters of
Shakespeare, vocabulary of 65. Ask for a hint when stuck, ask the tutor to check your
work to run the tests, and ask to move on when ready for the next milestone. Plan on
two to three sittings; eight milestones in all.

The tests decide when you're done - not the tutor. When they all pass, run
`python train.py`: three acts. The bigram (Module 2's model) trains first and posts its
~2.5 - that's the bar. Then your GPT - 209,729 parameters, 4 blocks, 4-head attention -
trains for about five minutes to val ~2.0. Then the model speaks: 500 characters grown
from a single newline, with speaker names, line breaks, and almost-words. Try
`python train.py --long` sometime - a bigger model, about half an hour, noticeably
better verse. (Honesty: the lecture's 10M-parameter model on a GPU reaches 1.48; yours
is the same architecture exactly, sized for a laptop CPU.)

**The pictures:** this module's `explore.ipynb` (`pip install jupyter matplotlib`) shows
what you built thinking - the causal averaging matrix, a real attention map on real
text, bigram-vs-GPT loss curves, and the same prompt sampled as training progresses,
from noise to speech.

Next up: **Module 8 - Let's build the GPT Tokenizer**, the last piece of the story:
why real GPTs don't read characters, and the byte-pair machinery that feeds them. Watch
for it on **[the course page](https://onlearn.app/projects/nn-zero-to-hero)**.

---

## More from Onlearn

This is **Module 7** of **[Neural Networks: Zero to Hero](https://onlearn.app/projects/nn-zero-to-hero)**
on Onlearn — guided, build-it-yourself projects with an AI tutor that checks your work against
the tests.

**[→ Explore more courses](https://onlearn.app/projects)**
