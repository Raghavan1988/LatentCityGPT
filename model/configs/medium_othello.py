"""
LatentWorldsGPT — Othello full-reproduction config.

Sized for the 50k-games (~3M tokens) corpus. Between small.py (10.7M
params, overfits on 250k tokens) and small_music.py (1.4M params, too
small for richer Othello state).

Architecture choice: aim for ~1:1 params:tokens ratio, matching the
published Othello-GPT scaling (80M params / 60M tokens).

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: full reproduction of Othello-GPT board-state probe.    │
  │  Target:  linear probe (Nanda B-vs-W) ≥ 90 %, MLP ≥ 90 %.        │
  │  Runtime: ~15–25 min on Apple MPS.                                │
  └─────────────────────────────────────────────────────────────────┘
"""

# ── Architecture (~4M params) ──
block_size = 128   # ≥ max Othello game length
n_layer    = 4     # vs 6 in small.py
n_head     = 4     # vs 6 in small.py
n_embd     = 256   # vs 384 in small.py
dropout    = 0.2   # bit more regularization than small.py's 0.1
bias       = False

# ── Training schedule ──
# 5000 iters at batch 64 × block 128 = 8.2k token-examples per iter,
# ~41M total. With ~3M train tokens that's ~14 visits per token-position,
# but each token is sampled in many different contexts.
max_iters                   = 5_000
warmup_iters                = 300
lr_decay_iters              = 5_000
batch_size                  = 64
gradient_accumulation_steps = 1

# ── Optimizer ──
learning_rate = 3e-4
min_lr        = 3e-5
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# ── Eval / logging ──
eval_interval = 250         # train/val/gen CE+ppl every 250 iters
eval_iters    = 50
log_interval  = 20
