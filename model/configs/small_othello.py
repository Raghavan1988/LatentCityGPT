"""
LatentWorldsGPT — Othello-GPT reproduction config.

Same architecture as `small.py` but with:
  - block_size = 128 (down from 256; Othello games are ≤67 tokens)
  - eval_interval = 100 (down from 250; user wants more granular
    train / val / gen loss + perplexity tracking)

Designed for the ~250k-token random-games corpus from
`data/prepare_othello.py`. With small.py's 10.7M params and 3,673
visits/token, this should comfortably learn Othello rules.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: reproduce Li 2022 / Nanda 2023's board-state probe     │
  │           (~90 % per-cell accuracy) as the load-bearing          │
  │           end-to-end framework sanity check.                      │
  │  Runtime: ~5–10 min on Apple MPS.                                 │
  └─────────────────────────────────────────────────────────────────┘
"""

# ── Architecture ──
block_size = 128   # vs 256 in small.py (Othello games ≤ ~67 tokens)
n_layer    = 6
n_head     = 6
n_embd     = 384
dropout    = 0.1
bias       = False

# ── Training schedule ──
max_iters                   = 2_000
warmup_iters                = 200
lr_decay_iters              = 2_000
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
eval_interval = 100   # train/val/gen CE+ppl every 100 iters
eval_iters    = 50
log_interval  = 10
