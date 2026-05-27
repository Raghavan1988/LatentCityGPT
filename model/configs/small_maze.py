"""
LatentWorldsGPT — Phase 4 maze navigation config.

Matches the architecture specified in the locked predictions file
`predictions/predictions_maze_navigation.md`:
- n_layer = 6, n_head = 6, n_embd = 192, block_size = 64
- ~2M parameters
- Corpus: ~1.4M train tokens (data/maze_8x8/)

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: clean fits on 8×8 maze corpora; matches locked         │
  │           predictions for the ex-ante experiment.                │
  │  Target:  data/maze_8x8 / data/maze_8x8_within_shuffled /        │
  │           data/maze_8x8_global_shuffled.                         │
  │  Runtime: ~30-60 min per condition on Apple MPS.                 │
  └─────────────────────────────────────────────────────────────────┘
"""

# ── Architecture (matches the locked predictions file) ──
block_size = 64        # short paths; median ~10–15 steps
n_layer    = 6
n_head     = 6         # n_embd must be divisible by n_head; 192/6=32 ✓
n_embd     = 192
dropout    = 0.2       # moderate regularization
bias       = False

# ── Training schedule ──
# With 1.4M train tokens and batch 64 × block 64 = 4096 tokens/iter,
# 5000 iters processes ~20M token-examples — ~15 visits per training
# token. Adequate for a 67-token vocab; we expect convergence well
# before max_iters.
max_iters                   = 5_000
warmup_iters                = 200
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
eval_interval = 200
eval_iters    = 50
log_interval  = 20
