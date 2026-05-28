"""
LatentWorldsGPT — Phase 3-b large Othello config (championship-games retrain).

Matches the published Othello-GPT config from Li et al. 2022 / Nanda 2023
as closely as is meaningful for the comparative-study claim:
- n_layer = 8, n_head = 8, n_embd = 512
- ~25M parameters
- Targeted corpus: championship-quality games from WTHOR archives
  (~70k+ tournament games, 50–60 moves each ≈ 4M training tokens after
  BOS/EOS framing — comparable scale to Li 2022's training mix).

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: tighten the comparative Othello positive control.     │
  │  Target:  3-class MLP per-cell probe ≥ 0.95 (matches published) │
  │  Runtime: ~30–60 min on H100 80GB, ~3–4 h on RTX 4090,           │
  │           ~12–18 h on Apple MPS (too slow for iteration).        │
  └─────────────────────────────────────────────────────────────────┘

Use with `data/othello_championship/` (produced by
`data/prepare_othello_championship.py` after downloading WTHOR archives).
"""

# ── Architecture (matches published Othello-GPT) ──
block_size = 128       # > max Othello game length (typically 60 moves)
n_layer    = 8
n_head     = 8         # n_embd / n_head = 64 (matches GPT-2 small)
n_embd     = 512
dropout    = 0.2
bias       = False

# ── Training schedule ──
# 10000 iters at batch 64 × block 128 = ~82M token-examples total.
# With ~4M train tokens that's ~20 visits/token in different contexts.
max_iters                   = 10_000
warmup_iters                = 500
lr_decay_iters              = 10_000
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
eval_interval = 500    # train/val/gen CE+ppl every 500 iters
eval_iters    = 100
log_interval  = 50
