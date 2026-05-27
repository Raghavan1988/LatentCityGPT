"""
LatentWorldsGPT — flight-phase (M4) config.

Sized for the flight corpus (~46k train tokens at 5s sampling). The
small_music.py config (1.4M params) overfits at 30:1 params:tokens ratio
on this corpus. This config targets ~6:1 ratio to match the music-expanded
clean-fit regime, plus heavier dropout.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: clean fit on flight corpora; train/val/gen aligned.    │
  │  Target:  no overfit; probe + transplant on a healthy model.     │
  │  Runtime: ~2-5 min on Apple MPS.                                  │
  └─────────────────────────────────────────────────────────────────┘
"""

# ── Architecture (~0.3M params) ──
block_size = 256
n_layer    = 2       # vs 3 in small_music.py
n_head     = 2       # n_embd must be divisible by this
n_embd     = 96      # vs 192 in small_music.py
dropout    = 0.4     # heavier than small_music.py's 0.3
bias       = False

# ── Training schedule ──
# 500 iters at batch 64 × block 256 = 16k token-examples per iter,
# 8M total. With ~46k train tokens that's ~175 visits per token-position
# during training. Eval every 50 to catch overfit early.
max_iters                   = 500
warmup_iters                = 50
lr_decay_iters              = 500
batch_size                  = 64
gradient_accumulation_steps = 1

# ── Optimizer ──
learning_rate = 3e-4
min_lr        = 3e-5
weight_decay  = 0.3        # heavier weight decay
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# ── Eval / logging ──
eval_interval = 50          # eval frequently to catch overfit early
eval_iters    = 50
log_interval  = 10
