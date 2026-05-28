"""
LatentWorldsGPT — Phase 3-c cities scale-demo config.

A meaningfully larger cities model than `small.py` (~11M params). The
scale-demo question this config answers: do the probe + transplant
patterns observed at 10M-param scale persist when we train a model
roughly 5× larger on the full Manhattan corpus?

- n_layer = 12, n_head = 12, n_embd = 768
- ~50M parameters
- block_size = 512 (covers ~all Manhattan routes; median ≈ 50 steps)

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: scale demonstration for the cities domain.             │
  │  Target:  reproduce the multi-seed probe + transplant pattern   │
  │           at 5× params; lift the small-model concern.            │
  │  Runtime: ~6-10 h on H100 80GB, ~1 day on A100 80GB, ~2-3 days  │
  │           on RTX 4090. Not feasible on Apple MPS at this scale. │
  │  Memory:  ~50M params × fp32 ≈ 200MB weights; activations at    │
  │           batch 32 × block 512 × n_embd 768 × n_layer 12 ≈ 7GB. │
  │           Fits comfortably in 24GB+ VRAM; gradient checkpointing │
  │           not required.                                          │
  └─────────────────────────────────────────────────────────────────┘

Use with `data/manhattan/` (or regenerate with more walks for cleaner
convergence at this scale; see notes below).

NOTE on corpus size: current Manhattan corpus has ~2.74M train tokens.
For a 50M-param model that's a 0.055:1 tokens:params ratio (far below
Chinchilla optimum). Two options:
  (a) train as-is — model will be data-starved but the comparative
      question only requires reproducing the patterns, not
      pre-training-quality convergence.
  (b) regenerate Manhattan corpus with --n_walks 200000 to produce
      ~8M+ tokens (~3 hours of data generation; recommended for
      higher quality results).
"""

# ── Architecture (~50M params; GPT-2 small width with 12 layers) ──
block_size = 512   # covers ≥95% of Manhattan routes (median ~50 steps)
n_layer    = 12
n_head     = 12    # n_embd / n_head = 64 (matches GPT-2 small style)
n_embd     = 768
dropout    = 0.1   # less dropout — more data, less risk of overfitting
bias       = False

# ── Training schedule ──
# At batch 32 × block 512 = 16k token-examples per iter. 30,000 iters
# = ~480M token-examples. With ~3M train tokens that's ~160 visits per
# token in different contexts.
max_iters                   = 30_000
warmup_iters                = 1_500
lr_decay_iters              = 30_000
batch_size                  = 32
gradient_accumulation_steps = 2     # effective batch 64; helps stability at this scale

# ── Optimizer ──
learning_rate = 6e-4              # higher LR for larger model (GPT-2 small uses 6e-4)
min_lr        = 6e-5
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# ── Eval / logging ──
eval_interval = 1_000   # train/val/gen CE+ppl every 1000 iters
eval_iters    = 100
log_interval  = 100
