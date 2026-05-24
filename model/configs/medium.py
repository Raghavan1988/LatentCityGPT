"""
LatentCityGPT — medium config (~25–30M params).

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: full-scale / production-run config.                    │
  │  Targets: Boston- and South Bay-scale cities (vocab up to ~50k). │
  │  Runtime: ~12–24h on a single CUDA GPU (A100/H100/4090) on a     │
  │           full-sized corpus. Don't pick this for smoke iteration.│
  └─────────────────────────────────────────────────────────────────┘

Knob explanations: see model/configs/small.py — same fields, larger values.

Why block_size=512 here: South Bay's median shortest-path length is ~90 nodes,
max ~470. 512 covers >95% of routes whole; with 256 the model would only see
chunks of long routes and have to stitch context across boundaries.

Why batch_size dropped to 32 (and accumulation = 2): each sequence is 2× longer,
so per-step memory roughly doubles. We accumulate to keep effective batch = 64.
"""

# ── Architecture ──
block_size = 512
n_layer    = 8
n_head     = 8
n_embd     = 512
dropout    = 0.1
bias       = False

# ── Training schedule ──
max_iters                   = 30_000
warmup_iters                = 500
lr_decay_iters              = 30_000
batch_size                  = 32
gradient_accumulation_steps = 2     # effective batch_size = 64

# ── Optimizer ──
learning_rate = 2e-4
min_lr        = 2e-5
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# ── Eval / logging ──
eval_interval = 1_000
eval_iters    = 100
log_interval  = 25
