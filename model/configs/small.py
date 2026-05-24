"""
LatentCityGPT — small config (~10–15M params).

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: fast-iteration default for smoke runs.                 │
  │  Targets: City of London / Manhattan-scale cities (vocab ≤ ~5k). │
  │  Runtime: ~5 min on CUDA GPU, ~15–30 min on Apple MPS,           │
  │           several hours on CPU (not recommended).                │
  └─────────────────────────────────────────────────────────────────┘

How to read these knobs:

  ARCHITECTURE
    block_size      — max sequence length the model attends over in one window.
                      Pick so most routes fit whole. 256 covers ≥90% of
                      Manhattan-scale routes (median len ~50).
    n_layer         — transformer depth. More layers → richer representations
                      but slower and harder to train.
    n_head          — attention heads per block. n_embd must be divisible by this.
    n_embd          — residual-stream width. Embedding params = vocab × n_embd
                      so this is the dominant memory cost on big-vocab cities.
    dropout         — standard regularization for small datasets.

  TRAINING SCHEDULE
    max_iters       — number of optimization steps. NOT epochs — we sample
                      random windows from the stream forever.
    warmup_iters    — linear ramp from 0 → learning_rate over this many iters.
    lr_decay_iters  — cosine decay from learning_rate → min_lr over this range.
    batch_size      — sequences per micro-batch. Each is block_size long.
    gradient_accumulation_steps — micro-batches per optimizer step. Bump if
                      you can't fit batch_size on your device.

  OPTIMIZER (AdamW)
    learning_rate   — peak after warmup.
    min_lr          — cosine floor (typically learning_rate / 10).
    weight_decay    — applied only to 2-D params (Linear/Embedding weights);
                      biases and LayerNorm gains skip decay. See
                      model.configure_optimizers().
    beta1, beta2    — Adam moment decay rates.
    grad_clip       — global-norm clip threshold; prevents loss spikes.

  EVAL / LOGGING
    eval_interval   — every N iters: estimate train/val/gen CE, maybe save ckpt.
    eval_iters      — average over this many batches when estimating eval loss.
    log_interval    — print one-line train-loss update every N iters.
"""

# ── Architecture ──
block_size = 256
n_layer    = 6
n_head     = 6
n_embd     = 384
dropout    = 0.1
bias       = False

# ── Training schedule ──
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
eval_interval = 500
eval_iters    = 50
log_interval  = 10
