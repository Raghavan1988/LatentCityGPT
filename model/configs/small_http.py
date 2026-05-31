"""
LatentWorldsGPT — HTTP request log domain config (refined v2).

Sized for the NASA-HTTP corpus with PER-FIELD tokenization (4 tokens
per request, vocab ~55, ~4-8M training tokens, sessions up to 30
requests = up to 122 tokens including BOS/EOS).

Anti-overfit + carry-through-friendly design:

  - Smaller embedding dimension (n_embd=128) reflects the small vocab.
    A model that's too wide for the vocab size memorizes per-token
    co-occurrence statistics rather than learning sequential structure.
  - n_layer=4: enough depth for carry-through routing to compose
    across multiple attention hops, not so deep that overfit risk
    explodes.
  - Lower dropout than v1 (0.20 vs 0.30): carry-through is a routing
    phenomenon that benefits from stable attention patterns; very
    heavy dropout breaks those patterns and would risk a spurious
    null on Feature A.
  - block_size=128: covers sessions up to 31 requests (124 tokens +
    BOS/EOS); max_session_len in the data prep is capped at 30.
  - max_iters=5000 with eval_interval=100 and aggressive early-stop
    semantics: monitor val/train divergence and abort if seen.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Purpose: second pre-registered ex-ante test of the graded      │
  │  N-criterion's carry-through prediction on an applied domain.   │
  │  Targets: Feature A (first-request size_bin; carry-through;     │
  │  predicted encoded) and Feature B (cumulative count of large-   │
  │  response tokens; predicted null).                              │
  │  Runtime: ~1.5-2 hours per condition on Apple MPS.              │
  └─────────────────────────────────────────────────────────────────┘
"""

# ── Architecture (~700k params for vocab ~55) ──
block_size = 128       # covers session of 30 requests × 4 tokens + BOS/EOS
n_layer    = 4
n_head     = 4         # n_embd / n_head = 32
n_embd     = 128
dropout    = 0.20      # down from 0.30 to let carry-through routing stabilize
bias       = False

# ── Training schedule ──
# At batch 64 × block 128 = 8,192 token-examples per iter. 5,000 iters =
# ~41M token-examples. With ~5-7M train tokens that's 6-8 visits/token.
max_iters                   = 5_000
warmup_iters                = 300
lr_decay_iters              = 5_000
batch_size                  = 64
gradient_accumulation_steps = 1

# ── Optimizer ──
learning_rate = 3e-4
min_lr        = 3e-5
weight_decay  = 0.2
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# ── Eval / logging ──
eval_interval = 100    # catch val/train divergence early
eval_iters    = 50
log_interval  = 25
