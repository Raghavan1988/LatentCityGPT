"""
LatentCityGPT — model architecture (decoder-only causal transformer).

WHAT THIS FILE DOES, IN ONE PICTURE
====================================

   model.forward(idx, targets=None) does this:

           idx : (B, T) integer tensor of intersection IDs (in [0, vocab_size))
            │
            ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  token_embedding (vocab_size × n_embd)   →  (B, T, n_embd)       │
   │                +                                                 │
   │  position_embedding (block_size × n_embd, learned)               │
   └──────────────────────────────────────────────────────────────────┘
            │
            ▼
   ┌── repeat × n_layer ──────────────────────────────────────────────┐
   │   residual stream                                                 │
   │      │                                                            │
   │      ├── LayerNorm → CausalSelfAttention ──┐  (mixes positions)   │
   │      │                                     │                      │
   │      +─────────────────────────────────────┘                      │
   │      │                                                            │
   │      ├── LayerNorm → MLP (4× expand, GELU) ─┐ (per-position xform)│
   │      │                                      │                     │
   │      +──────────────────────────────────────┘                     │
   └──────────────────────────────────────────────────────────────────┘
            │
            ▼
       LayerNorm
            │
            ▼
   lm_head (Linear n_embd → vocab_size)        # weight-tied with token_embedding
            │
            ▼
       logits : (B, T, vocab_size)
            │
            ▼ (only if targets given)
   cross_entropy(logits, targets, ignore_index=PAD)
            │
            ▼
       loss : scalar


THE ONE RULE — enforced here
============================
This file only ever consumes integer token IDs. There is no coords.csv loader,
no float input, nothing positional in the geographic sense. The only "position"
the model knows is sequence-index 0..block_size-1 (the standard transformer
position embedding, totally unrelated to lat/lon). We assert on dtype in
forward() and the data loader asserts on value range — together those make
"a coordinate accidentally entered the model" a loud, immediate failure rather
than a silent corruption of the result.


KEY DESIGN CHOICES (and why)
============================
- nanoGPT-style: minimal moving parts, every piece readable in one screen.
- Pre-norm (LayerNorm before each sub-layer): more stable than post-norm.
- Weight tying — lm_head shares its weight matrix with token_embedding. Saves
  vocab_size × n_embd parameters and tends to improve perplexity. For South Bay
  (vocab≈46k) at n_embd=384, that's ~17M parameters saved.
- PyTorch ≥2.0's `F.scaled_dot_product_attention`: fused softmax + masking,
  Flash Attention on CUDA, basically free speed.
- AdamW with weight decay applied to 2-D params only (the GPT-3 recipe): 1-D
  things (biases, LayerNorm gains) get no decay. This is implemented in
  configure_optimizers().
- PAD=0 is ignored in the loss via ignore_index. We don't currently emit PAD in
  the data pipeline, but masking is cheap insurance.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Reserved token indices. Must match data/prepare_city.py exactly; mismatching
# these is the kind of bug that silently produces nonsense without erroring.
PAD, BOS, EOS = 0, 1, 2


@dataclass
class GPTConfig:
    """All architecture hyperparameters in one place. Loaded from a config file
    by train.py. `vocab_size` is *not* hardcoded — train.py fills it in from
    `data/<city>/meta.pkl` so the model is sized for the specific city's graph.
    """
    block_size: int = 256       # max sequence length the model can see at once
    vocab_size: int = 50_000    # **set at runtime from meta.pkl**, never hardcoded
    n_layer: int = 6            # number of transformer blocks
    n_head: int = 6             # attention heads per block (must divide n_embd)
    n_embd: int = 384           # residual-stream / embedding width
    dropout: float = 0.1
    bias: bool = False          # nanoGPT default: no bias on Linear/LayerNorm (faster, no quality loss)


class LayerNorm(nn.Module):
    """LayerNorm with an option to disable bias. PyTorch's built-in always has
    bias; we want to match nanoGPT's bias=False default for the small win."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        # `weight` is the learnable gain γ; initialized to 1
        self.weight = nn.Parameter(torch.ones(ndim))
        # `bias` is the learnable shift β; optional
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        # F.layer_norm normalizes the last dim to zero-mean unit-var, then applies γ, β.
        # eps=1e-5 matches PyTorch's nn.LayerNorm default.
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    "Causal" means position t can only attend to positions ≤ t. This is what
    makes the model autoregressive (predicts the next token from past tokens
    alone — no peeking at the future).

    Picture of one head's computation (Q, K, V live in the same residual stream):

         x  (B, T, n_embd)
          │
          ├──→ Linear → Q  (B, T, head_dim)   "what am I looking for?"
          ├──→ Linear → K  (B, T, head_dim)   "what do I have to offer?"
          └──→ Linear → V  (B, T, head_dim)   "if you pick me, this is what you get"
                              │
                              ▼
                attn = softmax( (Q @ K^T) / sqrt(head_dim) )    ← masked: t can't see >t
                              │
                              ▼
                            attn @ V → (B, T, head_dim)

    All heads run in parallel, their outputs are concatenated, then a final
    Linear projects back to n_embd width.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Combined projection: x → [Q | K | V]. Single matmul is faster than three.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back to the residual stream's width.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Dropouts inside attention (on the softmax) and on the output.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # PyTorch ≥2.0 has a fused attention kernel. Use it when available; it
        # handles softmax + causal mask + dropout in one shot, and on CUDA uses
        # Flash Attention.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Fallback: pre-compute the lower-triangular mask we'll use manually.
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.shape  # batch, seq len, embedding dim
        head_dim = C // self.n_head

        # Project to Q, K, V and split.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # each (B, T, C)
        # Reshape so heads are an explicit dim: (B, n_head, T, head_dim).
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.flash:
            # Fused: softmax( Q K^T / √d ) V  with causal mask and dropout.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,   # apply the upper-triangular mask
            )
        else:
            # Manual path (older PyTorch). Same math, slower.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)        # (B, h, T, T)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v                                                  # (B, h, T, head_dim)

        # Merge heads: (B, h, T, head_dim) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Project back + dropout
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Per-position feedforward block. Expand 4×, GELU, project back.

    This is where most of the model's "compute" lives. Attention shuffles
    information across positions (what node attends to what), MLP transforms
    that information at each position independently.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd,     4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd,     bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One transformer block = pre-norm attention sub-layer + pre-norm MLP
    sub-layer, each wrapped in a residual connection.

       x ─→ LN → Attn ─→ + ─→ LN → MLP ─→ +  → next block
       │                ↑                ↑
       └────────────────┘                │
                         └───────────────┘

    Pre-norm (LN before the sub-layer, not after) is GPT-2/3 standard. It's
    more stable than post-norm and lets you train deeper models without
    learning-rate warmup-then-collapse cliffs.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # attention sub-layer (with residual)
        x = x + self.mlp(self.ln_2(x))     # MLP sub-layer (with residual)
        return x


class GPT(nn.Module):
    """The whole LatentCityGPT model. Forward signature:

        idx     : LongTensor (B, T), values in [0, vocab_size).
        targets : LongTensor (B, T) of next-token labels (optional).

    Returns:
        logits  : FloatTensor (B, T, vocab_size).
        loss    : scalar CE loss if targets given, else None.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # The whole transformer in one ModuleDict for clean state_dict naming.
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),    # token embedding
            wpe  = nn.Embedding(config.block_size, config.n_embd),    # position embedding (learned)
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))
        # Output head: n_embd → vocab_size logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ── Weight tying ──
        # Make lm_head.weight literally the same tensor as wte.weight.
        # Both projections are then (vocab_size, n_embd) and share parameters.
        # See Press & Wolf (2017); also used by GPT-2.
        self.lm_head.weight = self.transformer.wte.weight

        # Small-init for stable training (GPT-2 recipe).
        self.apply(self._init_weights)
        # Special scaled init on residual projections: 0.02 / √(2 · n_layer).
        # This keeps the residual stream from blowing up as depth grows.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Print a one-liner summary so anyone running training sees what they got.
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT: {n_params/1e6:.2f}M parameters  "
              f"(vocab_size={config.vocab_size:,}, n_layer={config.n_layer}, "
              f"n_head={config.n_head}, n_embd={config.n_embd}, block_size={config.block_size})")

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # ── THE ONE RULE check ──
        # The model must only ever see integer token IDs. Anything else is a bug.
        # (The value-range check happens in the data loader on every batch — see train.py.)
        assert idx.dtype in (torch.long, torch.int32, torch.int64), \
            f"idx must be integer tensor, got {idx.dtype}"

        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"

        # 1. Embed: tokens + positions, then dropout.
        tok_emb = self.transformer.wte(idx)                            # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.transformer.wpe(pos)                            # (T, n_embd) broadcasts over batch
        x = self.transformer.drop(tok_emb + pos_emb)                   # (B, T, n_embd)

        # 2. Transformer stack.
        for block in self.transformer.h:
            x = block(x)

        # 3. Final LayerNorm and LM head.
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                       # (B, T, vocab_size)

        # 4. If labels given, compute cross-entropy loss.
        loss = None
        if targets is not None:
            # F.cross_entropy expects flat (N, C) and (N,) inputs.
            # ignore_index=PAD ⇒ positions where target==PAD don't contribute to the loss.
            # BOS/EOS *do* contribute — the model should learn route boundaries.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD,
            )

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: tuple[float, float], device: str):
        """AdamW with the GPT-3 weight-decay recipe: decay only 2-D params.

        Biases and LayerNorm gains are 1-D — they shouldn't be decayed because
        decay there pulls them toward zero, which hurts (a LayerNorm gain of 0
        kills the signal). 2-D weights (Linear, Embedding) get the standard
        decay; this is empirically what GPT-3 / nanoGPT use.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() <  2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Use the fused AdamW kernel on CUDA if available (measurably faster).
        use_fused = (device == "cuda") and \
                    ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra = dict(fused=True) if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressive sampling. Given prefix `idx` of shape (1, T), produce
        up to max_new_tokens additional tokens. Stops early if EOS is sampled.

        Used by eval/valid_edge.py's full-route mode (where we ask: "given a
        BOS+start_node prompt, can the model walk a valid route through the city?").
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size from the right (the model can't see more).
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                          else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                  # focus on last position
            if top_k is not None:
                # Zero out everything but the top-k for sharper sampling.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if next_token.item() == EOS:
                break
        return idx
