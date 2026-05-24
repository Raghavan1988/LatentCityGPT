"""
LatentCityGPT — training loop.

WHAT THIS FILE DOES, IN ONE PICTURE
====================================

  ┌──────────────────────────────────────────────────────────────────┐
  │  ONE TRAINING STEP                                               │
  └──────────────────────────────────────────────────────────────────┘

      train.bin (flat uint16/uint32 stream of token IDs)
            │
            │  sample B random windows of (block_size + 1) tokens each
            ▼
      window: [a, b, c, ..., z]      (length block_size + 1)
            │
            ├─→ x = window[:-1]      input sequence
            └─→ y = window[1:]       targets (shifted by one)
            │
            │  ONE-RULE check: assert 0 <= x.min() and x.max() < vocab_size
            ▼
      model(x, y) → logits, loss
            │
            │  loss = cross_entropy(logits, y, ignore_index=PAD)
            ▼
      loss.backward()  →  grad-clip  →  optimizer.step()  →  cosine LR

  ┌──────────────────────────────────────────────────────────────────┐
  │  EVERY `eval_interval` ITERATIONS                                │
  └──────────────────────────────────────────────────────────────────┘

      for split in {train, val, gen}:
          mean_CE = average cross-entropy over `eval_iters` batches
          perplexity = exp(mean_CE)
      print one line of all six numbers
      if val_CE improved → save checkpoint to checkpoints/best.pt

      The checkpoint stores: model weights, GPTConfig, vocab_size, iter, val
      metrics, and the data_dir. Eval scripts (eval/valid_edge.py, eval/probe.py)
      reconstruct the model from this alone.


WHAT WE OPTIMIZE vs WHAT WE WATCH
==================================
  Optimizes:   cross-entropy on train.bin (the only gradient signal).

  Watches:
    - train CE/perplexity        — sanity, should drop.
    - val CE/perplexity          — overfitting check.
    - gen CE/perplexity          — generalization (routes to held-out destinations).

  Does NOT watch directly here:
    - valid-edge rate            — see eval/valid_edge.py.
    - linear-probe R²            — see eval/probe.py (Phase 4).


USAGE
=====
    # Smoke run on the smallest city.
    python model/train.py --config model/configs/small.py --data_dir data/london_city

    # Manhattan with the small config, custom seed.
    python model/train.py --config model/configs/small.py --data_dir data/manhattan --seed 7

    # The eventual full-scale run.
    python model/train.py --config model/configs/medium.py --data_dir data/manhattan


DEVICE / PRECISION NOTES
========================
Device priority: CUDA > MPS (Apple Silicon) > CPU. AMP (mixed-precision) only
runs on CUDA; MPS uses fp32 (autocast on MPS is partial and not worth the
debugging cost yet). CPU runs work but are *slow* and only useful for sanity
checks on the smallest city.
"""

import argparse
import importlib.util
import math
import pickle
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig, PAD  # noqa: F401 — PAD imported for reference/clarity


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Import a config .py file as a module and return its module-level
    variables as a plain dict. We use importlib (not exec) so the config can
    legitimately do imports of its own if it ever needs to."""
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: v for k, v in vars(mod).items() if not k.startswith("_") and not callable(v)}


def pick_device() -> str:
    """CUDA > MPS > CPU. Print-friendly string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    """Best-effort determinism: Python, numpy, torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch_sampler(stream: np.memmap, block_size: int, batch_size: int,
                       device: str, vocab_size: int):
    """Closure that, when called, samples one batch from `stream`.

    What it produces:
      x : LongTensor (batch_size, block_size)  — input windows
      y : LongTensor (batch_size, block_size)  — same windows shifted by 1 (targets)

    The windows are drawn from independent uniform random positions; they may
    overlap and may cross route boundaries (BOS/EOS in the middle). That's how
    nanoGPT trains — the model learns the BOS/EOS structure too.
    """
    def get_batch():
        # Random starting indices. We need block_size+1 tokens to cut both x and y.
        ix = torch.randint(0, len(stream) - block_size - 1, (batch_size,))
        # Build the windows. .tolist() because indexing np.memmap with a tensor int
        # is finicky; plain Python ints are robust.
        x = torch.from_numpy(np.stack([stream[i     : i     + block_size] for i in ix.tolist()])).long()
        y = torch.from_numpy(np.stack([stream[i + 1 : i + 1 + block_size] for i in ix.tolist()])).long()

        # ── THE ONE RULE check on every batch. ──
        # If anything outside [0, vocab_size) ever shows up, it means either the
        # data pipeline or the loader corrupted the stream. Loud failure is what
        # we want.
        assert int(x.min()) >= 0 and int(x.max()) < vocab_size, \
            f"out-of-vocab token in batch: min={int(x.min())} max={int(x.max())} vocab_size={vocab_size}"

        # Move to device. pin_memory + non_blocking lets CUDA overlap H2D copy
        # with compute on the GPU.
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y
    return get_batch


def lr_at_iter(it: int, warmup_iters: int, lr_decay_iters: int,
               learning_rate: float, min_lr: float) -> float:
    """Schedule: linear warmup 0 → peak, then cosine peak → min_lr.

      ▲ lr
      │       _____________
      │      /             \\__
      │     /                 \\__
      │    /                     \\__
      │   /                         \\___    ← cosine
      │  /
      │ /     ← linear warmup
      └──────────────────────────────────────▶ iter
        0  warmup_iters         lr_decay_iters
    """
    # 1) Linear warmup.
    if it < warmup_iters:
        # +1's avoid lr=0 at iter 0 (zero-step optimization is a waste).
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) Cosine decay floor — past lr_decay_iters we stay at min_lr.
    if it > lr_decay_iters:
        return min_lr
    # 3) Cosine in between.
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, batch_fns: dict, eval_iters: int) -> dict[str, float]:
    """For each split, average cross-entropy over `eval_iters` batches.
    Returns dict mapping split -> mean CE (a python float)."""
    out = {}
    was_training = model.training
    model.eval()
    for split, get_batch in batch_fns.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch()
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to a model/configs/*.py")
    p.add_argument("--data_dir", required=True, help="path to data/<city>/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="checkpoints",
                   help="directory for checkpoints (best.pt). Created if missing.")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    # ── Load dataset metadata + streams ──
    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    dtype = np.dtype(meta["dtype"])
    print(f"data: {data_dir}  vocab_size={vocab_size:,}  dtype={dtype.name}")

    # memmap so we don't have to load the whole .bin into RAM. For a 100MB file
    # that doesn't matter; for the eventual full-scale corpus it might.
    streams = {split: np.memmap(data_dir / f"{split}.bin", dtype=dtype, mode="r")
               for split in ("train", "val", "gen")}
    for split, arr in streams.items():
        print(f"  {split}.bin: {len(arr):,} tokens")

    # ── Build model sized for this city ──
    config = GPTConfig(
        block_size=cfg["block_size"],
        vocab_size=vocab_size,           # ← from meta.pkl, never hardcoded
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
        bias=cfg["bias"],
    )
    model = GPT(config).to(device)

    # ── Optimizer + (optional) AMP ──
    optimizer = model.configure_optimizers(
        weight_decay=cfg["weight_decay"],
        learning_rate=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        device=device,
    )

    use_amp = (device == "cuda")
    if use_amp:
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        scaler  = torch.amp.GradScaler()
    else:
        amp_ctx = nullcontext()
        scaler  = None

    # ── Batch samplers per split ──
    batch_fns = {
        split: make_batch_sampler(streams[split], cfg["block_size"],
                                  cfg["batch_size"], device, vocab_size)
        for split in ("train", "val", "gen")
    }

    # ── Output dir for checkpoints ──
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    best_val_loss = float("inf")

    # ── Training loop ──
    print(f"\n── training: {cfg['max_iters']:,} iterations ──")
    t0 = time.time()
    accum = cfg.get("gradient_accumulation_steps", 1)

    for it in range(cfg["max_iters"]):
        # Set learning rate per the schedule.
        lr = lr_at_iter(it, cfg["warmup_iters"], cfg["lr_decay_iters"],
                        cfg["learning_rate"], cfg["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward, with optional gradient accumulation.
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for micro in range(accum):
            x, y = batch_fns["train"]()
            with amp_ctx:
                _, loss = model(x, y)
                # When accumulating, average the loss across micro-batches so the
                # effective gradient matches batch_size × accum.
                loss = loss / accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_value += loss.item() * accum  # report unscaled

        # Grad clip + step.
        if use_amp:
            if cfg["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

        # ── Log + eval ──
        if it % cfg["log_interval"] == 0:
            elapsed = time.time() - t0
            print(f"iter {it:>6d}  train_loss={loss_value:.4f}  "
                  f"lr={lr:.2e}  t={elapsed:.1f}s")

        if it > 0 and it % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, batch_fns, cfg["eval_iters"])
            ppl = {s: math.exp(l) for s, l in losses.items()}
            print(f"  eval @ {it:>6d}  "
                  f"train CE={losses['train']:.4f} ppl={ppl['train']:7.2f}  |  "
                  f"val   CE={losses['val']:.4f} ppl={ppl['val']:7.2f}  |  "
                  f"gen   CE={losses['gen']:.4f} ppl={ppl['gen']:7.2f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                # Save everything the eval/probe/causal scripts need to rebuild
                # the model from this file alone.
                torch.save({
                    "model_state":    model.state_dict(),
                    "config":         asdict(config),
                    "vocab_size":     vocab_size,
                    "iter":           it,
                    "val_loss":       losses["val"],
                    "val_perplexity": ppl["val"],
                    "data_dir":       str(data_dir),
                }, ckpt_path)
                print(f"  → saved checkpoint to {ckpt_path}  (val_ppl={ppl['val']:.2f})")

    # ── Final eval after the last iteration ──
    losses = estimate_loss(model, batch_fns, cfg["eval_iters"])
    ppl = {s: math.exp(l) for s, l in losses.items()}
    print("\n── final ──")
    print(f"train ppl={ppl['train']:.2f}   val ppl={ppl['val']:.2f}   gen ppl={ppl['gen']:.2f}")
    print(f"best val loss (saved): {best_val_loss:.4f}  ppl={math.exp(best_val_loss):.2f}")
    print(f"checkpoint: {ckpt_path}")
    print(f"\nNext: python eval/valid_edge.py --ckpt {ckpt_path} --data_dir {data_dir} --split val")


if __name__ == "__main__":
    main()
