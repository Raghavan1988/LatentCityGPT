"""Zero-ablation per-layer — set each block's output to zero and
measure the drop in next-token accuracy.

This is the destruction-side complement to activation patching: rather
than substituting a different value (transplant), we set the value to
zero and see how much that breaks the model.

For each block L, we replace the block's residual contribution with
zero (i.e., x_out = x_in for that block, skipping the block entirely)
and measure the next-token accuracy drop vs the unablated baseline.

We can ablate at two granularities:
  - block (default): zero the entire block delta (attn + mlp combined)
  - sub-layer: zero attn only, or mlp only

Usage:
    python eval/zero_ablation.py \\
        --ckpt checkpoints/best.pt \\
        --data_dir data/london_city \\
        --granularity block \\
        --n_positions 5000 \\
        --seeds 0 1 2 3 4
"""
import argparse
import pickle
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from model.model import GPT, GPTConfig

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_stream(data_dir, split="gen"):
    meta = pickle.loads((Path(data_dir) / "meta.pkl").read_bytes())
    dtype = np.uint32 if meta.get("dtype_str") == "uint32" else np.uint16
    arr = np.fromfile(Path(data_dir) / f"{split}.bin", dtype=dtype)
    return torch.from_numpy(arr.astype(np.int64))


def sample_positions(stream, block_size, n_positions, rng):
    pos = rng.choice(len(stream) - block_size - 1, size=n_positions, replace=False)
    pos.sort()
    return pos


@contextmanager
def block_zero_hook(model, block_idx, granularity):
    """Install a hook that zeroes the targeted block's residual contribution."""
    block = model.transformer.h[block_idx]
    handles = []

    if granularity == "block":
        # Replace block.forward with an identity (output = input).
        orig_forward = block.forward
        def identity_forward(x):
            return x
        block.forward = identity_forward
        try:
            yield
        finally:
            block.forward = orig_forward
    elif granularity == "attn_only":
        orig_attn = block.attn.forward
        block.attn.forward = lambda x: torch.zeros_like(x)
        try:
            yield
        finally:
            block.attn.forward = orig_attn
    elif granularity == "mlp_only":
        orig_mlp = block.mlp.forward
        block.mlp.forward = lambda x: torch.zeros_like(x)
        try:
            yield
        finally:
            block.mlp.forward = orig_mlp
    else:
        raise ValueError(f"unknown granularity: {granularity}")


@torch.no_grad()
def accuracy(model, idxs, targets):
    logits, _ = model(idxs)
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def run_seed(model, stream, n_positions, block_size, granularity, seed):
    rng = np.random.default_rng(seed)
    pos = sample_positions(stream, block_size, n_positions, rng)
    n_layer = model.config.n_layer

    # Pre-fetch all batches into a list (positions × tensors) so we can re-use across ablations.
    batches = []
    batch_size = 16
    for batch_start in range(0, len(pos), batch_size):
        batch_pos = pos[batch_start:batch_start + batch_size]
        idxs = torch.stack([stream[p:p + block_size] for p in batch_pos]).to(DEVICE)
        targets = torch.stack([stream[p + 1:p + block_size + 1] for p in batch_pos]).to(DEVICE)
        batches.append((idxs, targets, len(batch_pos)))

    def measure():
        acc_acc = 0
        n_acc = 0
        for idxs, targets, n in batches:
            acc_acc += accuracy(model, idxs, targets) * n
            n_acc += n
        return acc_acc / n_acc

    baseline_acc = measure()
    per_layer_acc = []
    for L in range(n_layer):
        with block_zero_hook(model, L, granularity):
            per_layer_acc.append(measure())
    return baseline_acc, per_layer_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--split", choices=("val", "gen"), default="gen")
    p.add_argument("--granularity", choices=("block", "attn_only", "mlp_only"),
                   default="block",
                   help="What to zero: the full block delta, attn only, or mlp only")
    args = p.parse_args()

    model, ckpt = load_model(args.ckpt)
    stream = load_stream(args.data_dir, args.split)
    block_size = model.config.block_size
    n_layer = model.config.n_layer

    all_baseline = []
    all_per_layer = defaultdict(list)
    for seed in args.seeds:
        baseline, per_layer = run_seed(model, stream, args.n_positions, block_size, args.granularity, seed)
        all_baseline.append(baseline)
        for L, acc in enumerate(per_layer):
            all_per_layer[L].append(acc)
        deltas = "  ".join(f"L{L}:{(per_layer[L] - baseline):+.4f}" for L in range(n_layer))
        print(f"  seed {seed}: baseline={baseline:.4f}  drops:  {deltas}")

    print()
    print("=" * 78)
    print(f"ZERO ABLATION — accuracy drop when block L is zeroed (granularity={args.granularity})")
    print(f"(mean ± std, {len(args.seeds)} seeds)")
    print("=" * 78)
    print(f"  ckpt: {args.ckpt}  block_size={block_size}  n_layer={n_layer}")
    baseline_mean = np.mean(all_baseline)
    baseline_std = np.std(all_baseline, ddof=1) if len(all_baseline) > 1 else 0
    print(f"  baseline accuracy: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print()
    print("  Block        Ablated acc           Drop (baseline − ablated)")
    for L in range(n_layer):
        vals = np.array(all_per_layer[L])
        drops = np.array(all_baseline) - vals
        print(f"  L{L:<10d} {vals.mean():.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.4f}    "
              f"{drops.mean():+.4f} ± {drops.std(ddof=1) if len(drops) > 1 else 0:.4f}")

    drops_per_layer = [np.mean(all_baseline) - np.mean(all_per_layer[L]) for L in range(n_layer)]
    peak_L = int(np.argmax(drops_per_layer))
    print()
    print(f"HEADLINE — most important block by ablation drop = L{peak_L} "
          f"(drop = {drops_per_layer[peak_L]:+.4f})")


if __name__ == "__main__":
    main()
