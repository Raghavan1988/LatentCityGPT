"""Logit lens — apply the trained final LayerNorm + unembed to each
layer's residual stream, plot per-layer next-token accuracy.

For each block index L (0..n_layer), we take the residual stream AFTER
that block and project it through the final LN + lm_head, then measure
argmax-next-token accuracy on held-out positions.

The standard finding (Logit Lens, nostalgebraist 2020) is that
transformers often commit to the answer at an intermediate layer; later
layers refine confidence rather than change the answer.

Usage:
    python eval/logit_lens.py \\
        --ckpt checkpoints/best.pt \\
        --data_dir data/london_city \\
        --n_positions 5000 \\
        --seeds 0 1 2 3 4
"""
import argparse
import pickle
from collections import defaultdict
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


@torch.no_grad()
def residuals_per_layer(model, idx):
    """Return list of per-layer residual streams: [embed, after L0, ..., after L_{n-1}]."""
    B, T = idx.shape
    tok_emb = model.transformer.wte(idx)
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    residuals = [x.clone()]
    for block in model.transformer.h:
        x = block(x)
        residuals.append(x.clone())
    return residuals


@torch.no_grad()
def lens_accuracy(model, idx, targets):
    """For each layer's residual, apply final_LN + lm_head, return next-token accuracy."""
    residuals = residuals_per_layer(model, idx)  # [L+1] of (B, T, n_embd)
    accs = []
    for res in residuals:
        x = model.transformer.ln_f(res)
        logits = model.lm_head(x)  # (B, T, V)
        preds = logits.argmax(dim=-1)
        accs.append((preds == targets).float().mean().item())
    return accs


def run_seed(model, stream, n_positions, block_size, seed):
    rng = np.random.default_rng(seed)
    pos = sample_positions(stream, block_size, n_positions, rng)
    accs_acc = None
    batch_size = 16
    for batch_start in range(0, len(pos), batch_size):
        batch_pos = pos[batch_start:batch_start + batch_size]
        idxs = torch.stack([stream[p:p + block_size] for p in batch_pos]).to(DEVICE)
        targets = torch.stack([stream[p + 1:p + block_size + 1] for p in batch_pos]).to(DEVICE)
        accs = lens_accuracy(model, idxs, targets)
        if accs_acc is None:
            accs_acc = [a * len(batch_pos) for a in accs]
        else:
            accs_acc = [acc + a * len(batch_pos) for acc, a in zip(accs_acc, accs)]
    return [a / len(pos) for a in accs_acc]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--split", choices=("val", "gen"), default="gen")
    args = p.parse_args()

    model, ckpt = load_model(args.ckpt)
    stream = load_stream(args.data_dir, args.split)
    block_size = model.config.block_size
    n_layer = model.config.n_layer

    all_accs = defaultdict(list)
    for seed in args.seeds:
        accs = run_seed(model, stream, args.n_positions, block_size, seed)
        for L, a in enumerate(accs):
            all_accs[L].append(a)
        print(f"  seed {seed}: " + "  ".join(f"L{L}={a:.4f}" for L, a in enumerate(accs)))

    print()
    print("=" * 78)
    print(f"LOGIT LENS — next-token accuracy by layer (mean ± std, {len(args.seeds)} seeds)")
    print("=" * 78)
    print(f"  ckpt: {args.ckpt}  block_size={block_size}  n_layer={n_layer}")
    print("  Layer        Accuracy (mean±std)")
    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layer)]
    for L, label in enumerate(layer_labels):
        vals = np.array(all_accs[L])
        print(f"  {label:8s}     {vals.mean():.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.4f}")

    print()
    means = [np.mean(all_accs[L]) for L in range(n_layer + 1)]
    peak_L = int(np.argmax(means))
    print(f"HEADLINE — best lens layer = {layer_labels[peak_L]} (acc={means[peak_L]:.4f})")


if __name__ == "__main__":
    main()
