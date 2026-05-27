"""Direct Logit Attribution per-layer (DLA).

For each block L, compute its direct additive contribution to the
logit for the correct next token. The decomposition is:

    final_residual = embed + Σ_L block_L_delta
    final_logit_for_correct_token = unembed(LN_final(final_residual))[correct]

Since LN_final is approximately linear, we can compute each component's
direct contribution as:

    contrib_L = unembed(LN_final_linearized(block_L_delta))[correct]

where LN_final_linearized uses the scaling factor computed from the
full final residual on the actual input (cached per position).

Output: per-layer mean ± std DLA across 5 seeds. Positive contribution
means that block pushes the correct-next-token logit up.

Usage:
    python eval/dla.py \\
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


@torch.no_grad()
def per_block_deltas(model, idx):
    """Return list of (B, T, n_embd) tensors:
        [embed_total, attn_L0_delta, mlp_L0_delta, attn_L1_delta, ..., final_residual]
    Each block contributes attn-output and mlp-output as separate residual deltas.
    """
    B, T = idx.shape
    tok_emb = model.transformer.wte(idx)
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_emb = model.transformer.wpe(pos)
    embed_total = model.transformer.drop(tok_emb + pos_emb)

    deltas = [embed_total.clone()]
    x = embed_total
    for block in model.transformer.h:
        # attn sub-layer
        attn_delta = block.attn(block.ln_1(x))
        deltas.append(attn_delta.clone())
        x = x + attn_delta
        # mlp sub-layer
        mlp_delta = block.mlp(block.ln_2(x))
        deltas.append(mlp_delta.clone())
        x = x + mlp_delta
    return deltas, x  # x is the final residual before final_LN


@torch.no_grad()
def dla_per_layer(model, idx, targets):
    """For each block sub-layer, compute its direct contribution to the
    correct-next-token logit, summed across positions."""
    deltas, final_res = per_block_deltas(model, idx)  # final_res: (B, T, n_embd)
    B, T, D = final_res.shape

    # Compute per-position LayerNorm scale from the final residual.
    # We treat LN as the linear operator x → (x − mean) * weight / std,
    # using the std computed from the *full* final residual.
    mean = final_res.mean(dim=-1, keepdim=True)
    var = final_res.var(dim=-1, keepdim=True, unbiased=False)
    eps = 1e-5
    inv_std = 1.0 / torch.sqrt(var + eps)  # (B, T, 1)
    ln_weight = model.transformer.ln_f.weight  # (n_embd,)

    # The unembed direction for each correct-next-token position.
    # lm_head.weight is (vocab, n_embd) — the row for target token t is the projection vector.
    # We need (B, T, n_embd) of unembed-directions.
    unembed_w = model.lm_head.weight  # (vocab, n_embd)
    target_dirs = unembed_w[targets]  # (B, T, n_embd)

    # Per-position DLA of each delta: project (delta − delta_mean) * inv_std * ln_weight onto target_dir.
    # Mean of delta along the n_embd axis is per-position; but LN_final's mean is taken over the FULL residual.
    # We use the full-residual mean and inv_std for linearization (standard logit-lens-style fold).
    contribs = []
    for delta in deltas:
        centered = delta - mean
        scaled = centered * inv_std * ln_weight  # (B, T, n_embd)
        # Project onto target direction
        contrib = (scaled * target_dirs).sum(dim=-1)  # (B, T)
        contribs.append(contrib.mean().item())
    return contribs


def sample_positions(stream, block_size, n_positions, rng):
    pos = rng.choice(len(stream) - block_size - 1, size=n_positions, replace=False)
    pos.sort()
    return pos


def run_seed(model, stream, n_positions, block_size, seed):
    rng = np.random.default_rng(seed)
    pos = sample_positions(stream, block_size, n_positions, rng)
    contribs_acc = None
    batch_size = 16
    for batch_start in range(0, len(pos), batch_size):
        batch_pos = pos[batch_start:batch_start + batch_size]
        idxs = torch.stack([stream[p:p + block_size] for p in batch_pos]).to(DEVICE)
        targets = torch.stack([stream[p + 1:p + block_size + 1] for p in batch_pos]).to(DEVICE)
        contribs = dla_per_layer(model, idxs, targets)
        if contribs_acc is None:
            contribs_acc = [c * len(batch_pos) for c in contribs]
        else:
            contribs_acc = [acc + c * len(batch_pos) for acc, c in zip(contribs_acc, contribs)]
    return [c / len(pos) for c in contribs_acc]


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

    all_contribs = defaultdict(list)
    for seed in args.seeds:
        contribs = run_seed(model, stream, args.n_positions, block_size, seed)
        for L, c in enumerate(contribs):
            all_contribs[L].append(c)
        print(f"  seed {seed}: " + "  ".join(f"c{L}={c:+.4f}" for L, c in enumerate(contribs)))

    # Labels: embed, then alternating attn_Li / mlp_Li for each block
    labels = ["embed"] + [f"{kind}_L{i}" for i in range(n_layer) for kind in ("attn", "mlp")]

    print()
    print("=" * 78)
    print(f"DIRECT LOGIT ATTRIBUTION — per-component direct contribution to correct logit")
    print(f"(mean ± std, {len(args.seeds)} seeds)")
    print("=" * 78)
    print(f"  ckpt: {args.ckpt}  block_size={block_size}  n_layer={n_layer}")
    print("  Component        Contribution (mean±std)")
    for L, label in enumerate(labels):
        vals = np.array(all_contribs[L])
        print(f"  {label:14s}   {vals.mean():+.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.4f}")

    print()
    means = [np.mean(all_contribs[L]) for L in range(len(labels))]
    peak_L = int(np.argmax(means))
    print(f"HEADLINE — largest direct contribution = {labels[peak_L]} ({means[peak_L]:+.4f})")


if __name__ == "__main__":
    main()
