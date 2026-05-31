"""
LatentWorldsGPT — HTTP within-position (single-k) probe for Feature B.

Design A in the position-confound follow-up. Restricts the probe to a
single fixed session position k (default k=5). At fixed k, every probe
example has identical position-bin / positional embedding contribution,
so the trained-vs-untrained gap cannot be attributed to the model
encoding position.

Target: cumulative_large_response_binned (Feature B from the locked
predictions). Class-balanced sampling because Feature B at k=5 has a
~47%/20%/33% split.

If trained MLP at any layer is close to untrained MLP at that layer
(say, within ~0.03), the framework's null prediction for Feature B is
**confirmed under position-controlled probing**. If it lands clearly
above (>0.10), the null is **falsified at fixed k** — the model encodes
aggregation beyond what positional encoding contributes.

This is a follow-up control to the locked predictions experiment, not
a pre-registered prediction.

Usage:
    python eval/probe_http_within_position.py \\
        --ckpt checkpoints/http_real/best.pt \\
        --data_dir data/nasa_http \\
        --seeds 0 1 2 3 4 \\
        --fixed_k 5
"""
import argparse
import csv
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

PAD, BOS, EOS = 0, 1, 2


def load_targets_at_k(data_dir, fixed_k, splits=("val", "gen")):
    """Returns Feature B labels and session_ids at sz-slot probe positions
    with request_idx_in_session == fixed_k."""
    targets = {}
    session_of = {}
    with open(data_dir / "http_state.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] not in splits: continue
            if row["field_type"] != "sz": continue
            if int(row["request_idx_in_session"]) != fixed_k: continue
            key = (row["split"], int(row["token_pos"]))
            targets[key] = int(row["cumulative_large_response_binned"])
            session_of[key] = int(row["session_id"])
    return targets, session_of


@torch.no_grad()
def build_dataset(model, streams, sampled_keys, target_dict, session_of,
                  block_size, device):
    model.eval()
    split_order = list(streams.keys())
    offsets, parts, cursor = {}, [], 0
    for s in split_order:
        offsets[s] = cursor
        parts.append(streams[s])
        cursor += len(streams[s])
    combined = np.concatenate(parts).astype(np.int64)

    n_layer = model.config.n_layer
    X_layers = [[] for _ in range(n_layer + 1)]
    y_list = []
    sess_list = []
    batch_size = 16
    batch_keys = []
    for key in sampled_keys:
        batch_keys.append(key)
        if len(batch_keys) >= batch_size:
            _process(model, batch_keys, combined, offsets, block_size, device,
                     target_dict, session_of, X_layers, y_list, sess_list)
            batch_keys = []
    if batch_keys:
        _process(model, batch_keys, combined, offsets, block_size, device,
                 target_dict, session_of, X_layers, y_list, sess_list)
    X = np.stack([np.concatenate(X_layers[L], axis=0) for L in range(n_layer + 1)], axis=0)
    return X, np.array(y_list, dtype=np.int64), np.array(sess_list, dtype=np.int64)


def _process(model, batch_keys, combined, offsets, block_size, device,
             target_dict, session_of, X_layers, y_list, sess_list):
    batch_idxs = []
    for split, pos in batch_keys:
        gp = offsets[split] + pos
        ctx_start = max(0, gp - block_size + 1)
        ctx = combined[ctx_start:gp + 1]
        if len(ctx) < block_size:
            ctx = np.concatenate([np.full(block_size - len(ctx), PAD), ctx])
        batch_idxs.append(ctx)
    idx_tensor = torch.from_numpy(np.stack(batch_idxs)).to(device)
    layer_acts = cache_layer_activations(model, idx_tensor)
    for L, act in enumerate(layer_acts):
        X_layers[L].append(act[:, -1, :].cpu().numpy())
    for split, pos in batch_keys:
        y_list.append(target_dict[(split, pos)])
        sess_list.append(session_of[(split, pos)])


class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.lin = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.lin(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_classes))
    def forward(self, x): return self.net(x)


def train_eval(probe, Xtr, ytr, Xte, yte, device, epochs=30, lr=3e-3):
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    yte_t = torch.from_numpy(yte).long().to(device)
    n = len(Xtr_t); batch = min(256, n)
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            ix = idx[s:s + batch]
            logits = probe(Xtr_t[ix])
            loss = F.cross_entropy(logits, ytr_t[ix])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        preds = probe(Xte_t).argmax(dim=-1)
        return (preds == yte_t).float().mean().item()


def position_split(n, train_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(train_frac * n)
    return perm[:ntr], perm[ntr:]


def session_split(sess_ids, train_frac, seed):
    unique = np.unique(sess_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_train = int(train_frac * len(unique))
    train_sess = set(perm[:n_train].tolist())
    train_ix = np.array([i for i, s in enumerate(sess_ids) if s in train_sess])
    test_ix = np.array([i for i, s in enumerate(sess_ids) if s not in train_sess])
    return train_ix, test_ix


def class_balanced_sample(targets, per_class_n, seed):
    """Returns class-balanced list of keys."""
    by_class = defaultdict(list)
    for k, c in targets.items():
        by_class[c].append(k)
    rng = np.random.default_rng(seed)
    sampled = []
    for c, ks in by_class.items():
        rng.shuffle(ks)
        sampled.extend(ks[:per_class_n])
    rng.shuffle(sampled)
    return sampled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fixed_k", type=int, default=5,
                   help="Session position to restrict probe data to.")
    p.add_argument("--per_class_n", type=int, default=3000,
                   help="Positions per Feature B class (×3 classes = total).")
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    print(f"fixed_k: {args.fixed_k}")

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device); trained.load_state_dict(ckpt["model_state"])
    trained.eval()

    print(f"\nLoading Feature B targets at k={args.fixed_k}...")
    targets, session_of = load_targets_at_k(data_dir, args.fixed_k, splits=("val", "gen"))
    print(f"  probe positions at k={args.fixed_k}: {len(session_of):,}")
    c = Counter(targets.values())
    total = sum(c.values())
    print(f"  Feature B distribution: {dict(sorted(c.items()))}")
    print(f"  majority class fraction: {max(c.values())/total:.4f}")

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {s: np.asarray(np.memmap(data_dir / f"{s}.bin", dtype=dtype, mode="r"))
               for s in ("val", "gen")}

    all_results = {
        "trained_pos": defaultdict(list),
        "trained_session": defaultdict(list),
        "untrained_pos": defaultdict(list),
        "untrained_session": defaultdict(list),
    }

    n_classes = max(targets.values()) + 1

    for seed in args.seeds:
        print(f"\n# SEED {seed}")
        torch.manual_seed(seed); np.random.seed(seed)
        untrained = GPT(config).to(device).eval()

        sampled = class_balanced_sample(targets, args.per_class_n, seed)
        print(f"  class-balanced sample: {len(sampled):,} positions "
              f"({args.per_class_n} per class × {n_classes} classes)")

        print(f"  building TRAINED dataset...")
        t0 = time.time()
        X_t, y_t, sess_t = build_dataset(
            trained, streams, sampled, targets, session_of,
            config.block_size, device)
        print(f"  {X_t.shape[1]:,} positions × {X_t.shape[0]} layers ({time.time()-t0:.1f}s)")

        print(f"  building UNTRAINED dataset...")
        t0 = time.time()
        X_u, y_u, sess_u = build_dataset(
            untrained, streams, sampled, targets, session_of,
            config.block_size, device)
        print(f"  {X_u.shape[1]:,} positions × {X_u.shape[0]} layers ({time.time()-t0:.1f}s)")

        def sweep(X, y, sess, label, cond_key):
            pos_tr, pos_te = position_split(len(y), args.probe_train_frac, seed)
            ses_tr, ses_te = session_split(sess, args.probe_train_frac, seed)
            for split_name, tr_ix, te_ix in [("pos", pos_tr, pos_te),
                                              ("session", ses_tr, ses_te)]:
                if len(tr_ix) == 0 or len(te_ix) == 0:
                    continue
                print(f"    {label} {split_name}-level (train n={len(tr_ix)}, test n={len(te_ix)})")
                print(f"      Layer       Lin      MLP")
                for L in range(X.shape[0]):
                    Xtr, ytr = X[L, tr_ix], y[tr_ix]
                    Xte, yte = X[L, te_ix], y[te_ix]
                    lin = LinearProbe(Xtr.shape[1], n_classes)
                    mlp = MLPProbe(Xtr.shape[1], n_classes)
                    lin_acc = train_eval(lin, Xtr, ytr, Xte, yte, device, epochs=args.epochs)
                    mlp_acc = train_eval(mlp, Xtr, ytr, Xte, yte, device, epochs=args.epochs)
                    lab = "embed" if L == 0 else f"L{L-1}"
                    print(f"      {lab:<8}  {lin_acc:.4f}   {mlp_acc:.4f}")
                    all_results[f"{cond_key}_{split_name}"][L].append((lin_acc, mlp_acc))

        sweep(X_t, y_t, sess_t, "TRAINED", "trained")
        sweep(X_u, y_u, sess_u, "UNTRAINED", "untrained")

    print(f"\n{'=' * 78}\nAGGREGATE — mean ± std over {len(args.seeds)} seeds\n{'=' * 78}")
    for cond, by_layer in all_results.items():
        if not by_layer: continue
        print(f"\n  {cond}")
        print(f"    Layer       LinAcc(mean±std)     MLPAcc(mean±std)")
        for L in sorted(by_layer.keys()):
            lins = np.array([r[0] for r in by_layer[L]])
            mlps = np.array([r[1] for r in by_layer[L]])
            lab = "embed" if L == 0 else f"L{L-1}"
            print(f"    {lab:<8}    {lins.mean():.4f} ± {lins.std(ddof=1):.4f}      {mlps.mean():.4f} ± {mlps.std(ddof=1):.4f}")

    print(f"\n{'=' * 78}\nHEADLINE — Feature B at fixed k={args.fixed_k}\n{'=' * 78}")
    for cond in ("trained_pos", "trained_session", "untrained_pos", "untrained_session"):
        results = all_results.get(cond, {})
        if not results: continue
        lin_means = {L: np.mean([r[0] for r in results[L]]) for L in results}
        mlp_means = {L: np.mean([r[1] for r in results[L]]) for L in results}
        best_lin = max(lin_means, key=lin_means.get)
        best_mlp = max(mlp_means, key=mlp_means.get)
        lin_s = np.std([r[0] for r in results[best_lin]], ddof=1)
        mlp_s = np.std([r[1] for r in results[best_mlp]], ddof=1)
        lab_lin = "embed" if best_lin == 0 else f"L{best_lin-1}"
        lab_mlp = "embed" if best_mlp == 0 else f"L{best_mlp-1}"
        print(f"  {cond:<22} linear best {lab_lin}: {lin_means[best_lin]:.4f} ± {lin_s:.4f}")
        print(f"  {cond:<22} MLP    best {lab_mlp}: {mlp_means[best_mlp]:.4f} ± {mlp_s:.4f}")


if __name__ == "__main__":
    main()
