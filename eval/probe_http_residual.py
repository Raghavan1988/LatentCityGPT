"""
LatentWorldsGPT — HTTP residual-after-position probe for Feature B.

Design B3 in the position-confound follow-up. Tests the framework's
null claim on Feature B (cumulative_large_response_binned) after
controlling for session-position structure across all probe positions.

Procedure
---------
1. Load Feature B labels across all sz-slot probe positions
   (request_idx_in_session >= 1).
2. For each seed × split:
   (a) Pool a sample of positions (proportional across k).
   (b) Split into probe-train / probe-test (position-level + session-level).
   (c) Compute per-position-k baseline P(class | k) from PROBE-TRAIN
       data only (Laplace-smoothed, alpha=1). This is the "position
       baseline" that captures everything position-only can predict.
   (d) Compute residual: y_onehot - baseline(k), for both train and
       test, using the same train-derived baseline.
   (e) Train a regression probe (Linear or MLP, MSE loss) to predict
       the 3-dim residual from layer activations.
   (f) Compute R² on test residual.
3. Compare trained-vs-untrained R² per layer.

Interpretation
--------------
- Untrained-model R² is expected ≈ 0 (random projections can't predict
  a content-defined residual).
- If trained R² ≈ untrained R² → activations contain no Feature B
  signal beyond position → **null confirmed** under position-statistical
  control.
- If trained R² > untrained R² → activations encode Feature B beyond
  what position predicts → **null falsified** under statistical control.

This is the statistical-control complement to Design A (single fixed
k, literal control). Both look at the same null claim from different
angles; agreement = high confidence in the verdict.

Usage:
    python eval/probe_http_residual.py \\
        --ckpt checkpoints/http_real/best.pt \\
        --data_dir data/nasa_http \\
        --seeds 0 1 2 3 4
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
N_CLASSES_B = 3


def load_targets(data_dir, splits=("val", "gen")):
    """Returns Feature B labels, position-k labels, session_ids at all
    sz-slot probe positions."""
    targets = {}
    pos_k = {}
    session_of = {}
    with open(data_dir / "http_state.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] not in splits: continue
            if row["field_type"] != "sz": continue
            req_idx = int(row["request_idx_in_session"])
            if req_idx < 1: continue
            key = (row["split"], int(row["token_pos"]))
            targets[key] = int(row["cumulative_large_response_binned"])
            pos_k[key] = req_idx
            session_of[key] = int(row["session_id"])
    return targets, pos_k, session_of


@torch.no_grad()
def build_dataset(model, streams, sampled_keys, target_dict, pos_k, session_of,
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
    k_list = []
    sess_list = []
    batch_size = 16
    batch_keys = []
    for key in sampled_keys:
        batch_keys.append(key)
        if len(batch_keys) >= batch_size:
            _process(model, batch_keys, combined, offsets, block_size, device,
                     target_dict, pos_k, session_of, X_layers, y_list, k_list, sess_list)
            batch_keys = []
    if batch_keys:
        _process(model, batch_keys, combined, offsets, block_size, device,
                 target_dict, pos_k, session_of, X_layers, y_list, k_list, sess_list)
    X = np.stack([np.concatenate(X_layers[L], axis=0) for L in range(n_layer + 1)], axis=0)
    return (X, np.array(y_list, dtype=np.int64),
            np.array(k_list, dtype=np.int64),
            np.array(sess_list, dtype=np.int64))


def _process(model, batch_keys, combined, offsets, block_size, device,
             target_dict, pos_k, session_of, X_layers, y_list, k_list, sess_list):
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
        k_list.append(pos_k[(split, pos)])
        sess_list.append(session_of[(split, pos)])


def position_baseline(y_train, k_train, n_classes, alpha=1.0):
    """Returns dict {k: probability vector of shape (n_classes,)} computed
    on probe-training data only, Laplace-smoothed with alpha."""
    counts = defaultdict(lambda: np.full(n_classes, alpha, dtype=np.float64))
    for y, k in zip(y_train, k_train):
        counts[k][y] += 1.0
    baseline = {k: c / c.sum() for k, c in counts.items()}
    return baseline


def apply_baseline(y, k, baseline, n_classes):
    """Returns residual array of shape (n, n_classes) = y_onehot - baseline(k).
    Falls back to uniform if k unseen in training."""
    uniform = np.full(n_classes, 1.0 / n_classes)
    residuals = np.zeros((len(y), n_classes), dtype=np.float32)
    for i, (yi, ki) in enumerate(zip(y, k)):
        oh = np.zeros(n_classes, dtype=np.float64)
        oh[yi] = 1.0
        bl = baseline.get(int(ki), uniform)
        residuals[i] = oh - bl
    return residuals


class LinearRegProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.lin(x)


class MLPRegProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)


def r2_score(pred, true):
    """Coefficient of determination across all dims jointly."""
    ss_res = np.mean(np.sum((pred - true) ** 2, axis=1))
    mean_true = true.mean(axis=0, keepdims=True)
    ss_tot = np.mean(np.sum((true - mean_true) ** 2, axis=1))
    if ss_tot < 1e-12: return 0.0
    return 1.0 - ss_res / ss_tot


def train_eval_reg(probe, Xtr, rtr, Xte, rte, device, epochs=30, lr=3e-3):
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    rtr_t = torch.from_numpy(rtr).float().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    rte_arr = rte
    n = len(Xtr_t); batch = min(256, n)
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            ix = idx[s:s + batch]
            pred = probe(Xtr_t[ix])
            loss = F.mse_loss(pred, rtr_t[ix])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred_te = probe(Xte_t).cpu().numpy()
    return r2_score(pred_te, rte_arr)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=10_000)
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device); trained.load_state_dict(ckpt["model_state"])
    trained.eval()

    print("\nLoading Feature B targets across all k...")
    targets, pos_k, session_of = load_targets(data_dir, splits=("val", "gen"))
    print(f"  probe positions: {len(session_of):,}")
    fb_dist = Counter(targets.values())
    print(f"  Feature B distribution: {dict(sorted(fb_dist.items()))}")
    k_dist = Counter(pos_k.values())
    print(f"  position-k distribution (top 10): "
          f"{dict(sorted(k_dist.items())[:10])} ...")

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
    import random as _rnd
    keys_and_labels = list(targets.items())

    for seed in args.seeds:
        print(f"\n# SEED {seed}")
        torch.manual_seed(seed); np.random.seed(seed)
        untrained = GPT(config).to(device).eval()

        _rng = _rnd.Random(seed)
        _rng.shuffle(keys_and_labels)
        sampled = [k for k, _ in keys_and_labels[:args.n_positions]]

        print(f"  building TRAINED dataset...")
        t0 = time.time()
        X_t, y_t, k_t, sess_t = build_dataset(
            trained, streams, sampled, targets, pos_k, session_of,
            config.block_size, device)
        print(f"  {X_t.shape[1]:,} positions × {X_t.shape[0]} layers ({time.time()-t0:.1f}s)")

        print(f"  building UNTRAINED dataset...")
        t0 = time.time()
        X_u, y_u, k_u, sess_u = build_dataset(
            untrained, streams, sampled, targets, pos_k, session_of,
            config.block_size, device)
        print(f"  {X_u.shape[1]:,} positions × {X_u.shape[0]} layers ({time.time()-t0:.1f}s)")

        def sweep(X, y, k_arr, sess, label, cond_key):
            pos_tr, pos_te = position_split(len(y), args.probe_train_frac, seed)
            ses_tr, ses_te = session_split(sess, args.probe_train_frac, seed)
            for split_name, tr_ix, te_ix in [("pos", pos_tr, pos_te),
                                              ("session", ses_tr, ses_te)]:
                if len(tr_ix) == 0 or len(te_ix) == 0:
                    continue
                # Compute baseline on PROBE-TRAIN data only
                baseline = position_baseline(y[tr_ix], k_arr[tr_ix], N_CLASSES_B)
                rtr = apply_baseline(y[tr_ix], k_arr[tr_ix], baseline, N_CLASSES_B)
                rte = apply_baseline(y[te_ix], k_arr[te_ix], baseline, N_CLASSES_B)

                # Quick sanity: per-position baseline strength = 1 - var(residual)/var(y_oh)
                y_oh_te = np.zeros((len(te_ix), N_CLASSES_B), dtype=np.float32)
                for i, yi in enumerate(y[te_ix]):
                    y_oh_te[i, yi] = 1.0
                baseline_pred_te = y_oh_te - rte
                baseline_r2 = r2_score(baseline_pred_te, y_oh_te)
                print(f"    {label} {split_name}-level (n_train={len(tr_ix)}, n_test={len(te_ix)})")
                print(f"      position-only baseline R² on y_oh: {baseline_r2:.4f}")
                print(f"      Layer       Lin R²     MLP R²")
                for L in range(X.shape[0]):
                    Xtr = X[L, tr_ix]; Xte = X[L, te_ix]
                    lin = LinearRegProbe(Xtr.shape[1], N_CLASSES_B)
                    mlp = MLPRegProbe(Xtr.shape[1], N_CLASSES_B)
                    lin_r2 = train_eval_reg(lin, Xtr, rtr, Xte, rte, device, epochs=args.epochs)
                    mlp_r2 = train_eval_reg(mlp, Xtr, rtr, Xte, rte, device, epochs=args.epochs)
                    lab = "embed" if L == 0 else f"L{L-1}"
                    print(f"      {lab:<8}  {lin_r2:+.4f}   {mlp_r2:+.4f}")
                    all_results[f"{cond_key}_{split_name}"][L].append((lin_r2, mlp_r2))

        sweep(X_t, y_t, k_t, sess_t, "TRAINED", "trained")
        sweep(X_u, y_u, k_u, sess_u, "UNTRAINED", "untrained")

    print(f"\n{'=' * 78}\nAGGREGATE — mean ± std over {len(args.seeds)} seeds\n{'=' * 78}")
    for cond, by_layer in all_results.items():
        if not by_layer: continue
        print(f"\n  {cond}")
        print(f"    Layer       LinR²(mean±std)        MLPR²(mean±std)")
        for L in sorted(by_layer.keys()):
            lins = np.array([r[0] for r in by_layer[L]])
            mlps = np.array([r[1] for r in by_layer[L]])
            lab = "embed" if L == 0 else f"L{L-1}"
            print(f"    {lab:<8}    {lins.mean():+.4f} ± {lins.std(ddof=1):.4f}     {mlps.mean():+.4f} ± {mlps.std(ddof=1):.4f}")

    print(f"\n{'=' * 78}\nHEADLINE — Feature B residual-after-position\n{'=' * 78}")
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
        print(f"  {cond:<22} linear best {lab_lin}: R²={lin_means[best_lin]:+.4f} ± {lin_s:.4f}")
        print(f"  {cond:<22} MLP    best {lab_mlp}: R²={mlp_means[best_mlp]:+.4f} ± {mlp_s:.4f}")


if __name__ == "__main__":
    main()
