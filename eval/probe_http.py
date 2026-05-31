"""
LatentWorldsGPT — HTTP log sequences probe.

Tests the two pre-registered features from
`predictions/predictions_http_log_sequences.md`:

  1. Feature A: `first_request_size_bin` (carry-through prediction)
     — uses CLASS-BALANCED sampling at the probe level to avoid
     majority-class saturation. The locked predictions file does not
     specify sampling strategy; we use balanced sampling so the
     trained-vs-untrained gap is the meaningful metric.

  2. Feature B: `cumulative_large_response_binned` (null prediction)
     — uses regular (proportional) sampling; the 3-class distribution
     {0, 1, 2+} is already reasonably balanced (30/24/46 empirically).

Probe positions are restricted to `field_type == 'sz'` (size_bin
slots) of request_idx >= 1, as specified in the locked predictions
file.

Multi-seed protocol: outermost loop varies untrained init, activation
sampling, and probe-training RNG together (matches probe_maze.py).

Two splits per feature:
  - position-level: random partition of probe positions (weak baseline)
  - session-level: partition sessions into disjoint sets — the honest
    test of representation generalization.

Usage:
    python eval/probe_http.py \\
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

# Per locked predictions: probe at sz-field-type positions of req_idx >= 1
TARGET_NAMES = ("feature_a_first_request_size_bin",
                "feature_b_cumulative_large_response_binned")


# ─────────────────────────────────────────────────────────────────────
# 1. Load targets from http_state.csv
# ─────────────────────────────────────────────────────────────────────

def load_targets(data_dir: Path, splits=("val", "gen")):
    """Returns:
       targets[target_name][(split, token_pos)] = label_int
       session_idx_of[(split, token_pos)] = session_id
       (Only probe positions are included: field_type == 'sz' and
       request_idx_in_session >= 1.)
    """
    targets = {t: {} for t in TARGET_NAMES}
    session_of = {}
    with open(data_dir / "http_state.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] not in splits:
                continue
            if row["field_type"] != "sz":
                continue
            if int(row["request_idx_in_session"]) < 1:
                continue
            key = (row["split"], int(row["token_pos"]))
            targets[TARGET_NAMES[0]][key] = int(row["first_request_size_bin"])
            targets[TARGET_NAMES[1]][key] = int(row["cumulative_large_response_binned"])
            session_of[key] = int(row["session_id"])
    return targets, session_of


def class_balanced_sample(keys_and_labels, max_per_class, rng):
    """Return a list of keys, sampled so each class has min(class_count,
    max_per_class) entries. The class with the fewest entries
    determines the per-class cap; pass max_per_class=None to use that
    automatic cap."""
    by_class = defaultdict(list)
    for k, lbl in keys_and_labels:
        by_class[lbl].append(k)
    if max_per_class is None:
        max_per_class = min(len(v) for v in by_class.values())
    keys = []
    for lbl, ks in by_class.items():
        ks = list(ks)
        rng.shuffle(ks)
        keys.extend(ks[:max_per_class])
    rng.shuffle(keys)
    return keys


# ─────────────────────────────────────────────────────────────────────
# 2. Build probe dataset
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_probe_dataset(model, streams, sampled_keys, target_dict, session_of,
                        block_size, device):
    """For each sampled (split, token_pos) key, cache per-layer
    activations at that position. Returns (X, y, session_ids)."""
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
            _process_batch(model, batch_keys, combined, offsets, block_size, device,
                           target_dict, session_of, X_layers, y_list, sess_list)
            batch_keys = []
    if batch_keys:
        _process_batch(model, batch_keys, combined, offsets, block_size, device,
                       target_dict, session_of, X_layers, y_list, sess_list)

    X = np.stack([np.concatenate(X_layers[L], axis=0) for L in range(n_layer + 1)], axis=0)
    y = np.array(y_list, dtype=np.int64)
    sess = np.array(sess_list, dtype=np.int64)
    return X, y, sess


def _process_batch(model, batch_keys, combined, offsets, block_size, device,
                   target_dict, session_of, X_layers, y_list, sess_list):
    batch_idxs = []
    for split, pos in batch_keys:
        global_pos = offsets[split] + pos
        ctx_start = max(0, global_pos - block_size + 1)
        ctx = combined[ctx_start:global_pos + 1]
        if len(ctx) < block_size:
            ctx = np.concatenate([np.full(block_size - len(ctx), PAD), ctx])
        batch_idxs.append(ctx)
    idx_tensor = torch.from_numpy(np.stack(batch_idxs)).to(device)
    layer_acts = cache_layer_activations(model, idx_tensor)
    for L, act in enumerate(layer_acts):
        X_layers[L].append(act[:, -1, :].cpu().numpy())
    for split, pos in batch_keys:
        key = (split, pos)
        y_list.append(target_dict[key])
        sess_list.append(session_of[key])


# ─────────────────────────────────────────────────────────────────────
# 3. Probes (re-use the maze probe heads)
# ─────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.lin = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.lin(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)


def train_eval(probe, Xtr, ytr, Xte, yte, device, epochs=30, lr=3e-3):
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    yte_t = torch.from_numpy(yte).long().to(device)
    n = len(Xtr_t)
    batch = min(256, n)
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
        acc = (preds == yte_t).float().mean().item()
    return acc


def position_split(n, train_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(train_frac * n)
    return perm[:ntr], perm[ntr:]


def session_level_split(session_ids, train_frac, seed):
    unique = np.unique(session_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_train = int(train_frac * len(unique))
    train_sess = set(perm[:n_train].tolist())
    train_ix = np.array([i for i, s in enumerate(session_ids) if s in train_sess])
    test_ix = np.array([i for i, s in enumerate(session_ids) if s not in train_sess])
    return train_ix, test_ix


def run_seed_layer_sweep(target_name, X, y, n_classes, train_ix, test_ix,
                         device, epochs, label):
    if len(train_ix) == 0 or len(test_ix) == 0:
        return []
    rows = []
    n_layer = X.shape[0]
    print(f"\n  {label}  [{target_name}]  n_classes={n_classes}  "
          f"train={len(train_ix):,} test={len(test_ix):,}")
    print(f"    Layer       Lin      MLP")
    for L in range(n_layer):
        Xtr = X[L, train_ix]; ytr = y[train_ix]
        Xte = X[L, test_ix];  yte = y[test_ix]
        in_dim = Xtr.shape[1]
        lin = LinearProbe(in_dim, n_classes)
        mlp = MLPProbe(in_dim, n_classes)
        lin_acc = train_eval(lin, Xtr, ytr, Xte, yte, device, epochs=epochs)
        mlp_acc = train_eval(mlp, Xtr, ytr, Xte, yte, device, epochs=epochs)
        lab = "embed" if L == 0 else f"L{L - 1}"
        print(f"    {lab:<8}  {lin_acc:.4f}   {mlp_acc:.4f}")
        rows.append((lin_acc, mlp_acc))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions_a", type=int, default=10_000,
                   help="Per-class cap × n_classes; used for class-balanced Feature A.")
    p.add_argument("--n_positions_b", type=int, default=10_000,
                   help="Total cap for Feature B (proportional sampling).")
    p.add_argument("--probe_train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--skip_untrained", action="store_true")
    p.add_argument("--skip_session_split", action="store_true")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    print(f"running {len(args.seeds)} seed(s): {args.seeds}")

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device)
    trained.load_state_dict(ckpt["model_state"])
    trained.eval()
    print(f"loaded ckpt: iter={ckpt.get('iter', '?')}  val_ppl={ckpt.get('val_perplexity', float('nan')):.4f}")

    print("\nLoading probe targets ...")
    targets, session_of = load_targets(data_dir, splits=("val", "gen"))
    print(f"  probe positions (sz of req_idx>=1): {len(session_of):,}")

    # Class distributions
    for tgt in TARGET_NAMES:
        c = Counter(targets[tgt].values())
        total = sum(c.values())
        print(f"  [{tgt}] classes: {dict(c)}")
        print(f"    majority class fraction: {max(c.values())/total:.4f}")

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {s: np.asarray(np.memmap(data_dir / f"{s}.bin", dtype=dtype, mode="r"))
               for s in ("val", "gen")}

    # all_results[target][cond][L] = list of (lin_acc, mlp_acc)
    all_results = {
        tgt: {
            "trained_pos": defaultdict(list),
            "trained_session": defaultdict(list),
            "untrained_pos": defaultdict(list),
            "untrained_session": defaultdict(list),
        } for tgt in TARGET_NAMES
    }

    for seed in args.seeds:
        print(f"\n{'#' * 78}\n# SEED {seed}\n{'#' * 78}")
        torch.manual_seed(seed); np.random.seed(seed)
        untrained = None if args.skip_untrained else GPT(config).to(device).eval()
        py_rng = np.random.default_rng(seed).bit_generator
        rand = np.random.default_rng(seed)

        for tgt_idx, tgt in enumerate(TARGET_NAMES):
            tgt_dict = targets[tgt]
            keys_and_labels = [(k, lbl) for k, lbl in tgt_dict.items()]
            n_classes = max(tgt_dict.values()) + 1

            # Class-balanced sampling for Feature A; proportional for Feature B
            if tgt == TARGET_NAMES[0]:  # Feature A
                # Compute per-class cap: equal positions per class
                c = Counter(tgt_dict.values())
                per_class = min(args.n_positions_a // n_classes,
                                min(c.values()))
                import random as _rnd
                _rng = _rnd.Random(seed)
                sampled = class_balanced_sample(keys_and_labels, per_class, _rng)
                print(f"\n[{tgt}] CLASS-BALANCED sampling: {per_class} positions per class × {n_classes} classes = {len(sampled):,} total")
            else:
                # Proportional random sample
                import random as _rnd
                _rng = _rnd.Random(seed)
                _rng.shuffle(keys_and_labels)
                sampled = [k for k, _ in keys_and_labels[:args.n_positions_b]]
                print(f"\n[{tgt}] PROPORTIONAL sampling: {len(sampled):,} positions")

            print(f"  building probe dataset for TRAINED (seed={seed})...")
            t0 = time.time()
            X_t, y_t, sess_t = build_probe_dataset(
                trained, streams, sampled, tgt_dict, session_of,
                config.block_size, device)
            print(f"  collected {X_t.shape[1]:,} positions × {X_t.shape[0]} layers ({time.time()-t0:.1f}s)")

            X_u = y_u = sess_u = None
            if untrained is not None:
                print(f"  building probe dataset for UNTRAINED (seed={seed})...")
                t0 = time.time()
                X_u, y_u, sess_u = build_probe_dataset(
                    untrained, streams, sampled, tgt_dict, session_of,
                    config.block_size, device)
                print(f"  collected {X_u.shape[1]:,} positions × {X_u.shape[0]} layers ({time.time()-t0:.1f}s)")

            pos_train_t, pos_test_t = position_split(len(y_t), args.probe_train_frac, seed)
            rows = run_seed_layer_sweep(tgt, X_t, y_t, n_classes,
                                         pos_train_t, pos_test_t, device, args.epochs,
                                         f"TRAINED POSITION-LEVEL (seed {seed})")
            for L, (lin, mlp) in enumerate(rows):
                all_results[tgt]["trained_pos"][L].append((lin, mlp))

            if not args.skip_session_split:
                ses_tr, ses_te = session_level_split(sess_t, args.probe_train_frac, seed)
                if len(ses_tr) > 0 and len(ses_te) > 0:
                    rows = run_seed_layer_sweep(tgt, X_t, y_t, n_classes,
                                                 ses_tr, ses_te, device, args.epochs,
                                                 f"TRAINED SESSION-LEVEL (seed {seed})")
                    for L, (lin, mlp) in enumerate(rows):
                        all_results[tgt]["trained_session"][L].append((lin, mlp))

            if untrained is not None:
                pos_train_u, pos_test_u = position_split(len(y_u), args.probe_train_frac, seed)
                rows = run_seed_layer_sweep(tgt, X_u, y_u, n_classes,
                                             pos_train_u, pos_test_u, device, args.epochs,
                                             f"UNTRAINED POSITION-LEVEL (seed {seed})")
                for L, (lin, mlp) in enumerate(rows):
                    all_results[tgt]["untrained_pos"][L].append((lin, mlp))
                if not args.skip_session_split:
                    ses_tr_u, ses_te_u = session_level_split(sess_u, args.probe_train_frac, seed)
                    if len(ses_tr_u) > 0 and len(ses_te_u) > 0:
                        rows = run_seed_layer_sweep(tgt, X_u, y_u, n_classes,
                                                     ses_tr_u, ses_te_u, device, args.epochs,
                                                     f"UNTRAINED SESSION-LEVEL (seed {seed})")
                        for L, (lin, mlp) in enumerate(rows):
                            all_results[tgt]["untrained_session"][L].append((lin, mlp))

    # ─────────────────────────────────────────────────────────────
    # Aggregate
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}\nAGGREGATE — mean ± std over {len(args.seeds)} seeds\n{'=' * 78}")
    for tgt in TARGET_NAMES:
        print(f"\n  [{tgt}]")
        for cond in ("trained_pos", "trained_session", "untrained_pos", "untrained_session"):
            results = all_results[tgt][cond]
            if not results:
                continue
            print(f"\n  {cond}")
            print(f"    Layer       LinAcc(mean±std)     MLPAcc(mean±std)")
            for L in sorted(results.keys()):
                lins = np.array([r[0] for r in results[L]])
                mlps = np.array([r[1] for r in results[L]])
                lab = "embed" if L == 0 else f"L{L - 1}"
                lin_s = f"{lins.std(ddof=1):.4f}" if len(lins) > 1 else "0.0000"
                mlp_s = f"{mlps.std(ddof=1):.4f}" if len(mlps) > 1 else "0.0000"
                print(f"    {lab:<8}    {lins.mean():.4f} ± {lin_s}      {mlps.mean():.4f} ± {mlp_s}")

    print(f"\n{'=' * 78}\nHEADLINE — best layer by mean over {len(args.seeds)} seed(s)\n{'=' * 78}")
    for tgt in TARGET_NAMES:
        print(f"\n  [{tgt}]")
        for cond in ("trained_pos", "trained_session", "untrained_pos", "untrained_session"):
            results = all_results[tgt][cond]
            if not results:
                continue
            lin_means = {L: np.mean([r[0] for r in results[L]]) for L in results}
            mlp_means = {L: np.mean([r[1] for r in results[L]]) for L in results}
            best_lin_L = max(lin_means, key=lin_means.get)
            best_mlp_L = max(mlp_means, key=mlp_means.get)
            lin_s = np.std([r[0] for r in results[best_lin_L]], ddof=1) if len(results[best_lin_L]) > 1 else 0
            mlp_s = np.std([r[1] for r in results[best_mlp_L]], ddof=1) if len(results[best_mlp_L]) > 1 else 0
            lab_lin = "embed" if best_lin_L == 0 else f"L{best_lin_L - 1}"
            lab_mlp = "embed" if best_mlp_L == 0 else f"L{best_mlp_L - 1}"
            print(f"    {cond:<22} linear best {lab_lin}: {lin_means[best_lin_L]:.4f} ± {lin_s:.4f}")
            print(f"    {cond:<22} MLP    best {lab_mlp}: {mlp_means[best_mlp_L]:.4f} ± {mlp_s:.4f}")


if __name__ == "__main__":
    main()
