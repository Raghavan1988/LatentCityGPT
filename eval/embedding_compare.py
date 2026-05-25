"""
LatentCityGPT — embedding-comparison experiment.

Hypothesis: the trained transformer's token embedding (`wte`) is approximately
a node2vec embedding of the city's street graph. If true, the "geographic
clustering" that the linear probe finds in `wte` is mostly classical
co-occurrence-based representation learning — not something specific to
transformer training. The transformer's specific contribution would then be
in what its higher layers DO with this representation.

WHAT THIS FILE DOES, IN ONE PICTURE
====================================

   trained model checkpoint          city street graph (graph.gpickle)
            │                                  │
            │                                  │ node2vec random walks
            │  read transformer.wte             │ + Word2Vec
            ▼                                  ▼
       wte matrix  (n_real_nodes, n_embd)      node2vec matrix  (n_real_nodes, n_embd)
            │                                  │
            └────────────────┬─────────────────┘
                             ▼
                Procrustes alignment + Linear CKA
                             │
                             ▼
              alignment R² and CKA between the two embeddings
                             │
                             ▼ optional
            train one linear probe on wte, one on node2vec,
            compare R² at predicting (lat, lon)


METRICS
=======
- **Procrustes R²**: best orthogonal-rotation + scaling + translation alignment.
  R² = 1 - normalized disparity. Bounded in [0, 1]. High R² = the two
  embeddings are essentially the same up to rotation/scale/translation.
- **Linear CKA** (centered kernel alignment): rotation-invariant similarity
  measure between two embedding sets. In [0, 1]. Used routinely in
  representation-comparison literature (e.g., Kornblith et al. 2019).
- **Probe R² parity**: train one linear probe wte → (x_m, y_m), another
  node2vec → (x_m, y_m). If they get comparable R², neither representation
  is "doing more work" for geographic decoding.

THE ONE RULE
============
This script reads coords.csv for the probe-parity comparison, but not for
the wte / node2vec similarity computation. None of the geographic data
ever flows into the model — we're comparing the model's wte (which was
learned only from token sequences) to node2vec on the graph (also no
coordinate data).


USAGE
=====
    python eval/embedding_compare.py \\
        --ckpt checkpoints/best.pt --data_dir data/london_city
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import networkx as nx
from scipy.spatial import procrustes
from node2vec import Node2Vec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from model import GPT, GPTConfig                            # noqa: E402
from probe import (                                          # noqa: E402
    load_coords_planar, LinearProbe, train_and_evaluate_probe,
)

N_RESERVED = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Get the two embeddings
# ─────────────────────────────────────────────────────────────────────────────

def load_wte_matrix(ckpt_path: str, device: str) -> tuple[np.ndarray, dict, dict, int]:
    """Load checkpoint, return wte for REAL-node rows only (drop PAD/BOS/EOS),
    plus itos/stoi for the same indexing, plus n_embd."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    wte_full = model.transformer.wte.weight.detach().cpu().numpy()
    wte_real = wte_full[N_RESERVED:]    # (n_real_nodes, n_embd)
    return wte_real, ckpt, config


def compute_node2vec(G: nx.MultiDiGraph, n_real_nodes: int, n_embd: int,
                     itos: dict, walk_length: int = 30, num_walks: int = 200,
                     window: int = 10, p: float = 1.0, q: float = 1.0,
                     workers: int = 4, seed: int = 0) -> np.ndarray:
    """Run node2vec on G; return embeddings ordered by token id (idx N_RESERVED..).

    Notes:
      - node2vec/gensim Word2Vec wants string node ids.
      - We use a simpler undirected view for random walks (treat the graph as
        undirected for clustering purposes — this is what node2vec typically
        does and what would best match the model's bag-of-neighbors learning).
    """
    Gu = G.to_undirected()
    n2v = Node2Vec(
        Gu, dimensions=n_embd,
        walk_length=walk_length, num_walks=num_walks,
        p=p, q=q, workers=workers, seed=seed, quiet=True,
    )
    w2v = n2v.fit(window=window, min_count=1, sg=1, seed=seed, workers=workers)

    # Order embeddings by token id so rows correspond between wte and n2v.
    emb_rows = []
    missing = 0
    for tok in range(N_RESERVED, N_RESERVED + n_real_nodes):
        node = itos[tok]
        key = str(node)
        if key in w2v.wv:
            emb_rows.append(w2v.wv[key])
        else:
            emb_rows.append(np.zeros(n_embd, dtype=np.float32))
            missing += 1
    if missing:
        print(f"  WARNING: {missing} node2vec embeddings missing; zero-filled")
    return np.stack(emb_rows).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Comparison metrics
# ─────────────────────────────────────────────────────────────────────────────

def procrustes_score(A: np.ndarray, B: np.ndarray) -> float:
    """Best orthogonal rotation + scaling alignment. scipy.spatial.procrustes
    standardizes both matrices and returns a disparity in [0, 1] where 0 is
    perfect alignment. We report R² = 1 - disparity.
    """
    _, _, disparity = procrustes(A, B)
    return 1.0 - float(disparity)


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Centered Kernel Alignment with linear kernel (Kornblith et al. 2019).
    Symmetric, scale-invariant, in [0, 1]. Higher = more similar."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = float(np.linalg.norm(A.T @ B, ord="fro") ** 2)
    den = float(np.linalg.norm(A.T @ A, ord="fro") * np.linalg.norm(B.T @ B, ord="fro"))
    return num / max(den, 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probe parity: train linear probe on each, compare R²
# ─────────────────────────────────────────────────────────────────────────────

def probe_parity(wte_real: np.ndarray, n2v_real: np.ndarray,
                  coords_real: np.ndarray, device: str,
                  seed: int = 0, train_frac: float = 0.8,
                  epochs: int = 300) -> dict:
    """Split real nodes 80/20. Train linear probe on each embedding separately
    to predict (x_m, y_m). Compare R² + median meters."""
    n = wte_real.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_frac)
    tr, te = perm[:n_train], perm[n_train:]

    results = {}
    for name, emb in (("wte", wte_real), ("node2vec", n2v_real)):
        probe = LinearProbe(emb.shape[1])
        r = train_and_evaluate_probe(
            probe,
            emb[tr], coords_real[tr],
            emb[te], coords_real[te],
            device,
            lr=1e-3, weight_decay=1e-3, epochs=epochs,
        )
        results[name] = r
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n2v_dim", type=int, default=None,
                   help="node2vec embedding dim (defaults to wte's n_embd)")
    p.add_argument("--walk_length", type=int, default=30)
    p.add_argument("--num_walks", type=int, default=200)
    p.add_argument("--n2v_p", type=float, default=1.0)
    p.add_argument("--n2v_q", type=float, default=1.0)
    p.add_argument("--probe_epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"     # cheap CPU work; no MPS needed for this experiment
    data_dir = Path(args.data_dir)

    # ── Load model's wte for real nodes ──
    print(f"\nLoading checkpoint: {args.ckpt}")
    wte_real, ckpt, config = load_wte_matrix(args.ckpt, device)
    n_real, n_embd = wte_real.shape
    print(f"  wte (real-node rows): shape ({n_real:,}, {n_embd})")
    print(f"  iter={ckpt.get('iter','?')}  "
          f"val_ppl={ckpt.get('val_perplexity', float('nan')):.4f}")

    # ── Load graph + itos ──
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    itos = meta["itos"]
    G = pickle.loads((data_dir / "graph.gpickle").read_bytes())
    print(f"\ngraph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

    # ── Run node2vec ──
    n2v_dim = args.n2v_dim or n_embd
    print(f"\nRunning node2vec (dim={n2v_dim}, walks={args.num_walks}, "
          f"length={args.walk_length}, p={args.n2v_p}, q={args.n2v_q}) ...")
    import time
    t0 = time.time()
    n2v_real = compute_node2vec(
        G, n_real, n2v_dim, itos,
        walk_length=args.walk_length, num_walks=args.num_walks,
        p=args.n2v_p, q=args.n2v_q, seed=args.seed,
    )
    print(f"  done in {time.time()-t0:.1f}s; node2vec shape {n2v_real.shape}")

    # ── Similarity metrics ──
    print("\nComputing Procrustes + CKA ...")
    if n2v_real.shape[1] != n_embd:
        print(f"  WARNING: dims differ ({n_embd} vs {n2v_real.shape[1]}). "
              f"Procrustes requires same-dim; padding with zeros.")
        if n2v_real.shape[1] < n_embd:
            pad = np.zeros((n_real, n_embd - n2v_real.shape[1]), dtype=np.float32)
            n2v_for_proc = np.concatenate([n2v_real, pad], axis=1)
        else:
            n2v_for_proc = n2v_real[:, :n_embd]
    else:
        n2v_for_proc = n2v_real

    proc_r2 = procrustes_score(wte_real, n2v_for_proc)
    cka = linear_cka(wte_real, n2v_real)

    # Compare wte against an UNTRAINED-RANDOM control of same shape — sanity check
    # that the metrics aren't trivially high.
    rng = np.random.default_rng(args.seed)
    random_emb = rng.standard_normal((n_real, n_embd)).astype(np.float32) * wte_real.std()
    proc_r2_rand = procrustes_score(wte_real, random_emb)
    cka_rand     = linear_cka(wte_real, random_emb)

    # ── Probe parity ──
    print("\nRunning probe parity ...")
    coords_xy, center_lat, center_lon = load_coords_planar(data_dir)
    coords_real = coords_xy[N_RESERVED : N_RESERVED + n_real].numpy()
    parity = probe_parity(
        wte_real, n2v_real, coords_real, device,
        seed=args.seed, epochs=args.probe_epochs,
    )

    # ── Report ──
    print("")
    print("═" * 78)
    print("EMBEDDING COMPARISON — SUMMARY")
    print("═" * 78)
    print(f"  n_real_nodes:        {n_real:,}")
    print(f"  wte dim:             {n_embd}")
    print(f"  node2vec dim:        {n2v_real.shape[1]}")
    print(f"  city extent:         ({coords_real[:,0].ptp():.0f} m × "
          f"{coords_real[:,1].ptp():.0f} m)")
    print("")
    print("  SIMILARITY metrics (wte vs node2vec):")
    print(f"    Procrustes R²:     {proc_r2:.4f}")
    print(f"    Linear CKA:        {cka:.4f}")
    print("")
    print("  SANITY: wte vs RANDOM embedding of same shape (should be near 0):")
    print(f"    Procrustes R²:     {proc_r2_rand:.4f}")
    print(f"    Linear CKA:        {cka_rand:.4f}")
    print("")
    print("  PROBE PARITY — predict (x_m, y_m) from each embedding:")
    print(f"    {'source':<10}  {'R²':<10}  {'med m':<10}  {'p90 m':<10}")
    for name, r in parity.items():
        print(f"    {name:<10}  {r['r2']:<10.4f}  "
              f"{r['median_m']:<10.1f}  {r['p90_m']:<10.1f}")
    print("")
    print("  Interpretation:")
    print("    • Procrustes/CKA → 1.0 vs node2vec  ⇒  wte is essentially node2vec")
    print("                                            and the transformer's added")
    print("                                            value is in higher layers.")
    print("    • Procrustes/CKA in (0.3, 0.7)      ⇒  partial overlap; structural")
    print("                                            similarity but not identical.")
    print("    • Probe R² parity (wte ≈ node2vec)   ⇒  geographic decodability is")
    print("                                            a node-embedding-co-occurrence")
    print("                                            phenomenon, not transformer-")
    print("                                            specific.")


if __name__ == "__main__":
    main()
