"""
LatentCityGPT — valid-edge rate evaluator.

WHAT THIS FILE DOES, IN ONE PICTURE
====================================

  The cross-entropy loss tells us the model is fitting SOME distribution. This
  script asks the harder question: does that distribution match the real city
  graph? This is the LatentCityGPT analogue of Othello-GPT's "legal move rate."

  ┌──────────────────────────────────────────────────────────────────┐
  │  MODE A — next-step valid-edge rate (the workhorse)              │
  └──────────────────────────────────────────────────────────────────┘

      stream from val.bin or gen.bin:
        ... BOS  n1  n2  n3  ...  nK  EOS  BOS  m1  m2  ...

      for each transition (n_i → n_{i+1}) where BOTH tokens are real nodes:
          let prefix = everything up to and including n_i
          pred = argmax over vocab of model( prefix ).logits[last]
          → does pred correspond to a real neighbor of n_i in graph.gpickle?

      ┌─────────────────────────────────────────────────────────┐
      │  Untrained baseline (random init):                       │
      │    ≈ avg_out_degree / vocab_size                         │
      │    e.g. Manhattan ~3 neighbors / 4546 vocab ≈ 0.07%      │
      │  Trained model: should approach 100%.                    │
      └─────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  MODE B — full-route generation                                  │
  └──────────────────────────────────────────────────────────────────┘

      sample N start nodes from the graph
      for each start:
          prompt = [BOS, start_token]
          let model generate up to `max_steps` more tokens or until EOS
          extract the real-node sequence [start, n1, n2, ..., (EOS)]
          this route is "fully valid" iff every consecutive pair (n_i, n_{i+1})
              is a real edge in the graph

      Stricter than next-step: even if next-step predictions are mostly valid,
      a 95% per-step rate compounds — over a 50-step route only ~7% would be
      fully valid. Full-route catches drift.


WHAT WE'RE PROVING (and what we are NOT)
========================================
This metric proves the model learned the city's *adjacency*. It does NOT yet
prove the model has a metric map — that's the linear-probe job in eval/probe.py
(Phase 4). Valid-edge is the necessary prerequisite: "earn the right to ask"
about emergent geometry by showing competent routing first.


THE ONE RULE
============
This script never reads coords.csv. It uses only graph adjacency (from
graph.gpickle) and token IDs. No coordinate ever enters the model — the model
never even sees this script's evaluation logic.


USAGE
=====
    # Next-step + full-route, evaluated on val split (in-distribution destinations).
    python eval/valid_edge.py --ckpt checkpoints/best.pt --data_dir data/london_city --split val

    # Generalization check: same but on gen split (held-out destinations).
    python eval/valid_edge.py --ckpt checkpoints/best.pt --data_dir data/london_city --split gen

    # Just full-route mode with more samples.
    python eval/valid_edge.py --ckpt checkpoints/best.pt --data_dir data/manhattan \\
        --mode full --n_routes 500 --max_steps 100

    # Bigger next-step sample (slower, lower variance).
    python eval/valid_edge.py --ckpt checkpoints/best.pt --data_dir data/manhattan \\
        --mode next --n_positions 100000
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import networkx as nx

# Make the model package importable when run as `python eval/valid_edge.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from model import GPT, GPTConfig, BOS, EOS  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str) -> tuple[GPT, dict]:
    """Reconstruct the model from a checkpoint saved by train.py.

    The checkpoint contains everything we need: a GPTConfig dict, the
    vocab_size, and the model state dict. So this script doesn't need to know
    which config was used or what city was trained on.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_artifacts(data_dir: Path):
    """Load graph + tokenizer from a city's data dir.

    Returns:
      G    : networkx MultiDiGraph (the SCC-restricted street network)
      itos : dict mapping token-id → OSM node id   (for "what node did the model emit?")
      stoi : dict mapping OSM node id → token-id   (for "what token represents this node?")
    """
    G = pickle.loads((data_dir / "graph.gpickle").read_bytes())
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return G, meta["itos"], meta["stoi"]


# ──────────────────────────────────────────────────────────────────────────────
# Mode A: next-step
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def next_step_valid_edge_rate(model: GPT, G: nx.MultiDiGraph, itos: dict,
                              stream: np.ndarray, block_size: int,
                              n_positions: int, device: str,
                              vocab_size: int) -> dict:
    """For random positions in `stream`, score the model's greedy next-token
    prediction against two things:

      (a) IS THE PREDICTION A REAL NEIGHBOR?
          The "valid-edge rate." This is the main number. Tolerant — it credits
          the model for any real neighbor, not just the labelled one (there are
          often multiple equally-good shortest paths).

      (b) DOES IT MATCH THE LABELLED TARGET?
          Top-1 accuracy. Stricter — credits only the exact next node from the
          original shortest path. Lower than (a) by definition.

    Why we use random samples rather than scanning every position: the eval
    splits have millions of tokens; we don't need to score them all. A few tens
    of thousands gives stable percentages.
    """
    rng = np.random.default_rng(0)
    n_neighbor_correct = 0
    n_top1_correct = 0
    n_scored = 0

    while n_scored < n_positions:
        # Sample one random window of (block_size + 1) tokens. We feed the first
        # block_size to the model and use the last token of the window as the
        # ground-truth target for the prediction at position (block_size - 1).
        start = int(rng.integers(0, len(stream) - block_size - 1))
        window = np.asarray(stream[start : start + block_size + 1])

        x = torch.from_numpy(window[:-1].astype(np.int64)).unsqueeze(0).to(device)
        # ONE-RULE defensive check.
        assert int(x.min()) >= 0 and int(x.max()) < vocab_size

        # Logits at every position in the prefix.
        logits, _ = model(x)             # (1, T, V)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (T,)

        # Walk positions t = 0 .. T-1. At each t, the model predicted window[t+1].
        # We score positions where both window[t] and window[t+1] are real nodes
        # (token ids >= 3) — i.e. the transition is "node A → node B", not a
        # BOS/EOS boundary.
        for t in range(len(window) - 1):
            prev_tok = int(window[t])
            true_tok = int(window[t + 1])
            if prev_tok < 3 or true_tok < 3:
                continue  # skip BOS→first-node, last-node→EOS, etc.
            pred = int(preds[t])
            cur_node = itos[prev_tok]

            # Validity check: prediction must be a real-node id AND a graph neighbor.
            if pred >= 3:
                pred_node = itos[pred]
                if G.has_edge(cur_node, pred_node):
                    n_neighbor_correct += 1
            # Top-1 accuracy: did we hit the exact same labelled node?
            if pred == true_tok:
                n_top1_correct += 1
            n_scored += 1
            if n_scored >= n_positions:
                break

    return {
        "valid_edge_rate": n_neighbor_correct / max(1, n_scored),
        "top1_accuracy":   n_top1_correct     / max(1, n_scored),
        "n_scored":        n_scored,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mode B: full-route generation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_route_validity(model: GPT, G: nx.MultiDiGraph, itos: dict, stoi: dict,
                        n_routes: int, max_steps: int, device: str,
                        temperature: float = 1.0) -> dict:
    """Sample N starting nodes, generate routes from each, score validity.

    Two numbers per run:

      fully_valid_route_rate
          fraction of routes where EVERY transition is a real edge.

      edge_validity_rate
          fraction of individual transitions across all routes that are real
          edges. Less strict, finer-grained.
    """
    real_nodes = sorted(n for n in G.nodes() if n in stoi)
    rng = np.random.default_rng(0)
    starts = rng.choice(len(real_nodes), size=n_routes, replace=False).tolist()

    n_fully_valid = 0
    edges_total = 0
    edges_valid = 0

    for s in starts:
        start_node = real_nodes[s]
        # Prompt the model with [BOS, start_token]. It then emits the rest.
        prompt = torch.tensor([[BOS, stoi[start_node]]], dtype=torch.long, device=device)
        out = model.generate(prompt, max_new_tokens=max_steps, temperature=temperature)
        seq = out.squeeze(0).tolist()

        # Walk the emitted sequence and extract real-node tokens up to EOS.
        # A mid-route PAD or BOS is illegal — flag as not-fully-valid.
        nodes = []
        valid_so_far = True
        for i, tok in enumerate(seq):
            if i == 0 and tok == BOS:
                continue        # the initial BOS prompt
            if tok == EOS:
                break           # legitimate route end
            if tok < 3:
                valid_so_far = False
                break           # PAD or unexpected BOS mid-route
            nodes.append(itos[tok])

        # Score every consecutive transition.
        for a, b in zip(nodes, nodes[1:]):
            edges_total += 1
            if G.has_edge(a, b):
                edges_valid += 1
            else:
                valid_so_far = False

        # A "fully valid" route is one with at least one transition, all edges real.
        if valid_so_far and len(nodes) >= 2:
            n_fully_valid += 1

    return {
        "fully_valid_route_rate": n_fully_valid / max(1, n_routes),
        "edge_validity_rate":     edges_valid / max(1, edges_total),
        "n_routes":               n_routes,
        "n_edges_seen":           edges_total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint produced by model/train.py")
    p.add_argument("--data_dir", required=True, help="data/<city>/ for the trained city")
    p.add_argument("--split", choices=("val", "gen"), default="val",
                   help="which .bin to use in next-step mode")
    p.add_argument("--mode", choices=("next", "full", "both"), default="both",
                   help="which mode(s) to run")
    p.add_argument("--n_positions", type=int, default=20_000,
                   help="(next-step) random positions to score")
    p.add_argument("--n_routes", type=int, default=200,
                   help="(full-route) number of starts to sample")
    p.add_argument("--max_steps", type=int, default=80,
                   help="(full-route) max generated tokens per route")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="(full-route) sampling temperature; 1.0 = honest")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() \
             else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    model, ckpt = load_model(args.ckpt, device)
    print(f"loaded ckpt: iter={ckpt.get('iter','?')}  "
          f"val_ppl={ckpt.get('val_perplexity','?')}")
    G, itos, stoi = load_artifacts(data_dir)
    print(f"graph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")
    vocab_size = ckpt["vocab_size"]

    # ── Mode A: next-step ──
    if args.mode in ("next", "both"):
        with open(data_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        dtype = np.dtype(meta["dtype"])
        stream = np.memmap(data_dir / f"{args.split}.bin", dtype=dtype, mode="r")

        result = next_step_valid_edge_rate(
            model, G, itos, np.asarray(stream),
            model.config.block_size, args.n_positions, device, vocab_size,
        )
        print(f"\n[next-step / {args.split}.bin]")
        print(f"  valid-edge rate  : {result['valid_edge_rate']*100:6.2f}%   (target → 100%)")
        print(f"  top-1 accuracy   : {result['top1_accuracy']*100:6.2f}%   (exact-label match)")
        print(f"  positions scored : {result['n_scored']:,}")

    # ── Mode B: full-route ──
    if args.mode in ("full", "both"):
        result = full_route_validity(
            model, G, itos, stoi,
            args.n_routes, args.max_steps, device, args.temperature,
        )
        print(f"\n[full-route generation, temperature={args.temperature}]")
        print(f"  fully-valid route rate : {result['fully_valid_route_rate']*100:6.2f}%   (target → 100%)")
        print(f"  edge validity (all)    : {result['edge_validity_rate']*100:6.2f}%   "
              f"({result['n_edges_seen']:,} transitions over {result['n_routes']} routes)")


if __name__ == "__main__":
    main()
