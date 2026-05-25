"""
LatentCityGPT — Phase 5 (clean version): activation transplant.

WHAT THIS FILE DOES, IN ONE PICTURE
====================================

  trained model (frozen)
        │
        │   STEP A — sample a "donor bank" of real activations
        │            for each real-node token T, find a position
        │            in val+gen where T is the current token, cache
        │            the residual at layer L (so we have one  a_T  per token T)
        │
  ┌─────┴────────────────────────────────────────┐
  │  token_id  →  a_T  (the model's representation │
  │                of "I'm at token_id" at layer L)│
  └────────────────────────────────────────────────┘
        │
        │   STEP B — for each test position with current token A:
        │            pick random target B (not A, not A's neighbor)
        │            run three forward passes:
        │              (1) UNPATCHED: original
        │              (2) TRANSPLANT: replace a_A with a_B at layer L
        │              (3) RANDOM:     replace a_A with a_RANDOM_OTHER at layer L
        │            measure next-token distribution at the test position
        ▼
   logits unpatched   logits transplanted   logits random-control
        │                    │                       │
        └──────────┬─────────┴──────────┬────────────┘
                   ▼                    ▼
       P(A's neighbors)         P(B's neighbors)
       across the three conditions


WHY THIS IS BETTER THAN `eval/causal.py`'s PINV-DIRECTION PATCH
==============================================================
The pseudoinverse-direction patch in `causal.py` is *probe-derived*: it
constructs a perturbation in the residual stream that the probe would
decode as a different location. But it does so via a direction in the
2-D probe-subspace, with magnitude scaled by the probe's geometry — and
this magnitude was empirically 7-13× the activation's own norm. The
patch overwhelms the residual, and the same "directional signal" shows
up on the destroyed-structure model that learned only co-occurrence.

The transplant approach uses the MODEL'S OWN residual from a real
position. It's a direct substitution of one piece of the model's
computation for another, bypassing the probe entirely. If the model
encodes "where I am" in its residual stream and uses it for next-token
prediction:
  - On the real-trained model:    P(B's nbrs) jumps up substantially.
  - On the destroyed-structure model: no jump (no learned location-use).

That's a clean, falsifiable test.


THE ONE RULE
============
We DO read graph.gpickle and stoi/itos (for selecting target B's
neighbors when scoring). We do NOT read coords.csv anywhere in this
file — this experiment is about the model's internal use of graph
structure, not about decoded geographic coordinates.


USAGE
=====
    python eval/transplant.py --ckpt checkpoints/best.pt \\
        --data_dir data/london_city --layer 3 --n_positions 200

    # Cross-model differential: run on both real and destroyed-structure
    # models, the differential is the experimental result.
"""

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from model import GPT, GPTConfig, BOS, EOS, PAD                  # noqa: E402
from probe import cache_layer_activations                         # noqa: E402

N_RESERVED = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build the activation donor bank: one real residual per real-node token
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_donor_bank(model: GPT, stream: np.ndarray, block_size: int,
                     layer: int, device: str, rng_seed: int = 0,
                     max_attempts_per_token: int = 5) -> dict:
    """For every real-node token id that appears in `stream`, find a position
    where that token is the current token, run the model, cache the residual
    stream at `layer`. Returns dict: token_id -> activation (n_embd,) on device.

    Why per-token: when we later transplant a_B for a_A, we want a real
    representation of the model processing token B. Multiple positions per
    token would let us average for noise reduction; for this experiment one
    per token is sufficient.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)
    s = stream.astype(np.int64)

    # Find candidate positions per token.
    # Stream-walking is fast; build position lists per token id.
    pos_by_token: dict[int, list[int]] = {}
    for i, tok in enumerate(s.tolist()):
        if tok < N_RESERVED:
            continue
        if i < block_size:
            continue   # need a left-context window of block_size
        pos_by_token.setdefault(int(tok), []).append(i)

    print(f"  donor bank: found positions for {len(pos_by_token):,} unique tokens")

    # For each token, pick a random position and cache its activation.
    # Batch by collecting several positions and running in parallel.
    donor_bank: dict[int, torch.Tensor] = {}
    batch_size = 32

    tokens_to_collect = list(pos_by_token.keys())
    rng.shuffle(tokens_to_collect)

    for batch_start in range(0, len(tokens_to_collect), batch_size):
        batch_tokens = tokens_to_collect[batch_start : batch_start + batch_size]
        windows = []
        target_positions_in_window = []   # position WITHIN the window of length block_size
        for tok in batch_tokens:
            positions = pos_by_token[tok]
            chosen_pos = int(rng.choice(positions))
            # Window: [chosen_pos - block_size + 1, chosen_pos] inclusive.
            # The "current token" is at index block_size - 1 within the window.
            start = chosen_pos - block_size + 1
            windows.append(s[start : start + block_size])
            target_positions_in_window.append(block_size - 1)
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
        acts_all = cache_layer_activations(model, idx_batch)
        acts_L = acts_all[layer]                              # (B, T, n_embd)
        for b, tok in enumerate(batch_tokens):
            t = target_positions_in_window[b]
            # Sanity: the token at this position should match `tok`.
            assert int(idx_batch[b, t].item()) == tok
            donor_bank[tok] = acts_L[b, t].clone()           # (n_embd,)
    return donor_bank


# ─────────────────────────────────────────────────────────────────────────────
# 2. Forward pass with REPLACE-style patching (not ADD)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_with_replacement(model: GPT, idx_batch: torch.Tensor, layer: int,
                              replace_positions: list[int],
                              replacement_vectors: torch.Tensor):
    """Run the model forward. After the residual at `layer` is produced (where
    layer=0 means after the input embedding, k means after block k), REPLACE
    the residual at (batch_row=b, seq_pos=replace_positions[b]) with
    replacement_vectors[b]. Continue forward through subsequent blocks and
    output head. Returns logits (B, T, V)."""
    B, T = idx_batch.shape
    device = idx_batch.device
    tok_emb = model.transformer.wte(idx_batch)
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    if layer == 0:
        for b in range(B):
            x[b, replace_positions[b]] = replacement_vectors[b]

    for i, block in enumerate(model.transformer.h, start=1):
        x = block(x)
        if i == layer:
            for b in range(B):
                x[b, replace_positions[b]] = replacement_vectors[b]

    x = model.transformer.ln_f(x)
    return model.lm_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. The transplant intervention experiment
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_transplant(model: GPT, layer: int, donor_bank: dict,
                   stream: np.ndarray, block_size: int,
                   G: nx.MultiDiGraph, itos: dict, stoi: dict,
                   n_positions: int, device: str,
                   rng_seed: int = 0, batch_size: int = 16) -> dict:
    """For each of n_positions test positions:
      - the current token is some A (real, has neighbors)
      - pick target B from donor_bank where B != A and B not in A's neighbors
      - look up a_B = donor_bank[B]
      - look up a_random = donor_bank[some random other token, not A or B]
      - three forward passes:
          unpatched: original
          transplant: replace residual at (test_position, layer) with a_B
          random_control: replace with a_random
      - score P(A's nbrs) and P(B's nbrs) under softmax at the test position
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)
    s = stream.astype(np.int64)

    # Cache neighbor tokens per real-node token.
    nbr_tokens_of: dict[int, set[int]] = {}
    for tok, node in itos.items():
        if tok < N_RESERVED or not G.has_node(node):
            continue
        nbr_tokens_of[tok] = set(stoi[n] for n in G.successors(node) if n in stoi)

    donor_tokens = list(donor_bank.keys())

    # Sample test positions.
    print(f"  sampling {n_positions} test positions...")
    test_records = []
    attempts = 0
    while len(test_records) < n_positions and attempts < n_positions * 100:
        attempts += 1
        start = int(rng.integers(0, len(s) - block_size - 1))
        # pick a position inside the window; not too close to either edge.
        t = int(rng.integers(8, block_size - 2))
        tok_A = int(s[start + t])
        if tok_A < N_RESERVED or tok_A not in donor_bank:
            continue
        nbrs_A = nbr_tokens_of.get(tok_A, set())
        if not nbrs_A:
            continue
        # Pick B: real, in donor bank, not A, not A's neighbor, has its own neighbors.
        for _ in range(50):
            tok_B = int(rng.choice(donor_tokens))
            if tok_B == tok_A or tok_B in nbrs_A:
                continue
            if not nbr_tokens_of.get(tok_B):
                continue
            break
        else:
            continue
        # Pick random-control token C: real, not A, not B.
        for _ in range(50):
            tok_C = int(rng.choice(donor_tokens))
            if tok_C == tok_A or tok_C == tok_B:
                continue
            break
        else:
            continue
        window = s[start : start + block_size].copy()
        test_records.append((window, t, tok_A, tok_B, tok_C))

    if len(test_records) < n_positions:
        print(f"  WARNING: collected only {len(test_records)}/{n_positions} valid positions")

    # Run interventions in mini-batches.
    print(f"  running interventions ({batch_size} per batch)...")
    results = {"unp_PA": [], "unp_PB": [],
               "trp_PA": [], "trp_PB": [],
               "rnd_PA": [], "rnd_PB": []}
    t0 = time.time()
    for batch_start in range(0, len(test_records), batch_size):
        batch = test_records[batch_start : batch_start + batch_size]
        B = len(batch)
        idx_batch = torch.from_numpy(
            np.stack([w for w, _, _, _, _ in batch]).astype(np.int64)
        ).to(device)
        positions = [t for _, t, _, _, _ in batch]
        a_B_vecs = torch.stack([donor_bank[tB] for (_, _, _, tB, _) in batch])
        a_C_vecs = torch.stack([donor_bank[tC] for (_, _, _, _, tC) in batch])

        # 1) unpatched: pass an identity-replacement (= original activation) trick
        #    The cleanest way is to re-run the forward without any replace.
        #    Implemented by replacing with the residual that's already there at layer L:
        #    cheaper to just compute logits normally.
        logits_unp, _ = model(idx_batch)
        # 2) transplant (replace with a_B)
        logits_trp = forward_with_replacement(model, idx_batch, layer,
                                               positions, a_B_vecs)
        # 3) random control (replace with a_C)
        logits_rnd = forward_with_replacement(model, idx_batch, layer,
                                               positions, a_C_vecs)

        for b in range(B):
            _, t, tok_A, tok_B, _ = batch[b]
            nbrs_A_list = list(nbr_tokens_of[tok_A])
            nbrs_B_list = list(nbr_tokens_of[tok_B])
            for label, logits in (("unp", logits_unp), ("trp", logits_trp), ("rnd", logits_rnd)):
                probs = F.softmax(logits[b, t], dim=-1).cpu().numpy()
                p_A = float(probs[nbrs_A_list].sum())
                p_B = float(probs[nbrs_B_list].sum())
                results[f"{label}_PA"].append(p_A)
                results[f"{label}_PB"].append(p_B)
    print(f"  done in {time.time()-t0:.1f}s")
    return results


def summarize(results: dict) -> dict:
    n = len(results["unp_PA"])
    out = {"n": n}
    for label in ("unp", "trp", "rnd"):
        out[f"{label}_PA"] = float(np.mean(results[f"{label}_PA"]))
        out[f"{label}_PB"] = float(np.mean(results[f"{label}_PB"]))
    # Effect sizes
    out["delta_PB_trp_over_unp"] = out["trp_PB"] - out["unp_PB"]
    out["delta_PB_trp_over_rnd"] = out["trp_PB"] - out["rnd_PB"]
    out["delta_PA_trp_over_unp"] = out["trp_PA"] - out["unp_PA"]
    # Per-position rate: does transplant beat random on P(B's nbrs)?
    n_trp_beats_rnd = sum(1 for i in range(n)
                          if results["trp_PB"][i] > results["rnd_PB"][i])
    out["trp_beats_rnd_rate"] = n_trp_beats_rnd / max(n, 1)
    return out


def print_report(layer: int, n_donors: int, summary: dict):
    print("")
    print("═" * 78)
    print("ACTIVATION TRANSPLANT — SUMMARY")
    print("═" * 78)
    print(f"  Replacement layer:                 L{layer}")
    print(f"  Donor-bank size (unique tokens):   {n_donors}")
    print(f"  Test positions:                    {summary['n']}")
    print("")
    print("  Mean probability mass over GRAPH NEIGHBORS of:")
    print(f"    P(A's neighbors)  unpatched      : {summary['unp_PA']:.4f}")
    print(f"    P(A's neighbors)  TRANSPLANT     : {summary['trp_PA']:.4f}   "
          f"Δ = {summary['delta_PA_trp_over_unp']:+.4f}")
    print(f"    P(A's neighbors)  random-control : {summary['rnd_PA']:.4f}")
    print("")
    print(f"    P(B's neighbors)  unpatched      : {summary['unp_PB']:.4f}")
    print(f"    P(B's neighbors)  TRANSPLANT     : {summary['trp_PB']:.4f}   "
          f"Δ = {summary['delta_PB_trp_over_unp']:+.4f}")
    print(f"    P(B's neighbors)  random-control : {summary['rnd_PB']:.4f}")
    print("")
    print("  Effect-size summary:")
    print(f"    TRANSPLANT lift on P(B's nbrs) over unpatched : "
          f"{summary['delta_PB_trp_over_unp']:+.4f}")
    print(f"    TRANSPLANT lift on P(B's nbrs) over random    : "
          f"{summary['delta_PB_trp_over_rnd']:+.4f}")
    print(f"    per-position: transplant_PB > random_PB in    : "
          f"{summary['trp_beats_rnd_rate']*100:.1f}% of cases")
    print("")
    print("  Interpretation:")
    print("    A model that USES its location representation should show:")
    print("      P(B's nbrs)_transplant >> P(B's nbrs)_random_control")
    print("    A model where the residual stream doesn't encode location-for-use")
    print("    (e.g. destroyed-structure model) should show:")
    print("      P(B's nbrs)_transplant ≈ P(B's nbrs)_random_control")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--layer", type=int, default=3,
                   help="residual-stream layer at which to substitute a_B for a_A")
    p.add_argument("--n_positions", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)

    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  iter={ckpt.get('iter','?')}  "
          f"val_ppl={ckpt.get('val_perplexity', float('nan')):.4f}  "
          f"vocab_size={config.vocab_size}  n_layer={config.n_layer}")
    assert 0 <= args.layer <= config.n_layer

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    itos = meta["itos"]
    stoi = meta["stoi"]

    val_stream = np.asarray(np.memmap(data_dir / "val.bin", dtype=dtype, mode="r"))
    gen_stream = np.asarray(np.memmap(data_dir / "gen.bin", dtype=dtype, mode="r"))
    combined   = np.concatenate([val_stream, gen_stream])

    G = pickle.loads((data_dir / "graph.gpickle").read_bytes())
    print(f"graph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

    print(f"\nBuilding donor bank at L{args.layer} ...")
    t0 = time.time()
    donor_bank = build_donor_bank(model, combined, config.block_size,
                                   args.layer, device, rng_seed=args.seed)
    print(f"  built in {time.time()-t0:.1f}s")

    print(f"\nRunning transplant intervention (n_positions={args.n_positions}) ...")
    results = run_transplant(
        model, args.layer, donor_bank, combined, config.block_size,
        G, itos, stoi, args.n_positions, device,
        rng_seed=args.seed, batch_size=args.batch_size,
    )
    summary = summarize(results)
    print_report(args.layer, len(donor_bank), summary)


if __name__ == "__main__":
    main()
