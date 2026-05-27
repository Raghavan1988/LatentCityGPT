"""
LatentWorldsGPT — flight activation transplant.

Mirrors `eval/transplant.py` (cities), `eval/transplant_othello.py`,
and `eval/transplant_music.py`. Picks positions in flight A (current
phase P_A, altitude bin α_A) and flight B (different phase P_B,
similar altitude bin α_A — to control for the lexical phase-from-altitude
component). Replaces residual at layer L of A's position with B's
residual. Measures whether the model's next-observation prediction
shifts toward observations typical of B's phase.

Two metrics:
  (i) max |Δp|  — total shift magnitude
  (ii) phase-shift specificity — P(predicted bins typical of B's phase)
       minus P(predicted bins typical of A's phase)

Random control: replace with residual from a third position C (different
phase from both A and B). Expected: transplant > random on phase-shift
metric; transplant ~ random on raw shift magnitude indicates the
transplant is doing phase-specific work, not just random perturbation.
"""

import argparse
import csv
import pickle
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

PAD, BOS, EOS = 0, 1, 2
N_RESERVED = 3


def token_to_bins(tok, n_vr, n_spd):
    if tok < N_RESERVED: return None
    t = tok - N_RESERVED
    alt_b = t // (n_vr * n_spd)
    rem = t - alt_b * (n_vr * n_spd)
    vr_b = rem // n_spd
    spd_b = rem - vr_b * n_spd
    return (alt_b, vr_b, spd_b)


def load_targets(data_dir):
    phases = {}
    flight_of = {}
    with open(data_dir / "flight_phase.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["split"], int(row["token_pos"]))
            phases[key] = row["phase"]
            flight_of[key] = int(row["flight_idx"])
    return phases, flight_of


@torch.no_grad()
def build_donor_bank(model, stream, phases, n_vr, n_spd, block_size, layer,
                     device, split_name, n_donors=800, rng_seed=0):
    model.eval()
    rng = np.random.default_rng(rng_seed)
    candidates = [pos for (s, pos), _ in phases.items()
                  if s == split_name and pos >= block_size and pos < len(stream)]
    rng.shuffle(candidates)
    donors = []
    batch_size = 32
    for batch_start in range(0, min(len(candidates), n_donors * 2), batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
        windows = []
        positions_in_window = []
        bs = []
        for pos in batch:
            win_start = pos - block_size + 1
            window = stream[win_start : win_start + block_size]
            if len(window) < block_size: continue
            windows.append(window)
            positions_in_window.append(block_size - 1)
            bs.append(pos)
        if not windows: continue
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
        acts = cache_layer_activations(model, idx_batch)
        acts_L = acts[layer]
        for b, pos in enumerate(bs):
            cur_tok = int(stream[pos])
            cur_bins = token_to_bins(cur_tok, n_vr, n_spd)
            donors.append({
                "residual": acts_L[b, positions_in_window[b]].clone(),
                "global_pos": pos,
                "window": np.asarray(windows[b]).copy(),
                "pos_in_window": positions_in_window[b],
                "phase": phases[(split_name, pos)],
                "cur_bins": cur_bins,
            })
            if len(donors) >= n_donors: break
        if len(donors) >= n_donors: break
    return donors


@torch.no_grad()
def forward_with_replacement(model, idx_batch, layer, replace_positions,
                              replacement_vectors):
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


def build_phase_token_sets(stream, phases, n_vr, n_spd, split_name):
    """Per phase, the set of token IDs that ever appear at positions tagged
    with that phase. Used to score 'predictions toward phase P typical tokens.'"""
    sets = defaultdict(set)
    for (s, pos), ph in phases.items():
        if s != split_name: continue
        if pos < len(stream):
            tok = int(stream[pos])
            if tok >= N_RESERVED:
                sets[ph].add(tok)
    return {k: list(v) for k, v in sets.items()}


@torch.no_grad()
def run_transplant(model, donors, phase_token_sets, layer, n_pairs, device,
                   alt_tolerance=1, rng_seed=0, batch_size=16):
    """Match A and B on similar alt_bin (within ±alt_tolerance) but different
    phase. C is a random control donor."""
    model.eval()
    rng = np.random.default_rng(rng_seed)
    n_phases = sorted(set(d["phase"] for d in donors))
    print(f"  donor phases: {dict(Counter(d['phase'] for d in donors))}")

    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        a_i, b_i, c_i = rng.integers(0, len(donors), size=3).tolist()
        if len({a_i, b_i, c_i}) < 3: continue
        a, b, c = donors[a_i], donors[b_i], donors[c_i]
        if a["cur_bins"] is None or b["cur_bins"] is None or c["cur_bins"] is None:
            continue
        # Match alt_bin within tolerance; differ phase
        if abs(a["cur_bins"][0] - b["cur_bins"][0]) > alt_tolerance: continue
        if a["phase"] == b["phase"]: continue
        if a["phase"] == c["phase"]: continue
        if b["phase"] == c["phase"]: continue
        pairs.append((a_i, b_i, c_i))
    print(f"  sampled {len(pairs)} matched-alt / differ-phase pairs")
    if not pairs:
        return None

    results = {
        "trp_maxdp": [], "rnd_maxdp": [],
        "trp_phaseB_gain": [], "rnd_phaseB_gain": [],
        "trp_phaseC_gain": [], "rnd_phaseC_gain": [],
    }
    t0 = time.time()
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        B = len(batch)
        windows = []; positions = []
        a_phases = []; b_phases = []; c_phases = []
        b_resids = []; c_resids = []
        for (a_i, b_i, c_i) in batch:
            a, b, c = donors[a_i], donors[b_i], donors[c_i]
            windows.append(a["window"])
            positions.append(a["pos_in_window"])
            a_phases.append(a["phase"]); b_phases.append(b["phase"]); c_phases.append(c["phase"])
            b_resids.append(b["residual"]); c_resids.append(c["residual"])
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
        b_vecs = torch.stack(b_resids); c_vecs = torch.stack(c_resids)

        logits_unp, _ = model(idx_batch)
        logits_trp = forward_with_replacement(model, idx_batch, layer, positions, b_vecs)
        logits_rnd = forward_with_replacement(model, idx_batch, layer, positions, c_vecs)

        for k in range(B):
            t = positions[k]
            p_unp = F.softmax(logits_unp[k, t], dim=-1).cpu().numpy()
            p_trp = F.softmax(logits_trp[k, t], dim=-1).cpu().numpy()
            p_rnd = F.softmax(logits_rnd[k, t], dim=-1).cpu().numpy()

            results["trp_maxdp"].append(float(np.max(np.abs(p_trp - p_unp))))
            results["rnd_maxdp"].append(float(np.max(np.abs(p_rnd - p_unp))))

            # phase-specific gain: P(tokens typical of phase X) under each condition
            def phase_mass(p_vec, phase):
                tokens = phase_token_sets.get(phase, [])
                if not tokens: return 0.0
                return float(p_vec[tokens].sum())
            results["trp_phaseB_gain"].append(phase_mass(p_trp, b_phases[k]) - phase_mass(p_unp, b_phases[k]))
            results["rnd_phaseB_gain"].append(phase_mass(p_rnd, b_phases[k]) - phase_mass(p_unp, b_phases[k]))
            results["trp_phaseC_gain"].append(phase_mass(p_trp, c_phases[k]) - phase_mass(p_unp, c_phases[k]))
            results["rnd_phaseC_gain"].append(phase_mass(p_rnd, c_phases[k]) - phase_mass(p_unp, c_phases[k]))
    print(f"  done in {time.time()-t0:.1f}s")
    return results, len(pairs)


def summarize(results, n_pairs):
    print(f"\nn_pairs: {n_pairs}\n")
    print(f"  {'Metric':<34}{'Transplant':>14}{'Random control':>18}{'Δ trp-rnd':>14}")
    print("  " + "─" * 80)
    for label, key in [
        ("max |Δ p|",                "maxdp"),
        ("P(B-phase tokens) gain",   "phaseB_gain"),
        ("P(C-phase tokens) gain",   "phaseC_gain"),
    ]:
        trp_v = float(np.mean(results[f"trp_{key}"]))
        rnd_v = float(np.mean(results[f"rnd_{key}"]))
        print(f"  {label:<34}{trp_v:>+14.4f}{rnd_v:>+18.4f}{trp_v-rnd_v:>+14.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", choices=("val", "gen"), default="gen")
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--n_donors", type=int, default=800)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--alt_tolerance", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device); model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded ckpt: iter={ckpt.get('iter','?')} val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}")
    print(f"layer = {args.layer} (of {config.n_layer} blocks)")

    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    n_vr = meta["n_vr"]; n_spd = meta["n_spd"]
    dtype = np.dtype(meta["dtype"])
    stream = np.asarray(np.memmap(data_dir / f"{args.split}.bin", dtype=dtype, mode="r"))
    phases, _ = load_targets(data_dir)
    print(f"{args.split}.bin: {len(stream):,} tokens; {len(phases):,} phase-labeled positions")

    phase_token_sets = build_phase_token_sets(stream, phases, n_vr, n_spd, args.split)
    print(f"phase token sets: {dict((k, len(v)) for k, v in phase_token_sets.items())}")

    donors = build_donor_bank(model, stream, phases, n_vr, n_spd, config.block_size,
                              args.layer, device, args.split,
                              n_donors=args.n_donors, rng_seed=args.seed)
    print(f"built {len(donors)} donors")
    if len(donors) < 50:
        print("not enough donors; exit"); return

    result = run_transplant(model, donors, phase_token_sets, args.layer,
                            args.n_pairs, device,
                            alt_tolerance=args.alt_tolerance,
                            rng_seed=args.seed,
                            batch_size=args.batch_size)
    if result is None: return
    metrics, n_pairs = result
    summarize(metrics, n_pairs)


if __name__ == "__main__":
    main()
