"""
LatentWorldsGPT — flight valid-step evaluator.

Flight-domain analog of cities `valid_edge.py`, music `valid_voice_step.py`,
and Othello `valid_othello_move.py`. Independently of any probe, asks:
does the model's greedy next-observation prediction correspond to a
PHYSICALLY PLAUSIBLE next state given the current observation?

Concretely, each token encodes a bin-tuple (alt_bin, vr_bin, spd_bin).
A predicted next-token is "physics-valid" iff:
  - |Δ alt_bin| ≤ 1     (altitude doesn't change more than 1 bin in 5s)
  - |Δ vr_bin| ≤ 1      (vertical rate is smoothly varying)
  - |Δ spd_bin| ≤ 1     (speed is smoothly varying)
  - Plus consistency:   if Δalt_bin > 0 then vr_bin must be > 2 (climbing);
                        if Δalt_bin < 0 then vr_bin must be < 2 (descending).

These rules approximate the physics constraints captured by the Sun et al.
fuzzy phase logic but operate purely on the discretized token space —
no phase label is read by this script.

THE ONE RULE
============
This script never reads flight_phase.csv. It REPLAYS physics rules on the
token stream — what the model would have to learn implicitly.

Usage:
    python eval/valid_flight_step.py --ckpt checkpoints/adsb_5s/best.pt \
        --data_dir data/adsb_5s --split gen
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
from model import GPT, GPTConfig, BOS, EOS, PAD  # noqa: E402

N_RESERVED = 3


def token_to_bins(tok, n_vr, n_spd):
    """Decode token id back to (alt_b, vr_b, spd_b)."""
    if tok < N_RESERVED:
        return None
    t = tok - N_RESERVED
    alt_b = t // (n_vr * n_spd)
    rem = t - alt_b * (n_vr * n_spd)
    vr_b = rem // n_spd
    spd_b = rem - vr_b * n_spd
    return (alt_b, vr_b, spd_b)


def is_physically_valid(cur_bins, next_bins, vr_level_bin):
    """Validate that next observation could plausibly follow current under
    physics constraints. Returns (valid_bool, reason_str)."""
    if cur_bins is None or next_bins is None:
        return False, "control-token"
    alt_c, vr_c, spd_c = cur_bins
    alt_n, vr_n, spd_n = next_bins
    if abs(alt_n - alt_c) > 1:
        return False, "alt-jump"
    if abs(vr_n - vr_c) > 1:
        return False, "vr-jump"
    if abs(spd_n - spd_c) > 1:
        return False, "spd-jump"
    # consistency: alt change direction matches vr
    if alt_n > alt_c and vr_n <= vr_level_bin:
        return False, "alt-up-vr-not-climb"
    if alt_n < alt_c and vr_n >= vr_level_bin:
        return False, "alt-down-vr-not-descent"
    return True, "ok"


def split_flights(stream):
    flights = []
    i = 0
    while i < len(stream):
        if stream[i] != BOS:
            i += 1; continue
        j = i + 1
        while j < len(stream) and stream[j] != EOS:
            j += 1
        if j - i >= 5:
            flights.append(np.asarray(stream[i : j + 1 if j < len(stream) else j]))
        i = j + 1
    return flights


@torch.no_grad()
def next_step_validity(model, flights, block_size, n_vr, n_spd,
                        vr_level_bin, device, vocab_size):
    """For each flight, run greedy next-token prediction at every position,
    check whether the predicted next-token is physics-valid given the
    current observation token. Aggregate over a held-out flight set."""
    from collections import Counter
    cats = Counter()
    n_scored = 0
    for flight in flights:
        seq = flight[:block_size] if len(flight) > block_size else flight
        x = torch.from_numpy(seq.astype(np.int64)).unsqueeze(0).to(device)
        assert int(x.min()) >= 0 and int(x.max()) < vocab_size
        logits, _ = model(x)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        # Score positions 1..len-2 (skip BOS and EOS)
        for t in range(1, len(seq) - 1):
            cur_tok = int(seq[t])
            if cur_tok < N_RESERVED:
                continue
            pred_tok = int(preds[t])
            cur_bins = token_to_bins(cur_tok, n_vr, n_spd)
            next_bins = token_to_bins(pred_tok, n_vr, n_spd)
            valid, reason = is_physically_valid(cur_bins, next_bins, vr_level_bin)
            cats["VALID" if valid else f"INVALID:{reason}"] += 1
            n_scored += 1
    return cats, n_scored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", choices=("val", "gen"), default="gen")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device); model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded ckpt: iter={ckpt.get('iter','?')}  val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}")

    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    n_vr = meta["n_vr"]; n_spd = meta["n_spd"]
    vr_level_bin = (n_vr - 1) // 2  # middle vr bin = "level"
    dtype = np.dtype(meta["dtype"])
    stream = np.asarray(np.memmap(data_dir / f"{args.split}.bin", dtype=dtype, mode="r"))
    flights = split_flights(stream)
    print(f"{args.split}.bin: {len(stream):,} tokens / {len(flights)} flights")

    cats, n_scored = next_step_validity(
        model, flights, config.block_size, n_vr, n_spd,
        vr_level_bin, device, config.vocab_size,
    )
    print(f"\n[next-step / {args.split}.bin]")
    print(f"  positions scored : {n_scored:,}")
    if n_scored == 0:
        print("  no positions scored — exiting"); return
    valid = cats.get("VALID", 0)
    print(f"  ── breakdown ──")
    for k in sorted(cats.keys()):
        v = cats[k]
        print(f"    {k:<28} : {v:>7,} ({v/n_scored*100:6.2f}%)")
    print(f"  ── headline ──")
    print(f"    VALID rate (physics-plausible next-token): "
          f"{valid/n_scored*100:6.2f}%   (target ≥ 95% for a model that learned flight physics)")


if __name__ == "__main__":
    main()
