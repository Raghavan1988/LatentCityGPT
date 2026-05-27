"""
LatentWorldsGPT — Milestone 4 (flight-phase) dataset preparation.

Turns real ADS-B trajectories (from `traffic` library quickstart sample,
or any traffic Traffic collection) into a token corpus for next-
observation language modeling, plus a phase-label side table the model
never sees.

THE ONE RULE (flight-phase edition): the flight-phase label per
position lives in `flight_phase.csv` and is read only by the probe /
eval code. The model's tokens encode only the (binned) physical
observation tuple: (alt_bin, vr_bin, speed_bin).

Token convention: 0=PAD, 1=BOS, 2=EOS; real tokens from 3.
Each token = one observation timestep = a discrete bin-tuple.

Outputs (in --out_dir):
    train.bin / val.bin / gen.bin   uint16 token streams (flight-level split)
    meta.pkl                         {vocab_size, stoi, itos, dtype, bin_edges, ...}
    flight_phase.csv                 split,flight_idx,token_pos,phase
"""

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np

PAD, BOS, EOS = 0, 1, 2
N_RESERVED = 3

# Bin edges — chosen to capture phase distinctions
ALT_EDGES = [0, 500, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]   # ft (10 bins)
VR_EDGES = [-99999, -1500, -500, 500, 1500, 99999]                                  # fpm (5 bins)
SPD_EDGES = [0, 100, 250, 400, 99999]                                               # knots (4 bins)
# Total real-tuple vocab: 10 * 5 * 4 = 200


def discretize(value, edges):
    """Return the bin index (0..len(edges)-2) for `value` given bin EDGES.
    Values below the first edge go to bin 0; values above the last go to the last bin."""
    if np.isnan(value):
        return None
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2


def bin_tuple_to_token_id(alt_b, vr_b, spd_b):
    """Encode (alt_b, vr_b, spd_b) → token id ≥ N_RESERVED.
    Layout: token = N_RESERVED + alt_b * (n_vr * n_spd) + vr_b * n_spd + spd_b
    """
    n_alt = len(ALT_EDGES) - 1
    n_vr  = len(VR_EDGES) - 1
    n_spd = len(SPD_EDGES) - 1
    return N_RESERVED + alt_b * (n_vr * n_spd) + vr_b * n_spd + spd_b


def total_vocab_size():
    n_alt = len(ALT_EDGES) - 1
    n_vr  = len(VR_EDGES) - 1
    n_spd = len(SPD_EDGES) - 1
    return N_RESERVED + n_alt * n_vr * n_spd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load flights
# ─────────────────────────────────────────────────────────────────────────────

def load_flights(sample_name="quickstart"):
    """Yield (flight_idx, dataframe) for each flight in the chosen sample."""
    import importlib
    samples = importlib.import_module("traffic.data.samples")
    coll = getattr(samples, sample_name)
    for i, f in enumerate(coll):
        yield i, f.data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Phase tagging
# ─────────────────────────────────────────────────────────────────────────────

def tag_phases(t, alt, spd, roc):
    """Tag each timestep with a phase label (str) using openap fuzzy logic."""
    from openap import phase
    fp = phase.FlightPhase()
    fp.set_trajectory(t, alt, spd, roc)
    return fp.phaselabel()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-flight tokenization
# ─────────────────────────────────────────────────────────────────────────────

def downsample_flight(df, sample_seconds):
    """Downsample a flight's per-second observations to one every
    `sample_seconds`. Returns numpy arrays (t, alt, spd, roc) at the
    downsampled timestamps + the original df indices kept."""
    import pandas as pd
    timestamps = [pd.Timestamp(x) for x in df['timestamp'].tolist()]
    t_full = np.array([(x - timestamps[0]).total_seconds() for x in timestamps],
                      dtype=np.float64)
    # Sample every `sample_seconds`. Take the first observation, then the next
    # observation whose time is at least sample_seconds after the previous kept.
    keep_idx = [0]
    next_t = t_full[0] + sample_seconds
    for i in range(1, len(t_full)):
        if t_full[i] >= next_t:
            keep_idx.append(i)
            next_t = t_full[i] + sample_seconds
    keep_idx = np.array(keep_idx, dtype=np.int64)
    alt = df['altitude'].fillna(0).to_numpy(dtype=np.float64)[keep_idx]
    spd = df['groundspeed'].fillna(0).to_numpy(dtype=np.float64)[keep_idx]
    roc = df['vertical_rate'].fillna(0).to_numpy(dtype=np.float64)[keep_idx]
    t = t_full[keep_idx]
    return t, alt, spd, roc, keep_idx


def encode_flight(df, sample_seconds):
    """Encode one flight: returns (tokens, phase_labels).

    tokens   : [BOS, t_1, t_2, ..., t_N, EOS]
    phase_labels: parallel; -1 sentinel for BOS / EOS; otherwise string phase
    """
    t, alt, spd, roc, _ = downsample_flight(df, sample_seconds)
    if len(t) < 10:
        return None, None
    phases = tag_phases(t, alt, spd, roc)
    if len(phases) != len(t):
        return None, None
    tokens = [BOS]
    phase_lbls = ["-"]
    for i in range(len(t)):
        alt_b = discretize(alt[i], ALT_EDGES)
        vr_b  = discretize(roc[i], VR_EDGES)
        spd_b = discretize(spd[i], SPD_EDGES)
        if None in (alt_b, vr_b, spd_b):
            continue
        tokens.append(bin_tuple_to_token_id(alt_b, vr_b, spd_b))
        phase_lbls.append(phases[i])
    tokens.append(EOS)
    phase_lbls.append("-")
    return tokens, phase_lbls


# ─────────────────────────────────────────────────────────────────────────────
# 4. Destroyed-structure controls
# ─────────────────────────────────────────────────────────────────────────────

def shuffle_within_flight(tokens, rng):
    """Weak control: shuffle real-token positions within each [BOS..EOS] span."""
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == BOS:
            j = i + 1
            while j < len(tokens) and tokens[j] != EOS:
                j += 1
            interior = list(tokens[i + 1 : j])
            rng.shuffle(interior)
            out.append(BOS)
            out.extend(interior)
            if j < len(tokens):
                out.append(EOS)
            i = j + 1
        else:
            out.append(tokens[i]); i += 1
    return out


def shuffle_globally(tokens, rng):
    """Strict control: shuffle real-token positions across the entire stream."""
    real_positions = [i for i, t in enumerate(tokens) if t >= N_RESERVED]
    real_values = [tokens[i] for i in real_positions]
    rng.shuffle(real_values)
    out = list(tokens)
    for pos, val in zip(real_positions, real_values):
        out[pos] = val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Splits + dump
# ─────────────────────────────────────────────────────────────────────────────

def split_flights(n_flights, val_frac, gen_frac, rng):
    indices = list(range(n_flights))
    rng.shuffle(indices)
    n_gen = int(n_flights * gen_frac)
    n_val = int(n_flights * val_frac)
    return (
        set(indices[n_gen + n_val :]),
        set(indices[n_gen : n_gen + n_val]),
        set(indices[:n_gen]),
    )


def pick_dtype(vocab_size):
    return np.uint16 if vocab_size < 2**16 else np.uint32


def dump(out_dir, splits, vocab_size):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dtype = pick_dtype(vocab_size)
    for name, payload in splits.items():
        arr = np.array(payload["tokens"], dtype=dtype)
        if arr.size:
            assert int(arr.min()) >= 0 and int(arr.max()) < vocab_size
        arr.tofile(out / f"{name}.bin")
    # build stoi/itos for (alt_b, vr_b, spd_b) → token id
    n_alt = len(ALT_EDGES) - 1
    n_vr  = len(VR_EDGES) - 1
    n_spd = len(SPD_EDGES) - 1
    stoi = {}
    itos = {}
    for alt_b in range(n_alt):
        for vr_b in range(n_vr):
            for spd_b in range(n_spd):
                tok = bin_tuple_to_token_id(alt_b, vr_b, spd_b)
                stoi[(alt_b, vr_b, spd_b)] = tok
                itos[tok] = (alt_b, vr_b, spd_b)
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({
            "vocab_size": vocab_size,
            "stoi": stoi, "itos": itos,
            "dtype": np.dtype(dtype).name,
            "pad": PAD, "bos": BOS, "eos": EOS,
            "alt_edges": ALT_EDGES, "vr_edges": VR_EDGES, "spd_edges": SPD_EDGES,
            "n_alt": n_alt, "n_vr": n_vr, "n_spd": n_spd,
        }, f)
    # phase target CSV
    with open(out / "flight_phase.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["split", "flight_idx", "token_pos", "phase"])
        for name, payload in splits.items():
            for flight_idx, start, end, phase_lbls in payload["flight_starts"]:
                for local_pos in range(end - start):
                    if phase_lbls[local_pos] == "-":
                        continue
                    w.writerow([name, flight_idx, start + local_pos,
                                phase_lbls[local_pos]])
    return dtype


# ─────────────────────────────────────────────────────────────────────────────
# 6. Driver
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(args, rng):
    print(f"[1/4] loading flights from traffic.{args.sample} ...")
    flights = list(load_flights(args.sample))
    print(f"      {len(flights)} flights available")

    print(f"[2/4] tokenizing (sample every {args.sample_seconds}s) + tagging phases ...")
    encoded = []
    for idx, df in flights:
        if args.limit is not None and len(encoded) >= args.limit:
            break
        toks, phases = encode_flight(df, args.sample_seconds)
        if toks is None:
            continue
        encoded.append((idx, toks, phases))
    print(f"      {len(encoded)} flights encoded (after min-length filter)")

    train_set, val_set, gen_set = split_flights(
        len(encoded), args.val_frac, args.gen_frac, rng)
    print(f"[3/4] split: train={len(train_set)} val={len(val_set)} gen={len(gen_set)}")

    splits = {
        "train": {"tokens": [], "flight_starts": []},
        "val":   {"tokens": [], "flight_starts": []},
        "gen":   {"tokens": [], "flight_starts": []},
    }
    for i, (flight_idx, toks, phases) in enumerate(encoded):
        if i in train_set:     name = "train"
        elif i in val_set:     name = "val"
        else:                  name = "gen"
        start = len(splits[name]["tokens"])
        splits[name]["tokens"].extend(toks)
        end = start + len(toks)
        splits[name]["flight_starts"].append((flight_idx, start, end, phases))

    if args.shuffle_within_flight:
        print("[3b/4] applying within-flight shuffle ...")
        for name in splits:
            splits[name]["tokens"] = shuffle_within_flight(splits[name]["tokens"], rng)
    if args.shuffle_globally:
        print("[3c/4] applying GLOBAL shuffle ...")
        for name in splits:
            splits[name]["tokens"] = shuffle_globally(splits[name]["tokens"], rng)

    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/adsb")
    p.add_argument("--sample", default="quickstart",
                   help="traffic.data.samples attribute name (default: quickstart)")
    p.add_argument("--sample_seconds", type=int, default=10,
                   help="downsample to one observation per N seconds")
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--gen_frac", type=float, default=0.10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle_within_flight", action="store_true")
    p.add_argument("--shuffle_globally", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    splits = build_corpus(args, rng)
    vocab = total_vocab_size()
    print(f"[4/4] writing to {args.out_dir} (vocab_size={vocab}) ...")
    dtype = dump(args.out_dir, splits, vocab)
    print("\ndone.")
    for name in ("train", "val", "gen"):
        n_tok = len(splits[name]["tokens"])
        print(f"  {name}.bin : {n_tok:>10,} tokens ({n_tok / max(1, vocab):.1f} visits/token)")
    print(f"  dtype     : {np.dtype(dtype).name}")


if __name__ == "__main__":
    main()
