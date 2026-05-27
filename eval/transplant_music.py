"""
LatentWorldsGPT — music transplant (causal check on voice-leading state).

The music-domain analog of `eval/transplant.py` (cities) and
`eval/transplant_othello.py` (Othello). Tests whether the music model's
residual stream causally encodes the "expected next pitch" via the
recent same-voice context.

Why voice-leading not beat/mode/chord
=====================================
Music's classification probes for beat/mode/chord all show trained ≈
untrained (`updateMay26_afternoon.md`) — those features aren't encoded.
But voice-leading rate is 98.99 % on held-out pieces, meaning the
model HAS encoded SOMETHING needed to predict next-same-voice pitches
within ~7 semitones. The transplant checks whether THAT encoded state
is causally responsible for predictions (it should be, if voice-leading
is what the model learned).

Setup
=====
For each position p, the model's prediction at p+1 should be a pitch
in voice V = (p+1 - bos_pos - 1) mod 4 close to the pitch at
position (p+1) - 4 = p - 3 (the same voice's previous pitch). Call this
"recent same-voice pitch for next slot" or RSVP.

Test:
  A: position p_A with RSVP_A.
  B: position p_B with RSVP_B such that |RSVP_A - RSVP_B| ≥ 12 semitones.
  Patch residual at A's position with B's residual.
  Score:
    ψ(X) = P(predicted pitch within ±7 semitones of X)
  Expect:
    Trained model:    ψ(B)_transplant > ψ(B)_unpatched (and > ψ(B)_random)
    Random control:   ψ(C)_random ≈ ψ(C)_unpatched   (no specificity)
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model"))

from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

PAD, BOS, EOS, REST = 0, 1, 2, 3
N_RESERVED = 4


# ─────────────────────────────────────────────────────────────────────────────
# 1. Walk stream → pieces → per-position (voice, recent-same-voice-pitch-at-next)
# ─────────────────────────────────────────────────────────────────────────────

def find_rsvp_per_position(stream, itos):
    """For each global position p in the stream, compute:
        rsvp[p] = MIDI pitch of the same-voice token at position (p+1)-4 = p-3,
                  IF that position is within the current [BOS..EOS] piece AND
                  is a real-pitch token (not BOS/EOS/PAD/REST).
                  Otherwise None.

    Used to label donor positions by "the pitch the model is conditioned on
    when predicting position p+1."
    """
    n = len(stream)
    rsvp = [None] * n
    bos_pos = -1
    for i in range(n):
        if stream[i] == BOS:
            bos_pos = i
        elif stream[i] in (EOS, PAD):
            bos_pos = -1
        # We care about positions i where i-3 >= bos_pos+1 (real-pitch slot)
        # and i+1 is still within the same piece (we'll check at sample time
        # that p+1 isn't EOS).
        if bos_pos >= 0 and i - 3 >= bos_pos + 1:
            prev_tok = int(stream[i - 3])
            if prev_tok >= N_RESERVED:
                rsvp[i] = itos[prev_tok]
    return rsvp


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build donor bank
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_donor_bank(model, stream, rsvp, block_size, layer, device,
                     n_donors=500, rng_seed=0):
    """Sample n_donors positions where rsvp is defined. Cache residual at
    `layer` and record the RSVP pitch.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)
    candidate_positions = [i for i, x in enumerate(rsvp)
                           if x is not None and i >= block_size]
    if len(candidate_positions) < n_donors:
        print(f"  WARNING: only {len(candidate_positions)} candidates available")
    rng.shuffle(candidate_positions)
    candidate_positions = candidate_positions[:n_donors]

    donors = []
    batch_size = 32
    for batch_start in range(0, len(candidate_positions), batch_size):
        batch = candidate_positions[batch_start : batch_start + batch_size]
        windows = []; pos_in_window = []
        for pos in batch:
            win_start = pos - block_size + 1
            windows.append(stream[win_start : win_start + block_size])
            pos_in_window.append(block_size - 1)
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
        acts = cache_layer_activations(model, idx_batch)
        acts_L = acts[layer]
        for b, pos in enumerate(batch):
            donors.append({
                "residual": acts_L[b, pos_in_window[b]].clone(),
                "rsvp": rsvp[pos],
                "global_pos": pos,
                "window": np.asarray(windows[b]).copy(),
                "pos_in_window": pos_in_window[b],
            })
    print(f"  donor bank: {len(donors)} positions")
    return donors


# ─────────────────────────────────────────────────────────────────────────────
# 3. Forward with replacement (mirror cities/transplant.py)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run transplant
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_transplant(model, donors, stoi, itos, layer, n_pairs, device,
                   rsvp_separation=12, voice_leading_band=7,
                   rng_seed=0, batch_size=16):
    """For n_pairs pairs (A, B, C) where:
       - A's RSVP and B's RSVP differ by ≥ rsvp_separation semitones
       - C is a random donor (also with RSVP ≥ rsvp_separation away from A AND from B)
    Run unpatched / transplant-B / random-C forward passes; score the
    probability mass on pitches within ±voice_leading_band semitones of
    A's RSVP, B's RSVP, C's RSVP.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)

    # Pre-compute "near pitches" for each unique RSVP
    # Token list of pitches within ±band semitones of a given MIDI value:
    def near_tokens(midi_target):
        return [stoi[m] for m in stoi
                if abs(m - midi_target) <= voice_leading_band]

    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 100:
        attempts += 1
        a_i, b_i, c_i = rng.integers(0, len(donors), size=3).tolist()
        if len({a_i, b_i, c_i}) < 3:
            continue
        a, b, c = donors[a_i], donors[b_i], donors[c_i]
        if abs(a["rsvp"] - b["rsvp"]) < rsvp_separation:
            continue
        if abs(a["rsvp"] - c["rsvp"]) < rsvp_separation:
            continue
        if abs(b["rsvp"] - c["rsvp"]) < rsvp_separation:
            continue
        pairs.append((a_i, b_i, c_i))
    print(f"  sampled {len(pairs)} test pairs (target {n_pairs}; "
          f"required |Δ RSVP| ≥ {rsvp_separation} semitones)")

    results = {"unp_PA": [], "unp_PB": [], "unp_PC": [],
               "trp_PA": [], "trp_PB": [], "trp_PC": [],
               "rnd_PA": [], "rnd_PB": [], "rnd_PC": []}
    t0 = time.time()
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        B = len(batch)
        windows = []; positions = []
        a_near = []; b_near = []; c_near = []
        b_residuals = []; c_residuals = []
        for (a_i, b_i, c_i) in batch:
            a, b_, c_ = donors[a_i], donors[b_i], donors[c_i]
            windows.append(a["window"])
            positions.append(a["pos_in_window"])
            a_near.append(near_tokens(a["rsvp"]))
            b_near.append(near_tokens(b_["rsvp"]))
            c_near.append(near_tokens(c_["rsvp"]))
            b_residuals.append(b_["residual"])
            c_residuals.append(c_["residual"])
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
        b_vecs = torch.stack(b_residuals)
        c_vecs = torch.stack(c_residuals)

        logits_unp, _ = model(idx_batch)
        logits_trp = forward_with_replacement(model, idx_batch, layer, positions, b_vecs)
        logits_rnd = forward_with_replacement(model, idx_batch, layer, positions, c_vecs)

        for k in range(B):
            t = positions[k]
            for label, logits in (("unp", logits_unp), ("trp", logits_trp), ("rnd", logits_rnd)):
                probs = F.softmax(logits[k, t], dim=-1).cpu().numpy()
                results[f"{label}_PA"].append(float(probs[a_near[k]].sum()))
                results[f"{label}_PB"].append(float(probs[b_near[k]].sum()))
                results[f"{label}_PC"].append(float(probs[c_near[k]].sum()))
    print(f"  done in {time.time() - t0:.1f}s")
    return results


def summarize(results):
    n = len(results["unp_PA"])
    out = {"n": n}
    for label in ("unp", "trp", "rnd"):
        for s in ("PA", "PB", "PC"):
            out[f"{label}_{s}"] = float(np.mean(results[f"{label}_{s}"]))
    out["delta_PB_trp_over_unp"] = out["trp_PB"] - out["unp_PB"]
    out["delta_PB_trp_over_rnd"] = out["trp_PB"] - out["rnd_PB"]
    out["delta_PA_trp_over_unp"] = out["trp_PA"] - out["unp_PA"]
    out["rate_trp_beats_rnd_on_PB"] = sum(
        1 for i in range(n)
        if results["trp_PB"][i] > results["rnd_PB"][i]
    ) / max(1, n)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", choices=("val", "gen"), default="gen")
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--n_donors", type=int, default=500)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--rsvp_separation", type=int, default=12,
                   help="min |Δ| between A's and B's recent-same-voice pitch")
    p.add_argument("--voice_leading_band", type=int, default=7)
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
    print(f"loaded ckpt: iter={ckpt.get('iter','?')}  val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}")
    print(f"layer = {args.layer} (of {config.n_layer} blocks)")

    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    dtype = np.dtype(meta["dtype"])
    stream = np.asarray(np.memmap(data_dir / f"{args.split}.bin",
                                  dtype=dtype, mode="r"))
    print(f"{args.split}.bin: {len(stream):,} tokens")

    print("\n[1/3] computing RSVP labels for each position ...")
    rsvp = find_rsvp_per_position(stream, itos)
    n_valid = sum(1 for x in rsvp if x is not None)
    print(f"  {n_valid:,} positions have a valid RSVP label")

    print(f"\n[2/3] building donor bank (n_donors={args.n_donors}) ...")
    donors = build_donor_bank(model, stream, rsvp, config.block_size,
                              args.layer, device, n_donors=args.n_donors,
                              rng_seed=args.seed)

    print(f"\n[3/3] running transplant interventions (n_pairs={args.n_pairs}) ...")
    results = run_transplant(model, donors, stoi, itos, args.layer,
                             args.n_pairs, device,
                             rsvp_separation=args.rsvp_separation,
                             voice_leading_band=args.voice_leading_band,
                             rng_seed=args.seed,
                             batch_size=args.batch_size)

    summary = summarize(results)
    print(f"\nn_scored: {summary['n']}")
    print(f"  {'condition':<12}{'P(near A)':>12}{'P(near B)':>12}{'P(near C)':>12}")
    for label in ("unp", "trp", "rnd"):
        print(f"  {label:<12}{summary[f'{label}_PA']:>12.4f}"
              f"{summary[f'{label}_PB']:>12.4f}"
              f"{summary[f'{label}_PC']:>12.4f}")
    print(f"\nEffect sizes:")
    print(f"  Δ P(near B's RSVP)  transplant − unpatched : "
          f"{summary['delta_PB_trp_over_unp']:+.4f}")
    print(f"  Δ P(near B's RSVP)  transplant − random    : "
          f"{summary['delta_PB_trp_over_rnd']:+.4f}")
    print(f"  Δ P(near A's RSVP)  transplant − unpatched : "
          f"{summary['delta_PA_trp_over_unp']:+.4f}")
    print(f"  rate(transplant > random on P(near B's RSVP)) : "
          f"{summary['rate_trp_beats_rnd_on_PB']*100:.1f}%")


if __name__ == "__main__":
    main()
