"""
LatentWorldsGPT — music transplant with beat-in-measure control.

Companion to `eval/transplant_music.py`. The RSVP transplant matched A
and B on voice but let RSVP differ — and got +0.804 lift. To isolate
whether the model encodes BEAT-IN-MEASURE specifically, this script:

  - Matches donors on RSVP (control for the local feature we know is
    encoded)
  - Forces donors A and B to DIFFER on beat-in-measure (e.g., A at
    beat 1, B at beat 3)

If the music model encodes beat, transplant should shift predictions
in a beat-specific direction (more than random control would).
If the model doesn't encode beat (consistent with the probe null at
chance), transplant should give ~0 specific effect — matching random
control on every distributional metric.

We measure several "did the prediction shift" metrics:
  - max |Δ p|  (the largest probability change for any single pitch)
  - KL divergence between unpatched and patched prediction
  - Cosine similarity between unpatched and patched logits
  - Whether the argmax changes

Per-pair we compare transplant vs random control on each metric. If
they're statistically indistinguishable → confirmed null.

Usage:
    python eval/transplant_music_beat.py --ckpt checkpoints/music_expanded/best.pt \
        --data_dir data/music_expanded --layer 1 --n_donors 1500 --n_pairs 200
"""

import argparse
import pickle
import sys
import time
from collections import Counter
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
# 1. Walk stream + compute (RSVP, beat-in-measure, voice) per position
# ─────────────────────────────────────────────────────────────────────────────

def find_position_metadata(stream, itos):
    """For each global position p compute the tuple (RSVP, beat_at_next, voice_at_next)
    where:
      - RSVP = MIDI pitch at position p-3 (if real-pitch, else None)
      - beat_at_next = which beat-in-measure (1..4) position p+1 belongs to
      - voice_at_next = which voice (S=0/A=1/T=2/B=3) position p+1 is in

    Returns parallel arrays of length len(stream); None where undefined.
    """
    n = len(stream)
    rsvp = [None] * n
    beat = [None] * n
    voice = [None] * n
    bos_pos = -1
    for i in range(n):
        if stream[i] == BOS:
            bos_pos = i
        elif stream[i] in (EOS, PAD):
            bos_pos = -1
        # RSVP for predicting p+1: pitch at (p+1)-4 = p-3
        if bos_pos >= 0 and i - 3 >= bos_pos + 1:
            prev_tok = int(stream[i - 3])
            if prev_tok >= N_RESERVED:
                rsvp[i] = itos[prev_tok]
        # beat_at_next and voice_at_next: properties of position p+1
        # p+1 is in piece if bos_pos >= 0 and p+1 is before next EOS
        if bos_pos >= 0 and i + 1 < n and stream[i + 1] not in (EOS, PAD, BOS):
            local_next = i + 1 - bos_pos  # 1..4 first quartet, 5..8 second, ...
            # local_next = 1 -> S of beat 1, 2 -> A of beat 1, ..., 5 -> S of beat 2
            voice[i] = (local_next - 1) % 4
            beat[i] = ((local_next - 1) // 4) % 4 + 1   # 1..4 in 4/4
    return rsvp, beat, voice


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build donor bank with metadata
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_donor_bank(model, stream, rsvp, beat, voice, block_size, layer,
                     device, n_donors=1500, rng_seed=0):
    model.eval()
    rng = np.random.default_rng(rng_seed)
    candidates = [i for i in range(len(stream))
                  if rsvp[i] is not None and beat[i] is not None
                  and voice[i] is not None and i >= block_size]
    if len(candidates) < n_donors:
        print(f"  WARNING: only {len(candidates)} candidates available")
    rng.shuffle(candidates)
    candidates = candidates[:n_donors]

    donors = []
    batch_size = 32
    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
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
                "rsvp": rsvp[pos], "beat": beat[pos], "voice": voice[pos],
                "global_pos": pos,
                "window": np.asarray(windows[b]).copy(),
                "pos_in_window": pos_in_window[b],
            })
    return donors


# ─────────────────────────────────────────────────────────────────────────────
# 3. Forward with replacement (shared shape)
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
# 4. Run beat-controlled transplant
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_transplant(model, donors, layer, n_pairs, device,
                   rsvp_tolerance=1, rng_seed=0, batch_size=16):
    """For n_pairs triples (A, B, C):
      - A and B have the SAME voice
      - A and B have RSVP within ±rsvp_tolerance semitones (matched)
      - A's beat ≠ B's beat (differ on beat-in-measure)
      - C is a random donor (different voice or random position) for control

    Score per pair:
      - max |Δ p|  for unpatched-vs-transplant and unpatched-vs-random
      - KL divergence (unp || trp) and (unp || rnd)
      - argmax changed under transplant? under random?

    If transplant >> random on these → beat encoding exists.
    If transplant ≈ random → null (consistent with probe finding).
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)

    # Index donors by (voice, beat) for matching
    by_voice_beat = {}
    for i, d in enumerate(donors):
        by_voice_beat.setdefault((d["voice"], d["beat"]), []).append(i)

    print(f"  donor distribution over (voice, beat):")
    for k, v in sorted(by_voice_beat.items()):
        print(f"    voice={k[0]} beat={k[1]} : {len(v)} donors")

    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        # Pick voice
        voices = list({v for (v, b) in by_voice_beat})
        voice = int(rng.choice(voices))
        # Pick two different beats with donors available
        beats = sorted({b for (v, b) in by_voice_beat if v == voice})
        if len(beats) < 2:
            continue
        beat_a, beat_b = rng.choice(beats, size=2, replace=False).tolist()
        # Pick A with voice & beat_a; B with voice & beat_b matching RSVP
        cands_a = by_voice_beat[(voice, beat_a)]
        cands_b = by_voice_beat[(voice, beat_b)]
        if not cands_a or not cands_b:
            continue
        a_i = int(rng.choice(cands_a))
        a = donors[a_i]
        # Find a B with matched RSVP
        b_candidates = [i for i in cands_b
                        if abs(donors[i]["rsvp"] - a["rsvp"]) <= rsvp_tolerance]
        if not b_candidates:
            continue
        b_i = int(rng.choice(b_candidates))
        # Random control C: any donor different from A, B
        all_other = [i for i in range(len(donors))
                     if i != a_i and i != b_i]
        c_i = int(rng.choice(all_other))
        pairs.append((a_i, b_i, c_i))
    print(f"  sampled {len(pairs)} matched-RSVP / differ-on-beat pairs")
    if len(pairs) == 0:
        print("  no usable pairs; exiting")
        return None

    metrics = {
        "trp_maxdp": [], "rnd_maxdp": [],
        "trp_kl":    [], "rnd_kl":    [],
        "trp_argmax_changed": [], "rnd_argmax_changed": [],
    }
    t0 = time.time()
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        B = len(batch)
        windows = []; positions = []
        b_residuals = []; c_residuals = []
        for (a_i, b_i, c_i) in batch:
            a = donors[a_i]
            windows.append(a["window"])
            positions.append(a["pos_in_window"])
            b_residuals.append(donors[b_i]["residual"])
            c_residuals.append(donors[c_i]["residual"])
        idx_batch = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
        b_vecs = torch.stack(b_residuals)
        c_vecs = torch.stack(c_residuals)

        logits_unp, _ = model(idx_batch)
        logits_trp = forward_with_replacement(model, idx_batch, layer, positions, b_vecs)
        logits_rnd = forward_with_replacement(model, idx_batch, layer, positions, c_vecs)

        for k in range(B):
            t = positions[k]
            p_unp = F.softmax(logits_unp[k, t], dim=-1).cpu().numpy()
            p_trp = F.softmax(logits_trp[k, t], dim=-1).cpu().numpy()
            p_rnd = F.softmax(logits_rnd[k, t], dim=-1).cpu().numpy()

            metrics["trp_maxdp"].append(float(np.max(np.abs(p_trp - p_unp))))
            metrics["rnd_maxdp"].append(float(np.max(np.abs(p_rnd - p_unp))))

            # KL(unp || patched), with epsilon for stability
            eps = 1e-10
            kl_trp = float(np.sum(p_unp * (np.log(p_unp + eps) - np.log(p_trp + eps))))
            kl_rnd = float(np.sum(p_unp * (np.log(p_unp + eps) - np.log(p_rnd + eps))))
            metrics["trp_kl"].append(kl_trp)
            metrics["rnd_kl"].append(kl_rnd)

            argmax_unp = int(p_unp.argmax())
            metrics["trp_argmax_changed"].append(int(p_trp.argmax()) != argmax_unp)
            metrics["rnd_argmax_changed"].append(int(p_rnd.argmax()) != argmax_unp)

    print(f"  done in {time.time() - t0:.1f}s")
    return metrics, len(pairs)


def summarize(metrics, n_pairs):
    print(f"\nn_pairs: {n_pairs}\n")
    print(f"  {'Metric':<30}{'Transplant':>15}{'Random control':>20}{'Δ (trp - rnd)':>18}")
    print("  " + "─" * 80)
    for label, key in [
        ("max |Δ p|  (mean)",       "maxdp"),
        ("KL(unp || patched) (mean)", "kl"),
        ("argmax changed (rate)",   "argmax_changed"),
    ]:
        trp_v = float(np.mean(metrics[f"trp_{key}"]))
        rnd_v = float(np.mean(metrics[f"rnd_{key}"]))
        delta = trp_v - rnd_v
        print(f"  {label:<30}{trp_v:>15.4f}{rnd_v:>20.4f}{delta:>+18.4f}")
    print()
    return {
        "trp_maxdp_mean": float(np.mean(metrics["trp_maxdp"])),
        "rnd_maxdp_mean": float(np.mean(metrics["rnd_maxdp"])),
        "trp_kl_mean":    float(np.mean(metrics["trp_kl"])),
        "rnd_kl_mean":    float(np.mean(metrics["rnd_kl"])),
        "trp_argmax_changed_rate": float(np.mean(metrics["trp_argmax_changed"])),
        "rnd_argmax_changed_rate": float(np.mean(metrics["rnd_argmax_changed"])),
        "n_pairs": n_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", choices=("val", "gen"), default="gen")
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--n_donors", type=int, default=1500)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--rsvp_tolerance", type=int, default=1,
                   help="max |Δ RSVP| (semitones) to consider A and B matched")
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
    print(f"matching tolerance: |Δ RSVP| ≤ {args.rsvp_tolerance} semitones")

    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    itos = meta["itos"]
    dtype = np.dtype(meta["dtype"])
    stream = np.asarray(np.memmap(data_dir / f"{args.split}.bin", dtype=dtype, mode="r"))
    print(f"{args.split}.bin: {len(stream):,} tokens")

    print("\n[1/3] computing position metadata (RSVP, beat, voice) ...")
    rsvp, beat, voice = find_position_metadata(stream, itos)
    n_valid = sum(1 for i in range(len(stream))
                  if rsvp[i] is not None and beat[i] is not None
                  and voice[i] is not None)
    print(f"  {n_valid:,} positions have full metadata")

    print(f"\n[2/3] building donor bank (n_donors={args.n_donors}) ...")
    donors = build_donor_bank(model, stream, rsvp, beat, voice,
                              config.block_size, args.layer, device,
                              n_donors=args.n_donors, rng_seed=args.seed)
    print(f"  built {len(donors)} donors")

    print(f"\n[3/3] running beat-controlled transplant ...")
    result = run_transplant(model, donors, args.layer, args.n_pairs, device,
                            rsvp_tolerance=args.rsvp_tolerance,
                            rng_seed=args.seed,
                            batch_size=args.batch_size)
    if result is None:
        return
    metrics, n_pairs = result
    summary = summarize(metrics, n_pairs)

    print(f"\n{'─'*78}\nINTERPRETATION\n{'─'*78}")
    delta_maxdp = summary["trp_maxdp_mean"] - summary["rnd_maxdp_mean"]
    delta_kl = summary["trp_kl_mean"] - summary["rnd_kl_mean"]
    if delta_maxdp < 0.05 and delta_kl < 0.05:
        print(f"  ✓ NULL CONFIRMED: transplant ≈ random control on every metric.")
        print(f"    Music model does NOT encode beat-in-measure in a way the")
        print(f"    framework can manipulate. Consistent with classification")
        print(f"    probe at chance.")
    elif delta_maxdp > 0.10 or delta_kl > 0.10:
        print(f"  ⚠ NULL CHALLENGED: transplant has a substantively larger")
        print(f"    effect than random control.")
        print(f"    The model may encode beat-in-measure in a way the linear")
        print(f"    classification probe cannot read. Investigate.")
    else:
        print(f"  ~ AMBIGUOUS: transplant slightly larger than random control")
        print(f"    but not substantively so. Likely weak/no beat encoding.")


if __name__ == "__main__":
    main()
