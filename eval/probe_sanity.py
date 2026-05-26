"""
Sanity probe for the music probe code.

Two probe targets — both per-position, both 4-class:

  (1) BLOCK_VOICE  = block-position T mod 4.
      Trivially encoded in the model's positional embedding (wpe).
      A linear classifier should recover this from any layer's
      activations at near-100% — even on a random-init model. If it
      does NOT, the probe code is broken (we're not reading the
      activations correctly).

  (2) PIECE_VOICE  = (stream_position - last_BOS_position - 1) mod 4.
      Equals the voice slot (S/A/T/B = 0/1/2/3) for tokens emitted by
      our 4-voice-per-beat encoding. Recoverable only from CONTENT —
      the model has to find BOS in its attention window or recognize
      the voice from pitch range. Trained model SHOULD get this; an
      untrained model with random embeddings probably won't.

  block_voice and piece_voice are equal when the block start aligns
  with a piece boundary; they differ by the (random) offset of the
  window into the piece otherwise. A working probe should show:

    block_voice : near-100% on trained AND untrained
    piece_voice : high on trained, near-chance on untrained
                  (and higher than block_voice would suggest)

Usage:
    python eval/probe_sanity.py --ckpt checkpoints/music_bach/best.pt \\
        --data_dir data/music_bach
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402
from probe import cache_layer_activations  # noqa: E402

PAD, BOS, EOS, REST = 0, 1, 2, 3


def compute_piece_voice_labels(stream):
    """For each position in stream, compute piece-local voice
    (0=S, 1=A, 2=T, 3=B) based on offset from most-recent BOS.
    Returns -1 for control-token positions and positions before any BOS."""
    labels = np.full(len(stream), -1, dtype=np.int64)
    bos_pos = -1
    for i, t in enumerate(stream):
        if t == BOS:
            bos_pos = i
        elif t == EOS or t == PAD:
            pass
        else:
            if bos_pos >= 0:
                labels[i] = (i - bos_pos - 1) % 4
    return labels


@torch.no_grad()
def build_probe_dataset(model, stream, voice_labels, block_size, n_positions,
                        device, rng_seed=0):
    model.eval()
    rng = np.random.default_rng(rng_seed)
    all_X = None
    all_piece_voice = []
    all_block_voice = []
    n_collected = 0
    while n_collected < n_positions:
        starts = rng.integers(0, len(stream) - block_size - 1, size=32)
        windows = [stream[s : s + block_size] for s in starts]
        idx = torch.from_numpy(np.stack(windows).astype(np.int64)).to(device)
        acts = cache_layer_activations(model, idx)
        if all_X is None:
            all_X = [[] for _ in range(len(acts))]
        for b in range(idx.shape[0]):
            for t in range(idx.shape[1]):
                pos = int(starts[b]) + t
                if pos >= len(voice_labels):
                    continue
                if voice_labels[pos] == -1:
                    continue
                for L in range(len(acts)):
                    all_X[L].append(acts[L][b, t].cpu().numpy())
                all_piece_voice.append(int(voice_labels[pos]))
                all_block_voice.append(t % 4)
                n_collected += 1
                if n_collected >= n_positions:
                    break
            if n_collected >= n_positions:
                break
    X = [np.stack(L) for L in all_X]
    return X, np.array(all_piece_voice), np.array(all_block_voice)


class LinearProbe(nn.Module):
    def __init__(self, in_dim, n):
        super().__init__()
        self.lin = nn.Linear(in_dim, n)

    def forward(self, x):
        return self.lin(x)


def train_eval(probe, Xtr, ytr, Xte, yte, device, epochs=30):
    probe = probe.to(device)
    Xtr = torch.from_numpy(Xtr).float().to(device)
    ytr = torch.from_numpy(ytr).long().to(device)
    Xte = torch.from_numpy(Xte).float().to(device)
    yte = torch.from_numpy(yte).long().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    best = -1.0
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), 512):
            ix = perm[i : i + 512]
            loss = F.cross_entropy(probe(Xtr[ix]), ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (probe(Xte).argmax(-1) == yte).float().mean().item()
            best = max(best, acc)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_positions", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    trained = GPT(config).to(device); trained.eval()
    untrained = GPT(config).to(device); untrained.eval()

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    streams = {
        s: np.asarray(np.memmap(data_dir / f"{s}.bin", dtype=dtype, mode="r"))
        for s in ("val", "gen")
    }
    combined = np.concatenate([streams["val"], streams["gen"]]).astype(np.int64)
    voice = compute_piece_voice_labels(combined)
    print(f"combined stream: {len(combined):,} tokens;"
          f" {(voice >= 0).sum():,} voice-labeled positions")

    for label, model in [("TRAINED", trained), ("UNTRAINED (random init)", untrained)]:
        print(f"\n=== {label} ===")
        X, y_piece, y_block = build_probe_dataset(
            model, combined, voice, config.block_size,
            args.n_positions, device, rng_seed=args.seed,
        )
        n = len(y_piece)
        perm = np.random.default_rng(args.seed).permutation(n)
        ntr = int(n * 0.8)
        tr_ix, te_ix = perm[:ntr], perm[ntr:]

        # Print majority-class baselines
        from collections import Counter
        cnt_p = Counter(y_piece.tolist())
        cnt_b = Counter(y_block.tolist())
        maj_p = max(cnt_p.values()) / n
        maj_b = max(cnt_b.values()) / n
        print(f"  majority-class baseline: piece_voice={maj_p:.3f}  "
              f"block_voice={maj_b:.3f}  (chance=0.25)")
        print(f"  {'Layer':<6}{'piece_voice':>14}{'block_voice':>14}")
        for L in range(len(X)):
            acc_p = train_eval(
                LinearProbe(X[L].shape[1], 4),
                X[L][tr_ix], y_piece[tr_ix], X[L][te_ix], y_piece[te_ix],
                device, args.epochs,
            )
            acc_b = train_eval(
                LinearProbe(X[L].shape[1], 4),
                X[L][tr_ix], y_block[tr_ix], X[L][te_ix], y_block[te_ix],
                device, args.epochs,
            )
            lay = "embed" if L == 0 else f"L{L}"
            print(f"  {lay:<6}{acc_p:>14.4f}{acc_b:>14.4f}")


if __name__ == "__main__":
    main()
