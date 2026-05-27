"""
LatentWorldsGPT — Othello valid-move-rate evaluator.

Othello-domain analog of cities `valid_edge.py` and music
`valid_voice_step.py`. Independently of any probe, asks: does the model
learn Othello rules? Specifically: does the greedy next-token prediction
correspond to a legal move (or a legitimate pass) on the board state
inherited from the prefix?

This metric is critical for interpreting the Othello-GPT probe result.
The published Li 2022 / Nanda 2023 board-state probe assumes the model
has learned to play. If our model's valid-move rate is too low, the
probe failure is "model didn't learn"; if valid-move rate is high but
probe still fails, the framework has a bug.

MODE A — next-step valid-move-rate
==================================
For each position in the stream with a real next-token target, run the
model on the prefix, greedy-decode, and check:
  - LEGAL_MOVE       : prediction is one of the legal moves
  - LEGAL_PASS       : prediction is PASS and there were no legal moves
  - ILLEGAL_PASS     : prediction is PASS but legal moves existed
  - ILLEGAL_MOVE     : prediction is a move-position but not legal
  - INVALID_TOKEN    : prediction is PAD/BOS/EOS (impossible)

Whose turn at position p: alternates starting with BLACK at position 1.
PASS doesn't break the alternation (next player still tries).

THE ONE RULE
============
This script never reads board_state.csv. Instead, it REPLAYS the game
internally to determine legal moves — which exactly matches what the
model has to do too.

Usage:
  python eval/valid_othello_move.py --ckpt checkpoints/othello/best.pt \\
      --data_dir data/othello --split gen
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from model import GPT, GPTConfig  # noqa: E402

# Mirror data/prepare_othello.py
PAD, BOS, EOS, PASS = 0, 1, 2, 3
N_RESERVED = 4
EMPTY, BLACK, WHITE = 0, 1, 2

# Re-import othello rules via the prepare_othello module
import importlib.util
_PREPARE = Path(__file__).resolve().parent.parent / "data" / "prepare_othello.py"
_spec = importlib.util.spec_from_file_location("prepare_othello", _PREPARE)
po = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(po)


def token_to_cell(tok):
    """Convert a real-move token to (r, c)."""
    if tok < N_RESERVED:
        return None
    pos = tok - N_RESERVED
    return (pos // 8, pos % 8)


def cell_to_token(r, c):
    return N_RESERVED + r*8 + c


# ─────────────────────────────────────────────────────────────────────────────
# Game replay
# ─────────────────────────────────────────────────────────────────────────────

def replay_game(tokens):
    """Replay a [BOS, ..., EOS] token sequence. Returns list of dicts, one
    per token position, with: board_before, board_after, color_to_move,
    legal_moves.

    states[p]:
      "board_before"   : board state BEFORE token p was applied
      "board_after"    : board state AFTER token p was applied
      "color_to_move"  : whose turn it WAS at position p
      "legal_moves"    : set of (r, c) tuples legal at this position
    """
    states = []
    board = po.initial_board()
    color = BLACK
    for p, tok in enumerate(tokens):
        if tok == BOS:
            states.append({
                "board_before": list(board), "board_after": list(board),
                "color_to_move": None, "legal_moves": set(),
            })
            continue
        if tok == EOS or tok == PAD:
            states.append({
                "board_before": list(board), "board_after": list(board),
                "color_to_move": None, "legal_moves": set(),
            })
            continue
        # Real move or PASS — `color` is the player about to move
        legal = set(po.legal_moves(board, color))
        state = {
            "board_before": list(board),
            "color_to_move": color,
            "legal_moves": legal,
        }
        if tok == PASS:
            pass  # board unchanged
        else:
            cell = token_to_cell(tok)
            if cell is not None and po.is_legal(board, cell[0], cell[1], color):
                board = po.apply_move(board, cell[0], cell[1], color)
        state["board_after"] = list(board)
        states.append(state)
        color = WHITE if color == BLACK else BLACK
    return states


# ─────────────────────────────────────────────────────────────────────────────
# Walk stream → list of game token-spans
# ─────────────────────────────────────────────────────────────────────────────

def split_games(stream):
    """Return list of np-array token spans, one per [BOS .. EOS] game."""
    games = []
    i = 0
    while i < len(stream):
        if stream[i] != BOS:
            i += 1; continue
        j = i + 1
        while j < len(stream) and stream[j] != EOS:
            j += 1
        # Include the EOS in the span
        end = j + 1 if j < len(stream) else j
        games.append(np.asarray(stream[i:end]))
        i = end
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Mode A — next-step
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def next_step_valid_move(model, games, block_size, device, vocab_size):
    """For each scored position in each game, score the model's greedy
    next-token prediction. Returns aggregate counters.
    """
    cats = Counter()
    n_scored = 0

    for game in games:
        if len(game) < 3:
            continue
        states = replay_game(game.tolist())
        # truncate to block_size for the forward pass
        seq = game[:block_size] if len(game) > block_size else game
        x = torch.from_numpy(seq.astype(np.int64)).unsqueeze(0).to(device)
        logits, _ = model(x)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        # The model's prediction at block-position t predicts position t+1.
        # We want to score position t+1 against the legal moves at that
        # position. The state at position t+1 has color_to_move and
        # legal_moves derived from board_after position t.
        for t in range(len(seq) - 1):
            next_pos = t + 1
            if next_pos >= len(states):
                break
            state = states[next_pos]
            if state["color_to_move"] is None:
                continue  # next position is BOS / EOS / PAD
            pred_tok = int(preds[t])
            legal = state["legal_moves"]
            n_scored += 1

            if pred_tok == PASS:
                if not legal:
                    cats["LEGAL_PASS"] += 1
                else:
                    cats["ILLEGAL_PASS"] += 1
            elif pred_tok in (PAD, BOS, EOS):
                cats["INVALID_TOKEN"] += 1
            else:
                cell = token_to_cell(pred_tok)
                if cell in legal:
                    cats["LEGAL_MOVE"] += 1
                else:
                    cats["ILLEGAL_MOVE"] += 1

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
    print(f"loaded ckpt: iter={ckpt.get('iter','?')}  "
          f"val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}")

    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    dtype = np.dtype(meta["dtype"])
    stream = np.asarray(np.memmap(data_dir / f"{args.split}.bin",
                                  dtype=dtype, mode="r"))
    games = split_games(stream)
    print(f"{args.split}.bin: {len(stream):,} tokens / {len(games)} games")

    cats, n_scored = next_step_valid_move(
        model, games, config.block_size, device, config.vocab_size,
    )
    print(f"\n[next-step / {args.split}.bin]")
    print(f"  positions scored : {n_scored:,}")
    if n_scored == 0:
        print("  (no positions scored — exiting)")
        return
    valid = cats["LEGAL_MOVE"] + cats["LEGAL_PASS"]
    print(f"  ── breakdown ──")
    for k in ("LEGAL_MOVE", "LEGAL_PASS",
              "ILLEGAL_PASS", "ILLEGAL_MOVE", "INVALID_TOKEN"):
        v = cats[k]
        print(f"    {k:<14} : {v:>7,} ({v/n_scored*100:6.2f}%)")
    print(f"  ── headline ──")
    print(f"    VALID rate (legal move OR legitimate pass) : "
          f"{valid/n_scored*100:6.2f}%   (target ≥ 95% for a model that "
          f"learned Othello rules)")


if __name__ == "__main__":
    main()
