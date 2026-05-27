"""
LatentWorldsGPT — Othello-GPT reproduction (Li 2022, Nanda 2023).

The end-to-end framework sanity check. If `eval/probe_othello.py` lands
the published ~90 % per-cell board-state probe accuracy on a model
trained from this pipeline, the entire framework (model + training +
activation extraction + multi-seed probe) is validated. Music's null
results are then principled (the N criterion fails for music), not
artifacts.

Othello rules (8x8):
  - Initial position: WHITE at (3,3) and (4,4); BLACK at (3,4) and (4,3).
  - BLACK moves first.
  - A move places a piece of the mover's color on an empty square such
    that, in at least one of 8 directions (N/S/E/W + 4 diagonals),
    there's a contiguous line of opponent pieces ending at one of the
    mover's own pieces. All such lines flip to the mover's color.
  - If no legal move, the player passes.
  - Game ends when both players pass consecutively or the board fills.

Token convention:
  PAD=0, BOS=1, EOS=2, PASS=3, board-cells start at 4.
  Move at (r, c) = token 4 + r*8 + c   (row-major).

THE ONE RULE (Othello edition): the probe-target table
`board_state.csv` is NEVER read by the model. Tokens are move positions
only. The model has to DERIVE the board from the move history — which
is exactly what makes the board-state probe meaningful.

Outputs (all in --out_dir):
  train.bin / val.bin / gen.bin    uint16 token streams
  meta.pkl                          {vocab_size, stoi, itos, ...}
  board_state.csv                   split,game_idx,token_pos,cell_states
                                    where cell_states is a hyphen-joined
                                    string of 64 integers in {0,1,2}.

Usage:
  python data/prepare_othello.py --n_games 5000 --out_dir data/othello
"""

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np

PAD, BOS, EOS, PASS = 0, 1, 2, 3
N_RESERVED = 4

EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Othello rules
# ─────────────────────────────────────────────────────────────────────────────

def initial_board() -> list[int]:
    """8x8 = 64-cell board, row-major. Center 4 cells pre-filled per
    Othello starting position."""
    b = [EMPTY] * 64
    b[3*8 + 3] = WHITE
    b[3*8 + 4] = BLACK
    b[4*8 + 3] = BLACK
    b[4*8 + 4] = WHITE
    return b


def flips_in_direction(board, r, c, dr, dc, color):
    """Return list of (r,c) positions that would be flipped if `color`
    plays at (r,c) and there's a captured line in direction (dr,dc).
    Returns [] if no captured line in that direction.
    """
    opp = WHITE if color == BLACK else BLACK
    line = []
    nr, nc = r + dr, c + dc
    while 0 <= nr < 8 and 0 <= nc < 8 and board[nr*8 + nc] == opp:
        line.append((nr, nc))
        nr += dr
        nc += dc
    if not line:
        return []
    if 0 <= nr < 8 and 0 <= nc < 8 and board[nr*8 + nc] == color:
        return line
    return []


def is_legal(board, r, c, color):
    """A move is legal if the cell is empty AND at least one direction
    produces flips."""
    if board[r*8 + c] != EMPTY:
        return False
    for dr, dc in DIRECTIONS:
        if flips_in_direction(board, r, c, dr, dc, color):
            return True
    return False


def legal_moves(board, color):
    """All legal (r,c) positions for `color`."""
    moves = []
    for r in range(8):
        for c in range(8):
            if is_legal(board, r, c, color):
                moves.append((r, c))
    return moves


def apply_move(board, r, c, color):
    """Apply a move (assumes legality already checked). Returns a NEW
    board. Flips all captured lines.
    """
    b = list(board)
    b[r*8 + c] = color
    for dr, dc in DIRECTIONS:
        flips = flips_in_direction(b, r, c, dr, dc, color)
        for fr, fc in flips:
            b[fr*8 + fc] = color
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random-game generation
# ─────────────────────────────────────────────────────────────────────────────

def play_random_game(rng: random.Random):
    """Generate one random game. Returns list of (action, board_state_after)
    where action is either ("move", r, c) or ("pass",).

    The game proceeds with BLACK first, alternating. Each turn the player
    picks a uniform random legal move; if none, passes. Two consecutive
    passes end the game.
    """
    board = initial_board()
    history = []   # list of (action_tuple, board_after_action)
    color = BLACK
    consecutive_passes = 0
    while consecutive_passes < 2 and sum(1 for c in board if c == EMPTY) > 0:
        moves = legal_moves(board, color)
        if not moves:
            history.append((("pass",), list(board)))
            consecutive_passes += 1
        else:
            r, c = moves[rng.randrange(len(moves))]
            board = apply_move(board, r, c, color)
            history.append((("move", r, c), list(board)))
            consecutive_passes = 0
        color = WHITE if color == BLACK else BLACK
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer():
    """64 board positions + 4 reserved (PAD/BOS/EOS/PASS) = vocab 68.
    Token for cell (r, c) = N_RESERVED + r*8 + c.
    PASS is a special control token (index 3).
    """
    stoi = {}
    itos = {}
    for r in range(8):
        for c in range(8):
            tok = N_RESERVED + r*8 + c
            stoi[(r, c)] = tok
            itos[tok] = (r, c)
    vocab_size = 64 + N_RESERVED
    return stoi, itos, vocab_size


def encode_game(history, stoi):
    """Turn a game's history into a token stream + per-position board state.

    tokens: [BOS, action1, action2, ..., actionN, EOS]
    boards: list, len = len(tokens). board[0] = pre-move (identity board)
            though we use the initial board here so probe targets are
            defined at BOS too. board[k] for k in [1..N] = board AFTER
            applying action k. board[N+1] (EOS) = same as board[N].
    """
    tokens = [BOS]
    boards = [initial_board()]
    for action, board_after in history:
        if action[0] == "pass":
            tokens.append(PASS)
        else:
            _, r, c = action
            tokens.append(stoi[(r, c)])
        boards.append(board_after)
    tokens.append(EOS)
    boards.append(boards[-1])
    return tokens, boards


# ─────────────────────────────────────────────────────────────────────────────
# 4. Splits + dump
# ─────────────────────────────────────────────────────────────────────────────

def split_games(n_games, val_frac, gen_frac, rng):
    indices = list(range(n_games))
    rng.shuffle(indices)
    n_gen = int(n_games * gen_frac)
    n_val = int(n_games * val_frac)
    gen = set(indices[:n_gen])
    val = set(indices[n_gen : n_gen + n_val])
    train = set(indices[n_gen + n_val :])
    return train, val, gen


def pick_dtype(vocab_size):
    return np.uint16 if vocab_size < 2**16 else np.uint32


def dump(out_dir, splits, stoi, itos, vocab_size):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = pick_dtype(vocab_size)

    for name, payload in splits.items():
        arr = np.array(payload["tokens"], dtype=dtype)
        if arr.size:
            assert int(arr.min()) >= 0 and int(arr.max()) < vocab_size
        arr.tofile(out / f"{name}.bin")

    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({
            "vocab_size": vocab_size, "stoi": stoi, "itos": itos,
            "dtype": np.dtype(dtype).name,
            "pad": PAD, "bos": BOS, "eos": EOS, "pass_": PASS,
            "n_cells": 64, "board_size": 8,
        }, f)

    # Probe target side table: per (split, game_idx, token_pos) the 64-cell
    # board state (empty=0, black=1, white=2).
    with open(out / "board_state.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "game_idx", "token_pos", "cells"])
        for name, payload in splits.items():
            for game_idx, start, end, boards in payload["game_starts"]:
                for local_pos in range(end - start):
                    global_pos = start + local_pos
                    board = boards[local_pos]
                    w.writerow([name, game_idx, global_pos,
                                "-".join(str(c) for c in board)])
    return dtype


# ─────────────────────────────────────────────────────────────────────────────
# 5. Driver
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(args, rng):
    print(f"[1/4] generating {args.n_games:,} random Othello games ...")
    games = [play_random_game(rng) for _ in range(args.n_games)]

    stoi, itos, vocab_size = build_tokenizer()
    print(f"[2/4] vocab_size = {vocab_size} (64 board cells + "
          f"{N_RESERVED} reserved)")

    train_set, val_set, gen_set = split_games(
        len(games), args.val_frac, args.gen_frac, rng,
    )
    print(f"[3/4] split: train={len(train_set):,}  val={len(val_set):,}  "
          f"gen={len(gen_set):,}")

    splits = {
        "train": {"tokens": [], "game_starts": []},
        "val":   {"tokens": [], "game_starts": []},
        "gen":   {"tokens": [], "game_starts": []},
    }
    game_lengths = []
    for game_idx, history in enumerate(games):
        if game_idx in train_set:
            name = "train"
        elif game_idx in val_set:
            name = "val"
        else:
            name = "gen"
        toks, boards = encode_game(history, stoi)
        start = len(splits[name]["tokens"])
        splits[name]["tokens"].extend(toks)
        end = start + len(toks)
        splits[name]["game_starts"].append((game_idx, start, end, boards))
        game_lengths.append(len(toks))

    print(f"      game length stats: min={min(game_lengths)} "
          f"median={int(np.median(game_lengths))} max={max(game_lengths)} "
          f"mean={np.mean(game_lengths):.1f}")
    return splits, stoi, itos, vocab_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/othello")
    p.add_argument("--n_games", type=int, default=5_000)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--gen_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    splits, stoi, itos, vocab_size = build_corpus(args, rng)

    print(f"[4/4] writing to {args.out_dir} ...")
    dtype = dump(args.out_dir, splits, stoi, itos, vocab_size)

    print("\ndone.")
    for name in ("train", "val", "gen"):
        n_tok = len(splits[name]["tokens"])
        print(f"  {name}.bin: {n_tok:>10,} tokens "
              f"({n_tok / max(1, vocab_size):.0f} visits/token)")
    print(f"  dtype     : {np.dtype(dtype).name}")


if __name__ == "__main__":
    main()
