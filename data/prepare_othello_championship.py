"""
data/prepare_othello_championship.py — championship Othello corpus from WTHOR.

Produces a training corpus matching the format expected by
`model/configs/large_othello.py` (Phase 3-b). The difference from
`data/prepare_othello.py` is the data source: championship games from
the WTHOR (World Thor) Othello tournament archive, rather than
randomly generated uniform-play games.

WTHOR background
─────────────────
WTHOR is the de-facto standard tournament database for Othello,
maintained by the French Othello Federation (FFO). Each year, the
FFO publishes binary `.wtb` archives of all recorded tournament games
from that year. The format has been stable since 1977; archives are
freely downloadable.

Where to get it
───────────────
- Primary mirror: https://www.ffothello.org/wthor/
- Alternative mirror (often easier): https://www.othello.dk/
- Annual files: WTH_1977.wtb, WTH_1978.wtb, ..., WTH_YYYY.wtb
- Typical total size: ~10-20 MB for all years combined, ~70k+ games

Download all annual files into a single directory, e.g.:
    mkdir -p data/wthor_raw
    cd data/wthor_raw
    # download every WTH_YYYY.wtb file there

WTHOR binary format (per game record, 68 bytes)
────────────────────────────────────────────────
Header (16 bytes at the start of each file):
    bytes 0-3  : century, year, month, day of file creation
    bytes 4-7  : number of game records (uint32 little-endian)
    bytes 8-11 : ... (other metadata; not needed)
    bytes 12-15: padding / reserved

Game record (68 bytes each, immediately following the header):
    bytes 0-1  : tournament label number (uint16)
    bytes 2-3  : black player number       (uint16)
    bytes 4-5  : white player number       (uint16)
    byte 6     : real black score (0-64)
    byte 7     : theoretical black score (perfect-play) (0-64)
    bytes 8-67 : 60 move bytes — each byte encodes one move as
                 col_1_8 * 10 + row_1_8 (e.g., 0x44 = position D4).
                 Zero bytes after game-end are padding.

Output (matches data/prepare_othello.py)
─────────────────────────────────────────
    train.bin / val.bin / gen.bin  — uint16 token streams
    meta.pkl                       — {vocab_size, stoi, itos, dtype, ...}
    board_state.csv                — per-position 64-cell state labels

Usage
─────
    # After downloading WTHOR archives to data/wthor_raw/:
    python data/prepare_othello_championship.py \\
        --wthor_dir data/wthor_raw \\
        --out_dir data/othello_championship \\
        --max_games 70000

Each game is validated against Othello rules; games with illegal moves
(should be rare in tournament play but the parser is conservative) are
silently dropped. The board_state.csv side table is built by replaying
each game move-by-move; this is the probe label source and never
enters the model.
"""

import argparse
import csv
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prepare_othello import (  # noqa: E402
    initial_board, is_legal, legal_moves, apply_move,
    build_tokenizer, pick_dtype,
)

# Token reserved indices (must mirror data/prepare_othello.py)
PAD, BOS, EOS, PASS = 0, 1, 2, 3
N_RESERVED = 4

# Board cell tokens 4..67 (64 cells); same as prepare_othello.py
def cell_to_token(row: int, col: int) -> int:
    return N_RESERVED + row * 8 + col


def token_to_cell(tok: int) -> tuple[int, int]:
    idx = tok - N_RESERVED
    return idx // 8, idx % 8


# ──────────────────────────────────────────────────────────────────────
# 1. Parse WTHOR binary files
# ──────────────────────────────────────────────────────────────────────

def parse_wthor_file(path: Path):
    """Yield move sequences (list of (row, col) tuples) from a .wtb file.

    Validates each game by replaying it against Othello rules; if the
    sequence is illegal at any point the game is skipped. Returns
    (move_list, black_score) per game.
    """
    data = path.read_bytes()
    if len(data) < 16:
        return
    # Header
    n_records = struct.unpack_from("<I", data, 4)[0]
    if 16 + n_records * 68 > len(data):
        # Some archives have a slightly different header size; try alt offset
        alt = (len(data) - 16) // 68
        if alt > 0:
            n_records = alt
        else:
            return
    offset = 16
    for _ in range(n_records):
        if offset + 68 > len(data):
            break
        record = data[offset:offset + 68]
        offset += 68
        # tournament_id = struct.unpack_from("<H", record, 0)[0]
        # black_id      = struct.unpack_from("<H", record, 2)[0]
        # white_id      = struct.unpack_from("<H", record, 4)[0]
        black_score = record[6]
        # theoretical = record[7]
        move_bytes = record[8:68]
        moves = []
        for b in move_bytes:
            if b == 0:
                break
            col = b // 10
            row = b % 10
            if not (1 <= col <= 8 and 1 <= row <= 8):
                moves = None
                break
            # WTHOR uses 1-indexed (col, row); convert to 0-indexed (row, col)
            moves.append((row - 1, col - 1))
        if moves is None:
            continue
        yield moves, black_score


# ──────────────────────────────────────────────────────────────────────
# 2. Replay a move sequence + validate against rules
# ──────────────────────────────────────────────────────────────────────

def replay_and_validate(moves):
    """Replay a WTHOR move sequence against Othello rules. Returns the
    per-move board states (one per move) and per-move colors (B or W),
    or None if any move is illegal.

    Othello convention: black moves first. Forced passes happen when a
    player has no legal move; WTHOR records the next actual move,
    omitting the implicit pass.
    """
    board = initial_board()
    color = -1  # black moves first; convention: -1 = black, +1 = white
    states_per_move = []
    colors_per_move = []
    for r, c in moves:
        # If current color has no legal moves, switch (implicit pass)
        if not legal_moves(board, color):
            color = -color
            if not legal_moves(board, color):
                return None  # game already over but moves remain
        if not is_legal(board, r, c, color):
            return None  # illegal move in the record
        board = apply_move(board, r, c, color)
        states_per_move.append([row[:] for row in board])
        colors_per_move.append(color)
        color = -color
    return states_per_move, colors_per_move


# ──────────────────────────────────────────────────────────────────────
# 3. Encode game → token sequence + side-table rows
# ──────────────────────────────────────────────────────────────────────

def encode_championship_game(moves, stoi):
    """Returns (tokens, side_table_rows_unfilled). The side-table rows
    don't yet have the split label (filled in by main())."""
    replayed = replay_and_validate(moves)
    if replayed is None:
        return None
    states, colors = replayed
    tokens = [stoi["<BOS>"]]
    side_rows = []  # one per token position
    side_rows.append({"current_move_token": tokens[-1], "board": None})  # BOS has no board
    for (r, c), state in zip(moves, states):
        tok = cell_to_token(r, c)
        if tok not in stoi.values():
            return None
        tokens.append(tok)
        side_rows.append({"current_move_token": tok, "board": state})
    tokens.append(stoi["<EOS>"])
    side_rows.append({"current_move_token": tokens[-1], "board": None})
    return tokens, side_rows


# ──────────────────────────────────────────────────────────────────────
# 4. Output (mirrors data/prepare_othello.py)
# ──────────────────────────────────────────────────────────────────────

def split_games(n_games, val_frac, gen_frac, rng):
    perm = rng.permutation(n_games)
    n_val = int(n_games * val_frac)
    n_gen = int(n_games * gen_frac)
    val = set(perm[:n_val].tolist())
    gen = set(perm[n_val:n_val + n_gen].tolist())
    train = set(perm[n_val + n_gen:].tolist())
    return train, val, gen


def dump(out_dir, splits, side_rows, stoi, itos, vocab_size, args):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = pick_dtype(vocab_size)
    for split_name in ("train", "val", "gen"):
        arr = np.array(splits[split_name], dtype=dtype)
        arr.tofile(out / f"{split_name}.bin")
    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "dtype": np.dtype(dtype).name,
        "dtype_str": "uint16" if dtype == np.uint16 else "uint32",
        "domain": "othello_championship",
        "source": "WTHOR",
        "max_games": args.max_games,
        "seed": args.seed,
    }
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    # board_state.csv side table
    with open(out / "board_state.csv", "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=[
            "split", "game_idx", "token_pos", "cell_row", "cell_col", "occupant",
        ])
        writer.writeheader()
        writer.writerows(side_rows)
    sizes = {k: len(splits[k]) for k in splits}
    print(f"wrote {out}")
    print(f"  train tokens: {sizes['train']:,}")
    print(f"  val tokens:   {sizes['val']:,}")
    print(f"  gen tokens:   {sizes['gen']:,}")
    print(f"  vocab size:   {vocab_size}")
    print(f"  side-table rows: {len(side_rows):,}")


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wthor_dir", required=True,
                   help="Directory containing WTH_YYYY.wtb files.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_games", type=int, default=70_000,
                   help="Cap on total games processed (across all years).")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--gen_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    wthor_dir = Path(args.wthor_dir)
    wtb_files = sorted(wthor_dir.glob("*.wtb"))
    if not wtb_files:
        print(f"ERROR: no .wtb files found in {wthor_dir}", file=sys.stderr)
        print(f"Download WTHOR archives from https://www.ffothello.org/wthor/", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(wtb_files)} WTHOR archive(s)")

    stoi, itos = build_tokenizer()
    vocab_size = max(stoi.values()) + 1

    rng = np.random.default_rng(args.seed)
    games = []
    for path in wtb_files:
        for moves, _score in parse_wthor_file(path):
            games.append(moves)
            if len(games) >= args.max_games:
                break
        if len(games) >= args.max_games:
            break
        print(f"  read {path.name}: {len(games):,} total games so far")
    print(f"\nTotal games parsed: {len(games):,}")

    # Split at game level
    train_ix, val_ix, gen_ix = split_games(len(games), args.val_frac, args.gen_frac, rng)

    splits = {"train": [], "val": [], "gen": []}
    all_side_rows = []
    n_kept, n_skipped = 0, 0
    for game_idx, moves in enumerate(games):
        if game_idx in val_ix:
            split_name = "val"
        elif game_idx in gen_ix:
            split_name = "gen"
        else:
            split_name = "train"
        encoded = encode_championship_game(moves, stoi)
        if encoded is None:
            n_skipped += 1
            continue
        tokens, side_rows = encoded
        offset = len(splits[split_name])
        for i, sr in enumerate(side_rows):
            if sr["board"] is None:
                continue
            for r in range(8):
                for c in range(8):
                    occ = sr["board"][r][c]
                    occupant = "empty" if occ == 0 else ("black" if occ == -1 else "white")
                    all_side_rows.append({
                        "split": split_name,
                        "game_idx": game_idx,
                        "token_pos": offset + i,
                        "cell_row": r,
                        "cell_col": c,
                        "occupant": occupant,
                    })
        splits[split_name].extend(tokens)
        n_kept += 1
        if n_kept % 5000 == 0:
            print(f"  encoded {n_kept:,} games ({n_skipped:,} skipped)")

    print(f"\nEncoded {n_kept:,} games ({n_skipped:,} skipped due to illegal moves)")
    dump(args.out_dir, splits, all_side_rows, stoi, itos, vocab_size, args)


if __name__ == "__main__":
    main()
