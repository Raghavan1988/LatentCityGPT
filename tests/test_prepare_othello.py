"""
Smoke test for data/prepare_othello.py — pure-Python, no network or torch.

Verifies the Othello rules (initial position, legal moves, flips), the
tokenizer bijection, encoded layout, no-probe-target-leakage (board-state
values never appear as raw token ids), and dump roundtrip.
"""

import argparse
import importlib.util
import pickle
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE = REPO_ROOT / "data" / "prepare_othello.py"

spec = importlib.util.spec_from_file_location("prepare_othello", PREPARE)
po = importlib.util.module_from_spec(spec)
spec.loader.exec_module(po)


def test_initial_board():
    b = po.initial_board()
    assert b[3*8 + 3] == po.WHITE
    assert b[3*8 + 4] == po.BLACK
    assert b[4*8 + 3] == po.BLACK
    assert b[4*8 + 4] == po.WHITE
    # All other cells empty
    for i in range(64):
        if i not in (3*8 + 3, 3*8 + 4, 4*8 + 3, 4*8 + 4):
            assert b[i] == po.EMPTY


def test_legal_moves_from_start():
    """BLACK has exactly 4 legal opening moves: c4, d3, e6, f5 (algebraic).
    In our (row, col) zero-indexed: (3, 2), (2, 3), (5, 4), (4, 5).
    """
    b = po.initial_board()
    moves = set(po.legal_moves(b, po.BLACK))
    expected = {(3, 2), (2, 3), (5, 4), (4, 5)}
    assert moves == expected, f"expected {expected}, got {moves}"


def test_apply_move_flips():
    """BLACK plays (3, 2). This should flip WHITE at (3, 3) → BLACK.
    Final board has BLACK at (3, 2), (3, 3), (3, 4), (4, 3); WHITE at (4, 4).
    """
    b = po.initial_board()
    assert po.is_legal(b, 3, 2, po.BLACK)
    new_b = po.apply_move(b, 3, 2, po.BLACK)
    assert new_b[3*8 + 2] == po.BLACK
    assert new_b[3*8 + 3] == po.BLACK  # flipped from WHITE
    assert new_b[3*8 + 4] == po.BLACK  # unchanged
    assert new_b[4*8 + 3] == po.BLACK  # unchanged
    assert new_b[4*8 + 4] == po.WHITE  # unchanged
    # Original board not mutated
    assert b[3*8 + 3] == po.WHITE


def test_tokenizer_bijection():
    stoi, itos, vocab_size = po.build_tokenizer()
    assert vocab_size == 68  # 64 cells + 4 reserved
    assert po.N_RESERVED == 4
    # 64 board positions → 64 distinct token ids in [4, 68)
    assert {stoi[(r, c)] for r in range(8) for c in range(8)} == set(range(4, 68))
    for r in range(8):
        for c in range(8):
            assert itos[stoi[(r, c)]] == (r, c)


def test_random_game_runs():
    rng = random.Random(0)
    history = po.play_random_game(rng)
    # Should produce at least a few moves; games rarely end in <10 plies
    assert len(history) >= 10
    # Each entry is (action_tuple, board_state)
    for action, board in history:
        assert action[0] in ("move", "pass")
        assert len(board) == 64
        for cell in board:
            assert cell in (po.EMPTY, po.BLACK, po.WHITE)


def test_encode_layout():
    stoi, _, vocab_size = po.build_tokenizer()
    # Build a tiny synthetic history: BLACK plays (3,2) (a real opening move)
    move = (("move", 3, 2), po.apply_move(po.initial_board(), 3, 2, po.BLACK))
    tokens, boards = po.encode_game([move], stoi)
    assert tokens[0] == po.BOS
    assert tokens[1] == stoi[(3, 2)]
    assert tokens[2] == po.EOS
    assert len(boards) == 3  # BOS + 1 move + EOS
    # board[0] is initial, board[1] is after move, board[2] is same as board[1]
    assert boards[0] == po.initial_board()
    assert boards[1] != boards[0]  # move changed something
    assert boards[2] == boards[1]


def test_no_probe_target_leakage():
    """Board-state values (0/1/2) must not appear as real-pitch token IDs.
    Real tokens are in [4, 68); board-state values are in {0, 1, 2}.
    The two namespaces overlap only on the RESERVED indices (PAD/BOS/EOS/PASS)
    which the probe code skips by construction.
    """
    stoi, _, vocab_size = po.build_tokenizer()
    rng = random.Random(0)
    history = po.play_random_game(rng)
    tokens, _ = po.encode_game(history, stoi)
    for t in tokens:
        assert t in (po.PAD, po.BOS, po.EOS, po.PASS) or \
               (po.N_RESERVED <= t < vocab_size)


def test_dump_roundtrip(tmp_path):
    args = argparse.Namespace(
        out_dir=str(tmp_path), n_games=20,
        val_frac=0.1, gen_frac=0.1, seed=0,
    )
    rng = random.Random(args.seed)
    splits, stoi, itos, vocab_size = po.build_corpus(args, rng)
    po.dump(tmp_path, splits, stoi, itos, vocab_size)

    meta = pickle.loads((tmp_path / "meta.pkl").read_bytes())
    dtype = np.dtype(meta["dtype"])
    assert meta["vocab_size"] == 68
    assert meta["n_cells"] == 64

    train_arr = np.fromfile(tmp_path / "train.bin", dtype=dtype)
    assert train_arr.size > 0
    assert int(train_arr.min()) >= 0
    assert int(train_arr.max()) < vocab_size

    # board_state.csv has the right header and at least one row
    lines = (tmp_path / "board_state.csv").read_text().splitlines()
    assert lines[0] == "split,game_idx,token_pos,cells"
    assert len(lines) > 1
    # Each cells field should decode to 64 integers in {0, 1, 2}
    for line in lines[1:10]:
        parts = line.split(",")
        cells = [int(x) for x in parts[3].split("-")]
        assert len(cells) == 64
        assert all(c in (0, 1, 2) for c in cells)


if __name__ == "__main__":
    import tempfile
    test_initial_board()
    test_legal_moves_from_start()
    test_apply_move_flips()
    test_tokenizer_bijection()
    test_random_game_runs()
    test_encode_layout()
    test_no_probe_target_leakage()
    with tempfile.TemporaryDirectory() as d:
        test_dump_roundtrip(Path(d))
    print("ok")
