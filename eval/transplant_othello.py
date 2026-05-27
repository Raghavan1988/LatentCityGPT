"""
LatentWorldsGPT — Othello activation transplant (Li 2022 / Nanda 2023's
third claim, reproduced in this codebase).

Mirrors `eval/transplant.py` (cities) for Othello. Pick game A position p
(board state B_A, color C to move) and game B position q (board state
B_B, same color C to move). Replace the residual at layer L of game A's
position p with the residual from game B's position q. Measure whether
the model's next-token distribution at position p shifts AWAY from B_A's
legal moves and TOWARD B_B's legal moves.

If yes: the residual stream causally encodes the board state — the
model uses the patched representation to compute its next-move
predictions, exactly as Li/Nanda's published result claims.

Headline metric: P(legal_moves(B_B)) under transplant - under unpatched.
Random control: replace with the residual from a third position with a
DIFFERENT board state, to confirm any shift is specific to B_B.

THE ONE RULE
============
This file does NOT read board_state.csv. It REPLAYS games internally
using `data/prepare_othello.py`'s rules to determine which cells are
legal moves on each board state — exactly what the model has to do too.

USAGE
=====
    python eval/transplant_othello.py \
        --ckpt checkpoints/othello_50k/best.pt \
        --data_dir data/othello_50k --layer 2 --n_positions 200
"""

import argparse
import importlib.util
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

# Re-import Othello rules via prepare_othello (avoid duplicating)
_PREPARE = HERE.parent / "data" / "prepare_othello.py"
_spec = importlib.util.spec_from_file_location("prepare_othello", _PREPARE)
po = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(po)

PAD, BOS, EOS, PASS = 0, 1, 2, 3
N_RESERVED = 4
BLACK, WHITE = 1, 2


def cell_to_token(r, c):
    return N_RESERVED + r * 8 + c


# ─────────────────────────────────────────────────────────────────────────────
# 1. Walk the stream into games and replay each (board states + turns)
# ─────────────────────────────────────────────────────────────────────────────

def split_games(stream):
    games = []
    i = 0
    while i < len(stream):
        if stream[i] != BOS:
            i += 1
            continue
        j = i + 1
        while j < len(stream) and stream[j] != EOS:
            j += 1
        end = j + 1 if j < len(stream) else j
        games.append((i, end, np.asarray(stream[i:end])))
        i = end
    return games


def replay_for_states(tokens):
    """Walk a game, return per-position (board_before, board_after,
    color_to_move, legal_moves_set) — same as valid_othello_move but
    keyed by position-index-within-game (0=BOS, etc.)."""
    states = []
    board = po.initial_board()
    color = BLACK
    for tok in tokens:
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
        legal = set(po.legal_moves(board, color))
        state = {"board_before": list(board), "color_to_move": color,
                 "legal_moves": legal}
        if tok != PASS:
            cell = ((tok - N_RESERVED) // 8, (tok - N_RESERVED) % 8)
            if po.is_legal(board, cell[0], cell[1], color):
                board = po.apply_move(board, cell[0], cell[1], color)
        state["board_after"] = list(board)
        states.append(state)
        color = WHITE if color == BLACK else BLACK
    return states


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the donor bank
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_donor_bank(model: GPT, stream: np.ndarray, block_size: int,
                     layer: int, device: str, n_donors: int = 1000,
                     rng_seed: int = 0):
    """Sample n_donors game-positions from the stream. For each, cache the
    residual at `layer` AND record the board state + color-to-move at the
    NEXT position (which is what the model will predict).

    Returns list of dicts: {residual, board_next, color_next, legal_moves_next,
    game_id, pos_in_game, window, pos_in_window}.

    Why "next position": when we transplant at position p, we're replacing
    the model's computation that produces logits at p — which predicts the
    token at p+1. The relevant board state is the one BEFORE p+1's move
    (= board AFTER p's move). The color is whoever is to move at p+1.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)
    games = split_games(stream)
    print(f"  donor bank: {len(games):,} games available")

    # Replay all games up-front; small games, fast.
    games_states = []
    for gid, (g_start, g_end, g_tokens) in enumerate(games):
        states = replay_for_states(g_tokens.tolist())
        games_states.append((gid, g_start, g_end, g_tokens, states))

    donors = []
    attempts = 0
    while len(donors) < n_donors and attempts < n_donors * 10:
        attempts += 1
        gid = rng.integers(0, len(games))
        _, g_start, g_end, g_tokens, states = games_states[gid]
        if len(g_tokens) < 8:
            continue
        # Pick a position p inside the game such that p+1 has a valid
        # color_to_move + non-empty legal_moves set
        candidates = [
            p for p in range(len(g_tokens) - 1)
            if states[p + 1]["color_to_move"] is not None
            and len(states[p + 1]["legal_moves"]) > 0
        ]
        if not candidates:
            continue
        p = int(rng.choice(candidates))

        # Build window of length block_size ending at (g_start + p)
        global_pos = g_start + p
        win_start = max(0, global_pos - block_size + 1)
        window = stream[win_start : win_start + block_size]
        if len(window) < block_size:
            continue
        pos_in_window = global_pos - win_start

        # Forward pass to cache residual at `layer`
        x = torch.from_numpy(np.asarray(window).astype(np.int64)).unsqueeze(0).to(device)
        acts = cache_layer_activations(model, x)
        residual = acts[layer][0, pos_in_window].clone()  # (n_embd,)

        donors.append({
            "residual": residual,
            "board_next": list(states[p + 1]["board_before"]),
            "color_next": states[p + 1]["color_to_move"],
            "legal_moves_next": states[p + 1]["legal_moves"],
            "game_id": gid, "pos_in_game": p,
            "window": np.asarray(window).copy(),
            "pos_in_window": pos_in_window,
        })
    print(f"  collected {len(donors)} donor positions")
    return donors


# ─────────────────────────────────────────────────────────────────────────────
# 3. Forward with replacement (copy from cities/transplant.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_with_replacement(model: GPT, idx_batch: torch.Tensor, layer: int,
                              replace_positions: list[int],
                              replacement_vectors: torch.Tensor):
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
# 4. Run transplant experiment
# ─────────────────────────────────────────────────────────────────────────────

def legal_moves_as_tokens(legal_set):
    """Convert set of (r, c) to list of token IDs."""
    return [cell_to_token(r, c) for r, c in legal_set]


@torch.no_grad()
def run_transplant(model: GPT, donors: list[dict], layer: int,
                   n_pairs: int, device: str, rng_seed: int = 0,
                   batch_size: int = 16) -> dict:
    """For n_pairs random pairs (A, B) of donors with the SAME color-to-move
    but DIFFERENT board state, run three forward passes:
      (1) unpatched: original
      (2) transplant: replace residual at A's position with B's residual
      (3) random control: replace with a third donor C's residual (also
          same color, different board)
    Score:
      P(legal_moves_A): mass on A's legal moves
      P(legal_moves_B): mass on B's legal moves
      P(legal_moves_C): mass on C's legal moves
    Headline: under transplant, P(B's moves) should rise; under random,
    P(C's moves) should NOT rise specifically toward B.
    """
    model.eval()
    rng = np.random.default_rng(rng_seed)

    # Index donors by color so we can match
    donors_by_color = {BLACK: [], WHITE: []}
    for i, d in enumerate(donors):
        donors_by_color[d["color_next"]].append(i)
    print(f"  donor color split: BLACK={len(donors_by_color[BLACK])}, "
          f"WHITE={len(donors_by_color[WHITE])}")

    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        color = BLACK if rng.random() < 0.5 else WHITE
        candidates = donors_by_color[color]
        if len(candidates) < 3:
            continue
        a_idx, b_idx, c_idx = rng.choice(candidates, size=3, replace=False).tolist()
        a, b, c = donors[a_idx], donors[b_idx], donors[c_idx]
        # Require different legal_moves sets (so the test is meaningful)
        if a["legal_moves_next"] == b["legal_moves_next"]:
            continue
        if b["legal_moves_next"] == c["legal_moves_next"]:
            continue
        pairs.append((a_idx, b_idx, c_idx))
    print(f"  sampled {len(pairs)} test pairs (target {n_pairs})")

    results = {"unp_PA": [], "unp_PB": [], "unp_PC": [],
               "trp_PA": [], "trp_PB": [], "trp_PC": [],
               "rnd_PA": [], "rnd_PB": [], "rnd_PC": []}
    t0 = time.time()
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        B = len(batch)
        windows = []; positions = []
        a_legal_lists = []; b_legal_lists = []; c_legal_lists = []
        b_residuals = []; c_residuals = []
        for (a_i, b_i, c_i) in batch:
            a, b_, c = donors[a_i], donors[b_i], donors[c_i]
            windows.append(a["window"])
            positions.append(a["pos_in_window"])
            a_legal_lists.append(legal_moves_as_tokens(a["legal_moves_next"]))
            b_legal_lists.append(legal_moves_as_tokens(b_["legal_moves_next"]))
            c_legal_lists.append(legal_moves_as_tokens(c["legal_moves_next"]))
            b_residuals.append(b_["residual"])
            c_residuals.append(c["residual"])
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
                results[f"{label}_PA"].append(float(probs[a_legal_lists[k]].sum()))
                results[f"{label}_PB"].append(float(probs[b_legal_lists[k]].sum()))
                results[f"{label}_PC"].append(float(probs[c_legal_lists[k]].sum()))
    print(f"  done in {time.time() - t0:.1f}s")
    return results


def summarize(results: dict) -> dict:
    n = len(results["unp_PA"])
    out = {"n": n}
    for label in ("unp", "trp", "rnd"):
        for s in ("PA", "PB", "PC"):
            out[f"{label}_{s}"] = float(np.mean(results[f"{label}_{s}"]))
    out["delta_PB_trp_over_unp"] = out["trp_PB"] - out["unp_PB"]
    out["delta_PB_trp_over_rnd"] = out["trp_PB"] - out["rnd_PB"]
    out["delta_PA_trp_over_unp"] = out["trp_PA"] - out["unp_PA"]  # should be negative (A's moves should LOSE mass)
    # Per-position rate
    n_trp_beats_rnd_on_B = sum(
        1 for i in range(n)
        if results["trp_PB"][i] > results["rnd_PB"][i]
    )
    out["rate_trp_beats_rnd_on_PB"] = n_trp_beats_rnd_on_B / max(1, n)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", choices=("val", "gen"), default="gen")
    p.add_argument("--layer", type=int, default=2,
                   help="layer at which to transplant (0=embed, 1..n_layer=block outputs)")
    p.add_argument("--n_donors", type=int, default=500)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() \
             else ("mps" if torch.backends.mps.is_available() else "cpu")
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
    dtype = np.dtype(meta["dtype"])
    stream = np.asarray(np.memmap(data_dir / f"{args.split}.bin", dtype=dtype, mode="r"))
    print(f"{args.split}.bin: {len(stream):,} tokens")

    print(f"\n[1/3] building donor bank (n_donors={args.n_donors}) ...")
    donors = build_donor_bank(
        model, stream, config.block_size, args.layer,
        device, n_donors=args.n_donors, rng_seed=args.seed,
    )

    print(f"\n[2/3] running transplant interventions (n_pairs={args.n_pairs}) ...")
    results = run_transplant(
        model, donors, args.layer, args.n_pairs, device,
        rng_seed=args.seed, batch_size=args.batch_size,
    )

    print(f"\n[3/3] summarizing ...")
    summary = summarize(results)
    print(f"\nn_scored: {summary['n']}")
    print(f"  {'condition':<12}{'P(A nbrs)':>12}{'P(B nbrs)':>12}{'P(C nbrs)':>12}")
    for label in ("unp", "trp", "rnd"):
        print(f"  {label:<12}"
              f"{summary[f'{label}_PA']:>12.4f}"
              f"{summary[f'{label}_PB']:>12.4f}"
              f"{summary[f'{label}_PC']:>12.4f}")
    print(f"\nEffect sizes:")
    print(f"  Δ P(B's legal moves)  transplant − unpatched : "
          f"{summary['delta_PB_trp_over_unp']:+.4f}  "
          f"(positive = transplant successfully shifts model toward B's legal moves)")
    print(f"  Δ P(B's legal moves)  transplant − random    : "
          f"{summary['delta_PB_trp_over_rnd']:+.4f}  "
          f"(positive = effect is SPECIFIC to B's residual, not just any patch)")
    print(f"  Δ P(A's legal moves)  transplant − unpatched : "
          f"{summary['delta_PA_trp_over_unp']:+.4f}  "
          f"(negative = transplant successfully MOVES AWAY FROM A's legal moves)")
    print(f"  rate(transplant > random on P(B nbrs)) : "
          f"{summary['rate_trp_beats_rnd_on_PB']*100:.1f}%  "
          f"(target → 100%)")

    print(f"\n{'─'*70}\nACCEPTANCE\n{'─'*70}")
    if summary["delta_PB_trp_over_unp"] > 0.05 and summary["delta_PB_trp_over_rnd"] > 0.03:
        print(f"  ✓ Transplant causally shifts model toward B's legal moves.")
        print(f"    This reproduces Li/Nanda's third claim: the residual stream's")
        print(f"    board-state encoding is causally responsible for next-move predictions.")
        print(f"    All 3 Othello-GPT claims now reproduced in this codebase.")
    else:
        print(f"  ✗ Transplant lift is weak. Investigate: layer choice, model")
        print(f"    training quality, donor-bank diversity, or pair-selection logic.")


if __name__ == "__main__":
    main()
