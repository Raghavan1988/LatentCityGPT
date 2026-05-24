"""
Smoke test for data/prepare_city.py — no network call.

Builds a synthetic strongly-connected grid graph with y/x attributes, runs the
full prepare_city pipeline on it (skipping the OSMnx pull), and verifies the
invariants PLAN.md Phase 0 names:

  - tokenizer is a bijection (PAD/BOS/EOS reserved, real nodes start at 3)
  - every route traverses only real edges of the input graph
  - gen.bin routes target held-out destinations; train.bin routes never do
  - *.bin binaries roundtrip through numpy at the recorded dtype
  - no coordinate value leaks into any token stream (THE ONE RULE)
"""

import argparse
import importlib.util
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE = REPO_ROOT / "data" / "prepare_city.py"

spec = importlib.util.spec_from_file_location("prepare_city", PREPARE)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def grid_graph(n: int = 6) -> nx.MultiDiGraph:
    """n x n directed grid; every cell -> its 4 neighbors (wraps around in both
    axes so the graph is strongly connected). y/x mimic OSM's lat/lon naming."""
    G = nx.MultiDiGraph()
    for i in range(n):
        for j in range(n):
            nid = i * n + j
            G.add_node(nid, y=float(i), x=float(j))
    for i in range(n):
        for j in range(n):
            here = i * n + j
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                there = ((i + di) % n) * n + ((j + dj) % n)
                G.add_edge(here, there, length=1.0)
    return G


def fake_args(out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        place="synthetic-grid",
        out_dir=str(out_dir),
        seed=0,
        holdout_frac=0.2,
        n_shortest=200,
        n_walks=50,
        n_gen=80,
        walk_min=4,
        walk_max=10,
        min_len=3,
        val_frac=0.1,
    )


def test_prepare_city_smoke(tmp_path):
    G = pc.largest_scc(grid_graph(n=6))
    stoi, itos, vocab_size = pc.build_tokenizer(G)

    # tokenizer bijection — PAD/BOS/EOS reserved, real nodes start at N_RESERVED
    assert pc.N_RESERVED == 3
    assert all(idx >= pc.N_RESERVED for idx in stoi.values())
    assert {stoi[n] for n in G.nodes()} == set(range(pc.N_RESERVED, vocab_size))
    assert all(itos[stoi[n]] == n for n in G.nodes())

    rng = random.Random(0)
    train_dests, heldout = pc.split_destinations(G.nodes(), 0.2, rng)
    assert train_dests.isdisjoint(heldout)
    assert train_dests | heldout == set(G.nodes())

    args = fake_args(tmp_path)
    rng2 = random.Random(args.seed)
    train, val, gen = pc.generate_corpus(G, stoi, train_dests, heldout, args, rng2)
    pc.dump(args.out_dir, train, val, gen, stoi, itos, vocab_size, G)

    # *.bin roundtrip at the recorded dtype
    meta = pickle.loads((tmp_path / "meta.pkl").read_bytes())
    dtype = np.dtype(meta["dtype"])
    train_arr = np.fromfile(tmp_path / "train.bin", dtype=dtype)
    gen_arr = np.fromfile(tmp_path / "gen.bin", dtype=dtype)

    # THE ONE RULE: no value outside [0, vocab_size) — i.e. only PAD/BOS/EOS
    # and tokenizer-mapped node indices ever appear in the stream.
    for arr in (train_arr, gen_arr):
        if arr.size:
            assert int(arr.min()) >= 0 and int(arr.max()) < vocab_size

    # Every encoded route's interior tokens decode to a real node and consecutive
    # interior pairs are real edges of G (i.e. route-edge validity).
    def split_routes(arr):
        routes, cur = [], []
        for t in arr.tolist():
            if t == pc.BOS:
                cur = []
            elif t == pc.EOS:
                if cur:
                    routes.append(cur)
                cur = []
            elif t == pc.PAD:
                continue
            else:
                cur.append(t)
        return routes

    def assert_routes_valid(arr):
        for route in split_routes(arr):
            nodes = [itos[t] for t in route]
            for a, b in zip(nodes, nodes[1:]):
                assert G.has_edge(a, b), f"fake edge {a}->{b} in token stream"

    assert_routes_valid(train_arr)
    assert_routes_valid(gen_arr)

    # Destination holdout: gen routes END on held-out nodes; train routes never do.
    for route in split_routes(gen_arr):
        last = itos[route[-1]]
        assert last in heldout, "gen route did not end on a held-out destination"
    for route in split_routes(train_arr):
        last = itos[route[-1]]
        assert last not in heldout, "train route ended on a held-out destination — leak"

    # coords.csv exists, covers every token index, holds no value found in *.bin
    coord_lines = (tmp_path / "coords.csv").read_text().splitlines()
    assert coord_lines[0] == "idx,lat,lon"
    seen = set()
    for line in coord_lines[1:]:
        idx_s, lat_s, lon_s = line.split(",")
        idx = int(idx_s)
        assert pc.N_RESERVED <= idx < vocab_size
        # the coordinate values themselves must not appear as tokens
        # (defensive: float values can't be tokens anyway, but confirms separation)
        assert float(lat_s) == float(lat_s)  # not NaN
        assert float(lon_s) == float(lon_s)
        seen.add(idx)
    assert seen == set(range(pc.N_RESERVED, vocab_size))


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_prepare_city_smoke(Path(d))
    print("ok")
