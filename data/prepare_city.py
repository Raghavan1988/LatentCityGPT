"""
LatentCityGPT — dataset preparation.

Turns a real city's street network into a token corpus for next-intersection
language modeling, plus a coordinate table the model never sees during training.

Outputs (all in --out_dir):
    train.bin   uint16/uint32 token stream — routes between TRAIN destinations
    val.bin     held-out routes between TRAIN destinations (in-distribution perplexity)
    gen.bin     routes targeting HELD-OUT destinations (generalization eval)
    meta.pkl    {vocab_size, stoi, itos, dtype, ...}
    coords.csv  idx,lat,lon  — the probe-side table; NEVER an input to the model
    graph.gpickle  the SCC-restricted networkx graph (for the eval harness later)

Token convention: 0=PAD, 1=BOS, 2=EOS, real intersections start at index 3.
One intersection = one atomic token. No coordinate ever enters the token stream.
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import networkx as nx

PAD, BOS, EOS = 0, 1, 2
N_RESERVED = 3


# ---------------------------------------------------------------------------
# 1. Get the city graph
# ---------------------------------------------------------------------------
def load_city_graph(place: str) -> nx.MultiDiGraph:
    """Pull a drivable street network from OpenStreetMap via OSMnx.

    Returns a directed multigraph: nodes carry y/x (lat/lon), edges carry length.
    Imported lazily so the rest of the pipeline (and the smoke test) runs without
    network access.
    """
    import osmnx as ox
    G = ox.graph_from_place(place, network_type="drive")
    return G


# ---------------------------------------------------------------------------
# 2. Restrict to the largest strongly-connected component
# ---------------------------------------------------------------------------
def largest_scc(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Keep the largest SCC so every source->target shortest-path query succeeds.

    On a directed driving network (one-ways), a random pair may have no legal
    path; the largest SCC guarantees reachability between any two kept nodes.
    """
    scc = max(nx.strongly_connected_components(G), key=len)
    H = G.subgraph(scc).copy()
    return H


# ---------------------------------------------------------------------------
# 3. Trivial tokenizer: OSM node id -> contiguous index
# ---------------------------------------------------------------------------
def build_tokenizer(G: nx.Graph):
    """A bijection node_id <-> token index. PAD/BOS/EOS occupy 0/1/2."""
    node_ids = sorted(G.nodes())
    stoi = {nid: i + N_RESERVED for i, nid in enumerate(node_ids)}
    itos = {i: nid for nid, i in stoi.items()}
    vocab_size = len(stoi) + N_RESERVED
    return stoi, itos, vocab_size


# ---------------------------------------------------------------------------
# 4 + 5. Route generation, with a held-out destination split
# ---------------------------------------------------------------------------
def split_destinations(nodes, holdout_frac, rng):
    """Reserve a fraction of nodes as held-out *destinations*.

    Held-out nodes may still appear as waypoints inside training routes; they
    just never serve as the explicit endpoint of a training route. Routes that
    target them go to gen.bin -> the generalization claim.
    """
    nodes = list(nodes)
    rng.shuffle(nodes)
    n_hold = int(len(nodes) * holdout_frac)
    heldout = set(nodes[:n_hold])
    train_dests = set(nodes[n_hold:])
    return train_dests, heldout


def shortest_path_route(G, src, dst):
    """length-weighted shortest path: realistic, goal-directed."""
    try:
        return nx.shortest_path(G, src, dst, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def random_walk_route(G, src, length, rng):
    """A cheap walk that gives even local coverage so quiet streets aren't starved."""
    route = [src]
    cur = src
    for _ in range(length - 1):
        succ = list(G.successors(cur))
        if not succ:
            break
        cur = rng.choice(succ)
        route.append(cur)
    return route


def encode(route, stoi):
    """[BOS, n1, n2, ..., EOS] as token indices."""
    return [BOS] + [stoi[n] for n in route] + [EOS]


def generate_corpus(G, stoi, train_dests, heldout, args, rng):
    """Produce three token streams: train, val (in-dist), gen (held-out dests)."""
    nodes = list(G.nodes())
    train_dest_list = list(train_dests)
    held_list = list(heldout)

    train_tokens, val_tokens, gen_tokens = [], [], []

    # --- shortest-path routes to TRAIN destinations -> train / val ---
    n_sp = args.n_shortest
    for i in range(n_sp):
        src = rng.choice(nodes)
        dst = rng.choice(train_dest_list)
        if src == dst:
            continue
        route = shortest_path_route(G, src, dst)
        if route is None or len(route) < args.min_len:
            continue
        toks = encode(route, stoi)
        # ~5% of train-destination routes become in-distribution validation
        (val_tokens if rng.random() < args.val_frac else train_tokens).extend(toks)

    # --- random walks -> train only (local coverage) ---
    # A walk may wander onto a held-out destination. Held-out nodes are allowed
    # as waypoints, but must never be the *endpoint* of a training route, or the
    # generalization claim leaks. So trim any trailing held-out nodes.
    for _ in range(args.n_walks):
        src = rng.choice(nodes)
        length = rng.randint(args.walk_min, args.walk_max)
        route = random_walk_route(G, src, length, rng)
        while route and route[-1] in heldout:
            route.pop()
        if len(route) < args.min_len:
            continue
        train_tokens.extend(encode(route, stoi))

    # --- shortest-path routes to HELD-OUT destinations -> gen.bin ---
    if held_list:
        for _ in range(args.n_gen):
            src = rng.choice(nodes)
            dst = rng.choice(held_list)
            if src == dst:
                continue
            route = shortest_path_route(G, src, dst)
            if route is None or len(route) < args.min_len:
                continue
            gen_tokens.extend(encode(route, stoi))

    return train_tokens, val_tokens, gen_tokens


# ---------------------------------------------------------------------------
# 6. Dump to disk in nanoGPT's expected format
# ---------------------------------------------------------------------------
def pick_dtype(vocab_size):
    return np.uint16 if vocab_size < 2**16 else np.uint32


def dump(out_dir, train, val, gen, stoi, itos, vocab_size, G):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = pick_dtype(vocab_size)

    for name, toks in [("train", train), ("val", val), ("gen", gen)]:
        arr = np.array(toks, dtype=dtype)
        # THE ONE RULE guard: token streams must contain only PAD/BOS/EOS and
        # tokenizer-mapped node indices in [0, vocab_size). Any value outside
        # that range means positional or other off-vocab data leaked in.
        if arr.size:
            assert int(arr.min()) >= 0 and int(arr.max()) < vocab_size, (
                f"{name}.bin contains out-of-vocab values "
                f"(min={int(arr.min())}, max={int(arr.max())}, vocab_size={vocab_size})"
            )
        arr.tofile(out / f"{name}.bin")

    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(
            {"vocab_size": vocab_size, "stoi": stoi, "itos": itos,
             "dtype": np.dtype(dtype).name,
             "pad": PAD, "bos": BOS, "eos": EOS},
            f,
        )

    # coordinate table — the probe-side ground truth, never a model input
    with open(out / "coords.csv", "w") as f:
        f.write("idx,lat,lon\n")
        for nid, idx in sorted(stoi.items(), key=lambda kv: kv[1]):
            y = G.nodes[nid].get("y")
            x = G.nodes[nid].get("x")
            assert y is not None and x is not None, (
                f"node {nid} (idx {idx}) is missing y/x; "
                "coords.csv must have a real coordinate for every token "
                "or the probe will silently train on garbage"
            )
            f.write(f"{idx},{y},{x}\n")

    with open(out / "graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    return dtype


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Cambridge, Massachusetts, USA")
    p.add_argument("--out_dir", default="data/city")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout_frac", type=float, default=0.10,
                   help="fraction of nodes reserved as held-out destinations")
    p.add_argument("--n_shortest", type=int, default=200_000)
    p.add_argument("--n_walks", type=int, default=50_000)
    p.add_argument("--n_gen", type=int, default=20_000)
    p.add_argument("--walk_min", type=int, default=10)
    p.add_argument("--walk_max", type=int, default=60)
    p.add_argument("--min_len", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.05)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"[1/6] pulling {args.place} from OpenStreetMap ...")
    G = load_city_graph(args.place)
    print(f"[2/6] restricting to largest SCC ...")
    G = largest_scc(G)
    print(f"      {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")
    print(f"[3/6] building tokenizer ...")
    stoi, itos, vocab_size = build_tokenizer(G)
    print(f"      vocab_size = {vocab_size:,}")
    print(f"[4/6] splitting destinations (holdout={args.holdout_frac}) ...")
    train_dests, heldout = split_destinations(G.nodes(), args.holdout_frac, rng)
    print(f"      {len(train_dests):,} train dests / {len(heldout):,} held out")
    print(f"[5/6] generating routes ...")
    train, val, gen = generate_corpus(G, stoi, train_dests, heldout, args, rng)
    print(f"[6/6] writing to {args.out_dir} ...")
    dtype = dump(args.out_dir, train, val, gen, stoi, itos, vocab_size, G)

    print("\ndone.")
    print(f"  train.bin : {len(train):>12,} tokens")
    print(f"  val.bin   : {len(val):>12,} tokens")
    print(f"  gen.bin   : {len(gen):>12,} tokens")
    print(f"  dtype     : {np.dtype(dtype).name}")


if __name__ == "__main__":
    main()
