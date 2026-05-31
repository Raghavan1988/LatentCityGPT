"""
data/prepare_http.py — NASA-HTTP request log corpus.

Builds a tokenized corpus from the NASA-HTTP access logs (publicly
hosted by the Internet Traffic Archive, LBL). This domain is the
prospective second pre-registered test of the graded N-criterion's
carry-through prediction (see `predictions/predictions_http_log_sequences.md`).

Per-field tokenization
──────────────────────
Each HTTP request is encoded as FOUR consecutive tokens, one per
field. This is deliberate: a sub-field of a compound token cannot be
cleanly probed for carry-through unless the model internally
decomposes its embedding into separable directions, and there is no
guarantee of that. Per-field tokenization makes each field a
positionally identifiable token, so the carry-through claim becomes
a clean token-level prediction.

Vocabulary (≈55 distinct field-value tokens after discretization):

    method   ∈ {GET, POST, HEAD, OTHER}                              (4)
    path_cat ∈ {top-30 URL prefixes by frequency, OTHER}             (31)
    status   ∈ {2xx, 3xx, 4xx, 5xx}                                  (4)
    size_bin ∈ {0, [1,100), [100, 1k), ..., [1M, ∞), missing}        (9)

Token layout per request (4 tokens):

    [method_token, path_cat_token, status_token, size_bin_token]

Session encoding:

    [BOS, m_1, p_1, s_1, sz_1, m_2, p_2, s_2, sz_2, ..., m_N, p_N, s_N, sz_N, EOS]

So a session of N requests is 4N + 2 tokens. With sessions capped at
30 requests, the longest session fits in block_size 128.

Outputs (under --out_dir)
─────────────────────────
    train.bin       uint16 token stream
    val.bin         uint16 token stream
    gen.bin         uint16 token stream
    meta.pkl        {vocab_size, stoi, itos, dtype, ...}
    http_state.csv  per-position side table for the probe:
        split, session_id, token_pos,
        request_idx_in_session,     (0 = first request, 1 = second, ...)
        field_type,                 (m / p / s / sz)
        first_request_size_bin,           ← Feature A (carry-through)
        cumulative_large_response_count   ← raw count for Feature B
        cumulative_large_response_binned  ← binned (0 / 1 / 2+)

Destroyed-structure controls
────────────────────────────
    --shuffle_within      shuffle 4-token request blocks within sessions
                          (preserves intra-request field order; destroys
                          inter-request session order)
    --shuffle_globally    globally permute the real-token alphabet

Usage
─────
    python data/prepare_http.py \\
        --raw_dir data/nasa_http_raw \\
        --out_dir data/nasa_http \\
        --n_path_cats 30

If --raw_dir does not contain the .gz files, the script prints
download URLs and exits.
"""
import argparse
import csv
import gzip
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PAD, BOS, EOS = 0, 1, 2
N_RESERVED = 3

DEFAULT_DOWNLOAD_URLS = [
    "http://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
    "http://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz",
]

# ──────────────────────────────────────────────────────────────────────
# 1. Parse Common Log Format
# ──────────────────────────────────────────────────────────────────────

CLF_RE = re.compile(
    r'^(?P<host>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<path>\S+)\s+(?P<protocol>\S+)?"\s+'
    r'(?P<status>\d+)\s+(?P<size>\S+)\s*$'
)

MONTH_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def parse_timestamp(ts: str) -> int:
    dt_part, _ = ts.rsplit(" ", 1) if " " in ts else (ts, "+0000")
    day_s, mon_s, year_s, h_s, m_s, sec_s = re.split(r"[/:]", dt_part)
    return int(datetime(
        int(year_s), MONTH_TO_NUM[mon_s], int(day_s),
        int(h_s), int(m_s), int(sec_s),
    ).timestamp())


def parse_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="latin-1", errors="replace") as f:
        for line in f:
            m = CLF_RE.match(line)
            if not m:
                continue
            try:
                ts = parse_timestamp(m["ts"])
            except (KeyError, ValueError):
                continue
            size_raw = m["size"]
            size = int(size_raw) if size_raw.isdigit() else -1
            status_raw = m["status"]
            try:
                status = int(status_raw)
            except ValueError:
                continue
            yield {
                "host": m["host"],
                "ts": ts,
                "method": m["method"],
                "path": m["path"],
                "status": status,
                "size": size,
            }


# ──────────────────────────────────────────────────────────────────────
# 2. Sessionization
# ──────────────────────────────────────────────────────────────────────

def sessionize(requests, idle_timeout=30 * 60):
    by_host = defaultdict(list)
    for r in requests:
        by_host[r["host"]].append(r)
    for host, reqs in by_host.items():
        reqs.sort(key=lambda x: x["ts"])
        cur = []
        last_ts = None
        for r in reqs:
            if last_ts is None or (r["ts"] - last_ts) <= idle_timeout:
                cur.append(r)
            else:
                if cur:
                    yield cur
                cur = [r]
            last_ts = r["ts"]
        if cur:
            yield cur


# ──────────────────────────────────────────────────────────────────────
# 3. Discretization
# ──────────────────────────────────────────────────────────────────────

# Size buckets: 0 | (0, 100) | [100, 1k) | [1k, 10k) | [10k, 100k) |
#               [100k, 1M) | [1M, 10M) | [10M+) | missing
# Indices 0..7 are real, 8 is "missing/unknown".
# Buckets >= 5 (≥ 100,000 bytes) are flagged as "large responses"
# for the Feature B computation.
SIZE_BUCKET_EDGES = [1, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
LARGE_RESPONSE_BIN_THRESHOLD = 5  # size_bin >= 5 counts as "large"


def size_bin(size: int) -> int:
    if size < 0:
        return 8
    if size == 0:
        return 0
    for i, edge in enumerate(SIZE_BUCKET_EDGES):
        if size < edge:
            return i + 1
    return len(SIZE_BUCKET_EDGES)


def status_bin(status: int) -> str:
    if 200 <= status < 300: return "2xx"
    if 300 <= status < 400: return "3xx"
    if 400 <= status < 500: return "4xx"
    if 500 <= status < 600: return "5xx"
    return "other"


def method_token(method: str) -> str:
    if method in ("GET", "POST", "HEAD"):
        return method
    return "OTHER"


def cumulative_large_binned(count: int) -> int:
    """Bin the cumulative-large-response counter into {0, 1, 2+}."""
    if count == 0: return 0
    if count == 1: return 1
    return 2


# ──────────────────────────────────────────────────────────────────────
# 4. Path-prefix discovery
# ──────────────────────────────────────────────────────────────────────

def discover_path_prefixes(requests, n_prefixes: int, depth: int = 2):
    counter = Counter()
    for r in requests:
        parts = r["path"].split("/")
        if len(parts) <= depth + 1:
            prefix = r["path"]
        else:
            prefix = "/".join(parts[: depth + 1]) + "/"
        counter[prefix] += 1
    return [p for p, _ in counter.most_common(n_prefixes)]


def categorize_path(path: str, top_prefixes: list[str]) -> str:
    for prefix in top_prefixes:
        if path.startswith(prefix):
            return prefix
    return "OTHER"


# ──────────────────────────────────────────────────────────────────────
# 5. Build per-field vocabulary
# ──────────────────────────────────────────────────────────────────────

def build_vocab(top_prefixes):
    """Build per-field vocabulary. Each field-value is its own token."""
    stoi = {"<PAD>": PAD, "<BOS>": BOS, "<EOS>": EOS}
    # Method tokens
    for m in ("GET", "POST", "HEAD", "OTHER"):
        stoi[f"m_{m}"] = len(stoi)
    # Path category tokens (use idx not full prefix to keep keys short)
    for i, p in enumerate(top_prefixes):
        stoi[f"p_{i}"] = len(stoi)
    stoi["p_OTHER"] = len(stoi)
    # Status tokens
    for s in ("2xx", "3xx", "4xx", "5xx", "other"):
        stoi[f"s_{s}"] = len(stoi)
    # Size bin tokens (0..8)
    for i in range(9):
        stoi[f"sz_{i}"] = len(stoi)
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos


def request_tokens(r, stoi, top_prefixes_idx):
    """Convert one HTTP request to 4 tokens: [m, p, s, sz]."""
    m_tok = stoi[f"m_{method_token(r['method'])}"]
    path_idx = top_prefixes_idx.get(categorize_path(r["path"], list(top_prefixes_idx.keys())), -1)
    if path_idx >= 0:
        p_tok = stoi[f"p_{path_idx}"]
    else:
        p_tok = stoi["p_OTHER"]
    s_tok = stoi[f"s_{status_bin(r['status'])}"]
    sz_b = size_bin(r["size"])
    sz_tok = stoi[f"sz_{sz_b}"]
    return [m_tok, p_tok, s_tok, sz_tok], sz_b


# ──────────────────────────────────────────────────────────────────────
# 6. Destroyed-structure controls
# ──────────────────────────────────────────────────────────────────────

def shuffle_within_request_blocks(seq, rng):
    """Shuffle the 4-token request blocks within a session, preserving
    intra-request field order (m, p, s, sz). This destroys inter-
    request order but keeps each request internally coherent."""
    assert seq[0] == BOS and seq[-1] == EOS
    interior = seq[1:-1]
    assert len(interior) % 4 == 0, "expected length multiple of 4 per request"
    n_reqs = len(interior) // 4
    blocks = [interior[i * 4:(i + 1) * 4] for i in range(n_reqs)]
    rng.shuffle(blocks)
    return [BOS] + [t for blk in blocks for t in blk] + [EOS]


def shuffle_globally(stream, vocab_size, rng):
    real_ids = list(range(N_RESERVED, vocab_size))
    shuffled = list(real_ids)
    rng.shuffle(shuffled)
    remap = dict(zip(real_ids, shuffled))
    return [remap.get(t, t) for t in stream]


# ──────────────────────────────────────────────────────────────────────
# 7. Output
# ──────────────────────────────────────────────────────────────────────

def pick_dtype(vocab_size):
    return np.uint16 if vocab_size < 65535 else np.uint32


def dump(out_dir, splits, side_rows, stoi, itos, vocab_size, args, top_prefixes):
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
        "domain": "http_log_sequences",
        "source": "NASA-HTTP",
        "tokenization": "per_field_4_tokens_per_request",
        "n_path_cats": args.n_path_cats,
        "top_prefixes": top_prefixes,
        "session_idle_timeout_s": args.session_idle_timeout_s,
        "min_session_len": args.min_session_len,
        "max_session_len": args.max_session_len,
        "shuffle_within": args.shuffle_within,
        "shuffle_globally": args.shuffle_globally,
        "seed": args.seed,
    }
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(out / "http_state.csv", "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=[
            "split", "session_id", "token_pos",
            "request_idx_in_session",
            "field_type",
            "first_request_size_bin",
            "cumulative_large_response_count",
            "cumulative_large_response_binned",
        ])
        writer.writeheader()
        writer.writerows(side_rows)
    sizes = {k: len(splits[k]) for k in splits}
    print(f"\nwrote {out}")
    print(f"  train tokens: {sizes['train']:,}")
    print(f"  val tokens:   {sizes['val']:,}")
    print(f"  gen tokens:   {sizes['gen']:,}")
    print(f"  vocab size:   {vocab_size}")
    print(f"  side-table rows: {len(side_rows):,}")


# ──────────────────────────────────────────────────────────────────────
# 8. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_path_cats", type=int, default=30)
    p.add_argument("--prefix_depth", type=int, default=2)
    p.add_argument("--session_idle_timeout_s", type=int, default=30 * 60)
    p.add_argument("--min_session_len", type=int, default=3)
    p.add_argument("--max_session_len", type=int, default=30,
                   help="Capped to ensure session fits in block_size=128.")
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--gen_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--shuffle_within", action="store_true")
    g.add_argument("--shuffle_globally", action="store_true")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    gz_files = sorted(raw_dir.glob("NASA_access_log*.gz"))
    if not gz_files:
        plain = sorted(raw_dir.glob("NASA_access_log*"))
        if plain:
            gz_files = plain
        else:
            print(f"ERROR: no NASA_access_log files found in {raw_dir}", file=sys.stderr)
            print("Download options:", file=sys.stderr)
            for u in DEFAULT_DOWNLOAD_URLS:
                print(f"  wget -P {raw_dir} {u}", file=sys.stderr)
            sys.exit(1)

    print(f"Parsing {len(gz_files)} NASA-HTTP log file(s)...")
    requests = []
    for path in gz_files:
        n_before = len(requests)
        requests.extend(parse_lines(path))
        print(f"  {path.name}: {len(requests) - n_before:,} parsed (total {len(requests):,})")

    print(f"\nDiscovering top-{args.n_path_cats} path prefixes (depth={args.prefix_depth})...")
    top_prefixes = discover_path_prefixes(requests, args.n_path_cats, args.prefix_depth)
    print(f"  examples: {top_prefixes[:5]}")
    top_prefixes_idx = {p: i for i, p in enumerate(top_prefixes)}

    print(f"\nSessionizing (idle timeout = {args.session_idle_timeout_s}s)...")
    sessions = []
    for sess in sessionize(requests, idle_timeout=args.session_idle_timeout_s):
        if args.min_session_len <= len(sess) <= args.max_session_len:
            sessions.append(sess)
    print(f"  kept {len(sessions):,} sessions (length in [{args.min_session_len}, {args.max_session_len}])")
    lengths = [len(s) for s in sessions]
    print(f"  median session length: {int(np.median(lengths))}, "
          f"p95: {int(np.percentile(lengths, 95))}")

    print(f"\nBuilding per-field vocabulary...")
    stoi, itos = build_vocab(top_prefixes)
    vocab_size = max(stoi.values()) + 1
    print(f"  vocab: {vocab_size} tokens ({vocab_size - N_RESERVED} field-value + 3 reserved)")

    print(f"\nSplitting sessions into train/val/gen...")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(sessions))
    n_val = int(len(sessions) * args.val_frac)
    n_gen = int(len(sessions) * args.gen_frac)
    val_ix = set(perm[:n_val].tolist())
    gen_ix = set(perm[n_val:n_val + n_gen].tolist())

    print(f"\nEncoding sessions + building side table...")
    splits = {"train": [], "val": [], "gen": []}
    side_rows = []
    py_rng = random.Random(args.seed)

    FIELD_TYPES = ["m", "p", "s", "sz"]

    for sess_id, sess in enumerate(sessions):
        split_name = "val" if sess_id in val_ix else ("gen" if sess_id in gen_ix else "train")

        # Encode session
        tokens = [BOS]
        first_size_bin = None
        size_bins_per_req = []
        for r in sess:
            req_toks, sz_b = request_tokens(r, stoi, top_prefixes_idx)
            tokens.extend(req_toks)
            size_bins_per_req.append(sz_b)
            if first_size_bin is None:
                first_size_bin = sz_b
        tokens.append(EOS)

        # Optionally shuffle 4-token request blocks (within-shuffle)
        if args.shuffle_within:
            tokens = shuffle_within_request_blocks(tokens, py_rng)
            # The "first_request" labels become ambiguous under shuffle.
            # We retain the ORIGINAL first request's size_bin as Feature A label.
            # The probe will read the residual at a probe position and try to
            # recover this label — whether the shuffle preserved a recoverable
            # signal is the empirical question.

        # Compute side-table rows for each non-BOS/EOS position
        offset = len(splits[split_name])
        cum_large = 0  # cumulative large-response counter (Feature B raw)
        for local_pos, tok in enumerate(tokens):
            if tok in (PAD, BOS, EOS):
                continue
            # Determine which request this position belongs to and its field type.
            # Interior position p_in_seq = local_pos - 1 (post-BOS offset).
            p_in = local_pos - 1
            req_idx = p_in // 4
            field_idx = p_in % 4
            field_type = FIELD_TYPES[field_idx]

            # If this token is a size_bin token, update the cumulative counter
            # BEFORE recording (so the row reflects "count including this token").
            if field_type == "sz" and not args.shuffle_within:
                # In the non-shuffled case, we know the size_bin from the position
                if req_idx < len(size_bins_per_req):
                    sb_at_this_req = size_bins_per_req[req_idx]
                    if sb_at_this_req >= LARGE_RESPONSE_BIN_THRESHOLD:
                        cum_large += 1
            elif field_type == "sz" and args.shuffle_within:
                # Under shuffle, position no longer reliably maps to request_idx.
                # We can still detect "large response" by looking up the token.
                sz_tok_name = itos.get(tok, "")
                if sz_tok_name.startswith("sz_"):
                    try:
                        sb = int(sz_tok_name.split("_")[1])
                        if sb >= LARGE_RESPONSE_BIN_THRESHOLD:
                            cum_large += 1
                    except (ValueError, IndexError):
                        pass

            side_rows.append({
                "split": split_name,
                "session_id": sess_id,
                "token_pos": offset + local_pos,
                "request_idx_in_session": req_idx,
                "field_type": field_type,
                "first_request_size_bin": first_size_bin,
                "cumulative_large_response_count": cum_large,
                "cumulative_large_response_binned": cumulative_large_binned(cum_large),
            })

        splits[split_name].extend(tokens)

    if args.shuffle_globally:
        print("\nApplying global token-alphabet shuffle...")
        real_ids = list(range(N_RESERVED, vocab_size))
        shuffled = list(real_ids)
        py_rng.shuffle(shuffled)
        remap = dict(zip(real_ids, shuffled))
        for split_name in splits:
            splits[split_name] = [remap.get(t, t) for t in splits[split_name]]
        # Side-table labels are ORIGINAL feature values; the probe targets
        # remain meaningful (label consistency within sessions preserved).

    dump(args.out_dir, splits, side_rows, stoi, itos, vocab_size, args, top_prefixes)


if __name__ == "__main__":
    main()
