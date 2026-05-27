"""
Smoke test for data/prepare_adsb.py — uses a hand-built synthetic flight
DataFrame to test the pipeline offline without the traffic library's
sample data dependency.

Verifies:
  - tokenizer bijection (PAD/BOS/EOS reserved; real tokens cover the
    full 10*5*4 = 200-tuple space)
  - encoding layout (BOS + N obs tokens + EOS; phase labels parallel)
  - no-probe-target-leakage: phase strings never appear as token IDs
  - shuffle_within_flight preserves per-flight set membership
  - shuffle_globally preserves the corpus multiset
  - dump roundtrip through numpy at recorded dtype
"""

import argparse
import importlib.util
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE = REPO_ROOT / "data" / "prepare_adsb.py"

spec = importlib.util.spec_from_file_location("prepare_adsb", PREPARE)
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)


def synthetic_flight(n_obs=120):
    """Build a synthetic flight DataFrame: climb → cruise → descent."""
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n_obs):
        t = base + pd.Timedelta(seconds=i)
        if i < n_obs // 3:                # climb
            alt = 500 + (i / (n_obs // 3)) * 30000
            spd = 150 + (i / (n_obs // 3)) * 250
            vr  = 2000
        elif i < 2 * n_obs // 3:           # cruise
            alt = 30500
            spd = 400
            vr  = 0
        else:                              # descent
            j = (i - 2 * n_obs // 3) / (n_obs // 3)
            alt = 30500 - j * 28000
            spd = 400 - j * 150
            vr  = -2000
        rows.append({
            "timestamp": t,
            "altitude": alt,
            "groundspeed": spd,
            "vertical_rate": vr,
        })
    return pd.DataFrame(rows)


def test_tokenizer_bijection():
    vocab = pa.total_vocab_size()
    assert vocab == 3 + 10 * 5 * 4   # PAD/BOS/EOS + 10 alt * 5 vr * 4 spd
    # Bin tuples → unique token ids in [3, vocab)
    seen = set()
    for alt_b in range(10):
        for vr_b in range(5):
            for spd_b in range(4):
                tok = pa.bin_tuple_to_token_id(alt_b, vr_b, spd_b)
                assert 3 <= tok < vocab
                seen.add(tok)
    assert len(seen) == 200


def test_discretize_edge_cases():
    # Below first edge → bin 0
    assert pa.discretize(-100, pa.ALT_EDGES) == 0
    # Above last edge → last bin
    assert pa.discretize(50000, pa.ALT_EDGES) == len(pa.ALT_EDGES) - 2
    # NaN → None
    assert pa.discretize(float("nan"), pa.ALT_EDGES) is None
    # Specific values
    assert pa.discretize(0, pa.VR_EDGES) == 2  # middle "level" bin
    assert pa.discretize(2000, pa.VR_EDGES) == 4  # "rapid climb"
    assert pa.discretize(-2000, pa.VR_EDGES) == 0  # "rapid descent"


def test_encode_synthetic_flight():
    df = synthetic_flight(n_obs=120)
    toks, phases = pa.encode_flight(df, sample_seconds=10)
    assert toks is not None
    # Layout: [BOS, real tokens..., EOS]
    assert toks[0] == pa.BOS
    assert toks[-1] == pa.EOS
    assert all(t >= pa.N_RESERVED for t in toks[1:-1])
    # Phase labels parallel; sentinels at boundaries
    assert len(phases) == len(toks)
    assert phases[0] == "-" and phases[-1] == "-"
    # We expect some non-"NA" phase labels in the interior
    real_phases = set(phases[1:-1])
    assert any(p != "-" for p in real_phases)


def test_no_probe_target_leakage():
    df = synthetic_flight(n_obs=120)
    toks, _ = pa.encode_flight(df, sample_seconds=10)
    vocab = pa.total_vocab_size()
    for t in toks:
        assert t in (pa.PAD, pa.BOS, pa.EOS) or (pa.N_RESERVED <= t < vocab)


def test_shuffles_preserve_what_they_should():
    from collections import Counter
    stream = [pa.BOS, 10, 20, 30, 40, pa.EOS, pa.BOS, 11, 22, 33, pa.EOS]
    rng = random.Random(0)

    # within-flight: per-span multiset preserved; BOS/EOS positions unchanged
    out = pa.shuffle_within_flight(stream, rng)
    bos_pos = [i for i, t in enumerate(stream) if t == pa.BOS]
    eos_pos = [i for i, t in enumerate(stream) if t == pa.EOS]
    assert [i for i, t in enumerate(out) if t == pa.BOS] == bos_pos
    assert [i for i, t in enumerate(out) if t == pa.EOS] == eos_pos
    assert sorted(out[1:5]) == sorted(stream[1:5])
    assert sorted(out[7:10]) == sorted(stream[7:10])

    # global: full corpus multiset preserved
    rng2 = random.Random(1)
    out_g = pa.shuffle_globally(stream, rng2)
    assert Counter(out_g) == Counter(stream)
    assert [i for i, t in enumerate(out_g) if t == pa.BOS] == bos_pos


def test_dump_roundtrip(tmp_path):
    # Build a tiny corpus directly (without going through the traffic library)
    df = synthetic_flight(n_obs=120)
    toks, phases = pa.encode_flight(df, sample_seconds=10)
    splits = {
        "train": {"tokens": list(toks), "flight_starts": [(0, 0, len(toks), phases)]},
        "val":   {"tokens": [], "flight_starts": []},
        "gen":   {"tokens": [], "flight_starts": []},
    }
    vocab = pa.total_vocab_size()
    pa.dump(tmp_path, splits, vocab)

    meta = pickle.loads((tmp_path / "meta.pkl").read_bytes())
    assert meta["vocab_size"] == vocab
    assert meta["n_alt"] == 10 and meta["n_vr"] == 5 and meta["n_spd"] == 4

    dtype = np.dtype(meta["dtype"])
    arr = np.fromfile(tmp_path / "train.bin", dtype=dtype)
    assert arr.size > 0
    assert int(arr.min()) >= 0 and int(arr.max()) < vocab

    lines = (tmp_path / "flight_phase.csv").read_text().splitlines()
    assert lines[0] == "split,flight_idx,token_pos,phase"
    assert len(lines) > 1  # at least some phase rows


if __name__ == "__main__":
    import tempfile
    test_tokenizer_bijection()
    test_discretize_edge_cases()
    test_encode_synthetic_flight()
    test_no_probe_target_leakage()
    test_shuffles_preserve_what_they_should()
    with tempfile.TemporaryDirectory() as d:
        test_dump_roundtrip(Path(d))
    print("ok")
