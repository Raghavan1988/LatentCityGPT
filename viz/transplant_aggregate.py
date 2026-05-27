"""Aggregate multi-seed transplant logs into a single mean ± std table.

Reads checkpoints/multiseed_w2/{cities,othello,flight,music}_*_seed{0..4}.log,
extracts the per-run lift / rate numbers, and prints a markdown table.
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "checkpoints" / "multiseed_w2"

# Each entry: (tag_prefix, list of regex patterns mapped to column names)
DOMAINS = {
    "Cities": [
        ("cities_london_real",   "London real"),
        ("cities_london_within", "London within-shuf"),
        ("cities_london_global", "London global-shuf"),
        ("cities_manhattan",     "Manhattan real"),
        ("cities_boston",        "Boston real"),
    ],
    "Othello": [
        ("othello", "Othello (50k)"),
    ],
    "Flight": [
        ("flight_real",   "real"),
        ("flight_within", "within-shuf"),
        ("flight_global", "global-shuf"),
    ],
    "Music — voice-leading": [
        ("music_real",   "real"),
        ("music_within", "within-shuf"),
        ("music_global", "global-shuf"),
    ],
    "Music — beat (control)": [
        ("music_beat_real", "real"),
    ],
}

# regex patterns (each script uses slightly different phrasing)
LIFT_UNP_PATTERNS = [
    r"TRANSPLANT lift on P\(B's nbrs\) over unpatched\s*:\s*([+-]?\d*\.?\d+)",  # cities
    r"Δ P\(B's legal moves\)\s+transplant − unpatched\s*:\s*([+-]?\d*\.?\d+)",  # othello
    r"Δ P\(near B's RSVP\)\s+transplant − unpatched\s*:\s*([+-]?\d*\.?\d+)",    # music voice-leading
    r"Δ P\(B-beat\)\s+transplant − unpatched\s*:\s*([+-]?\d*\.?\d+)",           # music beat (if formatted this way)
    r"^\s*P\(B-phase tokens\) gain\s+([+-]?\d*\.?\d+)\s+[+-]?\d*\.?\d+\s+[+-]?\d*\.?\d+",  # flight table
]
LIFT_RND_PATTERNS = [
    r"TRANSPLANT lift on P\(B's nbrs\) over random\s*:\s*([+-]?\d*\.?\d+)",
    r"Δ P\(B's legal moves\)\s+transplant − random\s*:\s*([+-]?\d*\.?\d+)",
    r"Δ P\(near B's RSVP\)\s+transplant − random\s*:\s*([+-]?\d*\.?\d+)",
    r"Δ P\(B-beat\)\s+transplant − random\s*:\s*([+-]?\d*\.?\d+)",
    r"^\s*P\(B-phase tokens\) gain\s+[+-]?\d*\.?\d+\s+[+-]?\d*\.?\d+\s+([+-]?\d*\.?\d+)",  # flight Δ trp-rnd column
]
RATE_PATTERNS = [
    r"transplant_PB > random_PB in\s*:\s*([\d.]+)%",
    r"rate\(transplant > random.*?\)\s*:\s*([\d.]+)%",
]


def find_first(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.MULTILINE)
        if m:
            return float(m.group(1))
    return None


def parse_seed_logs(tag):
    results = []
    for seed in range(5):
        path = LOG_DIR / f"{tag}_seed{seed}.log"
        if not path.exists():
            continue
        text = path.read_text()
        unp = find_first(LIFT_UNP_PATTERNS, text)
        rnd = find_first(LIFT_RND_PATTERNS, text)
        rate = find_first(RATE_PATTERNS, text)
        results.append((seed, unp, rnd, rate))
    return results


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:+.4f}"
    a = np.array(vals)
    return f"{a.mean():+.4f} ± {a.std(ddof=1):.4f}"


def fmt_rate(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    a = np.array(vals)
    if len(a) == 1:
        return f"{a[0]:.1f}%"
    return f"{a.mean():.1f} ± {a.std(ddof=1):.1f}%"


def main():
    lines = []
    lines.append("# Multi-seed transplant retrofit — summary\n")
    lines.append("All numbers are mean ± std over 5 seeds. Lift = transplant lift on")
    lines.append("P(B's relevant set), measured against (a) unpatched baseline and")
    lines.append("(b) random-direction patch control. Rate = fraction of test pairs")
    lines.append("where transplant lift exceeds the random control on that pair.\n")
    for domain, entries in DOMAINS.items():
        lines.append(f"## {domain}\n")
        lines.append("| Condition | Lift over unp | Lift over rnd | trp > rnd rate |")
        lines.append("|---|---|---|---|")
        for tag, label in entries:
            runs = parse_seed_logs(tag)
            unps = [r[1] for r in runs]
            rnds = [r[2] for r in runs]
            rates = [r[3] for r in runs]
            lines.append(f"| {label} | {fmt(unps)} | {fmt(rnds)} | {fmt_rate(rates)} |")
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
