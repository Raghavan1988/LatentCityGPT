"""Generate a per-domain transplant-lift bar chart from the W2 multi-seed logs."""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "checkpoints" / "multiseed_w2"
FIG_DIR = ROOT / "figs"

LIFT_UNP_PATTERNS = [
    r"TRANSPLANT lift on P\(B's nbrs\) over unpatched\s*:\s*([+-]?\d*\.?\d+)",
    r"Δ P\(B's legal moves\)\s+transplant − unpatched\s*:\s*([+-]?\d*\.?\d+)",
    r"Δ P\(near B's RSVP\)\s+transplant − unpatched\s*:\s*([+-]?\d*\.?\d+)",
    r"^\s*P\(B-phase tokens\) gain\s+([+-]?\d*\.?\d+)\s+[+-]?\d*\.?\d+\s+[+-]?\d*\.?\d+",
]

CONDITIONS = [
    ("cities_london_real",   "London real",        "Cities"),
    ("cities_london_within", "London within-shuf", "Cities"),
    ("cities_london_global", "London global-shuf", "Cities"),
    ("cities_manhattan",     "Manhattan",          "Cities"),
    ("cities_boston",        "Boston",             "Cities"),
    ("othello",              "Othello (50k)",      "Othello"),
    ("flight_real",          "Flight real",        "Flight"),
    ("flight_within",        "Flight within",      "Flight"),
    ("flight_global",        "Flight global",      "Flight"),
    ("music_real",           "Music real",         "Music VL"),
    ("music_within",         "Music within",       "Music VL"),
    ("music_global",         "Music global",       "Music VL"),
]

DOMAIN_COLORS = {
    "Cities":   "tab:blue",
    "Othello":  "tab:green",
    "Flight":   "tab:purple",
    "Music VL": "tab:orange",
}


def find_first(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.MULTILINE)
        if m:
            return float(m.group(1))
    return None


def parse(tag):
    vals = []
    for seed in range(5):
        p = LOG_DIR / f"{tag}_seed{seed}.log"
        if not p.exists():
            continue
        v = find_first(LIFT_UNP_PATTERNS, p.read_text())
        if v is not None:
            vals.append(v)
    return np.array(vals)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    labels, means, stds, colors = [], [], [], []
    for tag, label, domain in CONDITIONS:
        vals = parse(tag)
        if len(vals) == 0:
            continue
        labels.append(label)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0)
        colors.append(DOMAIN_COLORS[domain])

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Transplant lift over unpatched (mean ± std, 5 seeds)")
    ax.set_title("Multi-seed transplant lift across 12 (domain × condition) pairs")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    # legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=d) for d, c in DOMAIN_COLORS.items()]
    ax.legend(handles=legend, loc="upper right")
    out = FIG_DIR / "week2_transplant_lift.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
