"""
Figure 4: Cross-condition gradient bars.

For each (domain, feature) we show the encoding gap under the three
destroyed-structure conditions (real, within-shuffled, global-shuffled).
The expected pattern under the destroyed-structure prediction is
real > within > global. The figure shows where this prediction holds and
where it does not.

Generates figures/04_cross_condition_gradient.png.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Each entry: (label, [real, within, global], metric, note)
domains = [
    ("Cities London (geographic probe gap)",
     [0.550, 0.672, 0.012], "probe gap",
     "non-monotone (within > real)"),

    ("Music voice-leading (valid voice-leading rate)",
     [0.9625, 0.6433, 0.5591], "valid-step rate",
     "monotone, clean"),

    ("Flight phase (probe gap trained-untrained)",
     [0.105, 0.100, 0.053], "probe gap",
     "monotone, weak"),

    ("Maze starting cell (probe gap trained-untrained)",
     [0.152, 0.031, 0.154], "probe gap",
     "non-monotone (global == real)"),

    ("HTTP Feature A (probe gap, real predicted)",
     [0.168, 0.134, 0.163], "probe gap",
     "monotone-ish, all confirm"),

    ("HTTP Feature B at fixed k=5 (Design A probe gap)",
     [0.220, 0.199, 0.140], "probe gap",
     "monotone, all above +0.10 threshold"),
]

n = len(domains)
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
axes = axes.flatten()

condition_labels = ["real", "within-\nshuffled", "global-\nshuffled"]
condition_colors = ["#1f6feb", "#e08e0b", "#888"]

for i, (label, vals, metric, note) in enumerate(domains):
    ax = axes[i]
    xs = np.arange(3)
    bars = ax.bar(xs, vals, color=condition_colors, width=0.55,
                  edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

    # Add threshold line for probe gaps where +0.10 is the typical
    # null/encoding threshold or pre-registered falsifier
    if "probe gap" in metric:
        ax.axhline(0.10, color="#c92a2a", linewidth=0.9,
                   linestyle=":", alpha=0.7)
        ax.text(2.55, 0.10, "  +0.10\n  threshold",
                color="#c92a2a", fontsize=7, va="center", ha="left")

    ax.set_xticks(xs)
    ax.set_xticklabels(condition_labels, fontsize=9.5)
    ax.set_title(label, fontsize=9.5, pad=10)
    ax.set_ylabel(metric, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Bottom note
    ax.text(0.5, -0.30, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5,
            style="italic", color="#555")

fig.suptitle(
    "Cross-condition gradient: real / within-shuffled / global-shuffled\n"
    "(The framework's destroyed-structure prediction is real > within > global. "
    "Where it holds and where it does not is shown below each panel.)",
    fontsize=11, y=1.00,
)
fig.tight_layout()

out = HERE / "04_cross_condition_gradient.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
