"""
Figure 3: Per-layer ablation chart.

Shows where in the network each (domain, feature) representation lives.
Generates figures/03_per_layer_ablation.png.

Data sourced from the paper text:
- Othello: per-layer transplant lift over unpatched (real condition)
- Music voice-leading: per-layer transplant lift (real condition)
- Maze starting cell: per-layer probe gap trained-untrained (real)
- HTTP Feature A: per-layer probe gap (real)
- HTTP Feature B: per-layer probe gap (real, before position control)
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (label, layer_indices, values, metric_kind)
panels = [
    ("Othello board state\n(per-layer transplant lift)",
     ["embed", "L0", "L1", "L2", "L3"],
     [0.000, 0.040, 0.062, 0.108, 0.296],
     "transplant lift"),

    ("Music voice-leading\n(per-layer transplant lift)",
     ["embed", "L0", "L1", "L2"],
     [0.000, 0.035, 0.813, 0.889],
     "transplant lift"),

    ("Maze starting cell (real)\n(per-layer probe gap trained-untrained)",
     ["L0", "L1", "L2", "L3", "L4", "L5"],
     [0.005, 0.041, 0.119, 0.131, 0.142, 0.152],
     "probe gap"),

    ("HTTP Feature A (real)\n(per-layer probe gap trained-untrained)",
     ["embed", "L0", "L1", "L2", "L3"],
     [0.068, 0.127, 0.154, 0.164, 0.168],
     "probe gap"),

    ("HTTP Feature B (real, no position control)\n(per-layer probe gap trained-untrained)",
     ["embed", "L0", "L1", "L2", "L3"],
     [0.107, 0.235, 0.276, 0.291, 0.279],
     "probe gap"),

    ("HTTP Feature B at fixed k=5 (Design A)\n(per-layer probe accuracy: trained vs untrained)",
     ["embed", "L0", "L1", "L2", "L3"],
     None,
     "fixed-position accuracy"),
]

# Special two-line panel data for the last subplot
fixed_k_trained = [0.683, 0.855, 0.877, 0.913, 0.900]
fixed_k_untrained = [0.683, 0.694, 0.693, 0.695, 0.695]

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
axes = axes.flatten()

for i, (label, layers, values, kind) in enumerate(panels):
    ax = axes[i]
    xs = np.arange(len(layers))

    if i < 5:
        # standard single-line plot
        ax.plot(xs, values, marker="o", linewidth=2.2, markersize=7,
                color="#1f6feb")
        for x, y in zip(xs, values):
            ax.annotate(f"{y:+.3f}", xy=(x, y),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color="#444")
        ax.axhline(0, color="#bbb", linewidth=0.7, linestyle="--")
        if "probe gap" in kind:
            ax.axhline(0.10, color="#c92a2a", linewidth=0.7,
                       linestyle=":", alpha=0.7,
                       label="locked falsification threshold (HTTP/maze)")
            ax.legend(loc="upper left", fontsize=7, frameon=False)
    else:
        # Fixed-k probe: two lines (trained vs untrained)
        ax.plot(xs, fixed_k_trained, marker="o", linewidth=2.2,
                markersize=7, color="#1a8a3a", label="trained")
        ax.plot(xs, fixed_k_untrained, marker="s", linewidth=2.2,
                markersize=7, color="#888", label="untrained")
        for x, y in zip(xs, fixed_k_trained):
            ax.annotate(f"{y:.3f}", xy=(x, y),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color="#1a8a3a")
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax.set_xticks(xs)
    ax.set_xticklabels(layers, fontsize=9)
    ax.set_title(label, fontsize=9.5, pad=8)
    ax.set_xlabel("layer", fontsize=9)
    if kind == "transplant lift":
        ax.set_ylabel("transplant lift\n(over unpatched)", fontsize=9)
    elif kind == "probe gap":
        ax.set_ylabel("probe gap\n(trained - untrained)", fontsize=9)
    else:
        ax.set_ylabel("probe accuracy", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle(
    "Per-layer ablation: where each (domain, feature) representation lives\n"
    "(Othello and music: causal evidence via transplant. "
    "Maze, HTTP: descriptive evidence via probe gap.)",
    fontsize=11, y=1.00,
)
fig.tight_layout()

out = HERE / "03_per_layer_ablation.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
