"""Generate per-layer ablation plots from the multi-seed probe logs.

Reads checkpoints/multiseed_w1/probe_*.log, parses the aggregate
(mean ± std) tables, and produces one figure per domain with
trained-vs-untrained curves across layers.
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "checkpoints" / "multiseed_w1"
FIG_DIR = ROOT / "figs"

LAYER_LINE = re.compile(
    r"^\s*(embed|L\d+)\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)\s*$"
)


def parse_block(text, header_pattern):
    m = re.search(header_pattern, text)
    if not m:
        return None
    start = m.end()
    block = text[start : start + 1500]
    rows = []
    for line in block.splitlines():
        mm = LAYER_LINE.match(line)
        if mm:
            layer, lin_mu, lin_sd, mlp_mu, mlp_sd = mm.groups()
            rows.append(
                (
                    layer,
                    float(lin_mu),
                    float(lin_sd),
                    float(mlp_mu),
                    float(mlp_sd),
                )
            )
        elif rows and line.strip() == "":
            break
    return rows or None


def parse_log(path, trained_header, untrained_header):
    text = path.read_text()
    return parse_block(text, trained_header), parse_block(text, untrained_header)


def plot_cities():
    conditions = [
        ("london_real", "London real", "tab:blue", "-"),
        ("london_within_shuffled", "London within-shuf", "tab:orange", "--"),
        ("london_global_shuffled", "London global-shuf", "tab:red", ":"),
        ("manhattan_real", "Manhattan real", "tab:green", "-"),
        ("boston_real", "Boston real", "tab:purple", "-"),
    ]
    trained_hdr = r"TRAINED — NODE-LEVEL[^\n]*\(mean ± std"
    untrained_hdr = r"UNTRAINED — NODE-LEVEL[^\n]*\(mean ± std"

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, label, color, ls in conditions:
        path = LOG_DIR / f"probe_cities_grid_{tag}.log"
        if not path.exists():
            continue
        trained, untrained = parse_log(path, trained_hdr, untrained_hdr)
        if not trained:
            continue
        layers = [r[0] for r in trained]
        mlp_t = np.array([r[3] for r in trained])
        sd_t = np.array([r[4] for r in trained])
        mlp_u = np.array([r[3] for r in untrained])
        x = np.arange(len(layers))
        ax.errorbar(
            x, mlp_t, yerr=sd_t, label=f"{label} (trained)",
            color=color, linestyle=ls, marker="o", capsize=3,
        )
        ax.plot(
            x, mlp_u, color=color, linestyle=ls, alpha=0.3,
            marker="x", markersize=5,
        )
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(["embed"] + [f"L{i}" for i in range(1, 7)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Node-level MLP accuracy")
    ax.set_title("Cities grid-classification probe (mean ± std over 5 seeds)\nsolid markers = trained, faint × = untrained")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / "week1_cities_per_layer.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_othello():
    path = LOG_DIR / "probe_othello.log"
    trained_hdr = r"TRAINED — board-state\s+\(mean ± std"
    untrained_hdr = r"UNTRAINED — board-state\s+\(mean ± std"
    trained, untrained = parse_log(path, trained_hdr, untrained_hdr)
    if not trained:
        print(f"!! could not parse {path}")
        return

    layers = [r[0] for r in trained]
    x = np.arange(len(layers))
    mlp_t = np.array([r[3] for r in trained])
    sd_t_mlp = np.array([r[4] for r in trained])
    lin_t = np.array([r[1] for r in trained])
    sd_t_lin = np.array([r[2] for r in trained])
    mlp_u = np.array([r[3] for r in untrained])
    sd_u_mlp = np.array([r[4] for r in untrained])
    lin_u = np.array([r[1] for r in untrained])
    sd_u_lin = np.array([r[2] for r in untrained])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, mlp_t, yerr=sd_t_mlp, label="trained MLP", color="tab:blue", marker="o", capsize=3)
    ax.errorbar(x, lin_t, yerr=sd_t_lin, label="trained linear", color="tab:blue", marker="s", linestyle="--", capsize=3)
    ax.errorbar(x, mlp_u, yerr=sd_u_mlp, label="untrained MLP", color="tab:gray", marker="o", capsize=3)
    ax.errorbar(x, lin_u, yerr=sd_u_lin, label="untrained linear", color="tab:gray", marker="s", linestyle="--", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Per-cell accuracy (3-class)")
    ax.set_title("Othello board-state probe (mean ± std over 5 seeds)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / "week1_othello_per_layer.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_flight():
    conditions = [
        ("", "real", "tab:blue", "-"),
        ("_within_shuffled", "within-shuf", "tab:orange", "--"),
        ("_global_shuffled", "global-shuf", "tab:red", ":"),
    ]
    trained_hdr = r"TRAINED — FLIGHT-LEVEL\s+\(mean ± std"
    untrained_hdr = r"UNTRAINED — FLIGHT-LEVEL\s+\(mean ± std"

    fig, ax = plt.subplots(figsize=(7, 5))
    for suffix, label, color, ls in conditions:
        path = LOG_DIR / f"probe_flight{suffix}.log"
        if not path.exists():
            continue
        trained, untrained = parse_log(path, trained_hdr, untrained_hdr)
        if not trained:
            continue
        layers = [r[0] for r in trained]
        x = np.arange(len(layers))
        mlp_t = np.array([r[3] for r in trained])
        sd_t = np.array([r[4] for r in trained])
        mlp_u = np.array([r[3] for r in untrained])
        ax.errorbar(
            x, mlp_t, yerr=sd_t, label=f"{label} (trained)",
            color=color, linestyle=ls, marker="o", capsize=3,
        )
        ax.plot(x, mlp_u, color=color, linestyle=ls, alpha=0.3, marker="x")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["embed", "L1", "L2"])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Flight-level MLP accuracy (5-class phase)")
    ax.set_title("Flight phase probe (mean ± std over 5 seeds)\nsolid markers = trained, faint × = untrained")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / "week1_flight_per_layer.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_music():
    conditions = [
        ("", "real", "tab:blue", "-"),
        ("_within_shuffled", "within-shuf", "tab:orange", "--"),
        ("_global_shuffled", "global-shuf", "tab:red", ":"),
    ]
    targets = ["mode", "chord", "beat"]
    chance = {"mode": None, "chord": None, "beat": 0.2501}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for ax, tgt in zip(axes, targets):
        trained_hdr = rf"TRAINED — PIECE-LEVEL \[{tgt}\]\s+\(mean ± std"
        untrained_hdr = rf"UNTRAINED — PIECE-LEVEL \[{tgt}\]\s+\(mean ± std"
        for suffix, label, color, ls in conditions:
            path = LOG_DIR / f"probe_music{suffix}.log"
            if not path.exists():
                continue
            trained, untrained = parse_log(path, trained_hdr, untrained_hdr)
            if not trained:
                continue
            layers = [r[0] for r in trained]
            x = np.arange(len(layers))
            mlp_t = np.array([r[3] for r in trained])
            sd_t = np.array([r[4] for r in trained])
            mlp_u = np.array([r[3] for r in untrained])
            ax.errorbar(
                x, mlp_t, yerr=sd_t, label=f"{label}",
                color=color, linestyle=ls, marker="o", capsize=3,
            )
            ax.plot(x, mlp_u, color=color, linestyle=ls, alpha=0.3, marker="x")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["embed", "L1", "L2", "L3"])
        ax.set_xlabel("Layer")
        ax.set_title(f"music — {tgt}")
        ax.grid(True, alpha=0.3)
        if chance[tgt] is not None:
            ax.axhline(chance[tgt], color="black", linestyle=":", alpha=0.5, label=f"chance {chance[tgt]:.2f}")
        if tgt == "mode":
            ax.set_ylim(0, 1)
            ax.set_ylabel("Piece-level MLP accuracy")
        elif tgt == "chord":
            ax.set_ylim(0, 0.5)
        else:
            ax.set_ylim(0.2, 0.4)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Music probes — piece-level MLP accuracy (mean ± std over 5 seeds)\nsolid markers = trained, faint × = untrained")
    out = FIG_DIR / "week1_music_per_layer.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    plot_cities()
    plot_othello()
    plot_flight()
    plot_music()


if __name__ == "__main__":
    main()
