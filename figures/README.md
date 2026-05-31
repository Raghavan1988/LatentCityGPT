# Figures

Visual companions to `paper_draft.md` and `report.md`. Each figure
is self-contained and references the corresponding paper sections.

| # | File | Format | What it shows |
|---|---|---|---|
| 1 | [`01_experimental_arc.md`](01_experimental_arc.md) | Mermaid (renders on GitHub) | The pre-register → falsify → revise → re-test timeline with commit hashes and gap numbers inline |
| 2 | [`02_results_matrix.md`](02_results_matrix.md) | Markdown heatmap | Cross-domain × feature × condition outcome matrix with color-coded verdicts |
| 3 | [`03_per_layer_ablation.png`](03_per_layer_ablation.png) | PNG (matplotlib) | Where each (domain, feature) representation lives across transformer depth |
| 4 | [`04_cross_condition_gradient.png`](04_cross_condition_gradient.png) | PNG (matplotlib) | Real / within-shuffled / global-shuffled gap comparison; shows where the monotonicity prediction holds and where it does not |

The matplotlib figures are regeneratable from the source scripts:

```bash
python figures/03_per_layer_ablation.py
python figures/04_cross_condition_gradient.py
```

## Mapping to paper sections

| Figure | Best paired with |
|---|---|
| 1 (experimental arc) | `paper_draft.md` §4.11, `report.md` §4.10 |
| 2 (results matrix) | `paper_draft.md` §5.9, `report.md` §7.3 |
| 3 (per-layer ablation) | `paper_draft.md` §5.6, `report.md` §5.3 |
| 4 (cross-condition gradient) | `paper_draft.md` §3.3 and §5, `report.md` §4.2 and §5 |
