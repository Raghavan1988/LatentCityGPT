# Week 1 Multi-Seed Retrofit — Update (2026-05-27)

All probe headline numbers across the four trained domains (cities, Othello,
flight, music) have been recomputed under a multi-seed protocol with two
bug fixes applied. This document records the corrected numbers and the
reframed interpretation that follows.

## What changed

### Fix 1 — Model state was not loaded into "trained" probes

`eval/probe_othello.py`, `eval/probe_music.py`, `eval/probe_symgroup.py`, and
`eval/probe_sanity.py` previously constructed a `GPT(config)` instance but
never called `load_state_dict(ckpt["model_state"])`. The variable named
`trained` held a random-init model whose only difference from `untrained`
was the seed of the PyTorch RNG. Any comparison of "trained vs untrained"
under those scripts was uninformative.

`eval/probe_cities_grid.py` and `eval/probe_flight.py` were not affected.

### Fix 2 — Single-seed numbers replaced by multi-seed mean ± std

All five probe scripts now run an outermost seed loop that simultaneously
varies (a) the untrained-init random initialization, (b) the activation
sampling positions, and (c) the probe-training RNG. Per-layer accuracies
aggregate across 5 seeds with `ddof=1` standard deviation. Headlines are
selected by best-layer-by-mean to avoid the per-seed max-layer inflation
seen in earlier single-seed runs.

The cities-grid script already had a partial outermost loop; the others
were brought to the same protocol. Implementation in commit `37c31c2`.

## Corrected headline numbers

All numbers below are MLP-probe accuracy at the best layer (selected by
mean across seeds) under the honest split — node-level for cities,
per-cell mean for Othello, flight-level for flight, piece-level for
music. Linear-probe numbers are recorded in the raw logs under
`checkpoints/multiseed_w1/`.

### Cities (NODE-LEVEL, MLP, best layer)

| City × Condition       | Trained MLP        | Untrained MLP      | Gap     |
|------------------------|--------------------|--------------------|---------|
| London real            | 0.6423 ± 0.0545    | 0.0927 ± 0.0130    | +0.550  |
| London within-shuffled | 0.7393 ± 0.0505    | 0.0673 ± 0.0170    | +0.672  |
| London global-shuffled | 0.0993 ± 0.0248    | 0.0870 ± 0.0359    | +0.012  |
| Manhattan real         | 0.6092 ± 0.0141    | 0.1033 ± 0.0078    | +0.506  |
| Boston real            | 0.6667 ± 0.0205    | 0.1117 ± 0.0042    | +0.555  |

The pattern is clean across 3 cities × 3 destroyed-structure conditions on
London. Untrained MLP at ~0.10 reflects 10×10 grid + class-imbalance prior;
trained MLP at ~0.6–0.7 represents real geographic encoding. Global-shuffled
drops to untrained baseline, as expected.

The London within-shuffled point being higher than London real is consistent
with the May-24 decomposition reading: within-shuffling preserves geographic
clustering but destroys graph adjacency. The grid probe tests clustering, not
adjacency, and the within-shuffled model is free to specialize on clustering
without a graph-topology constraint.

Linear and MLP probes land within 5–10 pts of each other on node-level, so
the cities encoding is approximately linear (the Nanda-2023 strong claim
holds within our setup).

### Othello (per-cell MLP at L4)

| | Trained MLP        | Untrained MLP      | Gap     |
|--|--------------------|--------------------|---------|
| Othello-50k | 0.9399 ± 0.0012    | 0.5963 ± 0.0058    | +0.344  |

5-seed std of 0.0012 is the tightest we see — the trained Othello board-state
encoding converges to nearly identical accuracy across all sources of
variance. Trained MLP 0.9399 lands within 0.01 of the Li-et-al / Nanda
published ~94%.

### Flight (flight-level, MLP)

| Condition           | Trained MLP        | Untrained MLP      | Gap     |
|---------------------|--------------------|--------------------|---------|
| real                | 0.8817 ± 0.0792    | 0.7765 ± 0.1043    | +0.105  |
| within-shuffled     | 0.6785 ± 0.1834    | 0.5783 ± 0.1715    | +0.100  |
| global-shuffled     | 0.3766 ± 0.1512    | 0.3236 ± 0.1080    | +0.053  |

Trained MLP drops monotonically across the destroyed-structure conditions
(0.88 → 0.68 → 0.38). The trained-vs-untrained gap is small (+0.05 to +0.11)
because the untrained model on flight scores high in absolute terms (flight
phase is highly predictable from token frequencies alone).

Std on flight-level is large (0.08–0.18) — only 238 quickstart flights and
held-out flights show seed-dependence in which flights end up in test.

### Music (piece-level, MLP)

| Target × Condition         | Trained MLP        | Untrained MLP      | Gap     |
|----------------------------|--------------------|--------------------|---------|
| mode  · real               | 0.8198 ± 0.0154    | 0.7528 ± 0.0458    | +0.067  |
| mode  · within-shuffled    | 0.9428 ± 0.0119    | 0.8142 ± 0.0541    | +0.129  |
| mode  · global-shuffled    | 0.5952 ± 0.1168    | 0.5482 ± 0.0583    | +0.047  |
| chord · real               | 0.3035 ± 0.0202    | 0.2147 ± 0.0285    | +0.089  |
| chord · within-shuffled    | 0.1444 ± 0.0120    | 0.1259 ± 0.0103    | +0.018  |
| chord · global-shuffled    | 0.1137 ± 0.0085    | 0.1017 ± 0.0119    | +0.012  |
| beat  · real               | 0.2798 ± 0.0075    | 0.2740 ± 0.0077    | +0.006  |
| beat  · within-shuffled    | 0.2639 ± 0.0083    | 0.2632 ± 0.0108    | +0.001  |
| beat  · global-shuffled    | 0.2730 ± 0.0047    | 0.2740 ± 0.0087    | −0.001  |

Beat majority baseline is 0.2501. Trained and untrained both sit at chance
for beat in all three conditions; the gap is within 1σ of zero.

This is the most reframed result in the suite. The earlier "music probe is
null on all three targets" reading collapses once Fix 1 is applied:

- **Beat is null** (gap ≈ 0). Consistent with the N-criterion argument that
  voice-leading next-token prediction does not require beat-in-measure
  representation.
- **Mode is small but positive** (real gap +0.067; within-shuffled gap +0.129).
  Mode is lexically recoverable from the pitch set, and within-shuffling
  preserves pitch set, which is why the within-shuffled model can specialize
  on the lexical signal harder than the real model.
- **Chord shows a clear positive signal in the real condition** (real gap
  +0.089, ~3σ separation). This collapses under within-shuffling (+0.018)
  and global-shuffling (+0.012), consistent with chord being a local
  structural feature that depends on multi-pitch ordering.

The cleanest causal corroboration for music (voice-leading transplant
+0.804/+0.071/−0.010 from `updateMay26_night.md`) is consistent: the model
encodes the local pitch-set state required for next-pitch prediction.

## Per-domain summary

| Domain    | Trained MLP (real, honest split)        | σ scale     | Encoding evidence |
|-----------|------------------------------------------|-------------|------------------|
| Othello   | 0.9399 ± 0.0012  (per-cell)              | < 0.005     | strong           |
| Cities    | 0.61 – 0.67  (node-level, 3 cities)      | 0.01–0.05   | strong           |
| Flight    | 0.8817 ± 0.0792  (flight-level)          | 0.08–0.18   | moderate         |
| Music     | chord 0.3035 ± 0.0202 (piece-level)      | 0.01–0.06   | partial          |

The encoding-locality taxonomy from May-26 still holds. The change is in
music: the probe-side story is now "chord is encoded, beat is not, mode is
lexically recoverable" rather than "all three are null."

## Files written / changed

- Multi-seed protocol + load_state_dict fix landed in commit `37c31c2`.
- Raw probe logs: `checkpoints/multiseed_w1/probe_*.log`.
- Runner scripts: `repro/cities_grid_multiseed.sh`, `repro/stage2_runner.sh`.

## Next

- Per-layer accuracy figures from the new logs (Task 71).
- Update of `STATUS_vs_OTHELLO-GPT.md` once the figures are in.
- Symmetric-group methodology calibration remains open (Task 56 / Milestone 1
  per `pivot.md`).
