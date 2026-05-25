# PLAN.md — build plan after the 2026-05-24 pivot

The single-domain "Othello-GPT on cities" plan ran to completion and produced a
decomposition result (see `update_may24_final.md` and `STATUS_vs_OTHELLO-GPT.md`).
The project pivoted to a comparative study of where Othello-GPT-style results
extend; the master plan for the pivot lives in `pivot.md`. This file is the
phased build plan that subsumes both:

- **Phases 0–5** below are the original cities phases, marked DONE with summary
  notes. They constitute the worked example that the multi-domain milestones
  will reproduce.
- **Phases 6 onward** are the new multi-domain milestones from `pivot.md`,
  summarized here so PLAN.md is self-contained. See `pivot.md` for full detail
  on each.

Work-style invariant across all phases: small-first, smoke-test before scaling,
the per-domain analogue of THE ONE RULE (no probe-target value enters the
model's input).

---

## Phase 0 — Data (cities) ✅ DONE

`data/prepare_city.py` is written, smoke-tested, and four real-city corpora are on disk.

- Pipeline: OSM pull (OSMnx) → largest strongly-connected component → trivial
  tokenizer (`0=PAD, 1=BOS, 2=EOS`, real nodes from 3) → blended shortest-path
  + random-walk routes → destination holdout → nanoGPT-format outputs.
- `--place` accepts ≥1 names; OSMnx unions them.
- `--shuffle_routes` (weak destroyed-structure) and `--shuffle_globally` (strict
  destroyed-structure) flags added in the pivot work — both are reused per-domain.
- **Acceptance (met):** `tests/test_prepare_city.py` passes on a synthetic
  grid; pipeline produces real corpora at `data/{london_city, manhattan,
  boston, southbay, london_shuffled, london_global_shuffled}/`.

**Per-domain follow-ups** (Phase 6+): replicate the pipeline pattern, including
both destroyed-structure flags.

---

## Phase 1 — Model & training ✅ DONE

`model/model.py`, `model/train.py`, `model/configs/{small,medium}.py` written.
nanoGPT-style decoder-only transformer, weight-tied output head, PAD-masked
loss, dtype-asserted input.

- **Acceptance (met):** trained London / Manhattan / Boston smoke models. Val
  loss decreases smoothly; checkpoints load + generate routes without error.
- Three additional London variants trained for the cities decomposition result:
  real / within-route-shuffled / global-shuffled.

---

## Phase 2 — Intrinsic eval ✅ DONE

`eval/valid_edge.py` (next-step + full-route validity) and `eval/baselines.py`
(uniform / unigram / 1st-and-2nd-order Markov + long-range coherence) written.

- **Acceptance (met):** Real London valid-edge rate 99.7%/val, 99.3%/gen;
  baselines + GPT compared on the real → real protocol; GPT beats Markov-2
  on perplexity (1.23 val vs 1.46) and modestly on long-range coherence.
- **Open follow-up:** geographic-region holdout (`--split geographic` in
  `prepare_city.py`); LSTM baseline. Both deferred to multi-domain work.

---

## Phase 3 — Baselines ✅ DONE (LSTM still deferred)

Same `eval/baselines.py`. LSTM baseline is the one deferred item.

---

## Phase 4 — Probe suite ✅ DONE with node-level split

`eval/probe.py` with linear + MLP probes, per-layer sweep, untrained-model
control, **position-level + node-level splits** (the pivot's finding #2).

- **Acceptance (met):** linear-probe trained-vs-untrained gap is positive
  on the node-level split in all three cities (London +0.69, Manhattan +0.18,
  Boston +0.18 R²).
- **Pivot finding:** MLP probe is contaminated by lookup memorization in
  continuous-target settings; the "linear ≈ MLP" criterion for "linearly
  encoded" does not apply cleanly. The node-level split is the necessary
  control.
- **`eval/embedding_compare.py`** (added in pivot work) shows the trained
  `wte` is NOT a node2vec embedding of the graph; node2vec produces *stronger*
  geographic decodability than `wte` alone. The model's geographic signal is
  built by higher layers, not delivered by the embedding table.

---

## Phase 5 — Causal intervention ✅ DONE (corrected version)

`eval/causal.py` was written first using pseudoinverse-direction patching; it
did not isolate causal use of the representation (same statistical signal on
real and destroyed-structure models). **The file is preserved as the documented-
failure version.**

`eval/transplant.py` is the corrected Phase 5: substitute a real residual
`a_B` for `a_A` at layer L. Validated on three conditions:

| Condition                  | val ppl | P(A nbrs) | Transplant lift on P(B nbrs) |
|---|---:|---:|---:|
| Real London                | 1.65    | 0.984     | **+0.953**                   |
| Within-route shuffled      | 25.0    | 0.061     | +0.247                       |
| Global shuffled            | 313     | 0.006     | +0.000 (chance per-position) |

The differential decomposes the causal effect into a geographic-clustering
contribution (~+0.25, present in any geographically-co-occurring corpus) and
a sequence-trained graph-adjacency contribution (~+0.70, only in the real
model).

- **Acceptance (met):** patching reliably bends next-hop predictions toward
  the patched location's real neighbors, well above the random-direction
  control. Destroyed-structure control gives the predicted null.

---

## Phase 6 — Multi-domain expansion (from `pivot.md`)

The portfolio of additional domains, each applying the cities template
(data pipeline + small.py training + probe with node-level split + two-tier
destroyed-structure control + `eval/transplant.py`). Ordered by cost first,
risk first.

### Milestone 1 — Symmetric-group-GPT (methodology calibration)

**Effort:** 1–2 days. Lowest-risk; guaranteed-positive domain by construction.
- `data/prepare_symgroup.py` synthetic generator (words in Sₙ, target is
  resulting permutation).
- Train small.py; probe via `eval/probe.py` (with node-level split);
  destroyed-structure via within-word generator shuffle.
- **Acceptance:** node-level linear probe accuracy > 0.9 for n ≤ 8;
  destroyed-structure control drops to chance.

### Milestone 2 — Music (within-domain mixed-verdict experiment)

**Effort:** 2–3 days. Highest scientific value per day in the portfolio.
- `data/prepare_music.py` using `music21` Bach chorales.
- Train small.py.
- Compute three probe targets (key signature, current chord, beat position)
  via `music21` and run each through `eval/probe.py` with the standard
  destroyed-structure suite.
- **Acceptance:** key and chord probes show the cities-like failure mode (high
  R² on real AND on within-piece-shuffled). Beat probe shows the Othello-like
  positive (high accuracy on real, collapsed on shuffle).

### Milestone 3 — Dialog-state tracker (applied text)

**Effort:** 2–3 days.
- `data/prepare_multiwoz.py` (MultiWOZ + BPE-tokenize utterances).
- Train small.py with block_size adjusted for dialog lengths.
- Probe each slot's value at each turn; focus eval on *inferred* slots that
  fail surface-mention shortcuts.
- Destroyed-structure: shuffle turns within a dialog.
- **Acceptance:** per-slot probe accuracy with inferred-vs-surface breakdown.
  Destroyed-structure kills inferred-slot accuracy.

### Milestone 4 — Flight-phase (applied aviation / temporal integration)

**Effort:** 3–5 days. The cleanest Othello-fit in the applied portfolio.
- `data/prepare_adsb.py` using the `traffic` library + OpenSky Network.
- Discretize altitude / vertical-rate / ground-speed / heading; phase labels
  via Sun et al. fuzzy logic.
- Train small.py; probe + two-tier destroyed-structure control; reuse
  `eval/transplant.py` unchanged.
- **Acceptance:** layer-wise phase-probe accuracy figure; within-flight shuffle
  ablation; transplant intervention shows phase-conditioned shift in next-
  token distribution.

### Milestone 5 — Maze-GPT (applied spatial; optional)

**Effort:** 5–7 days. Optional / upside.
- `data/prepare_maze.py` (procedural maze generator + agent observation model).
- Probe for agent pose, wall configuration, explored-cell map.

### Milestone 6 — Paper assembly

**Effort:** 5–7 days.
- Unified figure across all domains with consistent probe protocol.
- Rewrite `CONTEXT.md` once domains land.
- Workshop-paper draft.

---

## Definition of done (project)

The pivot succeeded if, across at least four domains (cities + sym-group +
music + one of {dialog, flight-phase}), the package demonstrates:

1. A reproducible three-condition gradient (real / weak-destroyed /
   strict-destroyed) per domain.
2. A predictive characterization (D, N, ¬L criteria or a tightened version
   after Milestone 2) that explains why some probes succeed and others fail.
3. The methodological caveats (MLP-probe lookup contamination; within-route
   shuffle insufficient when set-membership is the relevant co-occurrence
   structure) demonstrated cleanly on the cities decomposition.

See `pivot.md` for the risk register, confidence summary, and decision points.

---

## Pointers

- `pivot.md` — master plan for the pivot; comprehensive milestones, risks,
  confidence summary.
- `update_may24_final.md` — empirical narrative of the cities decomposition
  session.
- `STATUS_vs_OTHELLO-GPT.md` — claim-by-claim comparison to the Othello-GPT
  literature; what cities establishes.
- `next_steps.md` — short concrete plan for the experiments that were run
  during the pivot session.
