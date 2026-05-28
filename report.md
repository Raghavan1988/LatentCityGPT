# LatentWorldsGPT — Progress Report (2026-05-27)

*Self-contained progress writeup; no project background required to
read.*

## 1. What this project is, in one paragraph

LatentWorldsGPT is a comparative interpretability study. The question:
**under what conditions do next-token transformer language models
spontaneously develop an internal representation of the world they're
modeling, and where in the network does that representation live?**
The seminal result is Li 2022 (Othello-GPT) and Nanda 2023's
follow-up: a transformer trained only to predict the next move in
Othello develops a residual-stream representation of the current board
state, recoverable by a simple linear probe. The natural follow-up is
whether this generalizes — does it happen in domains beyond games,
under what conditions, in what shape? This project tests that across
five domains (cities, Othello, flight phases, music, symmetric-group
walks) using a unified methodology, with ex-ante predictions on a
sixth (maze navigation) to test the framework's predictive power.

## 2. The framework — N-criterion

The driving hypothesis is what we call the **N-criterion**: a feature
F is encoded in the model's residual stream **iff** the next-token
prediction objective requires F. Equivalently: features the model
needs in order to predict the next token will emerge as residual
representations; features irrelevant to next-token prediction will
not, even if they're in principle decodable from the input.

This is falsifiable. It makes risky predictions in advance. The most
informative single test we run is one where the N-criterion predicts
a **null** (some feature should NOT be encoded), and we measure
whether the probe finds it anyway.

## 3. Domains studied

| Domain | What the model predicts | What N-criterion says | Result |
|---|---|---|---|
| Cities | Next intersection ID on a walk | Geographic location is required for next-step prediction | encoded (linearly) |
| Othello | Next legal move | Board state is required for legality | encoded |
| Flight (ADS-B) | Next discretized flight bin | Phase is partially required | partially encoded |
| Music | Next pitch | Voice-leading is required; chord/beat are not | voice-leading + chord encoded; beat null |
| Symgroup (S_8) | Next group element | Partial product partially required | partial signal |
| **Maze** (ex-ante) | **Next cell on optimal path** | **Row/col/distance required; starting cell NULL** | **predictions locked; experiment in flight** |

Each domain has three corpus conditions: **real** (the natural data),
**within-shuffled** (token order shuffled within each sequence —
preserves set-membership but destroys order), and **global-shuffled**
(all tokens reassigned uniformly — destroys both order and identity).
This is the standard destroyed-structure control: real models should
show signal that drops monotonically as structure is destroyed.

## 4. Methodology — what makes this defensible

### 4.1 Multi-seed mean ± std

Every probe and transplant number reported is the mean ± std over
5 independent seeds, where each seed simultaneously varies:
- The untrained-control random initialization
- The activation sampling positions
- The probe-training RNG

This is unusually rigorous for the interp literature. Single-seed
numbers were the norm; we found and fixed one earlier bug (4 probe
scripts had been comparing two random-init models against each other,
silently masking signal) that the multi-seed protocol caught.

### 4.2 Probe + transplant convergence

For every claim about encoding, we test it two ways:
- **Probe** (descriptive): can a linear / MLP probe recover the
  feature from the residual stream? Strong signal = feature is
  represented.
- **Transplant** (causal): if we replace the recipient's residual
  with the donor's at layer L, does the next-token prediction
  shift toward the donor's expected continuation? Strong signal =
  feature is causally used.

These two tests are independent; convergent positive results are
much harder to game.

### 4.3 Per-layer ablation

For every domain, we run both probe and transplant **at every layer**,
not just one. This shows *where* the world state lives in the network
and how it's built up across depth — turning the question
"is X encoded?" into "where in the model is X encoded, and how is it
computed across blocks?"

### 4.4 Linear-vs-MLP encoding test

Nanda's strong claim (2023) is that world states are encoded
**linearly** in the residual stream, not just non-linearly. We test
this directly: linear vs MLP probe accuracy at the best layer. If the
gap is small, the encoding is approximately linear. Across all
positive domain × condition pairs we've tested, the gap is ≤ 0.13 —
Nanda's strong claim holds cross-domain.

### 4.5 Pre-registered ex-ante predictions

For the maze domain, we wrote down quantitative predictions
**before any experiment was run or any maze model trained**, locked
them in a git commit (timestamp = `aa025b1`), and made the file
append-only after that commit. The git audit trail is the evidence
that predictions preceded the experiment.

This is rare in mechanistic interpretability work, where most
analyses are post-hoc. A confirm/falsify table from a pre-registered
experiment is significantly harder to dismiss as cherry-picking.

## 5. Results landed so far

### 5.1 Cities (node-level, MLP, best layer, 5 seeds)

| City × Condition | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| London real | 0.6423 ± 0.0545 | 0.0927 ± 0.0130 | **+0.55** |
| London within-shuffled | 0.7393 ± 0.0505 | 0.0673 ± 0.0170 | **+0.67** |
| London global-shuffled | 0.0993 ± 0.0248 | 0.0870 ± 0.0359 | +0.012 (null) |
| Manhattan real | 0.6092 ± 0.0141 | 0.1033 ± 0.0078 | **+0.51** |
| Boston real | 0.6667 ± 0.0205 | 0.1117 ± 0.0042 | **+0.55** |

Clean across 3 cities × 3 conditions on London. The within-shuffled
case scoring higher than real is the interesting finding: within-
shuffling preserves geographic clustering but destroys graph
adjacency, and the probe target is geographic clustering, so the
within-shuffled model specializes harder on it.

### 5.2 Othello (per-cell board-state, 5 seeds)

| | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| Othello-50k | 0.9399 ± 0.0012 | 0.5963 ± 0.0058 | **+0.34** |

The Othello 0.9399 matches the published Li 2022 / Nanda 2023 number
(~0.94) within 0.01. The 5-seed standard deviation is 0.0012, which
means our reproduction is more stable than the typical published
single-run number. This served as our framework-validation check
(see § 4.2 — both probe and transplant pass; § 4.3 — peak at L4; § 4.4
— linear-vs-MLP gap +0.131).

### 5.3 Flight phase (5 seeds)

Clean monotonic 3-condition gradient on the trained MLP at L2
(flight-level honest split):
- real: 0.8817 ± 0.0792
- within-shuffled: 0.6785 ± 0.1834
- global-shuffled: 0.3766 ± 0.1512

The gradient persists on transplant lift too (real +0.47 / within
+0.30 / global +0.00).

### 5.4 Music (5 seeds, piece-level honest split)

The most reframed result in the suite. The N-criterion predicts:
**voice-leading should be encoded** (you need it for next-pitch
prediction); **beat-in-measure should NOT** (voice-leading is locally
predictable without knowing beat). Multi-seed confirms:

- **Voice-leading transplant** (real): +0.889 ± 0.007 → strongly
  encoded and causally used.
- **Beat probe** (real): trained MLP 0.2798 ± 0.0075 vs untrained
  0.2740 ± 0.0077 — gap +0.006, **null within 1σ of zero**.
- **Beat transplant** (real): transplant Δ is *less* than the random
  control on every metric (Δ KL = −3.29, Δ argmax-changed = −0.50).
  Beat is not just unreadable correlationally; it is causally inert.
- **Chord probe** (real, piece-level): trained MLP 0.3035 ± 0.0202
  vs untrained 0.2147 ± 0.0285 — gap +0.089 (~3σ). Real signal.
  Drops to +0.018 under within-shuffle, confirming structure-
  dependence.

This convergence — probe null **and** transplant null on beat,
probe positive **and** transplant positive on voice-leading and
chord — is the cleanest case of the N-criterion making both positive
and negative predictions that hold.

### 5.5 Symgroup (S_8 partial-product, 5 seeds × 3 conditions)

| Variant | Trained MLP (word-level) | Untrained MLP | Gap |
|---|---|---|---|
| sa real | 0.3599 ± 0.0050 | 0.3074 ± 0.0039 | +0.053 (~10σ) |
| sa within-shuffled | 0.2879 ± 0.0018 | 0.2778 ± 0.0039 | +0.010 (null) |
| sa global-shuffled | 0.2913 ± 0.0040 | 0.2774 ± 0.0034 | +0.014 (null) |

Small but real positive on the honest split; destroyed-structure
controls null out cleanly. Symgroup didn't reach the absolute-
accuracy threshold for a "strong positive control" but the cross-
condition gradient is clean and the verdict is consistent with the
N-criterion (partial product is partially used for next-element
prediction).

### 5.6 Encoding-locality taxonomy (where state lives per domain)

Per-layer transplant ablation tells us not just *whether* the state
is encoded but *where* in the network it lives. Peak transplant lift
per domain (5 seeds):

| Domain | L0 (embed) | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|---:|
| London real | +0.735 | +0.815 | +0.901 | **+0.937** | +0.919 | +0.916 |
| Manhattan real | +0.820 | +0.853 | +0.928 | +0.969 | **+0.974** | +0.973 |
| Boston real | +0.869 | +0.748 | +0.756 | +0.868 | +0.918 | **+0.933** |
| Othello (50k) | +0.040 | +0.062 | +0.108 | **+0.296** | — | — |
| Music real (voice-leading) | +0.035 | +0.813 | **+0.889** | — | — | — |

Two interesting patterns:
1. **Peak layer shifts deeper as the world gets bigger.** London
   (663 nodes) peaks at L3, Manhattan (4,546) at L4, Boston (11,368)
   at L5. The bigger the world, the more depth the model uses to
   consolidate the representation.
2. **Music shows a huge L0 → L1 jump** (+0.035 → +0.813). The
   voice-leading state is **transformer-computed**, not embed-
   encoded. Cities, by contrast, has +0.74 already at the embed —
   the embed table itself carries most of the geographic structure
   (because each token = one geographic location). Music has 60
   pitch tokens shared across all pieces, so the embed can't encode
   "what's the voice-leading neighborhood of this token" — the
   transformer is forced to compute it from context.

### 5.7 Procrustes-aligned cities map overlay

For each cities model, we extract token embeddings, fit a linear
projection to (lat, lon), Procrustes-align with real London coords,
and plot the overlay. The result (in `figs/phase5_cities_overlay.png`):

- Real-trained model: median per-cell error 213 m
- Within-shuffled model: median 104 m (cleanest geographic clustering
  because the model is freed from learning graph adjacency)
- Global-shuffled model: median 444 m (essentially noise)

This is the headline figure that makes the decomposition story
immediately visible.

## 6. The ex-ante prediction experiment (in flight)

We picked **maze navigation** as the domain to test the N-criterion
predictively (committed: `predictions/predictions_maze_navigation.md`,
git timestamp `aa025b1`). Four predictions, written before any maze
model existed:

| Prediction | N-criterion verdict | Quantitative band |
|---|---|---|
| 1. Current cell row | encoded | MLP 0.70–0.97, gap ≥ 0.40 |
| 2. Current cell col | encoded | MLP 0.70–0.97, gap ≥ 0.40 |
| 3. Manhattan distance to goal | encoded | MLP 0.35–0.75, gap ≥ 0.20 |
| 4. **Starting cell** | **NULL** | **gap ≤ 0.10** (load-bearing risky claim) |

Cross-target predictions: row/col peak shallower than distance;
destroyed-structure controls null out monotonically; linear ≈ MLP
within 0.30.

**Current state**: maze training done on all 3 conditions; probe +
transplant in progress on the real maze condition and within-shuffled
condition; global-shuffled condition queued. Expected completion of
the confirm/falsify table: **tonight 21:00–22:00 PDT**.

If the starting-cell NULL holds (gap ≤ 0.10 as predicted), the
N-criterion will have made a risky negative prediction in advance
that confirmed. If it falsifies (gap > 0.15), the framework will need
revision on this domain — also informative, just in a different way.

## 7. Methodological assets — committed code and documentation

| Asset | Where | What it does |
|---|---|---|
| Probe pipeline | `eval/probe_*.py` | Multi-seed mean ± std probes across 5 domains |
| Transplant pipeline | `eval/transplant_*.py` | Multi-seed mean ± std causal-substitution per domain |
| Per-layer ablation | `repro/phase2_transplant_multiseed.sh` | Runs every probe and transplant at every layer × 5 seeds |
| Complementary causal-interp | `eval/dla.py`, `eval/logit_lens.py`, `eval/zero_ablation.py` | DLA + logit lens + zero-ablation; running across all checkpoints |
| Pre-registration protocol | `predictions/README.md` | 6-step protocol, lockdown rules, audit trail via git |
| Aggregators | `viz/transplant_aggregate.py`, `viz/symgroup_aggregate.py`, etc. | Parse raw logs into mean ± std summary tables |
| Visualization | `viz/overlay.py`, `viz/per_layer_plots.py`, `viz/transplant_plot.py` | Figures published in `figs/` |
| Phase 1 writeup | `update_phase1.md` | Probe-rigor results across all 4 domains |
| Phase 2 writeup | `update_phase2.md` | Transplant + linear-encoding + symgroup |
| Project status (high-level) | `CLAUDE.md`, `PLAN.md`, `pivot.md`, `STATUS_vs_OTHELLO-GPT.md` | Plan + current state + comparison to literature |

## 8. What's not yet done

### 8.1 Still in flight (autonomous, completes tonight)

- Phase 4-d probe + transplant on within-shuffled and global-shuffled
  maze conditions (queued)
- Phase 3-d/e/f remaining runs (~11 of 36, mostly slow zero-ablation
  on cities)
- Phase 4-e: writeup of the confirm/falsify table (after compute finishes)

### 8.2 Pending GPU rental (Phase 3-b + Phase 3-c)

- Phase 3-b: retrain Othello on championship-quality games (WTHOR
  archive). Configs and parser are ready; data download is one
  `wget` loop away. Estimated training: 1 hour on H100. Tightens
  the Othello comparison to "matches published exactly."
- Phase 3-c: train a ~50M-param cities model on full Manhattan. This
  is the **scale demonstration** that answers "does this work at
  larger model sizes?". Config is ready. Estimated training: 6-10
  hours on H100.

Single rental window for both: ~$30-40. Runbook in
`repro/phase3_bc_gpu_runbook.md`.

### 8.3 Writing (the bulk of remaining effort)

The experimental results are now overwhelmingly in hand. The bulk of
remaining work is **writing**, not running:
- A polished first draft (Phase 5-d) — 3-4 days of focused work.
- An outside-reader pass (Phase 5-e) — calendar-bound; 1-2 weeks.
- Limitations section, reproducibility check, final pass (Phase 6) —
  ~1 week.

Realistic calendar estimate from today: **2.5 to 3 weeks** to a
publication-quality draft.

## 9. Honest limitations

What I'd flag explicitly:

1. **Small-model scale.** Most models in this study are 4M–13M params.
   Bigger model + bigger corpus (Phase 3-c) would address this
   concern; it's queued, not done.
2. **Synthetic mazes (in flight).** The ex-ante prediction experiment
   uses synthetic 8×8 mazes generated by recursive backtracking. This
   is the cleanest possible test of the N-criterion framework, but a
   reader might want a real-world ex-ante test too.
3. **Symgroup as a partial positive control.** Symgroup didn't reach
   the absolute-accuracy threshold for a "strong positive control"
   (0.36 word-level MLP). It is a real cross-condition gradient but
   a partial-signal data point, not a clean encoded-feature
   demonstration.
4. **Cities probe interpretation needed care.** Position-level probes
   on small-vocab cities (London) inflate via memorization; the
   node-level honest split is the right test. We use node-level
   throughout, but a reader who only looks at position-level
   numbers will see misleadingly strong signal.
5. **No frontier-model probe.** The methodology has not been applied
   to a Llama-class model. This is out of scope (would need much
   larger compute budget) but is the natural follow-up.

## 10. Open questions an outside reader can usefully weigh in on

In rough order of how much external eyes help:

1. **The N-criterion framing.** Does the prose in this report
   accurately frame what the project is doing? Does it scan as
   coherent methodology or as marketing?
2. **The pre-registered maze predictions.** The file
   `predictions/predictions_maze_navigation.md` is locked but its
   wording could be sharpened. Does it have the right risk/safety
   trade-off?
3. **The cross-domain claim.** With 4 positive controls (cities,
   Othello, flight, music voice-leading) + 1 partial (symgroup) +
   1 in-flight (maze), is the cross-domain comparative claim
   defensible? Where would a skeptical reader push back?
4. **The limitations.** Are there limitations in § 9 that I've
   missed or that should be elevated to a stronger caveat?
5. **The path to a draft.** Is the 2.5–3 week timeline to a
   publication-quality draft realistic given the experimental
   evidence in hand?

## Appendix: links to the key artifacts

- This file: `report.md`
- Phase 1 writeup: `update_phase1.md`
- Phase 2 writeup: `update_phase2.md`
- Phase 3–6 plan: `plan_phases_3_6.md`
- Locked predictions: `predictions/predictions_maze_navigation.md`
- Status vs literature: `STATUS_vs_OTHELLO-GPT.md`
- High-level pivot rationale: `pivot.md`
- GPU runbook: `repro/phase3_bc_gpu_runbook.md`
- Headline figures: `figs/phase1_*_per_layer.png`,
  `figs/phase2_transplant_lift.png`, `figs/phase5_cities_overlay.png`
