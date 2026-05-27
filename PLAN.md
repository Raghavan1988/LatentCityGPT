# PLAN.md — build plan after the 2026-05-24 pivot

> **Status update — 2026-05-26 night (most recent).** Cross-domain
> transplant experiments completed (`updateMay26_night.md`). The corrected
> interpretation: cities + music are TOKEN-LOCAL (transplant lifts
> ~0.8-0.95); Othello is PREFIX-DERIVED (transplant ~0.1-0.2). Music
> transplant new (`eval/transplant_music.py`) — real +0.804 / within-
> shuffled +0.071 / global-shuffled −0.010 (clean 3-condition gradient).
> **Music's null on classification probes is now causally demonstrated
> as principled N-criterion failure:** voice-leading IS encoded
> (transplant works); beat/mode/chord are NOT encoded (probe at chance).
> Same model + same framework + different feature targets = opposite
> outcomes.

> **Status update — 2026-05-26 evening.** Beyond the original plan we have
> now reproduced **Othello-GPT from scratch in this codebase** as the
> framework-validation positive control. See `updateMay26_evening.md`.
> The plan didn't originally include this — Othello was supposed to be a
> literature anchor (cite, don't reproduce) — but the user asked "are you
> sure there's no bug?" after music's M2 first-pass, and the cleanest
> answer was a from-scratch reproduction. Result: 3-class MLP probe 91.19 %
> (vs published 94 %; within 3 pts), 3-class linear 77.15 % (in published
> 75-85 % range). Framework validated. See "Phase 7: Othello-GPT in-codebase
> reproduction (unplanned)" at the bottom for the build artifacts.

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

**Status (2026-05-26):** First-pass DONE, methodology calibration
**inconclusive**. See `updateMay26.md`. Two variants tried:
- *Random uniform words:* model converges to uniform-over-generators
  (no learnable signal); probe at majority-class baseline.
- *Self-avoiding walks on Cayley graph of S₈* (Othello-analog):
  three-condition val_ppl gradient lands cleanly (5.90 / 6.72 / 6.75
  real / within / global), but partial-product probe collapses to
  lexical-only signal (33% mean across element-probes; trained ≈
  untrained). The self-avoiding constraint apparently does not force
  full-product encoding — local recent-state info suffices for picking a
  legal next move. Probe code is NOT obviously broken (it finds the
  lexical signal that exists) but the task isn't a clean known-positive.

**Effort:** 1–2 days. ~~Lowest-risk; guaranteed-positive domain by construction.~~
Originally framed as lowest-risk; actual run showed designing a
synthetic task that *forces* permutation tracking is harder than
expected.
- `data/prepare_symgroup.py` synthetic generator with `--self_avoiding`
  flag. ✅ Done.
- Train small.py; probe via `eval/probe_symgroup.py`. ✅ Done.
- **Acceptance:** node-level linear probe accuracy > 0.9 for n ≤ 8. ❌
  Not met (achieved ~0.33 mean; trained ≈ untrained).
- **Next:** redesign task so each move's legality depends on *full*
  history (not just recent), or pre-pend an initial-permutation tokenized
  prefix that must be tracked. Out of scope for current sessions.

### Milestone 2 — Music: three load-bearing probes, three independent bets

**Status (2026-05-25 + 2026-05-26 morning + 2026-05-26 afternoon):**
First-pass + heavy-probe followup + retraction + expanded-corpus v2
ALL DONE. See `updateMay26_afternoon.md` for the final clean picture.
Headlines: voice-leading gradient (98.99/25.93/58.77 strict) is the
load-bearing positive; all classification probes (beat, mode, chord)
show trained ≈ untrained on PIECE-LEVEL under honest 4-seed reporting;
diagnosis is "N criterion fails — next-pitch objective doesn't require
beat/mode/chord encoding."

**Status (2026-05-25 + 2026-05-26, post-correction):** First-pass +
heavy-probe followup + correction DONE. See `updateMay25.md` and
`updateMay26.md` (which has a § Correction section). Headlines after
all three:
- **Voice-leading gradient is the load-bearing positive** (96.25% / 64.33%
  / 55.91% strict, real / within / global) — the cities valid-edge
  analog. Robust.
- **Mode probe is pure lexical artifact** — trained ≈ untrained across all
  three conditions; the original "60% cities-analogue leak" claim from
  updateMay25 was a random-embedding artifact, not a learned signal.
  Robust.
- **Beat probe sits at chance (~28–32%) in all conditions** with trained
  ≈ untrained on PIECE-LEVEL. Robust across multi-seed.
- **The "cities-style reversal in music" claim from updateMay26 is
  RETRACTED.** It was a single-seed × max-across-layers artifact;
  multi-seed verification (3 light seeds + 1 heavy seed) shows the
  within-shuffled model does NOT have a higher beat probe than the
  real model.
- **Joint outcome A (predicted mixed verdict) does NOT land.** Honest
  framing: cities-style methodology cautionary tale stands on its own
  in cities; music gives a clean voice-leading gradient + confirms
  mode-as-lexical-artifact, but does NOT replicate the cities reversal.

**Effort:** 2–3 days. Highest scientific value per day in the portfolio.
- `data/prepare_music.py` using `music21` Bach chorales.
- Train small.py.
- Compute three probe targets (key signature, current chord, beat position)
  via `music21` and run each through `eval/probe.py` with the two-tier
  destroyed-structure suite (within-piece shuffle + global shuffle), matching
  the cities decomposition.

**Each probe is an independent bet on whether sequence order is load-bearing
for that target in tonal music.** All three outcomes have substantial reward
or risk depending on direction; see pivot.md Milestone 2 for the full
outcome matrix.

- **Key probe** — predicted to *survive* shuffle (cities-analogue). If it
  collapses, key is sequence-trained (cadential context); ¬L stops being
  binary.
- **Chord probe** — predicted to *survive* shuffle. If it collapses,
  functional-harmony context is sequence-trained — a finer-grained Othello-
  positive than beat.
- **Beat probe** — predicted to *collapse* on shuffle (the load-bearing
  positive). If it survives, tonal pitch-class statistics leak beat
  information (strong beats favor I/V), and the within-domain positive role
  has to move elsewhere.

**Joint outcomes (2³ space → four qualitative cases):**

- **A.** Predicted mixed verdict (key + chord survive, beat collapses) →
  paper's originally-planned central figure lands.
- **B.** Universal cities-like failure (all three survive) → beat-leak is
  real; within-domain positive role moves to a less-leaky corpus or
  milestone.
- **C.** Universal Othello-like positive (all three collapse) → music is
  less leaky than expected; mixed-verdict claim fails but music becomes a
  clean positive analogue.
- **D.** Graded leak (any partial-mixed pattern) → ¬L reshapes from binary
  to a quantity; highest ceiling, highest write-up cost.

- **Acceptance:** the three-probe × three-condition table (real /
  within-piece-shuffled / global-shuffled) on a single trained model with
  node-level splits. Framing decision (which of A–D the paper centers on)
  is made *after* the table is in hand.

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

## Phase 7 — Othello-GPT in-codebase reproduction ✅ DONE (unplanned, 2026-05-26 evening)

Triggered by the question "are you sure there's no bug?" after music's
M2 first-pass. Original plan had Othello as a literature anchor (cite
Li 2022 / Nanda 2023, don't reproduce). This phase added a from-scratch
reproduction as the framework-validation positive control.

### Build artifacts

- `data/prepare_othello.py` — 8x8 Othello rules (placement, flips,
  legal moves), random-uniform-play game generator, tokenizer (PAD/BOS/
  EOS/PASS + 64 board cells = vocab 68), `board_state.csv` side table.
- `tests/test_prepare_othello.py` — 8 offline tests; all pass.
- `eval/valid_othello_move.py` — Othello analog of cities `valid_edge`
  and music `valid_voice_step`: greedy next-token prediction vs the
  set of legal Othello moves; reports LEGAL_MOVE / LEGAL_PASS /
  ILLEGAL_PASS / ILLEGAL_MOVE / INVALID_TOKEN breakdown.
- `eval/probe_othello.py` — per-cell board-state probe with
  multi-formulation support (occupancy binary, B-vs-W on occupied =
  Nanda formulation, 3-class empty/black/white).
- `model/configs/small_othello.py` — block_size 128, eval_interval 100;
  used for the initial 5k-game run (showed undertraining).
- `model/configs/medium_othello.py` — n_embd=256, n_layer=4, n_head=4,
  dropout=0.2, ~4M params. Sized for the 50k-games corpus; matches
  the published Othello-GPT params:tokens ratio (~1.3). No overfit
  through 5000 iters.

### Corpora and checkpoints

| Corpus | Games | Train tokens | Visits/token | Config | best val_ppl | Overfit? |
|---|---:|---:|---:|---|---:|---|
| `data/othello` | 5,000 | 250 k | 3,673 | small_othello.py (10.7 M) | 18.09 | Yes (after iter 700) |
| `data/othello_50k` | **50,000** | **2.5 M** | **36,729** | medium_othello.py (4 M) | **15.22** | **No** (train/val/gen aligned within 3 % through 5000 iters) |

### Probe results (50k corpus, the headline reproduction)

| Probe formulation | TRAINED | UNTRAINED | Gap | Published target | Match? |
|---|---:|---:|---:|---:|---|
| **3-class MLP** | **91.19 %** | 55.59 % | **+35.60** | ~94 % (Li 2022) | **✓ Within 3 pts** |
| **3-class LINEAR** | **77.15 %** | 52.54 % | **+24.61** | ~75-85 % (Li 2022) | **✓ In range** |
| Occupancy LINEAR | **94.85 %** | 67.61 % | **+27.24** | n/a | ✓ Near-ceiling |
| B-vs-W LINEAR (Nanda) | 69.90 % | 56.23 % | +13.67 | ~98 % (Nanda 2023) | Partial (needs more training) |

Valid-move rate: **82.18 %** (vs published ~95 %+; substantial progress
from the 5k run's 74.6 %).

### Acceptance: met

- 3-class MLP probe within 3 pts of published 94 %: ✅
- 3-class linear probe in published 75-85 % range: ✅
- Trained-vs-untrained gap > 20 pts on at least one formulation: ✅
  (multiple — MLP +35.6, linear +24.6, occupancy +27.2)
- Framework reliably finds learned features when N is satisfied: ✅

### What this validates

The codebase's full pipeline (model + training + activation extraction +
classification probe + multi-seed honest reporting) reliably reproduces
Othello-GPT's central result. The 3-domain comparative story now has
two positive controls (cities, Othello) and one principled null (music),
with Othello specifically reproduced in this codebase to settle the
"is the framework sound?" question.

---

## Pointers

- `pivot.md` — master plan for the pivot; comprehensive milestones, risks,
  confidence summary.
- `update_may24_final.md` — empirical narrative of the cities decomposition
  session.
- `updateMay25.md` — music M2 first-pass.
- `updateMay26.md` — heavy-probe sweep + sym-group methodology calibration
  (with retraction banner for the cities-style-reversal claim).
- `updateMay26_afternoon.md` — music M2 v2 (expanded corpus, smaller
  model, honest reporting).
- `updateMay26_evening.md` — Othello-GPT in-codebase reproduction.
- `STATUS_vs_OTHELLO-GPT.md` — claim-by-claim comparison to the Othello-GPT
  literature; what cities establishes, and now what the Othello in-codebase
  reproduction adds.
- `next_steps.md` — short concrete plan for the experiments that were run
  during the pivot session.
