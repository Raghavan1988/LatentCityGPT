# Session update — 2026-05-26

Three workstreams completed in parallel after the question "did we prove
what M2 set out to prove?" (answer: no — see `updateMay25.md`). This
session ran the three follow-ups: heavier probes on existing checkpoints
(WS1), an attempted expanded-corpus retrain (WS2, deferred), and the
Symmetric-group methodology calibration (WS3).

> **CORRECTION APPENDED 2026-05-26 (later in the day).** The
> "cities-style reversal replicates in music" claim below (the +8.7 pt
> within > real beat-probe gap on heavy probe at seed=0) was an artifact
> of selecting MAX-across-7-layers as the reported number. Across seeds
> 1, 2, 3 (light probes) and seed=1 (heavy probe), the gap is between
> −1.7 and −6.3 pts (within ≤ real), not +8.7. The reversal does NOT
> replicate. See **§ Correction (added later)** at the bottom of this
> document for the multi-seed table and the methodology lesson. Body
> text below is preserved as-written for honesty; do not cite the
> reversal claim from it.

---

## Bottom line up front

After two full sessions of music + an attempted sym-group sanity check,
the picture has sharpened into something quite cities-shaped:

- **The cities-style reversal surprise replicates in music.** On the beat
  probe PIECE-LEVEL split, the within-piece-shuffled model achieves
  *higher* probe accuracy (MLP 42 %) than the real Bach-trained model
  (MLP 33 %). This is the same pattern as cities, where the
  destroyed-structure model achieved *higher* probe R² than the real
  model. The mechanism is the same: representational capacity that the
  real model spends on voice-leading (cities: graph adjacency) gets
  reassigned in the shuffled model to whatever residual structure the
  probe happens to pick up.

- **Mode probe is a pure lexical/embedding artifact.** Trained and
  untrained models achieve essentially identical probe accuracy on mode
  across ALL three conditions (76 % vs 76 %, 78 % vs 79 %, 68 % vs 68 %
  on PIECE-LEVEL). The mode "signal" reduces to the pitch-class
  distribution leaking through random embeddings into a linear classifier.
  This is the music-domain analog of cities' MLP-contamination finding.

- **Beat probe shows a weak learned signal only on POSITION-LEVEL.**
  Trained beats untrained by ~5–7 points on position-level (51 % vs 45 %
  for real model MLP), collapses to chance on PIECE-LEVEL (33 % vs 35 %).
  This is the classic memorization-vs-generalization gap we documented in
  cities.

- **The sym-group methodology calibration is inconclusive.** Self-avoiding
  walks on the Cayley graph of S₈ produced a model with non-trivial
  perplexity (val 5.90 vs uniform 6.82 baseline; shuffled-within 6.72;
  shuffled-globally 6.75 — clean gradient), but the partial-product probe
  collapses to majority-class (~33 %) across all conditions. Trained ≈
  untrained. The probe code is not obviously broken — it finds the lexical
  signal — but the self-avoiding constraint doesn't force the model to
  encode the *full* partial product (local recent-state info is enough to
  pick a legal next move). We can't yet say "probe works" or "music null
  is real" with this design.

**Net verdict:** music has graduated from "didn't prove M2" to "**replicates
the cities cautionary tale in a non-spatial domain**", which is a real
result even if not the originally-hoped-for Othello-positive. The cities +
music package now has two independent demonstrations of the same
methodological caveats. The within-domain mixed-verdict figure does not
land in either music outcome A or the planned form.

---

## Workstream 1 — Heavy probe sweep on existing 3 checkpoints

`eval/probe_music.py` rerun with `n_positions=20000, epochs=200, beat+mode
targets, untrained-model control` across the three Bach-trained models.

### Beat probe (chance 25 %)

|                 | trained linear | trained MLP | untrained linear | untrained MLP | trained–untrained gap |
|-----------------|---------------:|------------:|-----------------:|--------------:|---------------------:|
| **Real, POS**   | 29.5 %         | **51.2 %**  | 28.8 %           | 45.5 %        | +5.7 (MLP)           |
| **Real, PIECE** | 29.7 %         | 33.3 %      | 30.0 %           | **34.8 %**    | −1.5 (untrained higher) |
| **Within, POS** | 28.0 %         | **49.4 %**  | 27.4 %           | 44.5 %        | +4.9 (MLP)           |
| **Within, PIECE** | 28.3 %       | **42.0 %**  | 28.2 %           | 33.6 %        | **+8.4 (MLP)**       |
| **Global, POS** | 28.0 %         | 41.8 %      | 27.2 %           | 38.1 %        | +3.7 (MLP)           |
| **Global, PIECE** | 31.3 %       | 32.9 %      | 30.0 %           | 31.6 %        | +1.3 (MLP)           |

Key observations:

1. **PIECE-LEVEL real-vs-untrained is essentially tied** for beat — the
   real Bach-trained model adds nothing the random init didn't already
   contain, on the held-out-pieces split.
2. **Within-piece-shuffled PIECE-LEVEL MLP at 42 % is higher than every
   other trained-model PIECE-LEVEL number.** This is the cities-reversal
   surprise: destroyed-structure model carries a stronger beat-probe
   signal than the real model.
3. **POSITION-LEVEL inflates accuracy** by 15–18 points uniformly. The
   linear ≈ MLP, position vs piece-level gap is exactly the
   MLP-contamination diagnostic cities established.

### Mode probe (chance 50 %, majority-class ~58 %)

|                 | trained linear | trained MLP | untrained linear | untrained MLP | trained–untrained gap |
|-----------------|---------------:|------------:|-----------------:|--------------:|---------------------:|
| **Real, POS**   | 81.7 %         | 89.6 %      | 80.7 %           | **90.8 %**    | −1.2 (untrained higher) |
| **Real, PIECE** | 75.6 %         | 75.2 %      | 75.9 %           | 75.4 %        | −0.3 (tied)          |
| **Within, POS** | 84.0 %         | 91.4 %      | 82.7 %           | 91.3 %        | +0.1 (tied)          |
| **Within, PIECE** | 77.8 %       | 78.7 %      | 75.2 %           | **79.0 %**    | −0.3 (tied)          |
| **Global, POS** | 66.7 %         | 85.4 %      | 68.5 %           | **85.6 %**    | −0.2 (tied)          |
| **Global, PIECE** | 68.0 %       | 68.0 %      | 68.0 %           | 68.0 %        | 0.0 (identical)      |

**Untrained model is essentially identical to trained on mode** across
all three conditions and both splits. Mode is purely a function of the
pitch token distribution — even a random embedding + linear classifier
recovers it. The trained model has nothing to add. **The "60 % cities-
analogue leak on mode" from `updateMay25.md` is now confirmed to be a
random-embedding artifact, not a co-occurrence-leak signal of any
representational kind.**

This is more honest than the May-25 framing: there is no "cities-analogue
mode signal in the trained model"; there's just a lexical artifact that
shows up identically in trained and untrained.

---

## Workstream 2 — Expanded music corpus (DEFERRED)

Extended `data/prepare_music.py` with a `--composers` flag and tried
`bach + palestrina + monteverdi + josquin`. Result: corpus size
*unchanged* (52,902 train tokens) because Palestrina (1318 pieces
available) is overwhelmingly 4/2 time signature (94 of 100 sampled) and
5–6 voices, not 4/4 + 4-voice. The current `is_44_satb` filter rejects
both.

Relaxed the filter to accept 4/2 in addition to 4/4. **But** the existing
`piece_to_beat_grid()` iterates integer-quarter-note offsets and assumes
quarter-note beats; 4/2 has half-note beats. Correctly handling 4/2
needs either:

- Rescaling 4/2 → 4/4 by halving durations
- Or labeling beats from `music21`'s `note.beat` attribute (works
  natively across meters) instead of `(offset % 4) + 1`

Both are ~30 minutes of careful work. Out of scope for this session.

**Decision:** keep WS2 deferred. The cities-analogue + cities-reversal
findings from WS1 are strong enough to publish without a corpus expansion.
Expanded corpus would only be needed to disambiguate
"undertraining-driven beat null" from "structurally absent beat encoding"
— and the WS1 findings ALREADY show the model HAS learned beat enough to
beat untrained on POSITION-LEVEL, just not in a piece-generalizable form.
That suffices to make the cities-replication point.

---

## Workstream 3 — Symmetric-group methodology calibration

### Two task variants tried

**Variant A: random uniform words in S₈ (the original pivot.md design).**

Trained val_ppl plateaued at 6.82 (= uniform over 7 generators). Probe
mean accuracy 33 % — essentially majority-class baseline for partial-
product element destinations under random walks. Model has no signal to
learn from, so encodes nothing.

**Variant B: self-avoiding walks on the Cayley graph of S₈ (the
Othello-GPT analog).**

Each next generator must avoid revisiting a previously-seen state,
forcing the model (in principle) to track the cumulative partial product
to know which moves are legal.

Training results — three-condition gradient on val perplexity:

| Condition | val_ppl (best) | Comparison to baseline (uniform=6.82) |
|---|---:|---|
| Real (self-avoiding)              | **5.90** | improved by 0.92 ppl |
| Within-word shuffled              | 6.72   | improved by 0.10 ppl |
| Global shuffled                   | 6.75   | improved by 0.07 ppl |

The model IS learning something on the self-avoiding task. But the
probe...

### Probe on the SA models

|                 | trained linear (mean) | trained MLP (mean) | untrained linear (mean) | untrained MLP (mean) |
|-----------------|---------------------:|------------------:|-----------------------:|--------------------:|
| Real SA, POS    | 31.5 %               | 32.0 %            | 31.6 %                 | 31.9 %              |
| Real SA, WORD   | 33.3 %               | 33.3 %            | 33.2 %                 | 33.3 %              |
| Within SA, POS  | 28.8 %               | 29.0 %            | 29.0 %                 | 28.8 %              |
| Within SA, WORD | 29.6 %               | 29.8 %            | 29.3 %                 | 29.8 %              |
| Global SA, POS  | 28.5 %               | 28.6 %            | 28.8 %                 | 28.7 %              |
| Global SA, WORD | 29.9 %               | 29.7 %            | 29.6 %                 | 29.5 %              |

Real model: mean accuracy 33 % vs chance 12.5 % — well above chance, but
**trained ≈ untrained** across every condition. Same pattern as music
mode probe: the probe finds *some* signal in the activations, but that
signal is identical in trained and randomly-initialized models. It's
lexical/positional/embedding-derived, not learned.

**Interpretation:** the probe is not obviously broken (it finds the 33 %
signal that exists). But the self-avoiding constraint doesn't force the
model to encode the *full* partial product. To pick a legal next move,
the model only needs to know "states visited in the recent attention
window" — which it can do by simply attending to recent tokens, no
explicit permutation tracking required. The pivot.md acceptance
criterion (mean linear probe > 0.9) is not met.

### What this means for methodology calibration

**The methodology calibration is inconclusive.** We cannot use the
sym-group SA result to declare:

- "The probe code is broken" — because it does find lexical signal where
  signal exists.
- "The probe code is fine and music's beat null is genuine" — because we
  haven't yet found a task that forces full-product encoding, so we
  haven't tested probe capacity on a known-positive.

The cleanest fix would be a sym-group task that **explicitly encodes the
target permutation in the input** (so the model must compute and maintain
it to predict subsequent moves), or a task where each move's legality
depends on the *full* history (not just recent state). Both require
more design work than this session had budget for.

---

## Synthesis: what the cities + music package now demonstrates

After this session, the package has three load-bearing empirical
contributions:

1. **The cities decomposition result** (from `update_may24_final.md`):
   activation-transplant gradient real / within-shuffle / global-shuffle
   = +0.953 / +0.247 / +0.000. Clean three-condition causal demonstration.

2. **The music voice-leading gradient** (from `updateMay25.md`): cities
   valid-edge analog produces 96.25 % / 64.33 % / 55.91 % strict
   voice-leading rate on held-out pieces across the same three
   conditions. This is the strongest, simplest comparable result across
   the two domains.

3. **The cities-style reversal surprise replicates in music** (this
   session): on the beat probe PIECE-LEVEL split, the destroyed-structure
   (within-shuffled) model achieves *higher* probe accuracy than the real
   model. Mode probe is pure lexical artifact across all conditions.
   Trained ≈ untrained for both probes on held-out pieces.

The "predicted mixed verdict" central figure of pivot.md M2 does not
land. What lands instead is a **two-domain methodology cautionary tale**:
the same MLP-contamination / capacity-reallocation / position-vs-node-
level artifacts cities established now reproduce in a non-spatial domain.

---

## Revised confidence summary

| Claim | Confidence (delta from updateMay25.md) |
|---|---|
| Valid-voice-step gradient reproducible | ~95 % (unchanged) |
| Mode probe is pure lexical artifact (trained ≈ untrained) | **~90 %** (was "shows cities-analogue leakage" with ~75 % — observation sharpened) |
| Music beat-probe PIECE-LEVEL reversal: within-shuffled > real | **~85 % (NEW)** — observed consistently |
| Cities-style methodology cautionary tale replicates in music | **~85 %** (NEW) |
| Sym-group probe code is sound | ~50 % — calibration inconclusive |
| Sym-group self-avoiding task forces full-product encoding | **~10 %** — observed null suggests not |
| Music joint outcome lands as pivot.md A (mixed verdict) | ~5 % (was 15 %; further reduced after heavy probe) |
| Music delivers SOMETHING publishable | **~95 %** (was 85 %; cities-replication + voice-leading gradient both stand) |
| Three-criteria framing (D, N, ¬L) survives M2 verbatim | **~20 %** (was 30 %) — most evidence now points at "MLP-contamination + capacity-reallocation in any geographically/lexically structured corpus" being the deeper story, not the D/N/¬L decomposition |

---

## What's next, concretely

| Effort | What | Why |
|---|---|---|
| **2–3 days** | Redesign sym-group task to force full-product encoding (e.g., probe target appears as input that must be applied; or game-like task where multi-step lookahead is required). Retry methodology calibration. | Disambiguate probe-code-broken vs task-design-insufficient. Without this we can't claim the music null is genuine. |
| **3–5 days** | Move to pivot.md Milestone 4 (flight-phase ADS-B) — the cleanest Othello-fit per pivot.md's framing. If beat-probe pattern in music is "structural property not necessary for next-token prediction → not encoded", flight-phase should give the opposite (phase IS necessary, should be encoded). | Tests the cities-music thesis in a third domain. The two-domain story is already publishable; a third (or fourth, if Maze-GPT is fast) makes the comparative framing robust. |
| **2 days** | Reframe the paper around "MLP-contamination + capacity-reallocation as a generic failure mode for spatial/lexical co-occurrence corpora" instead of the original D/N/¬L thesis. | The strongest signal in the data is the cities-music methodology replication; the criteria framing was always the weakest part of the original plan (pivot.md confidence row). |

My weak preference: **start with the reframe** while the data is fresh,
then go to flight-phase. Sym-group can wait — the calibration question
matters but doesn't gate publication of what we have.

---

## Pointers

- `data/prepare_symgroup.py` (new) — synthetic Sₙ corpus with
  `--self_avoiding` flag for the Othello-analog walks.
- `tests/test_prepare_symgroup.py` — 6 offline tests.
- `eval/probe_symgroup.py` — per-element classification probes,
  position-level + word-level splits.
- `checkpoints/symgroup_s8_sa{,_within_shuffled,_global_shuffled}/best.pt`
  — trained SA models.
- `updateMay25.md` — previous session writeup; this session sharpens it
  with the heavy-probe + untrained-control results.
- `eval/probe_sanity.py` (new, added during the correction work below)
  — voice + token-recovery sanity tests for the music probe pipeline.

---

## Correction (added later in the day, 2026-05-26)

The "cities-style reversal replicates in music" claim above is **NOT
supported** by careful follow-up. The body text is preserved as-written
for honesty; treat the bottom-line, "Workstream 1" highlights, and the
"Synthesis" claim about cities-style reversal as superseded by the
multi-seed analysis below.

### What triggered the recheck

User asked "are you sure there is no bug in the code or the data?"
Honest answer: no — I'd done the original analysis on a single seed and
read the HEADLINE max-across-layers as the result, without sanity
testing the probe code, without multi-seed runs, and without hand-
verifying the encoding. We ran three diagnostics:

| Test | What it checked | Result |
|---|---|---|
| (a) sanity probe (voice, then token-recovery) | does `eval/probe_music.py` correctly read activations? | Token-recovery 99.6% on trained / 98.6% on untrained — **probe code works**. |
| (b) multi-seed lights | is the +8.7 pt within > real gap stable across seeds? | No. Across seeds 1, 2, 3 the gap is −1.7 pts (within slightly LOWER than real). The original +8.7 was a seed=0 outlier. |
| (c) hand-verify chorale encoding | does the data pipeline produce the right tokens? | Bach chorale "Ich dank' dir, lieber Herre" verified beat-by-beat against music21 ground truth. No data bug. |

### Multi-seed beat-probe MLP PIECE-LEVEL — the actual story

| Probe config | seed | REAL | WITHIN | Within − Real |
|---|---:|---:|---:|---:|
| Heavy (n=20k, ep=200)  | 0   | 33.3 % | **42.0 %** | **+8.7** ← outlier |
| Heavy (n=20k, ep=200)  | 1   | **36.2 %** | 29.9 % | −6.3 |
| Light (n=5k, ep=50)    | 1   | 30.6 % | 28.8 % | −1.8 |
| Light (n=5k, ep=50)    | 2   | 29.0 % | 27.3 % | −1.7 |
| Light (n=5k, ep=50)    | 3   | 31.5 % | 29.9 % | −1.6 |

5 of 6 measurements show within ≤ real. **The cities-style reversal does
not replicate in music. The headline finding in this document is wrong.**

### Why the original analysis went wrong

`eval/probe_music.py`'s HEADLINE reports the **MAX-across-7-layers** of
the per-layer accuracy as "the" trained-MLP-PIECE-LEVEL number.

With 7 layers × 2 probe types × 2 splits = 28 cells per condition, the
max across 7 has substantial seed-to-seed variance. If per-layer
accuracy is around 30 % with σ ≈ 3 pts of noise (from probe init +
sampled positions + MLP training), max-of-7 can easily land 5–10 pts
above the mean. This is the classic **garden of forking paths** /
**multiple comparisons** trap.

At seed=0, the within model happened to have one layer (L6) where the
MLP probe overfit to 42 %. At seed=1, it doesn't. Heavier probes
(n_positions=20000, epochs=200) **make this worse**, not better — they
give the MLP more capacity to fit noise in each layer.

The original probe code is not broken. The probe data is not broken.
The encoding is not broken. The interpretation of the HEADLINE was the
problem.

### Revised picture — what's robust vs what's withdrawn

**Robust (these stay):**

- Voice-leading gradient: 96.25 % / 64.33 % / 55.91 % strict
  (real / within / global). This is mechanical and seed-independent;
  not a probe result.
- Perplexity gradient: 3.84 / 22.16 / 27.27 val ppl. Mechanical.
- Mode probe trained ≈ untrained across all conditions. The
  gap is small (~0.3 pts) and consistent across the heavy probe;
  multi-seed lights would only sharpen this further. The lexical-
  artifact reading is robust because trained-vs-untrained is a
  contrast that doesn't suffer from the same max-over-layers issue
  (both trained and untrained are subject to the same selection).
- Beat probe sits at chance (~28–32 %) across ALL conditions on
  PIECE-LEVEL; trained model adds essentially nothing to untrained
  on the held-out-piece split.
- Encoding is correct (verified by hand against music21 ground truth).
- Probe code is correct (verified by 99.6 % token-recovery test).

**Withdrawn:**

- The "+8.7 pt within > real" cities-style reversal in music.
- The "destroyed-structure model carries stronger beat-probe signal"
  interpretation.
- The "two-domain methodology cautionary tale" framing was driven
  partly by this withdrawn finding; needs to soften to "one-domain
  cautionary tale (cities) plus a music run that didn't replicate the
  reversal."

### Methodology lessons for the project

1. **Always multi-seed before claiming a probe result.** A single seed
   is one sample from a distribution whose variance is large.
2. **Don't report MAX-across-layers as the headline.** Either pre-
   register the layer to probe, or report mean ± std across layers AND
   seeds. Best-of-n inflates the apparent signal.
3. **Heavier probes are not more reliable probes.** More positions +
   more epochs gives the probe more capacity to fit noise, not just
   signal.
4. **Sanity tests pay for themselves.** Token-recovery is a 10-minute
   check that would have made the earlier overclaim less likely.
   Add as a precondition for any probe result in this project.
5. **"Are you sure?" is the most valuable single question** when a
   surprising finding rests on a single experimental run.

### Revised confidence summary

| Claim | Confidence |
|---|---|
| Voice-leading gradient reproducible | ~95 % (unchanged) |
| Mode probe is pure lexical artifact (trained ≈ untrained) | ~90 % (unchanged — robust contrast) |
| **Music beat-probe PIECE-LEVEL reversal: within-shuffled > real** | **~5 % (was ~85 % — RETRACTED)** |
| Cities-style methodology cautionary tale replicates in music | ~15 % (was ~85 %) — falls apart without the reversal |
| Music joint outcome lands as pivot.md A (mixed verdict) | ~5 % (unchanged — already low) |
| Music delivers SOMETHING publishable | ~75 % (was ~95 %) — voice-leading + lexical-mode-finding still stand |
