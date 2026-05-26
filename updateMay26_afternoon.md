# Session update — 2026-05-26 afternoon

Continuation of the morning's retraction session. Triggered by
"are you sure there is no bug in the code or the data?" → led to multi-
seed verification that retracted the cities-style reversal claim
(commit `c6cb747`). After the retraction, the question became "did
we prove what M2 set out to prove?" — answer: no — so the user
asked for two follow-ups in parallel:

1. **Methodology consolidation**: add multi-seed + mean-across-layers
   reporting to the probe so any future M2 number is reported
   honestly.
2. **Beat-null disambiguation**: fix 4/2 time-signature handling in
   `prepare_music.py`, rebuild an expanded music corpus
   (bach + palestrina + monteverdi + josquin), retrain with a
   smaller / better-regularized config, and rerun the probe under
   honest reporting.

Both done. This is the clean publishable M2 picture.

---

## Bottom line up front

The within-domain mixed-verdict figure pivot.md M2 envisioned does
**not** materialize. What lands instead is a **publishable
methodology paper** plus a **clean structural-metric gradient**:

- **Voice-leading rate (the music valid-edge analog) produces a clean
  three-condition gradient on the expanded corpus**: 98.99 % / 25.93 %
  / 58.77 % strict (≤ 7 semitones) for real / within-piece-shuffled /
  global-shuffled. The real model genuinely learned chorale-style
  voice leading; both destroyed-structure variants collapse.
- **Perplexity gradient is also clean**: 4.37 / 16.94 / 18.59 val ppl.
  Monotonic, train/val/gen aligned within 3 %, **no overfit**.
- **Classification probes (beat / mode / chord) on PIECE-LEVEL splits
  show trained ≈ untrained across all three conditions.** Under honest
  multi-seed (4 seeds × 7 layers) mean-across-layers reporting:
  - Beat probe sits at chance (~25 – 27 %) everywhere; trained model
    adds < 1 pt over random init. **Not an Othello-positive.**
  - Mode probe is at lexical baseline (~70 % on real / within where
    pitch set is preserved; ~54 % on global where it isn't). **Not a
    learned representation** — random embeddings recover it equally well.
- **Cities-style reversal (within > real on beat probe) DOES NOT
  EXIST.** Under multi-seed honest reporting, within is 25.4 % and
  real is 27.1 % on PIECE-LEVEL beat — statistically tied at chance.
  The +8.7 pt gap from this morning's heavy-probe-seed-0 was a
  max-of-7-layers fluke; commit `c6cb747` retracted it.

The paper now claims: **classification probes find what the model is
REQUIRED to encode for its training objective.** Music's next-pitch
objective doesn't require beat / mode / chord encoding — voice-leading
is locally predictable from same-voice context within ~16 tokens — so
classification probes find nothing. Generative-behavior metrics
(voice-leading rate) DO capture the same structural integrity. The
diagnostic distinction between the two is the methodological
contribution.

---

## What was done this session

### Code changes

- **`eval/probe_music.py`** — added `--seeds N1 N2 N3 …` (default
  `[0]`) and `--report_mode max|mean|both` (default `both`). For each
  layer, the probe is trained under each seed (re-initializing the
  classifier and re-shuffling the train/test split). The HEADLINE now
  shows:
  - `mean-across-layers&seeds: μ ± σ (n=L×S)` — the honest number.
  - `max-layer&seed: max-acc (at L_best)` — the inflated single-seed
    headline number.
  - `inflation (max − mean)` — flag for whether your finding is
    sensitive to best-of-N selection. Treat > 5 pts as a red flag.
- **`data/prepare_music.py`** — accepts 4/2 time signatures (was
  rejecting ~95 % of the Palestrina corpus). New helper
  `_beat_step_quarters` returns 2.0 for 4/2 and 1.0 for 4/4;
  `piece_to_beat_grid` iterates at the appropriate beat step.
  Encoding remains meter-agnostic — one token-tuple per beat.
- **`model/configs/small_music.py`** — smaller config sized for the
  music corpus (1.4 M params, n_embd=192, n_layer=3, dropout=0.3,
  max_iters=1000). Designed to not overfit on ≤ 200 k token corpora.
  The original `small.py` (10.7 M params) catastrophically overfit
  on Bach: val_ppl climbed from 3.84 (iter 500) to 280 (iter 1000).
- **`eval/probe_sanity.py`** (new in earlier commit) — voice +
  token-recovery sanity tests. Token recovery achieves 99.6 % on
  trained embed layer / 98.6 % on untrained — proves the probe
  pipeline correctly reads activations and trains classifiers.

### Corpora

| Corpus | Train tokens | Visits/token | val_ppl (best) |
|---|---:|---:|---:|
| `data/music_expanded` (real) | **358,246** | **5,778** | **4.37** |
| `data/music_expanded_within_shuffled` | 358,246 | 5,778 | 16.94 |
| `data/music_expanded_global_shuffled` | 358,246 | 5,778 | 18.59 |

763 pieces survive `4/4 OR 4/2` + SATB filter (vs 313 from Bach
alone). Vocab = 62 pitches + 4 reserved.

### Checkpoints

- `checkpoints/music_expanded/best.pt` — val_ppl 4.37 at iter 900.
  Train CE 1.46, val CE 1.47, gen CE 1.49 — aligned, no overfit.
- `checkpoints/music_expanded_within_shuffled/best.pt` — val_ppl 16.94.
- `checkpoints/music_expanded_global_shuffled/best.pt` — val_ppl 18.59.

---

## Results

### 1. Perplexity gradient (clean, monotonic, no overfit)

| Iter | Real (train/val/gen) | Within (train/val/gen) | Global (train/val/gen) |
|---|---|---|---|
| 100 | 21.6/22.2/22.6 | 18.7/18.8/18.6 | 25.0/25.3/25.6 |
| 500 | 5.11/5.18/5.25 | 17.07/17.21/16.87 | 19.25/19.37/19.98 |
| 900 | 4.30/4.37/4.45 | 16.75/16.94/16.55 | 18.46/18.59/19.13 |

The real model genuinely learns music structure (val_ppl 4.37 vs
uniform-over-62-vocab = 62, i.e., ~14× better than uniform).
Both shuffled models plateau near uniform-over-corpus-distribution.

### 2. Voice-leading gradient (the load-bearing structural metric)

`eval/valid_voice_step.py --split gen --strict_band 7 --loose_band 12`:

| Condition | Strict ≤7st | Loose ≤12st | Real-pitch pred | Median \|interval\| |
|---|---:|---:|---:|---:|
| Real expanded | **98.99 %** | 99.49 % | 99.51 % | 0 st |
| Within-shuffled | 25.93 % | 35.63 % | 41.36 % | 7 st |
| Global-shuffled | 58.77 % | 83.10 % | 99.04 % | 7 st |

The real-model voice-leading rate is **higher than the original
Bach-only model** (96.25 % strict). Median interval 0 reflects
Palestrina's slower harmonic rhythm (notes often held across beats).

Per-voice on real model: S 99.6 % / A 99.6 % / T 99.07 % / B 97.48 %
(Bass slightly looser, as expected — bass lines leap more).

The within-shuffled real-pitch-prediction rate (41 %) is unusually
low compared to the original Bach within-shuffled run (99.9 %). Cause:
Palestrina has more REST tokens (renaissance polyphony has frequent
voice entries / exits); the shuffled model often predicts REST as a
"safe" default. **Conditional on real-pitch predictions**, within
achieves 63 % strict and global achieves 60 % — comparable, both well
above chance for random pitch prediction (~30 %).

### 3. Honest multi-seed classification probe (the central result)

`eval/probe_music.py --seeds 0 1 2 3 --report_mode both --targets
beat mode --n_positions 10000 --epochs 80`:

**Beat probe, PIECE-LEVEL** (held-out pieces; chance = 25 %):

| Condition | Trained linear | Trained MLP | Untrained linear | Untrained MLP | T−U gap |
|---|---|---|---|---|---|
| Real          | 27.1 ± 0.7 % | 26.8 ± 0.3 % | 26.6 ± 0.5 % | 26.7 ± 0.3 % | < 1 pt |
| Within-shuf   | 25.3 ± 0.5 % | 25.4 ± 0.5 % | 25.0 ± 0.4 % | 25.6 ± 0.6 % | < 1 pt |
| Global-shuf   | 26.1 ± 0.8 % | 26.5 ± 0.3 % | 26.1 ± 0.6 % | 26.1 ± 0.3 % | 0 pt |

**All conditions at chance. Trained ≈ untrained throughout.** There is
no learned beat representation in any of the three models.

**Mode probe, PIECE-LEVEL** (chance 50 %, majority-class ~58 %):

| Condition | Trained linear | Trained MLP | Untrained linear | Untrained MLP | T−U gap |
|---|---|---|---|---|---|
| Real          | 64.6 ± 0.7 % | 70.3 ± 1.4 % | 64.9 ± 0.5 % | 69.4 ± 0.4 % | < 1 pt |
| Within-shuf   | 69.0 ± 4.3 % | 77.2 ± 6.2 % | 69.2 ± 3.7 % | 75.8 ± 4.7 % | < 2 pt |
| Global-shuf   | 53.5 ± 0.5 % | 54.0 ± 1.2 % | 52.9 ± 0.2 % | 53.6 ± 1.0 % | < 1 pt |

**Mode is a pure lexical artifact in all three conditions.** Trained
and untrained models recover mode equally well. Real and
within-shuffled both at ~70 % (per-piece pitch SET preserved → mode
linearly recoverable). Global-shuffle destroys per-piece pitch
distribution → drops to ~54 %.

### 4. Inflation diagnostic (the methodology cautionary tale)

For each (trained, probe-type, split) cell, we report
`max − mean` across 7 layers × 4 seeds. This is the inflation
single-seed max-of-layers reporting introduces.

| Cell | max − mean (pts) | Severity |
|---|---:|---|
| Beat PIECE-LEVEL trained linear (real) | +1.5 | small |
| Beat PIECE-LEVEL trained MLP (real) | +0.5 | small |
| Mode PIECE-LEVEL trained MLP (within) | **+6.7** | substantial |
| Mode POSITION-LEVEL trained MLP (within) | **+6.7** | substantial |
| Mode PIECE-LEVEL trained linear (within) | +5.6 | substantial |

**The inflation is largest exactly where the (now-retracted)
cities-style reversal claim originated**: within-shuffled mode/beat
MLP. The +8.7 pt seed=0 reversal anomaly was within the inflation
envelope of normal noise.

### 5. POSITION-LEVEL vs PIECE-LEVEL gap (memorization artifact)

| Probe | POSITION-LEVEL (mean) | PIECE-LEVEL (mean) | Gap |
|---|---:|---:|---:|
| Beat real, trained MLP | 26.6 % | 26.8 % | 0 pt |
| Beat within, trained MLP | 25.6 % | 25.4 % | 0 pt |
| Mode real, trained MLP | **83.9 %** | 70.3 % | **+13.6** |
| Mode within, trained MLP | **84.4 %** | 77.2 % | **+7.2** |
| Mode global, trained MLP | **70.0 %** | 54.0 % | **+16.0** |

Position-level inflates accuracy 7-16 pts via memorization. The
piece-level split (the music-domain analog of cities' node-level
split) is the only honest test.

---

## Why music's classification probes failed — the diagnosis

The original M2 outcome A predicted beat would be Othello-positive
(high on real, collapses on shuffle). The actual result is beat-at-
chance everywhere. Seven principled reasons, ranked by load-bearing-ness:

1. **Next-pitch prediction doesn't require beat / mode / chord
   encoding.** The model needs: current voice (free from positional
   embedding) + recent same-voice pitch (in attention) + other 3
   voices at the current beat. It does NOT need: explicit beat
   position, explicit mode, explicit Roman numeral. **The N criterion
   of pivot.md (state necessary for next-token prediction) fails
   here.** When N fails, ¬L doesn't matter — there's nothing to
   probe.

2. **Bach voice-leading is local.** ~16 tokens of context suffices.
   No global state needed (unlike Othello's full board state).

3. **4-voice fixed-order tokenization gives voice away for free.**
   Position-mod-4 = voice. The model never has to LEARN voice — it's
   in the positional embedding. Forcing the model to derive voice
   from content would require a tokenization that leaks the probe
   target.

4. **Mode is linearly recoverable from pitch distribution.** Minor
   keys use more flats; major keys use more sharps. A linear
   classifier on random embeddings can read this out (which is what
   untrained MLP at 69 % demonstrates). No learning needed.

5. **Within-piece shuffle leaves too much signal**. The shuffle
   destroys voice-leading but preserves per-piece pitch sets. The
   model still has something to fit (the per-piece marginal
   distribution), and learns that. But what it learns doesn't help
   the beat probe.

6. **The cities-style reversal was domain-specific to set-membership
   co-occurrence with the probe target.** In cities, within-route
   shuffle preserved geographic clustering (the probe target) within
   each route set. In music, the within-piece-preserved pitch set
   doesn't correlate with beat (the would-be probe target). No
   reversal channel.

7. **The pivot.md outcome matrix had an unstated assumption** that
   beat behaves like Othello-GPT's board state. It doesn't.

**Conclusion:** music gives us a **clean N-fails-but-D-and-¬L-hold
negative case**, not the within-domain mixed verdict we wanted. The
N criterion is more load-bearing than pivot.md originally suggested.

---

## What this means for the paper

The reframed paper claim:

> **Classification probes find what a model is REQUIRED to encode for
> its training objective. Generative-behavior metrics find what the
> model has learned to produce, whether or not it's encoded as a
> separable feature.** We demonstrate this distinction across two
> domains:
>
> - **Cities** (graph routing): adjacency IS required for valid-edge
>   prediction. Classification probes (linear ≈ MLP) find a
>   geographic representation; activation-transplant intervention
>   produces the expected three-condition causal gradient. *N
>   satisfied.*
> - **Music** (next-pitch prediction): beat / mode / chord are NOT
>   required for voice-leading prediction. Classification probes
>   find nothing. Voice-leading rate (the generative-behavior metric)
>   produces the clean three-condition gradient that classification
>   probes were supposed to. *N fails.*
>
> We additionally show that **single-seed best-of-N-layers probe
> reporting (a common practice in interpretability) systematically
> inflates apparent probe signal**, by 5-7 points in the regime we
> tested. The fix is mean-across-layers AND across-seeds reporting
> with max-vs-mean inflation as a diagnostic. Our original analysis
> exhibited this failure mode: a cities-style "destroyed-structure
> probe ≥ real probe" reversal observed at seed = 0 did not replicate
> under multi-seed reporting and has been retracted.

This is a **workshop paper**: cities + music gives a structural-metric
replication, a methodology contribution (honest probe reporting), AND
a diagnostic distinction (classification probes need N to hold;
generative-behavior metrics don't).

---

## Confidence summary (final, after this session)

| Claim | Confidence |
|---|---|
| Voice-leading three-condition gradient reproducible | ~97 % |
| Perplexity three-condition gradient reproducible | ~97 % |
| Mode probe trained ≈ untrained in ALL conditions | **~95 %** (multi-seed-confirmed; was ~90 % before) |
| Beat probe at chance in ALL conditions | **~95 %** (multi-seed-confirmed; was ~85 % before) |
| Cities-style "shuffled > real" reversal in music | **~3 %** (RETRACTED; was 85 % yesterday morning, 25 % yesterday evening, ~5 % after first multi-seed, now nailed at ~3 %) |
| The N-criterion-fails diagnosis for music | ~80 % |
| The single-seed-max-of-layers inflation finding generalizes | ~85 % |
| M2 is publishable as a workshop paper | **~85 %** (was ~75 % yesterday) |
| The original pivot.md M2 outcome A (mixed verdict) lands | **< 5 %** (it didn't) |

---

## What's next

1. **Update CLAUDE.md / PLAN.md / pivot.md** to reflect the final M2
   state. (Will do as part of this commit.)
2. **Sym-group methodology calibration: still inconclusive.** With
   the corrected music story, the sym-group question shifts from
   "is the probe code broken?" (answered no — token recovery is
   99.6 %) to "design a clean known-positive synthetic task that
   forces full-product encoding." Lower priority now.
3. **Next domain: pivot.md Milestone 4 — flight-phase.** This is
   the natural test of "N is satisfied here so classification probes
   should work." If flight-phase classification probes for phase ARE
   recoverable above the lexical baseline, the comparative thesis is
   strong (cities ✓, flight-phase ✓, music ✗ for principled
   structural reasons).
4. **Methodology paper polish.** The probe-reporting protocol could
   become a standalone contribution if we package it cleanly as
   `eval/probe_music.py`'s `--report_mode both` + `--seeds` pattern
   for re-use across domains. ~1 day of work to lift this into a
   shared `probekit.py`.

---

## Pointers

- `eval/probe_music.py` — multi-seed + honest reporting (added this
  session).
- `eval/probe_sanity.py` — voice + token-recovery sanity tests.
- `eval/valid_voice_step.py` — voice-leading rate evaluator (the
  music valid-edge analog).
- `data/prepare_music.py` — multi-composer + 4/2 time-sig support.
- `model/configs/small_music.py` — sized to fit without overfit on
  music corpora.
- `data/music_expanded{,_within_shuffled,_global_shuffled}/` — three
  corpora at 358 k train tokens each.
- `checkpoints/music_expanded{,_within_shuffled,_global_shuffled}/`
  — trained, no overfit.
- `updateMay25.md`, `updateMay26.md` — earlier session writeups; both
  have findings that this session sharpens or retracts. See the
  retraction banner at the top of `updateMay26.md`.
