# CLAUDE.md

Operational guide for working in this repo. Read `CONTEXT.md` for the scientific
framing and `PLAN.md` for the phased build plan. The current project framing
(after the 2026-05-24 pivot) is in `pivot.md`.

## What this is (one line)

A comparative study of where Othello-GPT-style emergent world representations
appear in next-token transformers, with cities as the first studied domain and
a portfolio of additional domains (symmetric-group, music, dialog, flight-phase,
maze) to map the conditions under which the result reproduces.

The cities domain is now a **decomposition** anchor: within the same domain we
have a model that learned the graph (real London, 99.7% valid-edge), a model
that learned only geographic clustering (within-route-shuffled London), and a
model that learned essentially nothing (global-shuffle London). Activation
transplant between residual-stream positions cleanly differentiates them
(P(B's nbrs) lift: +0.953 / +0.247 / +0.000).

## THE ONE RULE (for cities; generalize per-domain)

**Within the cities sub-project, no coordinate, lat/lon, distance, direction,
or any positional value may ever enter the model's token stream, vocabulary,
or training inputs.** Tokens are arbitrary intersection IDs; coordinates live
only in `coords.csv` and are read only by the probe/eval/viz code, never the
model. THE ONE RULE generalizes per-domain to "no probe-target value may
appear in the model's input."

## Repo map

```
data/
  prepare_city.py         # DONE — OSM -> tokens. --place takes >=1 names.
                          # --shuffle_routes (weak destroyed-structure),
                          # --shuffle_globally (strict destroyed-structure).
  <city>/                 # generated: train.bin val.bin gen.bin meta.pkl
                          # coords.csv graph.gpickle
  # Future per-domain pipelines (per pivot.md milestones):
  # prepare_symgroup.py   prepare_music.py   prepare_multiwoz.py
  # prepare_adsb.py       prepare_maze.py
tests/
  test_prepare_city.py    # synthetic-grid smoke test for the cities pipeline
model/
  model.py                # nanoGPT-style decoder-only transformer
  train.py                # training loop, reads *.bin + meta.pkl
  configs/                # small.py (smoke) + medium.py (full-scale)
eval/
  valid_edge.py           # next-step + full-route validity (Othello legal-move analogue)
  baselines.py            # uniform / unigram / 1st+2nd-order Markov + long-range coherence
  probe.py                # linear & MLP probes with position-level AND node-level splits
  causal.py               # PRESERVED — pseudoinverse-direction patching (documented
                          # failure mode; the pivot's finding #3)
  transplant.py           # NEW — clean Phase 5: substitute real a_B for a_A at layer L
  embedding_compare.py    # NEW — wte vs node2vec (Procrustes/CKA + probe parity)
viz/
  overlay.py              # Procrustes-aligned recovered-vs-true map (not yet written)
checkpoints/              # model weights (gitignored)
  best.pt                          # real London
  manhattan/                       # Manhattan
  boston/                          # Boston
  london_shuffled/                 # within-route-shuffled (weak destroyed-structure)
  london_global_shuffled/          # global-shuffled (strict destroyed-structure)
```

## Commands

```bash
# data (CPU, free; needs network for the OSM pull). python -u for live progress.
python -u data/prepare_city.py --place "Manhattan, New York, USA" --out_dir data/manhattan

# destroyed-structure variants (no graph adjacency, varying levels of clustering)
python -u data/prepare_city.py --place "City of London, ..." --shuffle_routes \
    --out_dir data/london_shuffled
python -u data/prepare_city.py --place "City of London, ..." --shuffle_globally \
    --out_dir data/london_global_shuffled

# train (MPS / CUDA / CPU)
python model/train.py --config model/configs/small.py --data_dir data/london_city

# evals on a checkpoint
python eval/valid_edge.py       --ckpt checkpoints/best.pt --data_dir data/london_city
python eval/baselines.py        --data_dir data/london_city --ckpt checkpoints/best.pt --coherence
python eval/probe.py            --ckpt checkpoints/best.pt --data_dir data/london_city
python eval/transplant.py       --ckpt checkpoints/best.pt --data_dir data/london_city  # clean Phase 5
python eval/embedding_compare.py --ckpt checkpoints/best.pt --data_dir data/london_city
```

## Conventions

- **Token reserved indices:** `0=PAD, 1=BOS, 2=EOS`; real tokens start at `3`.
  Fixed across data, model, and eval. Mirrors per-domain.
- **Data format:** `*.bin` are flat token streams (nanoGPT-style), dtype recorded
  in `meta.pkl` (`uint16` or `uint32`). Sequences concatenated as
  `[BOS, ..., EOS]` and chunked into `block_size` blocks at train time.
- **Three splits, three jobs:** `train.bin` (learn), `val.bin` (in-distribution
  perplexity), `gen.bin` (held-out generalization split — domain-specific).
- **Probe splits:** every probe is run with BOTH position-level and node-level
  splits. Position-level can be passed by lookup memorization; node-level is
  the probe-capacity-controlled test (the pivot's finding #2).
- **Destroyed-structure controls (two-tier per domain):** a *weak* control that
  shuffles within-sequence (preserves set-membership) and a *strict* control
  that shuffles globally across the corpus (breaks set-membership too). Cities
  uses `--shuffle_routes` and `--shuffle_globally`; new domains follow the
  same template.
- **Determinism:** every script takes `--seed` (default 0), reproducible.
- **Framework:** PyTorch. Keep the model nanoGPT-shaped, ~10–30M params.
- **Style:** small, testable functions; each eval script prints a single
  clear summary line of its primary metric.

## Dataset-sizing heuristics (cities — adapt per-domain)

- **Visits per token** during training: aim for ≥ 200; below ~100 the model
  cannot reliably learn each token's representation. Compute as
  `train_real_node_tokens / vocab_size`.
- **Block size vs sequence length**: cities median route length scales with
  city diameter — ~25 (City of London), ~50 (Manhattan), ~65 (Boston), ~90
  (South Bay). `block_size = 256` covers ≤~5k-vocab cities; `512` for sprawl.

## What "done" looks like for a task

Runs end-to-end on smoke-sized data, prints its primary metric, respects THE
ONE RULE (for cities) or its per-domain equivalent (probe targets cannot
appear in model input). Add an assertion at the data-pipeline boundary where
cheap.

## Current status

- [x] **Cities domain — DONE with decomposition result.** Three trained models
      on London (real / within-route-shuffled / global-shuffled), each with
      probe + transplant evaluation. See `update_may24_final.md` for the full
      numbers and `STATUS_vs_OTHELLO-GPT.md` for the comparison to the
      Othello-GPT lineage.
- [x] **Phase 5 (causal intervention)** implemented in `eval/transplant.py`
      (not the original `eval/causal.py` — see pivot.md finding #3).
- [x] **Methodology assets**: `eval/probe.py` with node-level split;
      `eval/embedding_compare.py` wte-vs-node2vec; `--shuffle_globally` in
      `data/prepare_city.py`.
- [x] Four real-city corpora built and on disk (smoke-sized):

      | City                 |  Nodes |  Vocab | Train tokens | Visits/node |
      |---                   |   ---: |   ---: |         ---: |        ---: |
      | City of London       |    663 |    666 |        1.13M |      ~1,600 |
      | Manhattan            |  4,543 |  4,546 |        2.74M |        ~590 |
      | Boston, MA           | 11,368 | 11,371 |        3.03M |        ~260 |
      | South Bay (MV+SV+SC) | 45,696 | 45,699 |        3.98M |         ~87 |

- [~] **Multi-domain expansion — Milestone 2 (music) + Milestone 1 (sym-group)
      methodology calibration DONE (2026-05-25 → 2026-05-26).**
      See `updateMay25.md` for the M2 first-pass writeup and `updateMay26.md`
      for the heavy-probe + sym-group session that followed.
      - **Mode probe is purely lexical (trained ≈ untrained in all conditions)**
        — robust across both Bach-only and expanded-corpus multi-seed runs.
      - **Original "cities-style reversal" claim RETRACTED 2026-05-26.** See
        `updateMay26.md` § Correction. The seed=0 anomaly was max-across-7-
        layers inflation; multi-seed verification (lights seeds 1/2/3 +
        heavy seed=1 + 4-seed honest probe on the expanded model) all show
        within ≈ real on PIECE-LEVEL beat (chance 25%, both at ~26%).
      - **M2 v2 with expanded corpus + smaller model (`updateMay26_afternoon.md`):**
        - Corpus 6.8× larger (358k train tokens; bach + palestrina +
          monteverdi + josquin); smaller model (1.4M params); no overfit.
        - Voice-leading gradient: 98.99% / 25.93% / 58.77% strict (real /
          within / global); cleanest comparable result.
        - Honest 4-seed probe: beat at chance (~27%) in ALL conditions
          with trained ≈ untrained. Mode at lexical baseline (~70% on
          real/within with pitch set preserved, ~54% on global with set
          destroyed). No classification probe lands as Othello-positive.
        - Diagnosis: N criterion (state necessary for next-token
          prediction) fails for beat/mode/chord in music — voice-leading
          is locally predictable, model doesn't need to encode global
          structural features.
- [x] **Cross-domain transplant gradient + corrected interpretation
      (2026-05-26 night, `updateMay26_night.md`).** Triggered by user's
      diagnostic worry "how confident are we cities ≠ Othello?". Ran:
      cities transplant on Manhattan + Boston (both ~+0.9, similar to
      London +0.953); music transplant new (`eval/transplant_music.py`,
      voice-leading state) → real +0.804, within-shuffled +0.071,
      global-shuffled −0.010 — clean 3-condition gradient mirroring
      cities. **The corrected interpretation: cities + music are
      token-local domains (transplant ~0.9); Othello is prefix-derived
      (transplant ~0.1-0.2). All three pass the causal residual-
      encoding test qualitatively.** Music's null on beat/mode/chord
      probes is now causally demonstrated as principled N-criterion
      failure: voice-leading IS encoded (transplant +0.804), beat/mode/
      chord are NOT encoded (probes trained ≈ untrained). Same model,
      same framework, different feature targets, opposite outcomes —
      cleanly explained by the next-pitch objective's requirements.
- [x] **Late-night diagnostic additions (2026-05-26).**
      Two more experiments in `updateMay26_night.md` § "Late-night
      addendum":
      (a) `eval/probe_cities_grid.py` — cities grid-classification
      probe (10×10 spatial grid, 100 cells). Resolves the MLP-
      contamination caveat from `STATUS_vs_OTHELLO-GPT.md` via node-
      level held-out tokens. London 66/62 % / Manhattan 57/61 % /
      Boston 55/65 % (lin/MLP); untrained 8-11 % across all. Linear ≈
      MLP within 5-10 pts → cities encoding IS linear (Nanda 2023's
      strong claim, now cleanly testable in cities).
      (b) `eval/transplant_music_beat.py` — music beat-controlled
      transplant. Matched-RSVP donors with different beat-in-measure
      produce LESS prediction shift than random control on every metric
      (max\|Δp\|, KL, argmax-changed). Beat is not just unreadable
      correlationally; it's causally inert.
      **Cross-domain claim now defensible at full strength:** cities
      and Othello both pass all 3 Li/Nanda claims; differ in encoding
      mechanism (token-local vs prefix-derived) and probe methodology
      (grid vs per-cell) but qualify as mechanistically interpretable
      in the same shape. Music: principled N-criterion failure on
      abstract features; positive on RSVP.
- [x] **Flight-phase Milestone 4 LANDED (2026-05-27, `updateMay27.md`).**
      Applied positive control: real ADS-B from `traffic` library
      (238 quickstart flights, 5s downsampled, 46k train tokens).
      `data/prepare_adsb.py` + `model/configs/tiny_flight.py` (0.27M
      params, 6:1 ratio, no overfit) + `eval/valid_flight_step.py` +
      `eval/probe_flight.py` + `eval/transplant_flight.py`.
      Results across 3 conditions (real / within / global):
      val_ppl 1.60 / 14.31 / 34.20; valid-physics rate 94 / 40 / 17 %;
      phase probe FLIGHT-LEVEL linear trained-untrained gap +20 / +12
      / +14 pts; transplant P(B-phase) gain trp−rnd +0.460 / +0.306 /
      +0.000. **Clean monotonic 3-condition gradient on all 4 metrics.**
      Flight occupies the middle of the encoding-locality spectrum
      (between token-local cities/music and prefix-derived Othello).
      pivot.md M4 milestone complete.
- [x] **Othello-GPT reproduced from scratch in this codebase
      (2026-05-26 evening, `updateMay26_evening.md`).** End-to-end
      validation that the framework finds learned features when N is
      satisfied:
      - 50k random uniform games → `data/othello_50k` (2.5M train
        tokens) → medium_othello.py (4M params, no overfit).
      - val_ppl 15.22, valid-move rate 82.2% (vs published ~95%+;
        approaching).
      - **3-class MLP probe: 91.19% per-cell mean (vs published
        ~94%; within 3 pts)**. Trained−untrained gap +35.6 pts.
      - **3-class LINEAR probe: 77.15%** (vs published 75-85%; in
        range).
      - Therefore: music null (probe at chance / lexical) is principled
        N-criterion failure, not a framework bug.
      - 3-domain comparative story now: cities + Othello (positive
        controls, probe works) + music (N fails, probe doesn't work).
      - **Sym-group methodology calibration inconclusive**: self-avoiding-walk
        task improved val_ppl (5.90 vs 6.82 uniform) but probe collapsed
        to lexical-only signal. Can't yet distinguish probe-code-broken vs
        task-design-insufficient.
      - **Voice-leading gradient (eval/valid_voice_step.py) is the
        cleanest comparable result** (96.25% / 64.33% / 55.91%
        real/within/global), structurally analogous to cities' valid-edge.
      - **Pipeline built**: `data/prepare_music.py` + `eval/probe_music.py`
        + `eval/valid_voice_step.py` (music-domain valid-edge analogue).
      - **Three corpora**: `data/music_bach{,_within_shuffled,_global_shuffled}/`
        — 313 chorales after 4/4+SATB filter, vocab=60, 52,902 train tokens
        each. Three trained checkpoints saved to `checkpoints/music_bach{,...}/best.pt`.
      - **Results**: voice-leading gradient clean (96.25% → 64.33% → 55.91%
        strict; cities valid-edge analogue lands); perplexity gradient clean
        (3.84 → 22.16 → 27.27); mode probe shows cities-analogue leakage
        (60/60/55 PIECE-LEVEL); **beat probe inconclusive** — all conditions
        sit at chance ~26%. Joint outcome doesn't fit any of pivot.md's
        A–D cleanly.
      - **Open**: resolve beat-probe null via heavier probe sweep (~30 min)
        and/or retrain with better-regularized config (~1 day); then revisit
        framing. Symmetric-group methodology calibration (Milestone 1) is
        now more attractive as an independent sanity check on the probe code.
