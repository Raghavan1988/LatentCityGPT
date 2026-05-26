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
        — this finding is robust across the heavy probe.
      - **Original "cities-style reversal" claim RETRACTED 2026-05-26 (later
        in day).** The +8.7 pt within > real beat-probe gap was from
        heavy probe at seed=0; reading max-across-7-layers as the
        headline inflated apparent signal. Multi-seed verification
        (lights seeds 1/2/3 + heavy seed=1) shows within ≤ real
        consistently. The reversal does NOT replicate. See updateMay26.md
        § Correction for the multi-seed table + methodology lesson.
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
