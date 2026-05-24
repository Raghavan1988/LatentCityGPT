# CLAUDE.md

Operational guide for working in this repo. Read `CONTEXT.md` for the full
scientific framing and `PLAN.md` for the phased build plan.

## What this is (one line)

LatentCityGPT trains a small GPT on sequences of street intersections (routes) through
a real city, and proves that an internal, linearly-decodable metric **map** of
the city emerges in its activations — even though the model never sees a single
coordinate. It is the Othello-GPT result transplanted onto real geography.

## THE ONE RULE (non-negotiable)

**No coordinate, lat/lon, distance, direction, or any positional value may ever
enter the model's token stream, vocabulary, or training inputs.** Tokens are
arbitrary integers (intersection IDs). Coordinates live only in `coords.csv` and
are touched only by the probe/eval/viz code, never by the model. If a change
would let positional information reach the model's input, **do not make it** —
the entire claim of the project dies. When in doubt, stop and flag it.

## Repo map

```
data/
  prepare_city.py     # DONE & smoke-tested — OSM -> tokens. See its docstring.
  <city>/             # generated: train.bin val.bin gen.bin meta.pkl coords.csv graph.gpickle
model/
  model.py            # nanoGPT-style GPT (decoder-only, causal)
  train.py            # training loop, reads *.bin + meta.pkl
  configs/            # small/medium model configs
eval/
  valid_edge.py       # valid-edge-rate scorer (the Othello "legal move" analogue)
  baselines.py        # uniform / unigram / 1st+2nd-order Markov / LSTM
  probe.py            # linear & MLP coordinate probes; trained/untrained/destroyed controls
  causal.py           # activation patching ("I'm at node B" intervention)
viz/
  overlay.py          # Procrustes-aligned recovered-vs-true map (the viral image)
checkpoints/          # model weights (gitignored)
```

## Commands

```bash
# data (CPU, free; needs network for the OSM pull)
python data/prepare_city.py --place "Cambridge, Massachusetts, USA" --out_dir data/cambridge

# train (GPU)
python model/train.py --config model/configs/small.py --data_dir data/cambridge

# evals (CPU/GPU)
python eval/valid_edge.py --ckpt checkpoints/best.pt --data_dir data/cambridge --split gen
python eval/baselines.py  --data_dir data/cambridge
python eval/probe.py      --ckpt checkpoints/best.pt --data_dir data/cambridge --probe linear
python eval/causal.py     --ckpt checkpoints/best.pt --data_dir data/cambridge
python viz/overlay.py     --ckpt checkpoints/best.pt --data_dir data/cambridge
```

## Conventions

- **Token reserved indices:** `0=PAD, 1=BOS, 2=EOS`; real nodes start at `3`.
  This is fixed across data, model, and eval — never renumber it.
- **Data format:** `*.bin` are flat token streams (nanoGPT-style), dtype recorded
  in `meta.pkl` (`uint16`, or `uint32` if vocab >= 65536). Routes are concatenated
  `[BOS, ...nodes..., EOS]` and chunked into `block_size` blocks at train time.
- **Three splits, three jobs:** `train.bin` = learn; `val.bin` = in-distribution
  perplexity; `gen.bin` = routes to **held-out destinations** (generalization).
  Never train on `val` or `gen`.
- **Determinism:** every script takes `--seed` (default 0) and must be reproducible.
- **Framework:** PyTorch. Keep the model close to nanoGPT — small, readable, ~10–30M params.
- **Style:** prefer small, testable functions. Each eval script must be runnable
  standalone and print a single clear summary line of its headline metric.

## What "done" looks like for a task

A task is done when it runs end-to-end on the smoke-sized data, prints its
headline metric, and (for eval/probe/causal) respects THE ONE RULE — i.e. it can
be audited to confirm no coordinate entered the model. Add an assertion where
cheap.

## Current status

- [x] `data/prepare_city.py` — written, smoke-tested on a synthetic grid (tokenizer
      bijection, route-edge validity, destination-holdout integrity, binary roundtrip).
- [ ] Everything else — see `PLAN.md`. Next up: `model/` (Phase 1).
