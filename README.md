# LatentCityGPT

A small GPT is trained on nothing but **sequences of street intersections** —
routes through a real city — using ordinary next-token prediction. It is never
shown a coordinate, an address, a distance, or a direction. The question this
project asks:

> Does an internal representation of the city — a **latent city** — emerge inside
> the model anyway, recoverable from its activations, even though the only thing
> it ever saw was arbitrary integer IDs?

This is the [Othello-GPT](https://arxiv.org/abs/2210.13382) emergent-world-model
paradigm transplanted from a synthetic board onto real geography. The headline
artifact is a probe-decoded overlay: a recognizable city reassembling itself out
of a model fed only anonymous tokens.

## What we expect to find (a ladder, not a single bet)

The name is deliberately robust to partial outcomes. The result is a ladder of
increasingly strong claims, and *any* of the first rungs is already a real finding:

1. **It learns the graph.** The model routes competently — its predicted next
   intersections are real neighbors. (Near-certain; route structure is latent in
   the sequences.)
2. **A map is decodable.** A linear probe recovers each intersection's position
   from the activations — possibly metric, possibly a topologically-faithful but
   distorted "rubber sheet." Either way, the latent city is there.
3. **The map emerged from training.** The trained model beats untrained and
   structure-destroyed controls, and generalizes to held-out nodes — so the
   geometry came from learning, not from the probe.

Stronger rungs (encoded *linearly*; *causally* steers predictions via activation
patching) are bonus tiers. The project is worth doing if only the first three land.

## The one rule

**No coordinate, distance, or direction ever enters the model's input.** Tokens
are arbitrary intersection IDs; coordinates live only in `coords.csv` and are
touched solely by the probe / eval / viz code. That separation is the entire
integrity of the project — if positional information reaches the model's input,
the claim dies.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build a city corpus (CPU, free; needs network for the OSM pull)
python data/prepare_city.py --place "Manhattan, New York, USA" --out_dir data/manhattan

# 2. Train (GPU)
python model/train.py --config model/configs/small.py --data_dir data/manhattan

# 3. Eval, probe, visualize
python eval/valid_edge.py --ckpt checkpoints/best.pt --data_dir data/manhattan --split gen
python eval/probe.py      --ckpt checkpoints/best.pt --data_dir data/manhattan --probe linear
python viz/overlay.py     --ckpt checkpoints/best.pt --data_dir data/manhattan
```

Suggested first city is **Manhattan** (instantly recognizable, ~4-5k nodes, under
the `uint16` ceiling). Pair it with an irregular core like **City of London** for
the "no grid to exploit" credibility result. Always run the graph pull alone first
and check the node count before generating a full corpus.

## Repo

| Path | What |
|---|---|
| `data/prepare_city.py` | OSM street network -> token corpus (done, smoke-tested) |
| `model/` | nanoGPT-style model + training loop |
| `eval/` | valid-edge rate, baselines, probe suite, causal patching |
| `viz/` | Procrustes-aligned recovered-vs-true map overlay (the shareable image) |

## Docs

- **`CLAUDE.md`** — operational guide (conventions, commands, status). Read by Claude Code each session.
- **`CONTEXT.md`** — full scientific framing, the hypothesis ladder, proof strategy, and traps to avoid.
- **`PLAN.md`** — phased build plan with per-phase acceptance criteria.

## Not to be confused with

There is a separate, unrelated **CityGPT** (Feng et al., KDD 2025,
[arXiv:2406.13948](https://arxiv.org/abs/2406.13948)). That work *fine-tunes large
LLMs on text instructions* to improve performance on urban benchmarks — it **gives**
the model spatial knowledge in language and measures task scores. LatentCityGPT is
the inverse: it **withholds** all spatial information and trains a small model from
scratch on bare intersection IDs, then asks whether spatial structure emerges
unsupervised. Different question (emergence vs. capability), different method
(from-scratch vs. fine-tune), different proof (probe + causal intervention vs.
benchmark).

## Status

Data pipeline done; four real-city corpora on disk (City of London, Manhattan,
Boston, South Bay tri-city — see `CLAUDE.md` for the dataset table). Model and
evals in progress — see `PLAN.md`.

## References

- Li et al., *Emergent World Representations* (Othello-GPT) — the probe + control +
  causal-intervention template.
- Perozzi et al., *DeepWalk* / Grover & Leskovec, *node2vec* — why graph structure
  is recoverable from random walks in the first place.
- Karpathy, *nanoGPT* — the model/training scaffold.
