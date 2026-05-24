# CONTEXT.md — LatentCityGPT

The full background. `CLAUDE.md` is the operational summary; this file is the
"understand the project deeply before changing anything" document.

## The result we are proving

Train a decoder-only transformer on nothing but **sequences of intersection IDs**
— routes through a real city — using ordinary next-token prediction. The model's
inputs are arbitrary integers with no positional meaning. The claim:

> A linearly-decodable **metric map** of the city emerges inside the model's
> activations. A linear probe recovers each intersection's true (lat, lon) from
> the residual stream with high R² and low median error in meters — despite the
> model never having seen a coordinate. The map emerges from training (an
> untrained net of the same architecture fails the probe), is encoded *linearly*,
> and *causally* drives the model's routing.

This is **Othello-GPT** (Li et al.) transplanted from a synthetic board onto real
geography. Othello-GPT showed an emergent board state recoverable by a probe;
LatentCityGPT shows an emergent *map*. The substrate matters: real geography
provides an objective metric ground truth (latitude/longitude in meters) against
which the recovered representation can be measured directly, and the methodology
sits in an established interpretability lineage.

## Why this substrate

Goals, in priority order: (1) demonstrate deep model comprehension — not API
fine-tuning, but mapping internal structure; (2) produce a result that is
visually inspectable against an objective ground truth (recovered coordinates
overlaid on the real street network); (3) relevance to work on spatial /
world-model reasoning in transformers. Real geography is chosen over a synthetic
maze because it grounds the recovered representation in a measurable metric
space (meters of error). A synthetic-grid version is kept only as a
**control/ablation** (you can dial structure up and down), not as a competing
substrate.

## Data representation

One training example = one **route**: a path through the street graph, e.g.
intersections `42 -> 17 -> 18 -> 91 -> 305`, remapped to contiguous token indices
and wrapped:

```
sequence: [BOS, 45, 20, 21, 94, 308, EOS]
input :   [BOS,  45,  20,  21,  94, 308]
target:   [ 45,  20,  21,  94, 308, EOS]
```

Standard causal LM: every position predicts the next intersection. Many routes
are packed into fixed-length blocks for efficiency, but conceptually each example
is one journey. The corpus blends **length-weighted shortest paths** (realistic,
goal-directed, keeps the model destination-aware over long routes) with **random
walks** (cheap local coverage so quiet streets aren't starved).

The coordinate table is a *separate* file (`coords.csv`):

```
idx,lat,lon
45,40.7128,-74.0060
20,40.7135,-74.0051
```

**This separation is the entire integrity of the project** (see THE ONE RULE in
`CLAUDE.md`). The tokenizer is a trivial bijection `osm_node_id <-> index`, with
`0/1/2` reserved. One node = one atomic token. This is not just the simplest
choice, it's the *required* one: tokenizing coordinate digits or turn-by-turn
directions would inject positional info into the input and collapse the claim.

## How we evaluate (three layers)

1. **Perplexity / cross-entropy** on `val.bin` — basic "is it learning" signal.
2. **Valid-edge rate** — the direct analogue of Othello-GPT's legal-move rate.
   Take the model's greedy next-node prediction; check it's a real neighbor of the
   current node in `graph.gpickle`. Extend to whole **generated routes**: sample
   from a start node and confirm every consecutive pair is a real edge. A model
   with no internal structure predicts globally-popular nodes and scores terribly;
   a model that learned connectivity scores high. This is the progress curve.
3. **Generalization** — valid-edge rate + perplexity on `gen.bin` (routes to
   **held-out destinations**). High here = it generalized over geometry rather
   than memorizing endpoints. Stronger variant: hold out a contiguous **sub-region**
   (geographic split), not just scattered nodes — see PLAN Phase 2.

None of these eval layers requires the coordinate table.

## The proof — and the trap

**The trap:** the instinct is "LatentCityGPT beats baselines at next-node prediction."
That is weak and partly false — a 1st-order Markov chain over the graph basically
*is* the adjacency matrix, so it's genuinely competitive at next-node prediction.
Staking the project on "my transformer predicts next nodes better than n-grams"
wins narrowly, unconvincingly, and misses the point.

**The actual contribution:** LatentCityGPT contains an emergent, linearly-decodable
metric map, and the baselines structurally **cannot**. So run two separate
comparisons with two different purposes:

| Comparison | Baseline | What it establishes |
|---|---|---|
| Is it a competent route model? (sanity) | uniform random; unigram frequency; 1st/2nd-order Markov; same-size LSTM | LatentCityGPT matches/beats on perplexity and **long-range** coherence (Markov forgets the destination; LatentCityGPT stays goal-directed). This only *earns the right* to make the real claim. |
| Does it hold a map? (**the contribution**) | the *same* LatentCityGPT architecture, **untrained / randomly initialized** | A linear probe recovers true coordinates far better from the trained model than the untrained one. The gap **is** the emergence — it came from learning, not from the probe's own capacity. |

Stack three controls to make the contribution airtight:

- **Probe-capacity control** — probe the raw token embeddings / IDs alone.
  They're arbitrary integers, so coordinate R² should be near zero. Proves the
  geometry lives in the deep representation, not the input.
- **Linear-vs-nonlinear** — if a linear probe nearly matches an MLP probe, the
  map is encoded *linearly* (the strong, lab-relevant claim), not merely "in there
  somewhere."
- **Destroyed-structure** — train an identical LatentCityGPT on routes where node
  identities are shuffled so adjacency is meaningless. The probe should now fail.
  Proves the recovered map comes from the real graph's structure, not probe overfit.

**The closer (correlation -> causation):** activation patching. Patch the residual
direction encoding "I'm at node B" while the model is actually at node A, and show
its next-hop distribution swings toward B's neighbors. A model whose predictions
bend when you edit its internal sense of location is unambiguously *using* a world
model.

## Primary metrics & the visual demonstration

Report coordinate **R²**, **median reconstruction error in meters**, and a
**Procrustes-aligned overlay** of recovered vs. true intersection positions. The
overlay is the visual demonstration of the result: a side-by-side rendering of
what the probe extracted from the model's internals against the city's actual
geometry, so a reader can verify the metric correspondence directly.

## The honest one-line claim

> LatentCityGPT is at least as good a route model as strong sequence baselines, and
> uniquely among them it builds an internal metric map of the city that emerged
> from training, is encoded linearly, and causally drives its predictions.

## Budget & infra

- **Data:** $0, CPU-only (OSMnx + networkx).
- **Training:** ~$50–150. A ~10–30M-param nanoGPT trains on a single rented
  A100/H100 (RunPod / Lambda) in well under 48h. Start with a small city
  (Cambridge / one borough) — don't size by vibes.
- **Hosting:** weights on Hugging Face (free); interactive demo on Streamlit /
  Vercel.

## Reference

- Li et al., "Emergent World Representations: ... Othello-GPT" — the template for
  the probe + control + causal-intervention methodology.
- Karpathy, nanoGPT — the model/training scaffold to stay close to.
