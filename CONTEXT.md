# CONTEXT.md — scientific framing after the 2026-05-24 pivot

`CLAUDE.md` is the operational summary; this file is "understand the project
deeply before changing anything." Read alongside `pivot.md` (master plan for
the pivot), `update_may24_final.md` (empirical narrative of the cities
decomposition), and `STATUS_vs_OTHELLO-GPT.md` (claim-by-claim comparison to
the literature).

---

## The question this project asks (post-pivot)

**Where do Othello-GPT-style emergent world representations actually appear in
next-token transformers, and why?**

Othello-GPT (Li et al. 2022) showed that a transformer trained on Othello move
sequences encodes the board state in its residual stream; Nanda 2023 sharpened
the claim to linearly-decodable + causally-used. The lineage spawned a series
of replications (Chess-GPT / Karvonen 2024 and others) without a systematic
account of *when the result should and should not be expected*. This project
attempts that account, with cities as the first studied domain.

---

## What the cities domain established

The single-domain framing the project started with — "an emergent metric map
of a real city appears in a next-token transformer trained on routes" — did
*not* survive contact with the destroyed-structure control. What survived,
and what was empirically established, is a richer decomposition:

### The three-condition causal gradient (cities, City of London)

Three trained models, same architecture, differing only in training data:

| Condition                | val ppl | P(A nbrs) | Probe R² (node-level lin.) | Transplant lift |
|---|---:|---:|---:|---:|
| Real London              | 1.65    | 0.984     | 0.64                       | **+0.953**      |
| Within-route shuffled    | 25.0    | 0.061     | 0.96                       | +0.247          |
| Global shuffled          | 313     | 0.006     | −0.02                      | +0.000 (chance) |

Reading the table: training data with full geographic + ordered structure
(real) produces both a strong probe signal and a strong causal effect of
substituting `a_B` for `a_A`. Destroying only the *order* (within-route
shuffle) preserves probe signal (in fact strengthens it) but reduces the
causal effect ~4×. Destroying set-membership too (global shuffle) collapses
both signals to the relevant null.

### Two distinct phenomena, both causally encoded, with different sources

The differential decomposes the model's causal use of its residual stream
into two contributions:

- **Geographic clustering** (~+0.247 of the transplant lift). Emerges from any
  training signal with geographic co-occurrence — including a permutation-
  invariant bag-of-tokens objective. Sequence order is *not* required.
- **Graph-adjacency-using behavior** (~+0.706 of the lift). Requires sequence-
  order training. Destroyed by within-route shuffle. The part that makes the
  trained model a competent route model.

Both are causally read out by activation transplantation. The clean
methodological observation is that the standard probe-based test for
"emergent world model" can confound these — a probe trained on
geographically-clustered embeddings will succeed even on a model that has no
graph-adjacency knowledge.

---

## The reframed thesis (working hypothesis)

> Emergent world representations in next-token transformers depend on three
> structural properties of the training distribution:
> - **(D) Discrete state** — so probes cannot lookup-memorize via
>   continuous-target shortcuts.
> - **(N) State necessary for next-token prediction** — so the model has no
>   choice but to maintain it.
> - **(¬L) State not reconstructible from sequence co-occurrence statistics
>   alone** — so the probe signal cannot be an artefact of pairwise token
>   statistics.
>
> We document one domain (cities) failing the third property cleanly, will
> document a within-domain natural experiment (music: key/chord vs beat
> probes) that isolates the third property under a fixed model, and additional
> positive domains (synthetic algebraic, applied aviation, applied dialog)
> that exhibit all three.

**Caveats** (per `pivot.md`):

- The three criteria are not independent. Discrete-state often implies non-
  leaky; necessity interacts with both. The cities-only data isolates ¬L
  cleanly but does not separate D or N.
- The thesis is *consistent with* the data, not *uniquely predicted by* it.
  A narrower, more defensible framing — "the co-occurrence leak is the
  dominant failure mode in spatially-structured corpora" — may end up being
  the right paper-ready version. Decision after Milestone 2 (music).
- Reviewer pressure may force dropping D from the framing (since the cities
  continuous-target issue is a probe-methodology problem, not a domain
  property) and presenting only N and ¬L.

---

## Methodology (the load-bearing parts)

### Probe with both position-level AND node-level splits

The standard probe protocol — train on 80 % of positions, test on the other
20 % — is *insufficient* in continuous-target settings with few unique tokens
(~10³). Both linear and MLP probes can pass position-level via per-token
lookup memorization. The node-level split (train on 80 % of unique token
IDs, test on the disjoint 20 %) is the probe-capacity-controlled version
and gives the honest signal. Run both; report both. The cities probe.py
implements this.

### Two-tier destroyed-structure control

A *single* destroyed-structure control is not enough. Within-sequence
shuffles preserve set-membership co-occurrence, which itself carries the
signal a probe will find on geographically-structured data. A *strict*
control (global token shuffle across the entire stream) is needed to
disentangle. Cities `data/prepare_city.py` has both `--shuffle_routes` (weak)
and `--shuffle_globally` (strict); the per-domain template should provide
both.

### Activation transplant, not pseudoinverse-direction patching

The Nanda-style "compute a residual direction encoding world-state and patch
in target-direction" intervention is, in our setting, contaminated: the
pseudoinverse-derived direction depends on the probe's geometry, and a probe
trained on geographically-clustered activations finds a direction in any
such model — including a destroyed-structure model that demonstrably cannot
route. The corrected intervention substitutes a *real* residual `a_B` from a
position where the model genuinely processes B for the residual `a_A` at a
position where it processes A; the model's own representation is used, not
the probe's. `eval/transplant.py` implements this.

### `wte` vs an off-the-shelf graph embedding

When a probe succeeds on a learned token embedding, a natural reviewer
question is "is this just node2vec (or word2vec on graph walks)?". Cities
`eval/embedding_compare.py` answers this directly: Procrustes alignment +
linear CKA + probe parity. The cities result: `wte` is NOT a node2vec
embedding (similarity to random matrix is comparable); a pure node2vec on
the bare graph achieves *stronger* probe R² than `wte`. The geographic
signal we found is not in the input embedding; it is built by the
transformer's higher layers. This comparison should be reproduced
per-domain where a natural off-the-shelf embedding exists.

---

## Per-domain template (used by every milestone in PLAN.md)

For each new domain in Phase 6:

1. **Pipeline.** `data/prepare_<domain>.py` that produces the same
   `train/val/gen.bin` + `meta.pkl` + per-domain ground-truth files
   (analogue of `coords.csv` and `graph.gpickle`). Two destroyed-structure
   flags (weak + strict).
2. **Train.** Reuse `model/train.py` with the existing small/medium configs.
3. **Intrinsic eval.** Reuse `eval/valid_edge.py` and `eval/baselines.py`
   patterns; replace the city-specific scoring with the domain's primary
   metric.
4. **Probe.** Reuse `eval/probe.py` unchanged; choose the probe target from
   the domain's ground-truth labels.
5. **Activation transplant.** Reuse `eval/transplant.py` unchanged.
6. **Embedding comparison.** Run `eval/embedding_compare.py` (or its
   per-domain analogue) against the natural baseline embedding for the
   domain.

THE PER-DOMAIN ONE RULE: no probe-target value may appear in the model's
input. For cities this is "no coordinates"; for music "no key signature
explicit in the token stream"; for dialog "no slot-value labels"; etc.

---

## What competing baselines establish (the "earn the right" gate)

For each domain, the model must first match or beat domain-appropriate
sequence baselines on next-token prediction. Beating baselines on the
*primary* task doesn't establish the world-model claim — that requires the
probe + transplant gradient — but it earns the right to make the claim by
showing the model is doing real work. Cities established this against 1st-
and 2nd-order Markov on the real graph + long-range coherence.

---

## Definition of done (project)

See PLAN.md's "Definition of done" — at least 4 domains, the predictive
characterisation (D, N, ¬L or a tightened version), and the methodological
caveats demonstrated cleanly.

---

## References

- Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., Wattenberg, M.
  (2022). *Emergent World Representations: Exploring a Sequence Model
  Trained on a Synthetic Task* — Othello-GPT.
- Nanda, N. (2023). *Actually, Othello-GPT Has a Linear Emergent World
  Representation* — the linear-encoding sharpening.
- Karvonen, A. (2024). *Emergent World Models and Latent Variable Estimation
  in Chess-Playing Language Models* — Chess-GPT, the closest replication.
- Karpathy, A. *nanoGPT* — the model + training scaffold.
