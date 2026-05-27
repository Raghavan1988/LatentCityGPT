# Pivot plan — from LatentCityGPT to a comparative study of where Othello-GPT extends

**Status:** Proposed 2026-05-24; updated 2026-05-24 evening with empirical
results from the must-do experiments. Supersedes the single-domain framing
in `CONTEXT.md` and `PLAN.md` but does not replace any code. Read alongside
`STATUS_vs_OTHELLO-GPT.md` and `update_may24_final.md` for the
data-pipeline-level results; see the new section "Empirical results from
the must-do experiments" below for the causal-intervention results that
strengthen the pivot.

> **Status update — 2026-05-26 evening.** Major progress beyond the
> original plan. Music M2 first-pass + retraction + expanded-corpus v2
> all done (`updateMay25.md`, `updateMay26.md`, `updateMay26_afternoon.md`).
> Othello-GPT was reproduced from scratch in this codebase as a
> framework-validation positive control (`updateMay26_evening.md`):
> 3-class MLP probe 91.19 % per-cell (vs published ~94 %), 3-class linear
> 77.15 % (in published 75-85 % range). The framework is validated; music's
> null is principled N-criterion failure; sym-group methodology calibration
> attempted but inconclusive. The "three criteria" framing has sharpened:
> the **N criterion** (state necessary for next-token prediction) is the
> single load-bearing requirement for classification probes to succeed.
> Where N holds (cities, Othello, conjecturally flight-phase), probes find
> the encoded state. Where N fails (music), they don't — regardless of
> how much training data you throw at it.

---

## TL;DR

The single-domain "Othello-GPT on cities" thesis did not survive contact
with the destroyed-structure control. What we found instead is more
interesting: a **methodological failure mode** in next-token-transformer
interpretability that has been under-appreciated, with a domain
(cities) that documents it cleanly. We pivot from a single replication
to a **comparative study across 5–6 domains** that maps out where the
Othello-GPT lineage of results applies and where it does not.

**Nothing is thrown away.** Cities is the load-bearing negative anchor;
all infrastructure (data pipeline, model, probe suite, node-level split,
destroyed-structure control, MLP-contamination diagnostic) transfers
directly to every new domain.

**Confidence.** Workshop-paper-publishable after ~3 weeks of focused
work: ~75%. Mainline-conference-publishable after ~6–8 weeks: ~40%.
The framing (three criteria) is the weakest part and may need to be
softened or sharpened. The empirical contributions (cities negative
result + at least two positive analogues + within-domain mixed-verdict
experiment) are robust.

---

## What we learned that forced the pivot

Three findings from the cities work, in order of importance.

1. **Within-route shuffle does not destroy geographic probe signal — it
   strengthens it.** Probe R² on a model trained on order-shuffled
   London routes was *higher* than on the real model (node-level 0.965
   vs 0.639). The mechanism is a capacity argument: the real model
   spends representational capacity on directed adjacency, which
   competes with geographic clustering; the shuffled model spends
   100% on clustering. **Implication: probe-decoded geographic
   structure in residual streams is downstream of geographic
   co-occurrence in training data, not specifically of sequence
   learning.**

2. **The MLP-vs-linear probe criterion does not transfer to
   continuous-target settings.** With ~10³ tokens and continuous (lat,
   lon) targets, an MLP probe achieves near-perfect R² on the
   *untrained* model via lookup memorisation. The standard "linear ≈
   MLP → linearly encoded" test from Nanda 2023 is inapplicable in our
   setting; a node-level (held-out-token) split is the only way to
   disambiguate genuine representation from lookup.

3. **Target-direction activation patching does not isolate use of the
   representation.** Pseudoinverse-direction patches show ~75–80%
   target-beats-random on both the real model and the destroyed-
   structure model — the latter being a model that demonstrably cannot
   route. The intervention measures sensitivity to patches in the
   probe's direction, which any clustered model exhibits.
   **Update 2026-05-24 evening:** the Phase 5 design has now been
   replaced with **activation transplant** (substitute a residual from
   a position where the model is genuinely in state B into a position
   where it is in state A), implemented in `eval/transplant.py` and run
   on three conditions. Results in the new section below; the
   intervention produces the predicted clean gradient and decomposes
   the model's causal effect into two distinct contributions.

These three findings are independently novel. They are what the pivot
preserves and amplifies.

---

## Empirical results from the must-do experiments (2026-05-24 evening)

The three "must-do" experiments listed in `next_steps.md` were implemented
and run on the existing London corpus + checkpoints. All three predictions
held; the cumulative result is stronger than the pivot originally anticipated.

### Result A — Activation transplant Phase 5: a three-condition causal gradient

`eval/transplant.py` was added. For each test position with current token A,
the residual at layer L=3 is replaced with a residual cached from a position
where the model is genuinely processing a randomly chosen target token B
(model-own representation, not probe-derived). Three trained models, all
small-config London, were tested side-by-side:

| Condition                      | val ppl | P(A's nbrs) unpatched | **Transplant lift on P(B's nbrs)** | target > random rate |
|---|---:|---:|---:|---:|
| **Real London**                | **1.65** | **0.984** | **+0.953** | **100.0%** |
| Within-route shuffled London   | 25.0    | 0.061     | +0.247    | 99.5%      |
| **Global-shuffled London**     | **313** | **0.006** | **+0.000** | **48.0% (chance)** |

The differential decomposes the model's residual-stream causal effect into
two distinct contributions:

- **~+0.247** of the lift comes from **GEOGRAPHIC CLUSTERING** alone. Any
  geographic co-occurrence in training produces it; B's "graph neighbors"
  overlap with B's geographic neighborhood, so substituting `a_B` causes
  the model to predict spatially-nearby tokens, ~25 % of which are
  coincidentally graph successors.
- **~+0.706** is **specifically the graph-adjacency-using representation**
  that requires sequence-order training. Destroyed when ordered adjacency
  is destroyed (within-route shuffle); present only in the real model.
- The global-shuffle control hits 48 % (chance) — when both ordered
  adjacency *and* set-membership co-occurrence are destroyed, no causal
  effect remains.

This is the clean Othello-GPT-style causal demonstration once the proper
intervention is used. **The decomposition — clustering and adjacency as
two distinct, independently destroyable representations both causally
encoded in the same residual stream — is the project's most interesting
contribution after this session.**

### Result B — `wte` vs node2vec: the input embedding is NOT node2vec

`eval/embedding_compare.py` was added. Compared the trained model's input
embedding to node2vec computed directly on the City of London street graph.

| Similarity metric                       | wte vs node2vec | wte vs RANDOM (sanity) |
|---|---:|---:|
| Procrustes R² (best orthogonal alignment) | **0.276**       | 0.333                  |
| Linear CKA                              | **0.340**       | 0.313                  |

The trained transformer's `wte` is structurally no more similar to node2vec
than to a random matrix of the same shape.

| Probe parity (predict (x_m, y_m), 80/20 node-level split) | R²    | Median error |
|---|---:|---:|
| `wte` (trained model)                                       | 0.21  | 569 m       |
| **node2vec on the graph** (no transformer)                  | **0.94** | **124 m** |

node2vec on the graph is *substantially* more geographically organized
than the trained model's `wte`. The transformer didn't end up rediscovering
node2vec; it learned something quantitatively worse at *that* task — because
its job was different (next-token prediction, with directed-adjacency
information competing for representational capacity).

**Consequence for the project's framing:** geographic decodability in the
residual stream is *not* attributable to a "node2vec-like clustering in
the input embedding." Whatever representation the activation-transplant
intervention causally manipulates, it is **built by the transformer's
higher layers from sequence context**, not handed to the model by `wte`.

### Result C — Strict destroyed-structure control completes the gradient

The new `--shuffle_globally` flag in `data/prepare_city.py` randomly
permutes every real-token position across the entire training stream
(preserving only BOS/EOS structure). A model trained on this corpus:

| Metric                                  | Value    | Interpretation |
|---|---:|---|
| Val perplexity                          | **313**  | Essentially uniform over the vocab |
| P(A's neighbors) unpatched              | **0.006** | Model knows nothing about the graph |
| Linear probe R² (node-level)            | **−0.02** | Probe collapses to chance |
| Linear probe R² (untrained-init baseline) | −0.07 | Trained ≈ untrained |
| Activation-transplant lift              | **+0.000** | No causal effect (48 % target>random) |

This is the destroyed-structure control PLAN.md Phase 4 originally
intended. The pivot.md within-route-shuffle finding ("set-membership
preservation keeps the probe signal alive") is now confirmed by its
converse: when set-membership is destroyed too, the probe signal
collapses.

### Cross-condition summary

| Condition                | val ppl | P(A nbrs) | Probe R² (node-level linear) | Transplant lift on P(B nbrs) |
|---|---:|---:|---:|---:|
| Real London              | 1.65    | 0.984     | 0.64                         | **+0.953**                   |
| Within-route shuffled    | 25.0    | 0.061     | 0.96                         | +0.247                       |
| Global-shuffled          | 313     | 0.006     | −0.02                        | +0.000                       |

Each metric moves monotonically and consistently across the destruction
gradient. The triple of metrics is the package's load-bearing empirical
contribution.

### Implications for the pivot

- The activation-transplant intervention is **implemented and validated**
  on the cities domain. It can be applied directly to any new domain in
  the milestones below — no new code, only new probe targets.
- The destroyed-structure control template is **two-tier**: `--shuffle_routes`
  (weak; preserves set-membership) and `--shuffle_globally` (strict; breaks
  set-membership). Both should be reproduced per-domain to test the same
  gradient.
- The cities work has **graduated from "negative anchor only" to
  "decomposition result"**: it provides both a methodology cautionary tale
  (within-route shuffle is insufficient) AND a clean positive causal
  demonstration (transplant gradient). Cities is now a central member of
  the portfolio, not just a cautionary entry.
- The `wte` ≠ node2vec finding is an additional methodological asset:
  reviewers asking "are you just rediscovering node2vec?" have a clean
  negative answer.

---

## The reframed thesis

The single-domain framing ("an emergent map appears in city-route
transformers") was wrong in detail. The reframed thesis is comparative:

> *Emergent world models in next-token transformers depend on three
> structural properties of the training distribution:*
> - **(D) Discrete state** — so probes cannot lookup-memorise via
>   continuous-target shortcuts.
> - **(N) State necessary for next-token prediction** — so the model
>   has no choice but to maintain it.
> - **(¬L) State not reconstructible from sequence co-occurrence
>   statistics alone** — so the probe signal cannot be an artefact of
>   pairwise token statistics.
>
> *We document one domain (cities) failing the third property cleanly,
> a within-domain natural experiment (music: key vs beat probes) that
> isolates the third property under a fixed model, and three positive
> domains (synthetic algebraic, applied aviation, applied dialog) that
> exhibit all three. The package supports a predictive theory of when
> Othello-GPT-style results replicate and identifies a methodological
> contamination mode that has been under-reported.*

**Honest caveats on the thesis:**

- The three criteria are not independent. Discrete-state often implies
  non-leaky; necessity interacts with both. The package isolates ¬L
  cleanly (via music) but does not cleanly isolate D or N.
- The thesis as stated is *consistent with* the data, not *uniquely
  predicted by* it. Stronger alternative framings — e.g., "the
  co-occurrence leak is the dominant failure mode in
  spatially-structured corpora" — may be more defensible.
- Mainline-venue publication may require dropping D from the framing
  (since the cities continuous-target issue is a probe-methodology
  problem, not a domain property) and presenting only N and ¬L as the
  load-bearing criteria.

---

## What we keep, what we pivot

| Asset from the original work | Status |
|---|---|
| `data/prepare_city.py` with `--shuffle_routes` flag | **Keep.** Becomes the template for new domains' data pipelines. |
| `model/` (nanoGPT-small + train loop + configs) | **Keep.** Reused unchanged across all domains. |
| `eval/probe.py` with node-level split + MLP contamination diagnostic | **Keep.** The node-level split is the project's most important methodological contribution; reused on every domain. |
| `eval/baselines.py` (Markov 1st/2nd order + long-range coherence) | **Keep.** Per-domain baselines reuse this structure. |
| `eval/causal.py` (pseudoinverse-direction patching) | **Pivoted.** File preserved as documented-failure version — its negative result is part of the methodology narrative. |
| `eval/transplant.py` (corrected activation-transplant intervention) | **NEW asset (added 2026-05-24 evening).** Implemented + validated on three conditions; produces the clean Phase 5 causal gradient. Reused unchanged for every new domain. |
| `eval/embedding_compare.py` (wte vs node2vec comparison + Procrustes/CKA + probe parity) | **NEW asset (added 2026-05-24 evening).** Reusable per-domain as a sanity check on "is the model just rediscovering an off-the-shelf embedding?" |
| `data/prepare_city.py` `--shuffle_globally` flag (strict destroyed-structure control) | **NEW asset (added 2026-05-24 evening).** Two-tier destruction template (`--shuffle_routes` weak + `--shuffle_globally` strict) reproducible per-domain. |
| London / Manhattan / Boston / South Bay corpora + London-shuffled + London-global-shuffled corpora and checkpoints | **Keep.** Cities is now the project's *decomposition* anchor — the trained London + both shuffled-London checkpoints together produce the three-condition gradient referenced throughout. |
| `CONTEXT.md` "metric map emerges" framing | **Pivot.** Edit to reflect the negative-anchor role; preserve as historical narrative. |
| `STATUS_vs_OTHELLO-GPT.md` and `update_may24_final.md` writeups | **Keep, prominently.** These are the most candid existing documentation of the empirical finding. |
| `tests/test_prepare_city.py` synthetic-grid smoke test | **Keep, generalise.** Add an equivalent synthetic-state smoke test for each new domain's pipeline. |

**Net pivot:** no code is deleted. New domain pipelines are added as
peer directories (`data/prepare_music.py`, `data/prepare_adsb.py`, etc.)
that follow the same conventions. The probe and evaluation code already
parameterises over data directories and is reused.

---

## The domain portfolio

Five new domains plus the existing cities work. Each row is graded
against the three criteria (✓ / ✗ / ~) and assigned a role.

| # | Domain | Type | World state probed | D | N | ¬L | Role | Effort | M1? |
|:--|---|---|---|:-:|:-:|:-:|---|---|:-:|
| 1 | **Cities (existing)** | Applied — urban routing | (lat, lon) per token | ✗ | ✗ | ✗ | **Negative anchor** | Done | Done |
| 2 | **Music — key/chord probe** | Theoretical + creative AI | Key signature or current chord | ✓ | ~ | ✗ | **Negative anchor** | 2–3 days | ✓ |
| 2′ | **Music — beat probe** | Theoretical + creative AI | Beat position in measure | ✓ | ✓ | ✓ | **Within-domain positive** (same model, different probe — the cleanest single test in the package) | (same data) | ✓ |
| 3 | **Symmetric-group-GPT** | Theoretical / algorithm-learning | Resulting permutation σ ∈ Sₙ | ✓ | ✓ | ✓ | **Theoretical positive anchor** | 1–2 days | ✓ |
| 4 | **Dialog-state tracker (MultiWOZ)** | Applied — conversational AI | Slot-value belief state | ✓ | ✓ | ~ | **Applied positive (text)** | 2–3 days | ✓ |
| 5 | **Flight-phase (ADS-B / OpenSky)** | Applied — aviation safety / ATC | Flight phase (8 classes) | ✓ | ✓ | ✓ | **Applied positive (time series)** | 3–5 days | ✓ |
| 6 | **Maze-GPT** | Applied — robotics / embodied AI | Agent pose, wall map, explored cells | ✓ | ✓ | ✓ | **Applied positive (spatial)** | 5–7 days | ✓ |

**Literature anchors** (cite, do not reproduce): Othello-GPT (Li 2022,
Nanda 2023), Chess-GPT (Karvonen 2024).

**Domain-coverage rationale.** The portfolio is constructed to (i) test
the thesis across multiple sectors so a domain-specific reviewer
cannot dismiss it as a quirk, (ii) include at least one within-domain
natural experiment (music key vs beat) that varies only one criterion
at a time, and (iii) keep the theoretical baseline (sym-group) for
methodology validation.

**Trim option.** If time is tight, the minimum viable comparative
paper is **Cities + Music + Sym-group + Flight-phase** (4 domains, 3
positive criteria covered, 2 sectors). Maze-GPT and Dialog-state are
upside that strengthens the audience reach but are not load-bearing for
the thesis.

---

## Execution plan and milestones

Ordered by cheapest first, lowest-risk first. Each milestone produces
a single committable artefact and can be reviewed before moving on.

### Milestone 1 — Methodology calibration (1–2 days)
**Domain:** Symmetric-group-GPT.

- Write `data/prepare_symgroup.py` (synthetic generator: sample word
  in Sₙ, compute resulting permutation, write `*.bin`).
- Train `small.py` on it.
- Run `eval/probe.py` against the permutation target.
- Validate: probe recovers the permutation cleanly; destroyed-structure
  control (shuffle generators within a word) collapses signal.

**Why first:** lowest risk, guaranteed-positive domain by construction.
If the methodology fails here, everything downstream is in question.
This is a sanity check, not a paper result.

**Definition of done:** node-level linear probe achieves >0.9 accuracy
on held-out tokens for n ≤ 8; destroyed-structure control drops to
near-chance.

### Milestone 2 — Music: three load-bearing probes, three independent bets (2–3 days)
**Domain:** Music.

- Write `data/prepare_music.py` using Bach chorales from `music21`.
- Train small.py.
- Compute three probe targets via `music21`: key, current chord, beat
  position.
- Run all three through `eval/probe.py` with node-level split AND the
  two-tier destroyed-structure control template (within-piece shuffle
  + global shuffle), matching the cities decomposition.

**Each probe is an independent bet on whether sequence order is
load-bearing for its target in tonal music. Each direction of each
outcome reshapes the paper in a different way — there is no "uninteresting"
result, but only one outcome lands the originally-imagined central figure.**

| Probe | Predicted | If survives shuffle | If collapses on shuffle |
|---|---|---|---|
| **Key signature** | survives (cities-like) | predicted; clean cities-analogue in a non-spatial domain — strengthens the ¬L co-occurrence-leak story | surprise: key is sequence-trained (cadential / modulation context); undermines ¬L as a clean binary and suggests tonal context does real representational work |
| **Current chord** | survives (cities-like) | predicted; same shape as key | functional-harmony context is sequence-trained — a finer-grained Othello-positive than beat; reshapes ¬L into a graded axis (key fails, chord passes, beat passes) |
| **Beat position** | collapses (Othello-like) | **bad outcome.** Beat reconstructible from pitch-class statistics (strong beats favor I/V in tonal music); tonal music too leaky to host the within-domain positive; forces fallback to a less-leaky corpus or moves the within-domain positive role to a different milestone | predicted; the paper's load-bearing positive figure |

**Joint outcomes worth naming (the 2³ outcome space groups into four
qualitative cases):**

- **A. Predicted mixed verdict** (key + chord survive, beat collapses)
  → paper's originally-planned central figure lands. The within-domain
  experiment isolates ¬L cleanly under a fixed model. Best case.
- **B. Universal cities-like failure** (all three survive)
  → beat-leak in tonal music is real. The within-domain positive role
  moves elsewhere (folk monophony, counterpoint exercises, or a different
  milestone entirely). Cities-style methodology cautionary tale extends
  to music, which is still a publishable result, just a different paper.
- **C. Universal Othello-like positive** (all three collapse)
  → tonal music is less leaky than expected. Music becomes a clean
  positive analogue and the "mixed verdict" claim fails — but the
  portfolio reweights toward "we have multiple clean positives plus a
  cities negative anchor", which is also publishable, just less
  framing-novel.
- **D. Graded leak** (any partial-mixed pattern, e.g. key survives,
  chord and beat collapse)
  → reshapes ¬L from a binary criterion into a quantity. Potentially the
  most scientifically interesting outcome; demands a more careful theory
  of what kinds of co-occurrence support what probe targets. Higher
  ceiling but also higher write-up cost.

**Why second:** highest scientific value per day. All three probes are
load-bearing, all four joint outcomes are publishable in some form, and
the experiment varies only one factor (probe target) while holding model
and data constant. The "predicted mixed verdict" is the best case, not
the only viable case.

**Definition of done:** the three-probe × three-condition table (real /
within-piece-shuffled / global-shuffled) on a single trained model, with
node-level split applied to each. The framing decision (which of A–D
the paper centers on) is made *after* the table is in hand, not before.

### Milestone 3 — First applied datapoint (2–3 days)
**Domain:** Dialog-state tracker.

- Write `data/prepare_multiwoz.py` (download dataset, tokenize
  utterances with BPE, extract slot-value labels per turn).
- Train small.py with adjusted block_size for dialog lengths.
- Probe for each slot's value at each turn; report on inferred-slot
  cases specifically (those that fail co-occurrence shortcuts).
- Destroyed-structure control: shuffle turns within a dialog.

**Why third:** broadest reviewer audience; lowest applied-domain
effort; tests the framing in a non-numeric, non-spatial domain.

**Definition of done:** per-slot probe accuracy, with inferred-vs-
surface-mentioned slot breakdown. Destroyed-structure control kills
inferred-slot accuracy but spares surface-mentioned slots.

### Milestone 4 — Second applied datapoint (3–5 days)
**Domain:** Flight-phase (ADS-B).

- Write `data/prepare_adsb.py` using the `traffic` library + OpenSky
  Network. Discretise altitude / vertical-rate / ground-speed /
  heading. Group records into flights; compute phase labels via Sun
  et al. fuzzy logic.
- Train small.py.
- Probe for phase classification at each layer.
- Destroyed-structure control: within-flight record shuffle (weak)
  *and* global token shuffle (strict, the gradient template).
- Apply the existing `eval/transplant.py` activation-transplant
  intervention against the trained model.

**Why fourth:** the cleanest Othello-fit of any applied domain
(phase truly requires temporal integration). The activation-transplant
infrastructure built on cities is reused unchanged.

**Definition of done:** layer-wise phase-probe accuracy figure; within-
flight shuffle ablation; activation-transplant intervention showing
phase-conditioned shift in next-token distribution.

### Milestone 5 — Spatial applied datapoint (5–7 days, optional)
**Domain:** Maze-GPT.

- Write `data/prepare_maze.py` (procedural maze generator + agent
  observation model).
- Train small.py on (action, observation) sequences.
- Probe for agent pose (with continuous-target caution), wall
  configuration (discrete, clean), explored-cell map.

**Why fifth and optional:** highest setup cost; weakest marginal
contribution given sym-group already plays the controlled-positive
role and flight-phase already plays the applied-with-temporal-
integration role. Include if budget allows; defer if not.

### Milestone 6 — Paper assembly (5–7 days)
- Unified figure across all domains using consistent probe protocol.
- Rewrite of `CONTEXT.md` and project README to reflect comparative
  framing.
- Workshop paper draft.

**Total realistic budget:** 18–27 days of focused work, depending on
whether Maze-GPT is included and how much friction arises. Calendar
time: 5–8 weeks assuming part-time effort.

---

## Risks and how we mitigate them

| Risk | Likelihood | Severity | Mitigation |
|---|:-:|:-:|---|
| Sym-group methodology validation fails (probe doesn't recover the permutation cleanly) | Low | High | If this happens, the issue is in the probe / model code, not the theory. Debug before proceeding to other domains. |
| Music beat probe does not collapse under within-piece shuffle as predicted | Medium | High | The originally-imagined central figure depends on beat collapsing. If beat probe survives, see Milestone-2 outcome B: move the within-domain positive role to a different corpus or milestone; the cities-style "co-occurrence leak in spatial domains" narrows the thesis but is still publishable. |
| Music key/chord probes collapse on within-piece shuffle (against prediction) | Low–Medium | Medium | Outcome C/D in Milestone 2 — tonal context is doing more sequence-trained work than expected. Reframes ¬L from a binary into a quantity; potentially the most interesting outcome but adds write-up complexity. |
| Music joint outcome lands as D (partial-mixed / graded leak) | Medium | Medium (upside) | This is the high-ceiling outcome but raises the write-up cost: ¬L stops being a binary criterion and needs a theory of *how much* co-occurrence supports *which* probe targets. Budget extra time for the framing section if D lands. |
| Flight-phase phase labelling produces too many ambiguous-phase records | Medium | Medium | Sun et al. algorithm has a "transition" / unclassified state; filter or include as own class. |
| Dialog-state probe is dominated by surface-mention leak (every slot value is mentioned verbatim) | High | Medium | Focus probe evaluation on *inferred* slots specifically. MultiWOZ has cases where the system infers from context; build the probe target around those. |
| Maze-GPT setup eats more than 7 days | Medium | Medium | Hard-cap effort at 7 days. If not running by then, defer to a follow-up paper. |
| Three-criteria framing rejected as "not independent" by reviewers | High | Low | Drop D from the framing (it's a probe-methodology issue, not a domain property); present only N and ¬L. The empirical contributions stand regardless of how the criteria are organised. |
| Confirmation bias — we built the plan around the cities work, the data may not cooperate | — | High | Run sym-group + music *first* (4–5 days total). Hold all framing decisions until music results are in. Be prepared to narrow the thesis if the predictions don't land. |
| Effort estimates are off by 2x | High | Low | Stated explicitly in this plan. Budget is 18–27 days; calendar 5–8 weeks. |

---

## Publication path

**Primary target:** a mech-interp or interpretability workshop at
NeurIPS / ICLR (e.g., NeurIPS XAI, ICLR Mechanistic Interpretability,
ATTRIB). These venues actively reward comparative empirical studies
that document failure modes. Confidence: ~75%.

**Secondary target:** if the package develops further (more domains,
formal theory, tooling release), an ICLR / NeurIPS / ACL mainline paper.
Confidence after additional ~6 weeks of work: ~40%.

**Tertiary target:** an arXiv preprint with an applied-domain follow-up
(e.g., a focused flight-phase or dialog-state paper at the relevant
venue) that builds on this work. Always available as a fallback.

**One-line pitch for reviewers:**
> *Where do emergent world models actually appear in next-token
> transformers? We give a predictive characterisation across five
> domains spanning theoretical, creative, navigational, conversational,
> and aviation settings — and document a previously-unreported
> contamination mode in the probing methodology that explains why
> world-model claims in some domains survive scrutiny and others
> do not.*

---

## Open questions and decision points

Things this plan does *not* resolve, that need decisions during
execution:

1. **Final criteria framing.** D + N + ¬L vs N + ¬L only. Resolve
   after music results are in.
2. **Whether to release a methodology tool.** A `probekit` Python
   package containing the node-level split, destroyed-structure
   control template, and activation-transplant intervention would
   substantially raise the citation potential. ~1 week of polish work
   on top of the paper budget. Decide after Milestone 4.
3. **Whether to include Maze-GPT.** Resolve at the start of Milestone
   5, given remaining budget.
4. **Whether to extend to Chess-GPT replication for an additional
   literature anchor.** Probably skip; cite Karvonen 2024 instead.
5. **What to do with the cities work as standalone material.** Could
   become a separate methodology paper ("MLP-probe contamination in
   continuous-target settings"). Or roll into the comparative paper as
   the negative anchor. Default: roll in.

---

## Confidence summary

Updated 2026-05-24 evening after the three must-do experiments landed.

| Claim | Confidence (was → now) |
|---|---|
| The pivot preserves all existing work and infrastructure | ~95% (unchanged) |
| Cities is a publishable negative-anchor / methodology result | ~85% → **~92%** (transplant decomposition strengthens it materially) |
| The three-condition causal gradient (real / within-shuffle / global-shuffle) is reproducible | **~95% (new row)** |
| The clustering-vs-adjacency decomposition holds up to scrutiny | **~80% (new row)** |
| Music joint outcome lands as A (predicted mixed verdict: key + chord survive, beat collapses) | ~40% → **~15%** (observed: beat probe at chance in ALL conditions, not high-on-real; see `updateMay25.md`) |
| Music joint outcome is A or D (any partial-mixed pattern → still publishable as a within-domain ¬L result) | ~60% → **~50%** |
| Music joint outcome is publishable in some form (any of A / B / C / D — none are dead ends) | ~90% → **~85%** (valid-voice-step gradient is the safety net) |
| Beat probe collapses on within-piece shuffle (the load-bearing single bet) | ~55% → **n/a** (beat probe doesn't reach a level that can meaningfully collapse — sits at chance everywhere; resolves to (a) no encoding / (b) undertraining / (c) thin probe — TBD) |
| Mode probe survives within-piece shuffle (cities-analogue prediction) | ~70% → **~75%** (observed 60% vs 60% PIECE-LEVEL, matches prediction) |
| Music valid-voice-step rate produces a clean three-condition gradient (NEW row — the music-domain analogue of cities valid-edge) | **~95% (NEW; observed first try at 96.25% / 64.33% / 55.91% strict)** |
| The wte≠node2vec finding generalises to other domains' embeddings | **~60% (new row)** — needs per-domain comparison |
| Sym-group + Music + Flight-phase + Dialog-state will be runnable on M1 at smoke scale | ~90% (unchanged) |
| The paper is workshop-publishable after Milestones 1–4 | ~75% → **~82%** (transplant gives the package a clean causal claim cities alone now supports) |
| The three-criteria framing as stated holds up to reviewer pressure | ~55% (unchanged — still the weakest link) |
| The paper is mainline-conference-publishable after Milestones 1–5 + theory tightening | ~40% → **~48%** (the cities decomposition is a stronger central exhibit than originally weighted) |
| Effort estimates are within 2x of reality | ~70% (unchanged) |

**Single-point confidence in the overall plan:** moderate-high, modestly
increased by the must-do experiments. The cities domain is more robust
than the pivot's original "negative anchor" framing suggested — it now
provides both the methodology cautionary tale *and* a clean
activation-transplant causal demonstration. The framing has known
weaknesses that we will know more about after Milestone 2. The right
next action is Milestone 1 (sym-group), which costs ≤2 days and validates
the methodology before we commit to the framing.
