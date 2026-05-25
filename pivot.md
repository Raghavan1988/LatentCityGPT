# Pivot plan — from LatentCityGPT to a comparative study of where Othello-GPT extends

**Status:** Proposed, 2026-05-24. Supersedes the single-domain framing in
`CONTEXT.md` and `PLAN.md` but does not replace any code. Read alongside
`STATUS_vs_OTHELLO-GPT.md` and `update_may24_final.md` for the empirical
results that motivate the pivot.

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
   probe's direction, which any clustered model exhibits. The Phase 5
   design needs to be replaced with **activation transplant** (substitute
   a residual from a position where the model is genuinely in state B
   into a position where it is in state A), which uses the model's own
   representation rather than a probe-derived direction.

These three findings are independently novel. They are what the pivot
preserves and amplifies.

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
| `eval/causal.py` (pseudoinverse-direction patching) | **Pivot.** Keep the file as the documented-failure version; add `eval/transplant.py` for the corrected activation-transplant intervention. |
| London / Manhattan / Boston / South Bay corpora and checkpoints | **Keep.** Cities is the negative anchor; the trained London checkpoint is referenced throughout. |
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

### Milestone 2 — The within-domain mixed-verdict figure (2–3 days)
**Domain:** Music.

- Write `data/prepare_music.py` using Bach chorales from `music21`.
- Train small.py.
- Compute three probe targets via `music21`: key, current chord, beat
  position.
- Run all three through `eval/probe.py` with node-level split and
  destroyed-structure control (within-piece pitch shuffle).
- **Critical predictions to verify:**
  - Key probe: high R² on real *and* shuffled → cities-like failure
    reproduced.
  - Chord probe: high R² on real and shuffled → same.
  - Beat probe: high accuracy on real, *collapsed* on shuffled →
    Othello-like positive.

**Why second:** highest scientific value per day. This is the single
experiment that varies only one criterion while holding the model and
data constant. If it lands as predicted, the paper has its load-bearing
empirical figure.

**Definition of done:** the three-probe table showing the predicted
mixed verdicts, with destroyed-structure ablation.

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
- Destroyed-structure control: within-flight record shuffle.
- Replace pseudoinverse patching with activation-transplant in
  `eval/transplant.py`.

**Why fourth:** the cleanest Othello-fit of any applied domain
(phase truly requires temporal integration); also the right place to
introduce the corrected causal intervention.

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
| Music beat probe does not collapse under within-piece shuffle as predicted | Medium | High | The framing-level prediction depends on this experiment. If beat probe survives shuffle, the criteria framing needs revision. **Decision point:** if music doesn't deliver the mixed-verdict figure, fall back to a narrower thesis ("co-occurrence leak in spatial domains"). |
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

| Claim | Confidence |
|---|---|
| The pivot preserves all existing work and infrastructure | ~95% |
| Cities is a publishable negative-anchor / methodology result | ~85% |
| The within-domain music mixed-verdict will land as predicted | ~70% |
| Sym-group + Music + Flight-phase + Dialog-state will be runnable on M1 at smoke scale | ~90% |
| The paper is workshop-publishable after Milestones 1–4 | ~75% |
| The three-criteria framing as stated holds up to reviewer pressure | ~55% |
| The paper is mainline-conference-publishable after Milestones 1–5 + theory tightening | ~40% |
| Effort estimates are within 2x of reality | ~70% |

**Single-point confidence in the overall plan:** moderate-high. The
empirical contributions are robust; the framing has known weaknesses
that we will know more about after Milestone 2. The right next action
is Milestone 1 (sym-group), which costs ≤2 days and validates the
methodology before we commit to the framing.
