# Predictive Interpretability: A Comparative Study of Emergent World Representations in Small Next-Token Transformers

## Abstract

When does a transformer language model trained only to predict the
next token spontaneously develop an internal representation of the
world it is modeling? Prior work (Li et al. 2022; Nanda et al. 2023)
shows such representations exist for at least one toy domain
(Othello); whether and how this generalizes has remained an open
question. We address it by combining two contributions.

**First**, we develop a methodological protocol for testing
emergent representations across domains: multi-seed mean ± std
reporting, probe + activation-patching convergence, per-layer
ablation, destroyed-structure controls (real / within-shuffled /
global-shuffled corpora), and pre-registered ex-ante prediction
experiments with a git audit trail. We apply this protocol to five
descriptive-evidence domains (cities, Othello, flight phases, music,
walks on the symmetric group) and two pre-registered ex-ante
prediction experiments (maze navigation and HTTP log sequences).

**Second**, we test a candidate framework we call the **N-criterion**:
the hypothesis that the predictive relevance of a feature for
next-token prediction drives whether it appears in the residual
stream. The two pre-registered experiments both falsify substantive
predictions of this framework. The maze starting-cell prediction
(claimed null because the feature is irrelevant to next-step choice)
turned out to be encoded. The HTTP cumulative-large-response-count
prediction (claimed null because it would require active aggregation)
was also encoded. In both cases the broader "predictive relevance
drives encoding" framing was too coarse.

What does survive testing is a specific mechanism we call
**architectural carry-through**: features present at positionally
distinct input slots persist in the residual stream regardless of
their predictive relevance, via the side effect of self-attention
copying. This mechanism was introduced after the maze falsification
to explain the starting-cell-encoded result, and was then
independently confirmed by the HTTP Feature A pre-registered
prediction. Carry-through is the one substantive claim of this
paper that survived ex-ante testing on two domains of different
shape.

The HTTP cumulative-count falsification also reveals a methodological
issue: position-in-session correlates strongly with the target
feature, and the trained model's superior positional representations
inflate the probe gap. We document this confound and propose a
position-control follow-up.

**Contributions**: (1) a methodological framework for pre-registered
emergent-representation experiments, including the discovery and
discipline of pre-registering both positive and null predictions;
(2) one specific validated mechanism (architectural carry-through),
predicted ex-ante and confirmed on two domains of different shape;
(3) one identified failure mode (position-correlation as probe
confound) future work should control for. We do not claim a strong
predictive theory of which features emerge as representations; the
two pre-registered tests have demonstrably weakened any such claim
we might have tried to make.

**Scope.** All models studied here are in the 4M-13M parameter range
on synthetic or semi-synthetic tasks. We do not claim that any of
these findings extend to frontier-scale language models on
natural-language data.

---

## 1. Introduction

### 1.1 The question

A transformer language model is trained on one objective: predict the
next token. It is not given access to the structure of the domain it
is being trained on. It does not see board states in Othello, it does
not see latitude/longitude coordinates in cities, it does not see
musical chord labels. It only sees tokens.

Despite this minimal training signal, prior work (Li et al., 2022;
Nanda et al., 2023) has shown that a transformer trained to predict
the next move in random Othello games develops, in its residual
stream, a representation of the current board state — recoverable
by a simple linear probe and causally usable to steer the model's
predictions. This finding has become one of the most-cited examples
that next-token training can produce something one might call a
"world model."

A natural follow-up question: **does this generalize?** If we move
beyond Othello, do other transformers in other domains develop
analogous representations of the structure of their domain? Under
what conditions? Where in the network does the representation live?

### 1.2 The gap

Most mechanistic-interpretability work on this question has three
methodological features that we view as limiting:

1. **Single domain, post-hoc.** A typical paper picks one domain
   (Othello, or some other), runs some probes, and reports what was
   found. This leaves the reader unable to distinguish a genuine
   pattern from a fishing expedition — the analyst could have probed
   many features and reported only the ones that worked.

2. **Single seed.** Probe accuracies are typically reported as a
   single number, not as a mean ± standard deviation across
   independent seeds. This leaves uncertain how much of any reported
   result is the noise floor of the measurement.

3. **No ex-ante predictions.** Almost no published probe study
   commits in writing to what its outcome should be before running
   the experiment. This makes it difficult to distinguish a framework
   that *predicts* the data from a framework that *fits* the data
   after the fact.

We address all three: comparative (five descriptive-evidence domains,
plus two pre-registered ex-ante experiments), multi-seed (five seeds
per probe and per transplant, with σ reported), and pre-registered
(both ex-ante experiments — maze navigation and HTTP log sequences —
have their predictions committed to a public git repository in
advance, with timestamps).

### 1.3 The contribution

This paper makes three contributions.

**First**, we develop a **methodological protocol** for testing
emergent residual-stream representations across domains. The protocol
combines five elements: (i) multi-seed mean ± std reporting; (ii)
probe + activation-patching convergence at the same layer; (iii)
per-layer ablation; (iv) destroyed-structure controls; and (v)
pre-registered ex-ante predictions, including pre-registered nulls,
with a git audit trail. None of these elements is individually novel;
their combination is rarer than the literature suggests and represents
a reasonable bar for representational-content claims about small
transformers.

**Second**, we test a candidate framework — the **N-criterion** —
that attempts to predict which features will be encoded in terms of
their predictive relevance for next-token prediction. We tested two
specific risky predictions of this framework on two pre-registered
domains. Both fell in informative ways. In each case, the broader
"predictive relevance drives encoding" framing turned out to be
too coarse: features irrelevant to the next-token objective persisted
in the residual stream when they sat in positionally distinct input
slots, and features that would have required active aggregation
were also encoded when they correlated with positional information.
The N-criterion as a strong predictive theory does not survive these
tests.

**Third**, what *does* survive ex-ante testing is a specific
mechanism we call **architectural carry-through**: features present
at positionally distinct input slots persist in the residual stream
regardless of predictive relevance, via the side effect of
self-attention copying. This mechanism was introduced to explain
the maze starting-cell falsification, and was then independently
predicted and confirmed by the HTTP Feature A ex-ante prediction
on an applied domain at a different shape. Carry-through is the
one substantive claim of this paper that survived pre-registered
testing on two different domains. We also report a specific failure
mode the HTTP experiment surfaced — position-correlation as a probe
confound — which we identify as a methodological caveat future
work should control for.

We do not claim a strong predictive theory of which features emerge
as representations. The pre-registered tests have demonstrably
weakened any such claim we might have tried to make. What remains
is a methodology, one validated mechanism, and one identified
failure mode.

### 1.4 What this paper is not

We do not study frontier-scale language models. We do not study
natural-language data. We do not claim that any of the representations
recovered here are "the world model" of the underlying domain in a
philosophically loaded sense. We study **whether, when, and where**
small transformers represent task-relevant structure in their
residual streams, with care taken about what counts as evidence.

---

## 2. The N-criterion

### 2.1 What we are trying to make precise

The intuition behind Othello-GPT and its successors is that a
transformer "needs to know" the board state in order to predict the
next legal move, and that this need pressures gradient descent to
construct an internal representation of board state. Generalizing
this intuition cleanly is harder than it sounds. The word "need" can
mean several different things — information-theoretically required,
required for optimal prediction, required for *achievable* prediction
at this capacity and training budget, or merely useful. Almost no
real feature is cleanly required or cleanly irrelevant; predictive
relevance is continuous.

We therefore state the N-criterion in two forms.

### 2.2 The strict form (a useful straw target)

The strict form, useful as a target to test against:

> **Strict N-criterion.** A feature F is encoded as a linearly
> recoverable representation in the residual stream of a trained
> next-token transformer **iff** F is required for next-token
> prediction on this corpus, where "required" is defined model-
> independently from the task's information structure.

This form is **falsifiable in both directions**. A feature predicted
required but not encoded would falsify the "if" direction. A feature
predicted irrelevant but found encoded would falsify the "only if"
direction. We will see below that the data does the second of these
in at least one pre-registered case.

### 2.3 The graded form (what we actually argue)

The graded form is what we believe the evidence supports:

> **Graded N-criterion.** The predictive relevance of a candidate
> feature F for next-token prediction is the dominant driver of
> whether F emerges as a linearly recoverable representation in the
> residual stream of a trained next-token transformer. Features that
> provide robust, reusable predictive information tend to be encoded;
> features that are predictively irrelevant tend to be absent —
> *except* when the architecture provides what we call
> **architectural carry-through**: the feature is already present
> in the input at a positionally distinct slot, and self-attention
> can copy it forward across layers at essentially zero cost.

This form is weaker than the strict one. It is also closer to what
the data actually shows. It is more **mechanistically specific** in
one important way: it identifies the asymmetry in how next-token
training shapes residual content.

**The asymmetry of training pressure.** The strict form implicitly
assumed two gradients act on the residual stream: a pressure to
*acquire* features that improve next-token prediction, and a
pressure to *remove* features that do not. The graded form keeps
the first and discards the second. Next-token training has no
explicit mechanism to penalize encoding a feature that is
irrelevant; it has only a mechanism to reward encoding a feature
that is relevant. Irrelevant features therefore persist by default
if they are structurally easy to keep, and disappear only if
keeping them actively crowds out something the loss does care about.

**Architectural carry-through, made concrete.** In a self-attention
transformer, a feature already present in the input at an
identifiable position (e.g., the first non-padding token, or a token
in a positional slot the model can attend to specifically) can be
routed forward through every subsequent layer at low cost. The
positional embedding tags the slot. Self-attention can copy its
content. Each successive block can re-emit a related representation.
There is no training signal pushing back. The feature persists not
because the objective demands it but because nothing actively
removes it.

**Two predictions, one framework.** The revised framework therefore
makes *different* predictions about two categories of predictively-
irrelevant features:

- **In-input at a distinct slot.** Features that sit in the input
  at a positionally identifiable location — e.g., the starting cell
  of a maze, the source IP of a TCP connection's first packet — are
  expected to persist in the residual stream via carry-through,
  *even though* they are predictively irrelevant. The strict
  framework would predict null here; the graded framework predicts
  encoding.
- **Must be actively computed.** Features that are not in the input
  but would need to be derived from it by transformer computation —
  e.g., the total count of retransmissions over a TCP connection,
  the beat-in-measure of a music token derived by counting since
  the last bar — are expected to be absent if they are
  predictively irrelevant. Computing them costs parameters and
  gradient; if the objective does not pay for that cost, no
  pressure exists to do the computation.

We will see in §6.4 and §7 that the **carry-through half of this
split survives ex-ante testing on two different domains**: maze
starting-cell encoded as predicted by carry-through (§7.1); HTTP
first-request `size_bin` encoded as predicted by carry-through (§7.2).

The **null half does not survive**. In the maze experiment, the
predictively-required feature distance-to-goal — which the framework
expected to be encoded — turned out to be null (§7.1). In the HTTP
experiment, the predictively-irrelevant computed feature
cumulative-large-response-count — which the framework expected to be
null — turned out to be encoded (§7.2). Both directions of the null
prediction failed across the two pre-registered tests.

We retain the graded form in this section as the candidate framework
the experiments were designed to test, because it is what the
predictions files were written against. The post-test reading is
that **carry-through is the surviving mechanism** and the broader
predictive-relevance framing has been demonstrably weakened. We
return to this in §9 (joint reading of both pre-registered tests).

### 2.4 What we explicitly do not claim

It is worth being precise about three things the N-criterion is *not*
saying.

It is not a claim about whether interpretability methods (probes,
activation patching) work; those methods are how we test it.

It is not a claim that features absent from the residual stream are
not used by the model; a feature might be computed on the fly in
attention without ever appearing as a clean residual direction. The
N-criterion is about *what is represented in the residual stream*,
not about *what the model uses*. These can come apart.

It is not a claim that frontier-scale language models on natural
language follow this pattern. Whether they do is an empirical
question on which our small-model results provide indirect, weak
evidence at best.

### 2.5 Where the scientific payload lives

In Othello-style results that pre-date this paper, the *positive*
direction of the criterion ("features the model needs are encoded")
is already established. Replicating that direction in a new domain is
useful but not surprising. The risky, load-bearing claim is the
*negative* direction: features predicted to be irrelevant should be
absent even though they are decodable from the input. The music
beat-in-measure null (§6.4) is the cleanest post-hoc instance, and the
pre-registered maze starting-cell experiment (§7) is its prospective
counterpart. The maze case is the central scientific test of the
paper.

---

## 3. Background and related work

**Othello-GPT.** Li et al. (2022) trained a transformer on sequences
of moves from randomly played Othello games. They showed that a
non-linear (MLP) probe could recover the current board state from
the model's intermediate residual stream with ~94% per-cell accuracy.
Nanda et al. (2023) followed up by showing that the representation
was approximately linear, provided the basis was chosen as
"mine/yours/empty" (player-relative) rather than "black/white/empty"
(absolute color). Both papers also showed that activation patching at
the recovered direction causally steered the model's next-move
predictions.

**Why "linear" is basis-dependent.** Nanda's result is a cautionary
note for any claim that representations are "linear." Whether a
feature is linearly decodable depends on which feature parameterization
you ask about. We adopt a single parameterization per domain (defined
from the task) and stick with it; we acknowledge this is a choice.

**Superposition.** Elhage et al. (2022) and follow-up work have shown
that small transformers store more features than they have residual
dimensions, by packing distinct features into overlapping directions.
This is directly relevant to our negative predictions: superposition
predicts that features not strictly required by the task may still
appear in the residual stream as "extra inventory." The strict
N-criterion is in tension with this; the graded one accommodates it.

**Probe-causality convergence.** A descriptive probe finding that a
feature is decodable from activations is consistent with the feature
being computed *by the probe* rather than *represented by the model*.
The standard defense is activation patching ("transplant") — replace
the residual at the candidate location with a residual from another
context and measure whether the model's predictions shift toward the
donor's expected continuation. Convergence (positive probe AND
positive transplant) is the standard for taking a representational
claim seriously. We adopt this throughout.

---

## 4. Methodology

We describe the protocol once here. Each domain in §5–§6 instantiates
this protocol with domain-appropriate token streams, probe targets,
and transplant metrics.

### 4.1 Models

All models are small decoder-only transformers (nanoGPT-style), with
between 4M and 13M parameters depending on the domain. Architecture
details (layers, embedding dimension, block size) are tabulated per
domain in the appendix and committed in the project repository.

### 4.2 Three corpus conditions per domain

For each domain we construct three training corpora:

- **Real**: the natural data — actual paths in the city, actual
  moves in the game, actual ADS-B tracks, etc.
- **Within-shuffled**: the same data with token order shuffled
  *within each sequence*. This preserves the set of tokens appearing
  in any given sequence but destroys their order.
- **Global-shuffled**: all tokens reassigned by a global permutation
  across the entire corpus. This destroys both token identity and
  sequence structure.

A trained model should show degraded encoding under within-shuffled
(structure destroyed but tokens intact) and near-baseline encoding
under global-shuffled (everything destroyed). This is our equivalent
of a randomized-control trial: a representation that survives under
global-shuffled is almost certainly an artifact of probe capacity
rather than a genuine learned feature.

### 4.3 Multi-seed protocol

Every probe and transplant number we report is the mean ± standard
deviation across **five independent seeds**, where each seed
simultaneously varies:

- The untrained-control random initialization
- The positions sampled for the probe dataset
- The probe-training optimizer's random state

This is more rigorous than typical probe papers (which often report
single-seed numbers) and exposed at least one bug during this study:
four of our probe scripts had been initializing both "trained" and
"untrained" models with random weights — silently masking the
trained-vs-untrained gap on the affected runs. The multi-seed
protocol caught this discrepancy because the seed-to-seed variance
in the affected runs disagreed with the others. The fix is committed
and the affected numbers were rerun.

### 4.4 Probes

For each probe target we train two probes from each layer's residual
stream activations:

- **Linear probe**: a single dense layer with cross-entropy loss.
- **MLP probe**: a one-hidden-layer ReLU network.

We compare the two to test Nanda's strong-claim conjecture that the
representation is linearly decodable. A small gap between linear and
MLP probes is consistent with linear encoding (in the chosen basis);
a large gap suggests the encoding requires non-linear extraction.

### 4.5 Splits

Position-level split (random partition of probe positions) is the
weak baseline; a node-level (or maze-level, piece-level, flight-level
— depending on domain) split is the honest test. We report both but
treat the honest split as the primary number. Position-level is
known to inflate via per-position memorization; we have seen this
inflation in our cities data (§6.1) and use it as a routine diagnostic.

### 4.6 Activation patching ("transplant")

For each (domain × layer) we run a causal-intervention experiment.
We pick a recipient context A and a donor context B from disjoint
sequences. We run a forward pass on B's prefix and cache the residual
stream after layer L. We then run a forward pass on A's prefix, but
substitute the cached B-residual at layer L. We measure the change
in the model's next-token distribution — specifically, the probability
mass placed on B's expected continuation. A random-direction control
substitutes a residual from a third unrelated context. The standard
effect-size summary is "P(B-continuation) under transplant minus
under unpatched" and "under transplant minus under random control."

A positive transplant lift means the substituted residual carried
information the model used to generate B-like continuations. Combined
with a positive probe at the same layer, this is the convergent
descriptive + causal evidence on which a representational claim can
rest.

### 4.7 Per-layer ablation

We run the probe and transplant at *every* layer, not just one. This
lets us identify where in the network the representation lives and
how it is built up across depth. Two patterns we look for: (i) where
the probe peaks, and (ii) whether transplant at the same layer also
peaks (convergence at a specific depth).

### 4.8 Pre-registration protocol for novel domains

For the maze domain (§7), we wrote down four quantitative predictions
**before any maze model was trained** and committed them to the
project's git repository. The commit timestamp is the audit trail.
The predictions file is append-only after the commit; any post-hoc
addition appears as a clearly-marked amendment. We do not believe
this fully removes the possibility of unconscious analysis bias, but
it makes a particular class of biases (post-hoc threshold tuning,
silent prediction revision) verifiable from outside.

### 4.9 Power for nulls — a caveat

A null result on a probe is intrinsically weaker evidence than a
positive: the probe might lack the capacity to recover the feature
even if it were encoded. In domains where we report nulls, we
adopt two partial mitigations: (i) we use the same probe family and
training budget as on the positive-control features in the same
domain — i.e., the probe demonstrably works elsewhere with the same
budget; and (ii) where possible, we corroborate with a transplant
null (a causal experiment, not just a descriptive one). The cleanest
case is music beat-in-measure (§6.4), where both probe and transplant
agree on a null.

We acknowledge that this does not constitute a formal statistical
power analysis. We discuss this as a limitation in §8.5.

### 4.10 The experimental arc: pre-register, falsify, revise, re-test

This section summarizes the scientific timeline of the paper as it
actually unfolded. The reader should be able to see the arc once, here,
before reading the experimental sections in detail. Every commit hash
referenced below is verifiable from the project repository using
`git log --diff-filter=A` on the corresponding predictions file.

**Step 1. Maze pre-registration (commit `aa025b1`, 2026-05-27).** We wrote
down four quantitative predictions and committed them to the project
repository before any maze data was generated or any maze model was
trained. The load-bearing prediction was P4: the starting cell of the
maze would *not* be encoded in the residual stream, because the next-step
prediction does not need to know where the path started. This was the
strict N-criterion's null direction operationalized on a domain where
"required" is definable from the task structure alone.

**Step 2. Maze trained; P4 falsified.** We then trained the maze model and
ran the probes. P4 returned a +0.15 trained-vs-untrained gap, above the
locked falsification threshold of +0.10. The strict N-criterion was wrong
on a domain we had committed to in writing. Because the predictions were
in git before the model was trained, we cannot retroactively claim we
predicted otherwise.

**Step 3. Framework revision: introducing architectural carry-through.**
The maze falsification forced the question: why does an irrelevant feature
persist in the residual stream? The simplest reading is that the starting
cell sits at a positionally distinct slot (the first non-BOS token of the
sequence), self-attention can copy that slot's content forward across
every subsequent layer at essentially zero cost, and the next-token
training objective never penalizes carrying it. We named this mechanism
*architectural carry-through* and adopted the *graded* form of the
N-criterion (§2.3), which retains predictive relevance as the dominant
driver of encoding but adds carry-through as a second mechanism for
input-borne features.

**Step 4. HTTP pre-registration (commit `3b25ed3`, 2026-05-31).** With the
graded form in hand, we designed a second pre-registered test deliberately
to span both categories of the graded form's two-category split. We wrote
two predictions and committed them to the repository before any NASA-HTTP
data was downloaded:

- Feature A: `size_bin` of the first request. Predicted *encoded* via
  carry-through (it sits at a positionally identifiable input slot).
- Feature B: cumulative count of large responses seen so far, binned.
  Predicted *null* because it would require active aggregation across
  positions, and the next-token objective does not pay for that
  computation.

**Step 5. HTTP trained; mixed verdict.** Feature A confirmed at +0.17
gap. The carry-through mechanism survives ex-ante testing on an applied
domain at a different shape from the one it was developed against.
Feature B apparently falsified at +0.29 gap, well above the locked +0.10
threshold. The graded form's null direction failed on its first ex-ante
test.

**Step 6. Position-control follow-up (post-hoc).** We worried that the
Feature B falsification might be inflated by the model's strong positional
representation rather than reflecting genuine aggregation encoding. We
ran three follow-up probes to test this:

- The position-control probe (a purely positional target) recovers at
  +0.43 gap, larger than the Feature B gap of +0.29.
- The within-position probe (Design A, fixed k=5) gives a Feature B gap
  of +0.22.
- The residual-after-position probe (Design B3) gives a Feature B R²
  gap of +0.47.

The three controls agree on the direction. Position-correlation accounts
for part of the Feature B signal but not all. After the most conservative
control, Feature B retains a +0.22 gap, still above the locked +0.10
threshold. The graded form's null direction is falsified even after
controls.

**Step 7. Where we land.** Architectural carry-through survives 2-for-2
across the two pre-registered domains. The broader N-criterion in either
strict or graded form does not. The paper's three contributions follow
directly from this arc: (i) the pre-register, falsify, revise, re-test
loop as a methodology discipline; (ii) architectural carry-through as
the one substantively predictive claim that survived two ex-ante tests;
and (iii) position-correlation as a methodological caveat that future
probe-based work should control for.

§7 gives the full empirical detail for Steps 1 through 6. §9 (Discussion)
interprets the arc.

---

## 5. Cross-domain results: where the framework predicts well

We present results in the order of how cleanly they instantiate the
N-criterion. We lead with Othello (the prior literature's positive
control, here independently reproduced) and music (the cleanest
positive + negative pair within a single domain). We treat cities
separately in §6.1 because — as we will discuss — the cities domain
has a subtle confound that makes it less informative than it first
appears.

### 5.1 Othello (independent reproduction of the prior literature)

We trained a transformer (~4M parameters) on 50,000 random uniform
Othello games, with no information about board state in the input.
Probe target: the per-cell occupancy (empty / black / white) at each
position in the game.

| | Trained MLP probe | Untrained MLP probe | Gap |
|---|---|---|---|
| Best layer (L4), per-cell mean | **0.9399 ± 0.0012** | 0.5963 ± 0.0058 | **+0.344** |

The trained-MLP number matches Li et al. (2022)'s published ~0.94
within 0.01. The five-seed standard deviation of 0.0012 is the
tightest in our entire study — the representation is extremely
stable across seeds, sampling, and probe training. Linear probe at
the same layer: 0.8093 ± 0.0059 (gap vs untrained: +0.255).
Linear-to-MLP gap at this layer: 0.131. This is at the upper end of
what we would call "approximately linear" and we flag it as such
in §8.4.

Per-layer transplant lift peaks at L3 (+0.296 over unpatched);
the trained probe peaks at L4. Convergent: both descriptive and
causal evidence locate the board state in the same depth band.

### 5.2 Music: a clean positive + negative within one domain

We trained a transformer (~1.4M parameters) on a corpus of Bach
chorales and related Renaissance polyphony. Probe targets:

- **Voice-leading state** (the next pitch one would expect from
  the current voice's prior pitches). **N-criterion prediction:
  encoded** — next-pitch prediction requires tracking the local
  voice-leading state.
- **Chord** (the harmonic content at the current beat).
  **N-criterion prediction: weakly encoded** — chord is partially
  required for predicting voicings, partially redundant.
- **Beat-in-measure** (which of 4 beats the current note falls on).
  **N-criterion prediction: NULL** — voice-leading is locally
  predictable without knowing the beat; the next-pitch objective
  does not require beat information.

Multi-seed results on the honest piece-level split:

| Target | Trained MLP | Untrained MLP | Gap | Verdict |
|---|---|---|---|---|
| Voice-leading (transplant lift over unpatched at L2) | +0.889 ± 0.007 | — | — | **encoded + causally used** |
| Chord | 0.3035 ± 0.0202 | 0.2147 ± 0.0285 | +0.089 | weakly encoded |
| Beat-in-measure | 0.2798 ± 0.0075 | 0.2740 ± 0.0077 | +0.006 | **null** |

The beat null is doubly corroborated. The transplant experiment on
beat — using matched-voice-leading donors that differ only in
beat-in-measure — shows that transplant changes the model's
predictions *less* than the random-direction control on every
measured metric (max-|Δp|: 0.273 ± 0.005 vs 0.493 ± 0.007;
argmax-changed rate: 0.470 vs 0.965; KL: 0.69 vs 3.98). This is
not just "the probe found nothing" — it is the active causal
finding that beat-in-measure cannot be used to steer the model
even when explicitly transplanted.

This is the cleanest single result in the paper for the strict
N-criterion: the negative direction is verified by both probe and
causal experiment.

**However**, the music beat null was identified after the experiment
was run, not before. The strongest available reading is that the
beat null *motivated* the design of the pre-registered maze
experiment in §7 (where the analogous "irrelevant feature" prediction
was committed to writing before any data was collected). Beat is the
post-hoc observation; maze starting-cell is its prospective
replication.

### 5.3 Per-layer transplant: where state is computed

Per-layer transplant lift shows a domain-distinctive pattern:

| Domain (real condition) | L0 (embed) | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| Othello (50k random uniform) | +0.040 | +0.062 | +0.108 | **+0.296** | — |
| Music (voice-leading) | +0.035 | +0.813 | **+0.889** | — | — |

Two patterns are worth flagging:

- **Othello** shows the cleanest case of a representation built up
  across depth: each layer adds substantial signal, peaking at the
  deepest block.
- **Music** shows a striking L0 → L1 jump (+0.035 → +0.813).
  Voice-leading state is essentially absent from the embedding table
  itself and is constructed by the first transformer block from
  surrounding context. This is the canonical "transformer-computed
  world state" pattern.

We will contrast both with the cities case in §6.1, where the
embedding table itself already carries most of the relevant
structure.

---

## 6. Cross-domain results: where the framework is less clean

### 6.1 Cities — a partial confound

We trained transformers (~11M parameters) on tokenized street-walk
sequences from real city street networks (London, Manhattan,
Boston). Each token is a unique intersection ID; each sequence is
a walk between two intersections. The probe target is the
intersection's geographic location, evaluated via a 10×10 grid
classification on a held-out-token split.

Headline results (mean ± std over 5 seeds, MLP probe, best layer,
honest split):

| City × Condition | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| London real | 0.6423 ± 0.0545 | 0.0927 ± 0.0130 | +0.550 |
| London within-shuffled | 0.7393 ± 0.0505 | 0.0673 ± 0.0170 | +0.672 |
| London global-shuffled | 0.0993 ± 0.0248 | 0.0870 ± 0.0359 | +0.012 |
| Manhattan real | 0.6092 ± 0.0141 | 0.1033 ± 0.0078 | +0.506 |
| Boston real | 0.6667 ± 0.0205 | 0.1117 ± 0.0042 | +0.555 |

Two observations.

**First**, the within-shuffled London model scores *higher* than
real London on this probe. We note this as a **finding**, not as
part of a monotonic destruction story (the destruction story we
introduced in §4.2 explicitly predicts real ≥ within ≥ global). One
plausible reading: within-shuffling destroys graph adjacency
structure but preserves the per-route set of intersections, so the
within-shuffled model is freed from learning graph topology and
can specialize harder on geographic clustering — which is precisely
what the probe measures. We did not predict this in advance and we
do not have a clean test of the explanation. It is a real result
that complicates the framing.

**Second**, and more important: per-layer transplant lift on
London real is already +0.735 at L0 (the embed). That is, more than
three quarters of the eventual transplant signal is in the
embedding table itself, before the transformer has done any
computation. In cities, each token is a unique location, and the
embedding table's geographic clustering arises essentially from
the co-occurrence statistics of intersections in walks. This is a
real and arguably interesting fact, but it is **different evidence
than the Othello / music pattern**, where most of the encoding
is constructed by the transformer across layers. We therefore
treat the cities decompositions in this paper as evidence that
embedding-table-encoded structure can mimic the surface phenomenology
of an emergent world model, rather than as a third clean example of
one. The most-persuasive single visual in our analysis — a Procrustes-
aligned overlay of decoded vs real London — is, by our own reading,
carrying the weakest version of the thesis. We display it for
completeness but lead the cross-domain argument with Othello and music.

### 6.2 Flight phases (ADS-B)

We trained a small transformer (~0.27M parameters) on real ADS-B
flight trajectories tokenized as discretized (altitude, vertical-rate,
speed) tuples. Probe target: the current flight phase (climb,
cruise, descent, level, ground).

Results (flight-level honest split, mean ± std over 5 seeds):

| Condition | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| Real | 0.8817 ± 0.0792 | 0.7765 ± 0.1043 | +0.105 |
| Within-shuffled | 0.6785 ± 0.1834 | 0.5783 ± 0.1715 | +0.100 |
| Global-shuffled | 0.3766 ± 0.1512 | 0.3236 ± 0.1080 | +0.053 |

Per-layer transplant lift on real: +0.47 at L1 (peak in a 2-layer
model). Within-shuffled +0.30. Global-shuffled +0.00. Clean
monotonic gradient on the causal side. The probe-side untrained
baselines are very high (0.78 on real) because flight phase is
heavily determined by single-token statistics — most of the
flight-phase signal can be recovered from token marginals alone.
The trained-untrained gap (+0.105 on real) is real but small;
the transplant gradient is the more interpretable evidence here.

We treat flight as an instance of the **graded** N-criterion working:
moderate predictive relevance produces moderate encoding magnitude,
with destroyed-structure controls behaving as expected on the
causal side.

### 6.3 Symmetric-group walks

We trained transformers (~10.7M parameters) on self-avoiding walks
on the symmetric group S_8. Probe target: the per-element partial
product after the current word prefix.

Headline (word-level honest split, multi-seed at parity across 3
corpus conditions):

| Variant | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| sa real | 0.3599 ± 0.0050 | 0.3074 ± 0.0039 | +0.053 |
| sa within-shuffled | 0.2879 ± 0.0018 | 0.2778 ± 0.0039 | +0.010 |
| sa global-shuffled | 0.2913 ± 0.0040 | 0.2774 ± 0.0034 | +0.014 |

The gap on real (+0.053) is statistically distinguishable from zero
but absolutely small (~5 percentage points on a 36% trained number).
Destroyed-structure controls null out as predicted on the honest
split. We treat symgroup as a **partial signal** — useful as a
data point showing the cross-condition gradient works on a non-
spatial algebraic task, but **not** as a clean positive control.

### 6.4 The music beat null (returning to §5.2)

We covered this above as part of the music section: beat-in-measure
shows a null on both probe (trained-untrained gap +0.006) and
transplant (transplant moves the predictions *less* than the random-
direction control). This is the cleanest single instance of the
negative direction of the N-criterion in our cross-domain suite.

We acknowledge two limitations here. First, the beat null was
identified post-hoc, not pre-registered. Second, although the
positive-control on voice-leading establishes that the same
probe / same training budget can recover music structure when it
is encoded, the beat null is still a single domain — we do not
have multiple independent negative-direction confirmations from
post-hoc analysis. This motivated the pre-registered maze
experiment, which is the central scientific test of the paper and
the subject of §7.

---

## 7. Pre-registered ex-ante experiments

The pre-registered experiments are the paper's central scientific
tests. We ran two of them, on domains of deliberately different
shape: maze navigation (synthetic, spatial-graph) and HTTP log
sequences (applied, event-stream). The two domains share the
methodology but differ in vocabulary scale (67 vs 52 tokens),
tokenization (one token per cell vs four tokens per request), and
session structure. The maze experiment was completed first, and its
falsification motivated the introduction of architectural
carry-through; the HTTP experiment then provided an independent test
of the carry-through mechanism on a different domain shape.

We discuss the two experiments in turn (§7.1 maze, §7.2 HTTP), then
present a joint reading of what survives both (§7.3).

### 7.1 Maze navigation

The maze experiment was the first pre-registered test. We chose
maze navigation specifically because (i) "required" can be defined
model-independently from the task structure, (ii) the destroyed-
structure controls are unambiguous to construct, (iii) the training
fits in laptop-scale compute. The predictions were committed to the
project repository **before any maze model was trained or any maze
data generated**, with the commit timestamp serving as the audit
trail (commit hash: `aa025b1`).

#### 7.1.1 The four predictions

| # | Probe target | N-criterion verdict | Quantitative prediction (best-layer trained MLP) |
|---|---|---|---|
| P1 | Current cell row | encoded | 0.70 – 0.97; gap ≥ 0.40 vs untrained |
| P2 | Current cell col | encoded | 0.70 – 0.97; gap ≥ 0.40 vs untrained |
| P3 | Manhattan distance to goal | encoded | 0.35 – 0.75; gap ≥ 0.20 vs untrained |
| P4 | Starting cell ID | **NULL** | **gap ≤ 0.10** vs untrained (the load-bearing risky claim) |

P4 was the load-bearing prediction: the next-step prediction does
not require knowledge of where the path *started*, only of where
the path *is now* and where the path is *going*. If the model retains
starting-cell information in the residual stream despite not
"needing" it, that is direct evidence against the strict N-criterion.

#### 7.1.2 Results

We trained three maze models (~2M parameters each) on 8×8 mazes,
one per corpus condition. We ran multi-seed probes (5 seeds) and
per-layer transplant on each. Honest split is maze-level (the test
set contains mazes whose walls the model never saw during training).

#### P1 and P2: row and column

| | Trained MLP | Untrained MLP | Verdict |
|---|---|---|---|
| Row | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | prediction methodologically flawed |
| Col | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | prediction methodologically flawed |

Both row and column are recoverable at 100% accuracy by *both*
trained and untrained models. The reason is that each cell is a
unique token (token IDs 3..66, with row = (id − 3) // 8 and
col = (id − 3) mod 8), so the row and column are deterministic
functions of token identity alone. The probe trivially decodes them
from the token, not from any learned residual representation. This
makes P1 and P2 uninformative as written. We flag this as a
**predictor's error**: at the time we wrote the prediction, we did
not consider that row/col are recoverable from token identity
without any residual encoding at all. The prediction is therefore
neither confirmed nor falsified — it was never a meaningful test.

A corrected version would have used a probe target that is *not*
a deterministic function of token identity. We discuss this in §8.

#### P3: distance to goal

| | Trained MLP | Untrained MLP | Gap | Verdict |
|---|---|---|---|---|
| Best layer (real) | 0.1462 ± 0.0064 | 0.1310 ± 0.0115 | +0.015 | **predicted band missed** (predicted ≥ 0.20) |

The trained MLP probe on distance-to-goal reaches 0.1462 — barely
above the untrained baseline (0.1310), and far below the predicted
band of 0.35-0.75 with a gap of ≥ 0.20. The N-criterion's *positive*
prediction on distance was falsified on this probe.

We discuss in §8 what this likely means. Briefly: the model can
predict the next path step locally (from current cell + recent
prefix) without ever explicitly computing distance-to-goal as a
residual feature. This is the standard caveat that "the model can
use it" does not entail "the model represents it as a feature."

#### P4: starting cell (the load-bearing NULL)

| | Trained MLP | Untrained MLP | Gap | Verdict |
|---|---|---|---|---|
| Position-level best layer (L5) | 0.2156 ± 0.0035 | 0.0509 ± 0.0044 | **+0.165** | **NULL FALSIFIED** |
| Maze-level honest best layer (L5) | 0.2024 ± 0.0166 | 0.0504 ± 0.0082 | **+0.152** | **NULL FALSIFIED** |

The starting-cell-NULL prediction is falsified. The trained model's
residual stream at L5 carries information about which cell the maze
path started at, recoverable by an MLP probe at ~0.20 (vs an
untrained baseline of ~0.05 and a chance baseline of 1/67 ≈ 0.015).
The probe gap of +0.15 is well above the predicted falsification
threshold of 0.10.

The signal builds up across depth (L1: gap +0.04; L2: +0.12;
L5: +0.15), suggesting the representation is constructed by the
transformer rather than carried as an embedding-table artifact.

#### Per-layer transplant

Per-layer transplant lift on P(B's next path step) under the
three conditions:

| Layer | Real maze | Within-shuffled | Global-shuffled |
|---|---:|---:|---:|
| L0 | +0.000 | +0.000 | +0.000 |
| L1 | +0.028 | −0.003 | +0.028 |
| L2 | +0.035 | −0.003 | +0.035 |
| L3 | +0.091 | −0.001 | +0.087 |
| L4 | +0.141 | +0.007 | +0.137 |
| L5 | +0.155 | +0.017 | +0.155 |

Real and global-shuffled produce nearly identical transplant lift
profiles. Within-shuffled is near zero throughout. This pattern is
*also* unexpected from a naive destroyed-structure prediction
(which would say global-shuffled should look like within-shuffled).
The reason is that global-shuffling preserves the *sequential*
co-occurrence structure of the path (token X consistently follows
token Y across the corpus) while destroying the *spatial*
interpretation of tokens. Within-shuffling destroys the sequential
co-occurrence structure too. The transplant metric we used is
sensitive to the former and not directly to the latter. We treat
this as a finding about what the transplant metric measures (it
captures next-token sequential structure, not spatial structure)
rather than as evidence about the model.

#### 7.1.3 What the maze experiment shows

The four predictions resolved as:

| Prediction | Verdict |
|---|---|
| P1 (row encoded) | methodologically flawed — predictor's error |
| P2 (col encoded) | methodologically flawed — predictor's error |
| P3 (distance encoded) | falsified — distance is NOT encoded |
| P4 (starting cell NULL) | **falsified — starting cell IS encoded** |

This is a partial falsification of the strict N-criterion. The
risky negative-direction prediction (the central scientific bet)
was lost. The interesting interpretation is that **the information
was present in the input** (the starting cell is the first
non-BOS token of the sequence, in a positionally distinct slot)
**and the model had no incentive to actively discard it.** In other
words: the strict iff is too strong because it does not account for
information that is *already there* and merely persists. The
graded version of the criterion accommodates this:
predictively-relevant features are encoded; predictively-irrelevant
features can also be encoded if they are essentially free to carry
forward.

The pre-registration discipline did its job. The result is
informative because the prediction was wrong; if we had not
committed to the prediction in advance, we could not have
distinguished a genuine framework failure from a post-hoc story.

#### 7.1.4 What this means for the framework

We had hoped P4 would confirm. It did not. This is the most
important single result in the paper, and we report it with the
weight it deserves.

**The strict N-criterion does not survive this test.** A feature
plainly irrelevant to next-token prediction, predicted to be absent
in advance via the audit-trail mechanism, was recovered from the
trained model's residual stream with substantial signal. The
biconditional "iff" framing of the strict form predicted this
should not happen; it did.

**The graded form does survive, with a specific mechanistic
addition.** Going from the strict form to the graded form is not
just weakening — it is *naming the mechanism* by which the failed
prediction failed. The starting-cell information persists in the
residual stream because:

(i) it is present in the input at a positionally distinct slot —
the first non-BOS token of the sequence — and the positional
embedding at that index is unique;
(ii) self-attention can copy that slot's content forward across
every subsequent layer at essentially zero parameter cost;
(iii) nothing in the next-token training objective penalizes
retaining a feature that does not help prediction. Training
rewards encoding *useful* features. It does not actively discard
irrelevant ones if they are structurally easy to keep.

We call this mechanism **architectural carry-through**, introduced
in §2.3. The maze starting-cell finding is its cleanest single
instance.

**The graded form makes a new testable prediction the strict form
could not.** The strict form predicted "irrelevant feature → no
encoding," undifferentiated across feature types. The graded form
makes a sharper prediction: predictively-irrelevant features will
be encoded *iff* they are structurally trivial to carry forward.
This splits previously-undifferentiated irrelevant features into
two categories — those that ride positional identity through the
network at low cost (encoded), and those that would require active
computation by the transformer (not encoded). The next pre-
registered test of this framework should select features
deliberately to span both categories; this would be a follow-up to
the present study with stronger expected power than re-testing
either category alone.

We adopt the graded form as our working position from this point in
the paper forward, with the caveat that the next pre-registered test
(HTTP, §7.2) provided additional falsifying evidence.

### 7.2 HTTP log sequences

The HTTP experiment was the second pre-registered test. It was
designed specifically to test the two-category split — carry-through
features vs computed features — that the graded form makes after
the maze experiment. Following the same protocol, we wrote two
quantitative predictions and committed them to the project repository
**before any HTTP data was downloaded or any HTTP model was trained**.
Commit hash: `3b25ed3`.

We chose HTTP request logs (the NASA-HTTP dataset from the Internet
Traffic Archive) specifically because (i) it is an applied domain
with public clean data, (ii) it has periodic structure (sessions of
client requests) that lets us define carry-through and computed
features cleanly, (iii) it is at a different shape from the maze
experiment so confirmation would be independent evidence.

#### 7.2.1 The two predictions

Per-field tokenization: each HTTP request is encoded as four
consecutive tokens — `(method, path_category, status_bucket,
size_bin)` — so sequences expose each field as its own positionally
identifiable slot. Vocabulary: ~52 tokens. Sessions: 3–30 requests,
grouped by client host with 30-min idle timeout.

| # | Probe target | Category per graded framework | Predicted outcome |
|---|---|---|---|
| P1 | Feature A: `size_bin` of the FIRST request | In-input at distinct slot (the size_bin token at sequence position 4); irrelevant to predicting later requests | **Encoded** via carry-through. Best-layer trained MLP gap ≥ 0.10. |
| P2 | Feature B: cumulative count of "large-response" tokens (`size_bin` ≥ 5) in prefix, binned {0, 1, 2+} | Must-be-actively-computed (requires aggregation across positions); irrelevant to next-request prediction | **Null**. Best-layer trained MLP gap ≤ 0.10. |

Cross-feature prediction: A gap ≥ B gap. This is the *structural*
form of the carry-through differentiation.

#### 7.2.2 Results

Setup as actually run: NASA-HTTP July + August 1995 archives,
~3.46M requests, 226k retained sessions after filtering, 8.08M
training tokens per condition, vocab 52, model n_layer=4 / n_head=4 /
n_embd=128 / ~0.81M params, 5,000 training iterations with
dropout=0.20. Training converged val_ppl ≈ 1.54-1.58 with train/val/gen
tightly matched — no overfit observed. Probes used class-balanced
sampling for Feature A (an unforeseen majority-class issue surfaced
during the smoke test — the locked file did not specify sampling
strategy, and balanced sampling was a methodological choice made at
probe time, documented as an amendment to the predictions file).

**Feature A** — `size_bin` of first request (real condition,
session-level honest split, 5-seed mean ± std):

| Layer | Trained MLP | Untrained MLP | MLP gap |
|---|---|---|---|
| embed | 0.317 ± 0.04 | 0.249 ± 0.03 | +0.068 |
| L0 | 0.367 ± 0.05 | 0.240 ± 0.02 | +0.127 |
| L1 | 0.391 ± 0.05 | 0.237 ± 0.02 | +0.154 |
| L2 | 0.405 ± 0.04 | 0.241 ± 0.03 | +0.164 |
| **L3** | **0.410 ± 0.04** | **0.243 ± 0.03** | **+0.168** |

Best-layer trained-vs-untrained MLP gap: **+0.168** (predicted ≥ 0.10).
**P1 confirmed.**

Cross-condition: within-shuffled gap +0.134 (in predicted band
[0.00, 0.15]), global-shuffled gap +0.163 (in predicted band
[0.05, 0.20]). All three conditions confirm the carry-through
prediction on Feature A.

**Feature B** — cumulative large-response count, binned (real
condition, session-level honest split):

| Layer | Trained MLP | Untrained MLP | MLP gap |
|---|---|---|---|
| embed | 0.692 ± 0.01 | 0.585 ± 0.01 | +0.107 |
| L0 | 0.829 ± 0.01 | 0.594 ± 0.01 | +0.235 |
| L1 | 0.872 ± 0.01 | 0.596 ± 0.01 | +0.276 |
| **L2** | **0.888 ± 0.004** | **0.597 ± 0.01** | **+0.291** |
| L3 | 0.876 ± 0.01 | 0.597 ± 0.01 | +0.279 |

Best-layer (and max-layer) trained-vs-untrained MLP gap: **+0.291**
(predicted ≤ 0.10). **P2 falsified by ~2.9× the threshold.**

Cross-condition: within-shuffled max gap +0.236, global-shuffled max
gap +0.303. All three conditions falsify, in the same direction.

**Cross-feature ordering**: Feature A gap = +0.168; Feature B gap =
+0.291. A < B. **Cross-feature ordering falsified.**

Per the predictions file's own framework-level verdict, "if ≥ 2 of
the above hold, the revised framework needs substantial revision on
this class of applied domains." Two of the three (P2 and the
cross-feature ordering) fall the wrong way. The HTTP experiment
falsifies the graded form's substantive claims, with the carry-through
half (P1) confirming cleanly.

#### 7.2.3 Why the Feature B falsification has two readings

The Feature B result has a methodological wrinkle worth surfacing.
Position-in-session strongly correlates with cumulative-large-response
count: later sessions positions tend to have higher counts. The
trained model develops sharper positional representations than the
untrained model (it needs to predict which of {method, path,
status, size_bin} comes next, which requires knowing position mod 4).
The probe may be reading position rather than aggregation.

Evidence consistent with this reading:
- Untrained baseline is already at 0.60 (well above the 0.33 chance
  for 3 classes), suggesting positional info alone gives substantial
  signal.
- **Global-shuffled produces an unchanged or larger gap** (+0.303 vs
  +0.291 on real). Under global-shuffled, the token-alphabet
  permutation destroys identity-based attention to "large-response
  tokens." If the trained model were actually identifying these
  tokens and counting them, this gap should collapse. It does not.
  Position-as-proxy is the more consistent explanation.

We document this as an unresolved interpretive ambiguity. A
follow-up control — probing a purely positional feature like
"request_idx_in_session, binned" with the same probe protocol —
would resolve it: if the position-control gap matches the
Feature B gap, the falsification is position-confounded and the
graded framework's claim about computed features remains testable
(just not by this probe target). If a residual gap remains after
controlling, the framework's claim was wrong about even simple
aggregations being absent. We did not pre-register this control;
it was identified post-hoc and is the natural follow-up.

#### 7.2.4 Position-control follow-up

We ran the natural follow-up control proposed in §7.2.3, plus two
complementary designs, after observing the Feature B ambiguity. None of
these were pre-registered; they are documented as post-hoc methodology
amendments.

**Position-control probe.** Probe target: `request_idx_in_session`,
binned into 4 classes (idx ≤ 3 → 0; ≤ 6 → 1; ≤ 12 → 2; else 3). This
is a deterministic function of position — purely positional with no
content-dependent component. The probe protocol is otherwise identical
to §7.2.2.

Headline (best-layer trained MLP, session-level honest split, 5 seeds):

| Condition | Trained MLP @ L2 | Untrained MLP best | Gap |
|---|---|---|---|
| Real | 0.836 ± 0.009 | 0.409 ± 0.014 | **+0.427** |
| Within-shuffled | 0.945 ± 0.007 | 0.404 ± 0.026 | **+0.541** |
| Global-shuffled | 0.828 ± 0.011 | 0.424 ± 0.011 | **+0.404** |

The position-control gap exceeds the Feature B gap in every condition.
Pure position information is recovered at +0.43 by the trained model;
Feature B at +0.29. This is the cleanest available evidence that the
Feature B gap is at least partly inflated by positional encoding.

**Within-position probe (Design A) at fixed k=5.** We restrict the
Feature B probe to the size_bin slot of request 5 (the position with
the highest probe-data volume and a non-degenerate class distribution,
47% / 20% / 33%). At fixed k=5, the positional embedding contribution
is constant across examples.

| Condition | Trained MLP @ L2 | Untrained MLP @ L2 | Gap |
|---|---|---|---|
| Real | 0.904 ± 0.006 | 0.685 ± 0.014 | **+0.220** |
| Within-shuffled | 0.884 ± 0.003 | 0.685 ± 0.014 | **+0.199** |
| Global-shuffled | 0.825 ± 0.008 | 0.685 ± 0.014 | **+0.140** |

After holding position literally constant, Feature B retains a positive
trained-vs-untrained gap in all three conditions, with a monotonic
real > within > global ordering.

**Residual-after-position probe (Design B3).** We fit a per-position
empirical baseline P(F | k) on the probe-train split (Laplace-smoothed),
compute the residual `y_onehot − baseline(k)` on both train and test,
and train a regression probe (linear and MLP) with MSE loss to predict
the residual from activations. The metric is R² on the test residual.

| Condition | Trained R² @ L2 | Untrained R² best layer | Gap |
|---|---|---|---|
| Real | +0.678 ± 0.026 | +0.210 ± 0.012 | **+0.468** |
| Within-shuffled | +0.639 ± 0.024 | +0.210 ± 0.012 | **+0.429** |
| Global-shuffled | +0.431 ± 0.022 | +0.210 ± 0.012 | **+0.222** |

After statistically controlling for per-position baseline, Feature B
retains a substantial trained R² gap in all three conditions, with the
same monotonic gradient.

**Joint interpretation.** Both Design A and Design B3 agree across all
three conditions: Feature B is encoded above untrained baseline *even
after controlling for position*. The original §7.2.3 ambiguity is
partially but not fully resolved by the position confound:

- The position-control gap (+0.43 real) exceeds the original Feature B
  gap (+0.29), confirming that positional encoding accounts for part
  of the apparent Feature B signal.
- The within-position gap (+0.22 real) is smaller than the original
  Feature B gap, consistent with position contributing approximately
  +0.07 to the original measurement.
- However, +0.22 at fixed position k=5 is still well above the
  pre-registered falsification threshold of 0.10. The framework's null
  prediction on Feature B is *still* falsified, by approximately 2.2×
  the threshold rather than the original 2.9×.
- The B3 residual probe at R² gap +0.47 confirms the same direction
  under a fully different probe construction.

The cross-condition gradient (real > within > global) survives both
controls. This is significant: even after literal and statistical
position control, Feature B encoding strength tracks training-corpus
structure. The encoding is structure-dependent, not just a position
artifact.

#### 7.2.5 What the HTTP experiment shows

Carry-through confirmed independently on an applied domain at a
different shape (P1 with gap +0.17 over predicted ≥ 0.10 threshold).
The graded framework's null-on-computed-features claim falsified
(P2 with gap +0.29 over the ≤ 0.10 threshold). The position-control
follow-up (§7.2.4) shows that position-correlation accounts for part
but not all of the apparent gap: after controlling for position via
both Design A (within-position at fixed k=5) and Design B3
(residual-after-position), Feature B retains a +0.22 trained-vs-
untrained gap, still well above the pre-registered 0.10 threshold.

The maze experiment had previously falsified the strict form and
motivated the addition of carry-through. The HTTP experiment falsifies
further substantive claims of the graded form while independently
confirming the carry-through mechanism that was added in response to
the maze data.

### 7.3 Joint reading of the two pre-registered tests

Across both experiments, the framework's track record on its risky
ex-ante predictions is uneven:

| Pre-registered claim | Maze result | HTTP result |
|---|---|---|
| Carry-through (input-slot irrelevant feature encoded) | ✓ predicted by the mechanism added after maze (starting-cell encoded) | ✓ predicted ex-ante before any HTTP data; confirmed (Feature A) |
| Null on computed irrelevant feature | n/a | ✗ falsified (Feature B encoded with gap +0.29) |
| Specific positive (predictively-relevant feature encoded) | ✗ falsified (distance to goal NOT encoded) | n/a |

**Carry-through: 2-for-2.** The mechanism predicted correctly when
tested in the maze starting-cell case (where it was introduced as
the explanatory mechanism), and again when tested in the HTTP
Feature A case (where it was a forward-looking pre-registered
prediction on a domain not used to develop the mechanism). This is
the kind of cross-domain ex-ante validation that gives the carry-
through claim meaningful empirical weight.

**The broader "predictive relevance drives encoding" framing: 0-for-3
on risky predictions.** Sometimes predictively-required features are
absent (maze distance-to-goal). Sometimes predictively-irrelevant
features are present even after the obvious position-correlation
confound is controlled for (HTTP cumulative count, +0.22 at fixed
position k=5 per §7.2.4). The direction of failure is different in
each case, but the pattern is consistent: the broader framing was
too coarse to predict the observed empirical pattern.

**Position-correlation as a probe confound.** The HTTP Feature B
result surfaced a specific methodological issue: features that
correlate strongly with positional information will appear encoded
in any trained model that develops sharp positional representations,
regardless of whether the model is actually computing the feature.
This is a contribution future probe-based interp work should
control for; the position-control follow-up in §7.2.4 demonstrates
that this confound accounts for approximately +0.07 of the +0.29
Feature B headline, with +0.22 remaining and still above the
pre-registered 0.10 falsification threshold.

What we adopt going forward (§9): we report what the pre-registered
tests showed without further revising the framework mid-paper. The
carry-through claim survives ex-ante testing on two domains; the
broader N-criterion does not. The methodology, the mechanism, and
the failure mode are the three things the paper contributes.

---

## 8. Limitations

We flag the following limitations explicitly.

### 8.1 Scale and external validity

All models in this study are 4M–13M parameters on synthetic or
semi-synthetic data. Whether the N-criterion (in any form) predicts
the representational content of frontier-scale language models on
natural-language data is **not** established by this study. Our
opening framing scopes the question to small models on structured
tasks accordingly.

### 8.2 "Required" remains imperfectly operationalized

We define "required" from task structure (model-independently)
wherever we can: for maze, P4's null claim followed from the
observation that the next path step is determined by current cell
and goal alone; for music, the beat null followed from the
observation that voice-leading is locally predictable without
beat. But this still leaves room for proxies, shortcuts, and
partial reliance. A feature can be *useful* without being
*necessary*. The graded N-criterion in §2.3 absorbs this, at the
cost of being weaker and less surprising.

### 8.3 Probe nulls have weak power

A null probe result is consistent with (a) the feature being
unencoded, (b) the feature being encoded non-linearly or in
attention (not the residual stream), or (c) the probe lacking
capacity. We partially mitigate via (i) using the same probe
family and budget that works on the positive controls in the same
domain, and (ii) corroborating nulls with transplant where possible
(music beat is the cleanest case). We do **not** offer a formal
statistical power analysis. We acknowledge this as the most
defensible single criticism of our null results.

### 8.4 Linear vs MLP gap at the upper end

We report a maximum linear-vs-MLP gap of 0.13 across our positive
cases. The Othello validation case is at 0.131 — at the upper end
of what we describe as "approximately linear." For the strict
Nanda strong claim (the representation is *linear*), 0.131 of
absolute accuracy on a 0.94-trained probe is ~14% of the available
headroom. We do not claim our results establish linear encoding in
the strict Nanda sense; we claim that the representation is
substantially recoverable by linear probes, consistent with linear
encoding in some basis. (Recall that linear decodability is
basis-dependent: Nanda's 2023 result was that Othello board state
is linearly decodable in the player-relative basis but not in the
absolute-color basis. We have not exhaustively searched basis
choices.)

### 8.5 The within-shuffled cities oddity

London within-shuffled scores higher on the cities probe than
London real (§6.1). This is contrary to the naive monotonic-
destruction prediction. We offer a plausible post-hoc reading
(within-shuffling frees the model to specialize on geographic
clustering by removing the graph-topology constraint) but we did
not predict this in advance and we do not have a clean test of the
explanation. It is a real result that we surface here rather than
explain away.

### 8.6 One pre-registered domain

The pre-registered maze experiment is one ex-ante test. The strong
form of the pre-registration argument requires multiple
independent tests; one falsified prediction does not allow us to
fully rule out post-hoc story-telling about other domains. We view
the maze experiment as a *first* prospective test of the
N-criterion's negative direction and a model for future
pre-registrations rather than as a conclusive cross-domain
demonstration.

### 8.7 P1 and P2 of the maze predictions were methodologically flawed

We acknowledge openly that two of the four maze predictions (row
and column) were not meaningful tests, because the probe target
was a deterministic function of token identity. This is a
predictor's error, caught on inspection of the data rather than by
the audit-trail mechanism itself. It does not change the falsification
verdict on P4; it does illustrate that pre-registration is only
as good as the prediction's construct validity.

### 8.8 Mechanistic depth is coarse

We use residual-stream probes and full-residual transplant. We do
not perform path patching, composition-score analysis, or
attention-head-level decomposition. The N-criterion as stated is
about *encoding existence at the residual-stream level*, not about
circuit structure. Going from "the feature is represented" to "this
specific circuit computes it" requires finer tools we have not
applied. We treat this as a scope choice, not an oversight.

---

## 9. Discussion

### 9.1 What the strict N-criterion does not survive

The pre-registered falsification on the maze starting-cell prediction
is the central empirical fact of this paper. The strict
biconditional — F is encoded iff next-token prediction requires F —
does not hold. A feature that is plainly irrelevant to the next-token
objective, predicted to be absent in advance, was recovered in the
residual stream of the trained model with substantial signal
(probe gap +0.15 over untrained, well outside our pre-registered
falsification threshold, at depth L5 of a 6-layer model). The
information persisted because the input made it trivially carryable,
not because the objective demanded it.

### 9.2 What the graded form does say

The graded form — predictive relevance is the **dominant driver** of
which features are linearly recoverable from the residual stream,
with architectural carry-through as a second mechanism for input-
borne features — remains consistent with the data. Music beat-in-
measure was correctly predicted to be absent (null on both probe and
transplant). Voice-leading was correctly predicted to be present.
Chord was correctly predicted to be weakly present. Symgroup partial
product shows a partial-signal pattern. Even the cities embedding-
table result is consistent with predictive co-occurrence statistics
driving emergent geographic clustering.

The trade-off should be made plain. The strict form is more
informative and more falsifiable; the graded form is more
defensible. The graded form is *not* simply the strict form
weakened — it identifies a specific mechanism (architectural carry-
through) that the strict form did not predict, makes that mechanism
testable, and uses it to explain previously incompatible results.
We do not have a way to make the strict form survive the maze
evidence. We therefore adopt the graded form as our working
hypothesis.

### 9.3 Beat null and starting-cell encoding are jointly informative

The music beat-in-measure result (§5.2 / §6.4) and the maze
starting-cell result (§7) appear superficially in tension: both
features are predictively irrelevant, but one was found null and
the other was found encoded. Under the strict form, this would be a
genuine inconsistency requiring some additional ad-hoc story to
reconcile.

Under the graded form, the two results are *jointly informative*
about the architectural carry-through mechanism:

- **Music beat-in-measure** is predictively irrelevant *and* not
  present at any single positional slot of the input. To recover
  beat, the model would have to count tokens since the last bar
  marker — an active computation across multiple positions. Carry-
  through does not apply. The graded framework predicts null.
  Observed: null on both probe and transplant.
- **Maze starting-cell** is predictively irrelevant *and* present
  at a single positionally distinct slot (the first non-BOS token,
  index 1). Carry-through applies. The graded framework predicts
  persistence. Observed: encoded at a probe gap of +0.15 over the
  untrained baseline, depth-built up to L5.

The two results together do exactly the work pre-registration is
supposed to do: they identify what the framework was wrong about
(the strict iff in its undifferentiated form), point at a specific
mechanism (carry-through), and provide one positive and one
negative case the mechanism correctly differentiates. The mechanism
was not extracted from a single case; it was the simplest reading
consistent with both.

This joint-reading move — the maze experiment's value comes in
part from how it explains a *previously post-hoc* observation
(music beat) — is also how a falsified pre-registered prediction
can earn back some of its scientific value. The starting-cell
falsification did not just lose a bet. It revealed why the beat
null could be predicted *in advance* by the revised framework, in a
way the strict framework could not have done.

### 9.4 A new risky prediction the revised framework should make

A revised framework that fits the existing data without making
new risky predictions is not scientifically informative. The graded
form does make such predictions, and they are concrete enough to
be pre-registered for a future experiment.

The next test we propose for the framework — and would commit to
in writing before running, following the same audit-trail protocol
used in §7 — is a domain in which we pre-register predictions for
*two* predictively-irrelevant features chosen deliberately to span
the carry-through split:

- **Feature A**: predictively irrelevant, present at a positionally
  distinct input slot. Predicted: encoded.
- **Feature B**: predictively irrelevant, not present at any single
  input slot, would require active computation to recover.
  Predicted: null.

Concretely, a TCP-state-recovery task would offer source IP of the
connection initiator (present at the first packet, irrelevant
post-handshake) as Feature A and the connection's total
retransmission count (must be computed across packets, irrelevant
to next-packet prediction) as Feature B. A program-execution-trace
task or a multi-turn dialog-state task could be structured
similarly. The choice of domain matters less than the construction:
two irrelevant features chosen to differ only in whether
carry-through applies.

A confirm on both predictions would earn the graded framework
real credibility on its negative direction — including a specific
positive prediction it could only have been making *because* the
maze starting-cell result happened. A falsification on either
would require further revision and would, in turn, be informative
about which part of the mechanism is wrong. Either outcome is more
useful than not running the test. This is the natural follow-up to
the present study and is explicitly out of scope for it.

### 9.5 What the cities case does and does not show

The cities Procrustes-aligned overlay is the most visually striking
result in our study. It is also, by our own analysis, the weakest
support for an emergent computed world model: most of the
geographic structure is in the embedding table, before any
transformer block has acted. Embedding-table-encoded structure
arising from token co-occurrence statistics is interesting
(particularly the within-shuffled finding that destroying graph
adjacency leaves geographic clustering intact and possibly even
sharpens it) but is qualitatively different evidence than the
Othello and music cases, where most of the encoding is built up
across transformer layers. We surface the cities result as
"evidence that domains exist where embedding-table structure can
mimic the surface phenomenology of an emergent world model" rather
than as a third clean example of one. Future work might
investigate when embedding-table structure suffices for next-token
prediction and when the transformer is forced to compute the state
internally.

### 9.6 Why the pre-registration matters even when the prediction loses

The maze experiment cost us our central scientific bet. The
starting-cell NULL was the cleanest single prediction in the paper
and it falsified. We report this without dressing it up. We argue
the experiment is more valuable *because* it falsified than it
would have been with a confirm: the audit trail of the
pre-registration makes it impossible to retroactively re-interpret
the framework's prediction to match the data. The result forces an
honest revision (the graded form in §2.3), which we adopt.

The negative result is also informative *about* the mechanism. The
fact that starting-cell information persists in the residual stream
because it is structurally easy to carry forward — a positionally
distinct early token in a self-attention architecture — is a more
specific empirical claim than the strict N-criterion would have
allowed. It points at a mechanistic story (architectural carry-
through of input features) that the strict framework would have
dismissed.

### 9.7 What this implies for interpretability methodology

We do not claim our methodology is sufficient. We do claim it is a
**reasonable bar** for representational-content claims about small
transformers. Specifically: a representational claim should be
supported by (i) multi-seed mean ± std on the honest split, (ii)
probe and transplant agreement at the same layer, (iii) destroyed-
structure controls behaving as expected, and (iv) pre-registered
predictions where the question is novel enough that post-hoc
story-telling is a real risk. None of these are individually
novel; the combination is rarer than the literature suggests.

### 9.8 What we do not claim

We do not claim the N-criterion (in either form) explains the
behavior of frontier-scale language models on natural language. We
do not claim that our cities, Othello, flight, or music models
have a "world model" in any philosophically loaded sense. We claim
only that the residual streams of these specific small models on
these specific tasks contain linearly recoverable representations
of specific features, that we have tested those representations
with reasonable rigor, and that the pattern across domains is
consistent with the graded N-criterion subject to the limitations
in §8.

---

## 10. Conclusion

We tested the hypothesis that small next-token transformers
spontaneously represent features that are predictively relevant
to their training objective, across five domains uniformly, with
a sixth domain (maze navigation) ex-ante pre-registered. We
found:

- The strict biconditional form of the hypothesis is **falsified**
  by the pre-registered maze starting-cell experiment.
- A graded form — predictive relevance is the dominant driver of
  what is linearly recoverable from the residual stream — survives
  with substantial qualifications, and is the version of the
  claim we adopt going forward.
- Across positive controls (Othello, music voice-leading, music
  chord), the multi-seed probe + transplant + per-layer protocol
  reproduces and tightens prior literature results.
- Across negative controls (music beat, intended maze starting-
  cell), the picture is mixed: beat behaves as predicted (null in
  both probe and transplant); starting-cell falsifies the
  prediction.
- Domains differ in how the encoding is constructed: Othello
  builds it across depth, music computes it sharply at L0→L1,
  cities pre-encodes it at the embedding table from token co-
  occurrence statistics. These are qualitatively different
  mechanisms with qualitatively different evidential weight for
  the world-model claim.

The methodological contribution we believe matters most is the
combination of multi-seed reporting, probe + transplant
convergence, per-layer ablation, destroyed-structure controls, and
pre-registration with a git audit trail. We hope this combination
becomes a reasonable bar for the small-model branch of the
mechanistic-interpretability literature.

---

## References

(Bibliography to be added in the final version. Key works
referenced in text:)

- Li et al. (2022). *Emergent World Representations: Exploring a
  Sequence Model Trained on a Synthetic Task.*
- Nanda et al. (2023). *Emergent Linear Representations in
  World Models of Self-Supervised Sequence Models.*
- Elhage et al. (2022). *Toy Models of Superposition.*

---

## Appendix A: Reproducibility

All code, data preparation pipelines, training configurations,
probe/transplant scripts, multi-seed runners, and the
pre-registered predictions file are publicly versioned at the
project repository. Each headline number reported in this paper
can be reproduced end-to-end on a laptop with Apple MPS in under
8 hours; the larger-scale conditions discussed in §8.1 require GPU
rental.

The pre-registration audit trail is verifiable via
`git log --diff-filter=A predictions/predictions_maze_navigation.md`,
which shows the commit hash `aa025b1` predating any maze model
training, data generation, or probe run.

## Appendix B: Per-domain configurations

(Tabular summary of model architecture, training hyper-parameters,
and corpus statistics per domain, to be included in the final
version.)
