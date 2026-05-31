# Results — HTTP log sequences

Companion to `predictions/predictions_http_log_sequences.md`
(locked at commit `3b25ed3` on 2026-05-31, before any NASA-HTTP
data was downloaded or any HTTP model was trained or any probe
was run).

## TL;DR

- **Prediction 1 (Feature A — carry-through encoded)**: ✓ **CONFIRMED** in
  all 3 conditions.
- **Prediction 2 (Feature B — null on computed feature)**: ✗ **FALSIFIED**
  in all 3 conditions.
- **Cross-feature ordering (A gap ≥ B gap)**: ✗ **FALSIFIED** in all 3
  conditions.

Per the locked file's framework-level verdict ("if ≥ 2 of the
above hold, the revised framework needs substantial revision on this
class of applied domains"), the graded N-criterion as stated does
not survive on the HTTP domain. The carry-through mechanism is
preserved; the "null on computed features" claim is too strong.

A methodological caveat applies to Feature B's falsification:
position-in-session strongly correlates with cumulative-large-
response count in this dataset, and the trained model's better
positional representation may account for part of the observed
gap. A follow-up control experiment is proposed.

## Setup as actually run

- **Data**: NASA-HTTP July + August 1995 access logs.
- **Sessions**: 226k retained (length 3-30 requests, 30-min idle).
- **Tokenization**: per-field, 4 tokens per request.
- **Vocab**: 52 tokens.
- **Train tokens**: 8,079,516 (same across all 3 conditions).
- **Model**: per locked config — n_layer=4, n_head=4, n_embd=128,
  block_size=128, dropout=0.20, ~0.81M params.
- **Training**: 5,000 iters; converged val_ppl ≈ 1.54-1.58 across
  all 3 conditions; no overfit observed (train/val/gen tightly
  matched throughout).
- **Probe positions**: size_bin slots of request 2+ (per locked file).
- **Sampling**: class-balanced for Feature A (8-class skewed
  distribution); proportional for Feature B (already balanced).
- **Seeds**: 5, outermost loop varying everything (per protocol).
- **Split**: session-level honest split is the headline; position-
  level reported in side tables.

## Class distributions in side table (probe positions)

| Target | Distribution | Majority |
|---|---|---|
| Feature A: first_request_size_bin (8 classes, sb 0–8 with sb 1 unobserved) | sb 0: 6.0%, sb 2: 0.1%, sb 3: 1.0%, sb 4: **75.1%**, sb 5: 15.2%, sb 6: 0.5%, sb 7: 0.02%, sb 8: 2.0% | **0.751** (sb 4) |
| Feature B: cumulative_large_response_binned (3 classes) | bin 0: 29.6%, bin 1: 24.3%, bin 2+: 46.1% | 0.461 (bin 2+) |

## Prediction 1 verdict — Feature A: `first_request_size_bin`

Predicted: best-layer trained MLP gap ≥ 0.10 over untrained on the
session-level honest split.

### Real condition (5-seed mean ± std, session-level)

| Layer | Trained linear | Trained MLP | Untrained linear | Untrained MLP | MLP gap |
|---|---|---|---|---|---|
| embed | 0.319 ± 0.04 | 0.317 ± 0.04 | 0.245 ± 0.03 | 0.249 ± 0.03 | +0.068 |
| L0 | 0.323 ± 0.04 | 0.367 ± 0.05 | 0.237 ± 0.02 | 0.240 ± 0.02 | +0.127 |
| L1 | 0.330 ± 0.04 | 0.391 ± 0.05 | 0.235 ± 0.02 | 0.237 ± 0.02 | +0.154 |
| L2 | 0.341 ± 0.05 | 0.405 ± 0.04 | 0.241 ± 0.02 | 0.241 ± 0.03 | +0.164 |
| **L3** | 0.337 ± 0.04 | **0.410 ± 0.04** | 0.241 ± 0.03 | **0.243 ± 0.03** | **+0.168** |

**Best-layer trained MLP**: 0.4103 ± 0.0404 at L3.
**Best-layer untrained MLP**: 0.2428 ± 0.0277 at L3.
**Gap at best layer**: **+0.168** (predicted ≥ 0.10).

**Verdict on Prediction 1 (real)**: **✓ CONFIRMED** (+0.168 ≥ 0.10).

### Cross-condition (Feature A best-layer trained-vs-untrained MLP gap)

| Condition | Trained MLP | Untrained MLP | Gap | Predicted band | In band? |
|---|---|---|---|---|---|
| real | 0.410 ± 0.04 | 0.243 ± 0.03 | **+0.168** | gap ≥ 0.10 | **✓** |
| within-shuffled | 0.426 ± 0.08 | 0.291 ± 0.03 | **+0.134** | gap ∈ [0.00, 0.15] | **✓** |
| global-shuffled | 0.402 ± 0.06 | 0.239 ± 0.02 | **+0.163** | gap ∈ [0.05, 0.20] | **✓** |

All three conditions confirm Feature A's carry-through prediction. The
architectural-carry-through mechanism named in the report's §2.3
generalizes from the maze synthetic-spatial setup to the applied
HTTP event-stream setup at matching scale.

## Prediction 2 verdict — Feature B: `cumulative_large_response_binned`

Predicted: best-layer trained MLP gap ≤ 0.10 over untrained at every
layer, on the session-level honest split.

### Real condition (5-seed mean ± std, session-level)

| Layer | Trained linear | Trained MLP | Untrained linear | Untrained MLP | MLP gap |
|---|---|---|---|---|---|
| embed | 0.658 ± 0.01 | 0.692 ± 0.01 | 0.553 ± 0.01 | 0.585 ± 0.01 | +0.107 |
| L0 | 0.748 ± 0.01 | 0.829 ± 0.01 | 0.555 ± 0.01 | 0.594 ± 0.01 | +0.235 |
| L1 | 0.785 ± 0.01 | 0.872 ± 0.01 | 0.554 ± 0.01 | 0.596 ± 0.01 | +0.276 |
| **L2** | **0.807 ± 0.01** | **0.888 ± 0.004** | 0.556 ± 0.01 | 0.597 ± 0.01 | **+0.291** |
| L3 | 0.794 ± 0.01 | 0.876 ± 0.01 | **0.556 ± 0.01** | **0.597 ± 0.01** | +0.279 |

**Best-layer trained MLP**: 0.8884 ± 0.0043 at L2.
**Best-layer untrained MLP**: 0.5974 ± 0.0101 at L2.
**Gap at best layer**: **+0.291**.
**Maximum gap across layers**: +0.291 at L2.

**Verdict on Prediction 2 (real)**: **✗ FALSIFIED** (+0.291 > 0.15 threshold).

### Cross-condition (Feature B max MLP gap across layers)

| Condition | Trained MLP | Untrained MLP | Max gap | Predicted | In band? |
|---|---|---|---|---|---|
| real | 0.888 ± 0.004 | 0.597 ± 0.01 | **+0.291** | gap ≤ 0.10 | **✗** |
| within-shuffled | 0.826 ± 0.01 | 0.590 ± 0.01 | **+0.236** | gap ≤ 0.10 | **✗** |
| global-shuffled | 0.896 ± 0.004 | 0.593 ± 0.01 | **+0.303** | gap ≤ 0.10 | **✗** |

All three conditions falsify. Notably, global-shuffled produces a
**larger** gap than real (+0.303 vs +0.291). This is informative: under
global-shuffled, the token-alphabet permutation should destroy the
identity-based signal that would let the model "attend to large-response
tokens specifically." The fact that the gap is undiminished tells us
the trained model is not actually identifying large-response tokens
and counting them. It is using some other signal that correlates
strongly with the cumulative count.

## Cross-feature prediction verdict

Predicted: Feature A best-layer gap ≥ Feature B best-layer gap (real).

| Feature | Best-layer trained-untrained MLP gap (real) |
|---|---|
| A (carry-through) | +0.168 |
| B (computed null) | +0.291 |

A − B = −0.123 (B is larger than A).

**Verdict on cross-feature ordering**: **✗ FALSIFIED** in all 3
conditions (same direction).

## Framework-level verdict

Per the locked file:

| Claim | Verdict |
|---|---|
| 1. Carry-through prediction | ✓ confirmed |
| 2. Null on computed feature | ✗ falsified |
| 3. Cross-feature ordering | ✗ falsified |

**2 of 3 fall the wrong way. The graded N-criterion as stated does not
survive on this domain.** Per the locked file's own falsification
table, this requires substantial revision on this class of applied
domains.

## What the data is actually telling us

The carry-through half of the graded framework (Prediction 1)
confirms cleanly. Architectural carry-through is real, observable on
an applied domain at the same scale as the maze synthetic case, and
behaves about as predicted under destroyed-structure controls.

The null half (Prediction 2) does NOT confirm. The trained model
recovers cumulative-large-response count at 0.89 MLP accuracy on the
honest session-level split, well above the 0.60 untrained baseline.
The gap is +0.29, 2.9× the falsification threshold.

Two readings of why the null failed:

### Reading A: simple aggregations ARE encoded

Self-attention naturally supports "sum-over-tokens-matching-pattern-X"
operations. A single head can attend uniformly to all prior size_bin
tokens; the output is a weighted average of their value vectors. If
the value projections expose size_bin magnitude as a scalar dimension,
the resulting residual stream component is effectively a running
"large-response intensity." This requires zero gradient pressure to
form — it's a default attention pattern.

Under this reading, the graded framework's "null on computed features"
prediction was too strong. **Aggregations expressible by single
attention heads are not "computed" in the sense the framework
meant** — they're closer to free side effects of the architecture.
The framework needs a third category: features computable by simple
attention-pattern aggregation should also be expected to appear in
the residual stream, regardless of predictive relevance.

### Reading B: position-confounded probe

Position-in-session strongly correlates with cumulative-large-response
count: late-session positions tend to have higher counts. The trained
model develops sharper positional representations than the untrained
model (it needs to predict which of {m, p, s, sz} comes next, which
requires knowing position mod 4). The probe may be reading position,
not aggregation.

Evidence consistent with Reading B:
- The untrained baseline is already at 0.60 (well above the 0.33 chance
  for 3 classes), suggesting positional info alone gives substantial
  signal.
- Global-shuffled (with token identity destroyed) preserves the gap
  unchanged or higher (+0.30 vs +0.29 on real). The model cannot be
  "identifying large-response tokens" in global-shuffled since their
  identity is permuted, yet the recovery is undiminished. This is
  more consistent with position-as-proxy than with active aggregation.

### Why both readings are informative

Reading A says the framework's mechanism inventory is incomplete.
Attention is more capable than the strict "active computation"
language suggested.

Reading B says the probe methodology has a confound that wasn't
controlled for. Position is encoded by training and correlates with
the probe target.

We do not have enough data to fully disambiguate. The locked file's
prediction P2 is falsified either way — under Reading A, the null
claim was too strong; under Reading B, the null claim was untestable
without a position-control feature.

A follow-up control experiment would resolve this: probe a feature
that is **purely positional** (e.g., "request_idx mod 4", which is a
function of position alone with no other content). The trained-vs-
untrained gap on that control gives a lower bound on the "position
contribution" to any other probe. If the cumulative-count gap drops
to the position-control level after controlling, Reading B is correct;
if a substantial residual gap remains, Reading A is correct.

This control was not pre-registered and is not within the scope of
the current experiment. It is documented as the natural follow-up.

## Cross-condition pattern is informative

Feature B accuracy + gap is roughly invariant across real, within-
shuffled, and global-shuffled:

| Condition | Trained MLP | Untrained MLP | Gap |
|---|---|---|---|
| real | 0.888 | 0.597 | +0.291 |
| within-shuffled | 0.826 | 0.590 | +0.236 |
| global-shuffled | 0.896 | 0.593 | +0.303 |

This is **further evidence for Reading B**. Genuine aggregation should
require the structure global-shuffled destroys (token identity needed
to "attend to large-response tokens"); the fact that gap is preserved
even when identity is permuted means whatever signal the trained
model recovers does not depend on token identity. Position is the
remaining candidate.

## Framework revision implied by these results

The graded N-criterion as stated:
> Features that are predictively irrelevant tend to be absent — *except*
> when the architecture provides architectural carry-through.

The HTTP results suggest a further amendment, regardless of which
reading dominates:

> Features that are predictively irrelevant tend to be absent — except
> when (a) the architecture provides architectural carry-through, OR
> (b) they are recoverable from any feature the model encodes for
> next-token prediction purposes. Position is one such feature on
> domains with periodic input structure. Other examples are likely.

This amendment is a real weakening of the framework. The "null on
predictively-irrelevant features" prediction is now hedged on what
ELSE the model has encoded for predictive reasons. Position is a
specific case but plausibly not the only one; any feature correlated
with another predictively-relevant feature inherits some of its
encoding.

This refines the framework but does not destroy it: carry-through is
still real (Prediction 1 confirms cleanly), the destroyed-structure
controls behave as predicted on the carry-through side, and the
positive direction of the framework (predictively-required features
ARE encoded) remains unchallenged.

## Comparison to the maze experiment

| Aspect | Maze (Phase 4, commit `aa025b1`) | HTTP (Phase 5, commit `3b25ed3`) |
|---|---|---|
| Domain shape | Synthetic spatial graph | Applied event stream |
| Vocab | 67 cells | 52 field-value tokens |
| Tokenization | One token per cell | Four tokens per request |
| Carry-through prediction | Starting-cell encoded (FALSIFIED — encoded but predicted null) | Feature A encoded (CONFIRMED — predicted encoded) |
| Computed-feature prediction | Distance encoded (FALSIFIED — predicted encoded, was null) | Feature B null (FALSIFIED — predicted null, was encoded) |
| What was learned | Strict iff is wrong; need to add carry-through mechanism | Carry-through generalizes; need to weaken null claim further (correlated-feature inheritance) |

Both experiments deliver framework-modifying information. The maze
data forced the addition of carry-through. The HTTP data forces an
additional weakening of the null direction.

The framework is being refined by data through pre-registration —
which is the discipline working as intended, just at a higher cost
to the framework's strength than originally hoped.

## Reproducibility

- Predictions locked: commit `3b25ed3` (2026-05-31)
- Infrastructure committed: commit `e5fa92a`
- Data corpus: `data/nasa_http/`, `data/nasa_http_within_shuffled/`,
  `data/nasa_http_global_shuffled/`
- Trained checkpoints: `checkpoints/http_{real,within_shuffled,global_shuffled}/best.pt`
- Probe outputs: `checkpoints/multiseed_phase5_http/probe_*.log`
- The pre-registration audit trail is verifiable via
  `git log --diff-filter=A predictions/predictions_http_log_sequences.md`
  showing commit `3b25ed3` predating any HTTP data download or model
  training.

## Amendments to the predictions file

A methodology amendment will be added documenting the class-balanced
probe sampling used for Feature A (the locked file did not specify
sampling strategy; balanced sampling was a methodological choice made
at probe time to avoid majority-class saturation on the 75% sb=4
class). This is documented as an amendment rather than a silent
deviation per the predictions/README.md protocol.
