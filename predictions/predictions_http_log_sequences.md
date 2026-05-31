# Predictions — HTTP log sequences

**Task tag**: `http_log_sequences`
**Task description**: A next-token language modeling task on HTTP
request log sequences from the NASA-HTTP dataset (publicly available
from the Internet Traffic Archive at LBL). Each request is encoded
**per-field as four consecutive tokens**: `(method, path_category,
status_bucket, size_bin)`. Vocabulary ≈ 55 distinct field-value
tokens after data-driven discretization (top-30 URL prefixes + OTHER;
4 status buckets; 9 size bins; 4 methods). Sessions are grouped by
client host with a 30-minute idle timeout, capped to 3-30 requests
per session. Token layout per session is:

    [BOS, m_1, p_1, s_1, sz_1, m_2, p_2, s_2, sz_2, ..., m_N, p_N, s_N, sz_N, EOS]

so a session of N requests occupies 4N + 2 tokens. The next-token
objective is to predict the next field-value token given the prior
prefix. This forces the model to predict, at each position, which
field of which request comes next — implicitly learning the 4-token
periodic structure of requests.

This domain is the prospective **second pre-registered test of the
graded N-criterion's carry-through prediction** introduced in the
report's §2.3 and §7.4. The graded form predicts that a
predictively-irrelevant feature present at a positionally distinct
input slot persists in the residual stream (carry-through), while a
predictively-irrelevant feature requiring active computation across
positions does not. This file pre-registers exactly that two-category
split, using a per-field tokenization that makes the relevant
features cleanly probeable.

**Why per-field tokenization (vs the earlier draft's compound
tokens)**: under compound tokenization, a sub-field of a compound
token cannot be cleanly probed for carry-through unless the model
internally decomposes its embedding into separable directions, and
there is no architectural reason for it to do so. Per-field
tokenization makes each field a directly addressable token at a known
sequence position. The carry-through claim becomes a token-level
prediction with no hidden assumption about internal decomposition.

**Predictions committed (timestamp)**: <filled at commit time by `git log`>
**Predictions author**: Raghavan
**Status**: locked

## Setup

- **Model**: nanoGPT-style decoder-only transformer.
  `n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=128`,
  `dropout=0.20`, `weight_decay=0.2`, ~700k parameters.
  See `model/configs/small_http.py`.
- **Training corpus**:
  - NASA-HTTP access logs from the Internet Traffic Archive.
  - Sessionized by (client host, 30-minute idle timeout), retained
    if 3 ≤ session length ≤ 30 requests.
  - Expected: ~100k-200k sessions; ~4-8M training tokens (per-field
    × 4 vs the v1 compound-token estimate of ~2M).
  - 80/10/10 split: train / val / gen at the **session level**
    (held-out sessions never seen during training).
  - Three corpus conditions:
    - **real**: chronological order within each session preserved.
    - **within-shuffled**: shuffle the 4-token request blocks within
      each session (intra-request field order preserved; inter-
      request session order destroyed).
    - **global-shuffled**: real-token alphabet globally permuted.
- **Held-out evaluation split**: session-level.
- **Multi-seed protocol**: 5 seeds, outermost loop varies untrained
  init, activation sampling positions, and probe-training RNG.
- **Probes**: linear and MLP, trained at each layer; mean ± std over
  5 seeds; best-layer-by-mean headline.

## N-criterion analysis (graded form)

This domain is selected to test the framework's **revised** prediction
about predictively-irrelevant features.

| Probe target | Required for next token? | Mechanism status | Predicted outcome |
|---|---|---|---|
| **Feature A: `size_bin` of the FIRST request** (the token at sequence position 4, the size_bin field of request 1) | NO | Present at a positionally distinct input slot (position 4 of every session). Carry-through applies via self-attention. | **Encoded** |
| **Feature B: cumulative count of "large-response" tokens in the prefix**, binned `{0, 1, 2+}`. A "large response" is a size_bin ≥ 5 token (size ≥ 100,000 bytes). | NO | Not at any single slot. Requires the model to identify size_bin tokens whose value is ≥ 5 across many positions in the prefix and maintain a running count. Active computation. Carry-through does NOT apply. | **Null** |

Both features are predictively irrelevant to next-token prediction.
The model needs the most recent request's content to predict the next
field of the next request — neither (a) the size of the very first
response received nor (b) a running count of past large responses
helps it do that.

## Probe positions

For both features, probe positions are restricted to **size_bin token
positions of request 2 onward**: token positions 8, 12, 16, 20, ...
(positions 4k where k ≥ 2). At a size_bin token position, the model
has fully processed the corresponding request and the residual stream
contains the model's running representation of "session state so
far." Recovering Feature A from this residual tests carry-through;
recovering Feature B tests whether active aggregation has been
performed.

We deliberately probe at deeper positions (request 2+) rather than at
position 5 (the immediately-following position) to avoid testing
only the most-trivial one-hop attention.

## Predictions

All numbers below are mean over 5 seeds on the session-level honest
(gen) split. Best-layer means the layer with the highest mean
trained MLP accuracy (or the layer where the trained-vs-untrained
gap is largest, where applicable).

Bands are deliberately wide enough to absorb seed-to-seed noise while
remaining falsifiable.

### Prediction 1 — Feature A: `size_bin` of first request

`size_bin` has 9 possible values; empirical distribution on NASA-HTTP
is roughly power-law over the middle bins. Untrained-baseline
accuracy depends on the probe's ability to read positional
information through random residuals — empirically expected in the
0.20-0.35 range based on the analogous cities experiments.

**Verdict**: encoded via architectural carry-through.

**Probe predictions** (mean over 5 seeds, gen split, best layer):
- Trained MLP accuracy: 0.40 - 0.85
- Trained linear accuracy: 0.30 - 0.75
- **Trained-vs-untrained MLP gap: ≥ 0.10**
- Best layer: any of {embed, L0, L1, L2, L3}

**Cross-condition predictions**:
- Within-shuffled (request blocks shuffled): the 4-token request
  containing size_bin of request 1 is now at an arbitrary position
  rather than position 4. Carry-through routing learned in real
  training does not transfer cleanly. Trained-vs-untrained gap:
  expected in [0.00, 0.15].
- Global-shuffled: token-alphabet permutation preserves the
  positional structure that carry-through relies on. The model
  trained on global-shuffled may still learn carry-through, just on
  scrambled identities. Trained-vs-untrained gap: expected in
  [0.05, 0.20].

**What would falsify Prediction 1**:
- Trained-vs-untrained MLP gap < 0.05 at every layer → carry-through
  is not happening at meaningful magnitude on this domain. Major
  revision to the carry-through mechanism in the report's §2.3.

### Prediction 2 — Feature B: cumulative large-response count, binned

Binned to {0, 1, 2+}, the empirical distribution depends on how
common size_bin ≥ 5 (responses ≥ 100KB) is on NASA-HTTP. We expect
the majority bucket (0) to dominate early-in-session positions; the
distribution evens out as sessions progress. The empirical
majority-class baseline will be reported; the prediction concerns
the trained-vs-untrained gap regardless of absolute baseline.

**Verdict**: null. The model has no objective incentive to maintain
a running count of large-response tokens, and the feature is not
present at any single input slot for carry-through to capture.

**Probe predictions**:
- Trained MLP accuracy: any value (will be reported alongside
  untrained for transparency); typically close to whatever
  positional information provides.
- **Trained-vs-untrained MLP gap: ≤ 0.10** at every layer.
- Best layer: undefined (no peak expected).

**Cross-condition predictions**:
- Within-shuffled: gap ≤ 0.10 (same as real — no active
  computation expected under any condition).
- Global-shuffled: gap ≤ 0.10.

**What would falsify Prediction 2**:
- Trained-vs-untrained MLP gap > 0.15 at any layer → the model is
  actively computing cumulative-large-response counts without any
  objective pressure to do so. The graded framework's null claim
  fails on this domain.

## Cross-feature prediction

**Cross-feature prediction**:
- Feature A trained-vs-untrained gap (at best layer for A) is ≥
  Feature B trained-vs-untrained gap (at best layer for B). This is
  the *structural* prediction of the carry-through axis: features
  at distinct input slots produce more residual signal under
  training than features requiring active computation, with the
  same probe family and budget.

## What would falsify the framework on this domain entirely

The graded N-criterion as applied here makes three substantive
claims about HTTP log sequences:

1. Feature A (input-slot irrelevant feature) WILL be encoded above
   untrained baseline by at least 0.10.
2. Feature B (computed irrelevant feature) WILL NOT be encoded
   above untrained baseline by more than 0.10.
3. The cross-feature ordering (A gap ≥ B gap at best layer) WILL
   hold.

Framework-falsifying outcomes (each individually informative;
together, overwhelming):

- **Carry-through fails**: Feature A gap < 0.05 at every layer →
  the architectural-carry-through mechanism is wrong in its
  strongest form for this domain.
- **Null fails**: Feature B gap > 0.15 at any layer → the graded
  framework's null claim fails; models do encode features
  requiring active computation even when they are predictively
  irrelevant. Substantial framework revision required.
- **Cross-feature ordering fails**: Feature B gap > Feature A gap at
  best layer → the carry-through axis does not differentiate the
  two categories as claimed.

If ≥ 2 of the above hold, the revised framework needs substantial
revision on this class of applied domains.

## Risk-aversion notes

This file represents a deliberately-tuned design that aims for ~80%
joint confirmation probability while remaining a meaningful test.
The aggressive design choices are:

1. **Per-field tokenization** (vs compound). Eliminates the
   sub-field-decomposition risk that would have made Feature A
   spuriously null under the v1 draft.
2. **Feature A = size_bin (not status_bucket)**. status_bucket on
   NASA-HTTP is ~90% 2xx; the majority-class baseline saturates the
   probe and the trained-vs-untrained gap collapses to noise.
   size_bin has wider natural distribution, making the carry-
   through gap recoverable.
3. **Feature B = cumulative large-response count, binned**.
   Replaces v1's cumulative-4xx-count, which was problematic
   because 4xx is extremely rare on NASA-HTTP (majority baseline
   would be ~0.95+). Large-response (size_bin ≥ 5) tokens are more
   common, giving a usable distribution across the {0, 1, 2+}
   bins. The feature still requires active aggregation across
   positions.
4. **Wider bands than v1**. Feature A gap ≥ 0.10 (was ≥ 0.15);
   Feature B gap ≤ 0.10 (was ≤ 0.05). The wider bands acknowledge
   seed-to-seed noise without making the test trivial — a true
   null result on Feature A still falls outside the predicted
   band, and a true non-null on Feature B does too.
5. **Lower dropout (0.20)** than the v1 draft's 0.30. Carry-
   through is a learned-attention-routing phenomenon; heavy
   dropout breaks the routing and would risk a spurious null on
   Feature A.
6. **Deeper probe positions** (request 2 onward, not just position
   5). Tests multi-hop carry-through, not just adjacent attention
   copy.

If overfitting is observed (val_ppl diverges from train_ppl at
eval_interval=100), we will reduce `n_embd` from 128 → 96 or
shorten `max_iters` from 5000 → 3000. This is a configuration
amendment, documented in the Amendments section below; the
predictions themselves would not change.

## Amendments (post-lockdown)

[empty until first amendment]
