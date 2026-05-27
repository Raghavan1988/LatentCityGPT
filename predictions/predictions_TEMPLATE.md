# Predictions — <task tag>

**Task tag**: `<short kebab-case identifier — e.g., w4_maze, w4_tcp_state>`
**Task description**: <one-paragraph description of the domain and the
training corpus. Include: what the tokens are, what the next-token
objective is, how destroyed-structure controls will be built.>
**Predictions committed (timestamp)**: <ISO date, filled by the
committer at time of commit>
**Predictions author**: <name>
**Status**: locked / amended / superseded

## Setup

- **Model**: <architecture, params, n_layer, n_embd, etc.>
- **Training corpus**: <size, split, conditions: real / within-shuffled /
  global-shuffled>
- **Held-out evaluation split**: <node-level / piece-level /
  sequence-level — whatever the honest test is for this domain>
- **Multi-seed protocol**: 5 seeds, outermost loop varies untrained init,
  activation sampling, and probe-training RNG.

## N-criterion analysis

For each candidate probe target, state explicitly whether the
next-token objective requires the target to be encoded, and why.

| Probe target | Required for next token? | Mechanism | Prediction direction |
|---|---|---|---|
| <target 1> | YES / NO / PARTIAL | <brief mechanistic reasoning> | encoded / null / lexical |
| <target 2> | YES / NO / PARTIAL | <brief mechanistic reasoning> | encoded / null / lexical |
| <target 3> | YES / NO / PARTIAL | <brief mechanistic reasoning> | encoded / null / lexical |

## Predictions

### Prediction 1 — <target 1>

**N-criterion verdict**: <encoded / null / lexical>

**Probe predictions**:
- Best-layer trained MLP accuracy: <band, e.g., 0.75–0.85>
- Best-layer trained linear accuracy: <band>
- Trained-vs-untrained MLP gap: <band, e.g., ≥ 0.30>
- Best layer: <band, e.g., L2 or L3>

**Transplant predictions**:
- Peak transplant lift over unpatched: <band, e.g., 0.5–0.8>
- Peak transplant lift over random control: <band>
- Peak layer: <band>

**Destroyed-structure control predictions**:
- Within-shuffled probe MLP: <band>
- Within-shuffled transplant lift: <band>
- Global-shuffled probe MLP: <band — typically ≈ untrained baseline>
- Global-shuffled transplant lift: <band — typically ≈ 0.0 ± σ>

**What would falsify Prediction 1**:
- <Specific outcome 1, e.g., "Best-layer MLP < 0.50 — would mean the
  feature is not recoverable at all, contradicting the N-criterion
  verdict.">
- <Specific outcome 2, e.g., "Peak layer > L+2 from predicted —
  would mean the encoding location is wrong by 2 layers.">
- <Specific outcome 3, e.g., "Global-shuffled lift > 0.10 — would
  contradict the destroyed-structure null.">

### Prediction 2 — <target 2>

[Same structure as Prediction 1]

### Prediction 3 — <target 3>

[Same structure as Prediction 1]

## Cross-target predictions

Predictions about relationships ACROSS targets:

- **Encoding ordering**: <e.g., "Target 1 lift > Target 2 lift > Target 3
  null">
- **Layer ordering**: <e.g., "Target 1 peak deeper than Target 2 peak">
- **Destroyed-structure gradient**: <e.g., "Real > within > global on
  all three encoded targets, monotone">

## What would falsify the framework on this domain entirely

A failure that would invalidate the N-criterion as applied to this
domain (not just one prediction):

- <Outcome 1, e.g., "If a target predicted to be null (e.g., starting
  cell in a maze) shows trained-vs-untrained gap > 0.10, the
  N-criterion's negative claim fails on this domain.">
- <Outcome 2, e.g., "If destroyed-structure controls fail to
  monotonically reduce lift, the framework's claim about structure
  dependence is wrong here.">
- <Outcome 3>

## Amendments (post-lockdown)

If any update is made to this file after the initial commit, append
the change here with a timestamp and reason. **Do not edit the
predictions above.**

[empty until first amendment]
