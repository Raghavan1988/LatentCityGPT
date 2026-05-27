# Predictions — EXAMPLE (not a real prediction; do not lock)

This is a worked example showing what a filled predictions file looks
like. The task is fictional. **This file is NOT a live prediction**
and is committed only to make the template concrete for future use.

---

**Task tag**: `example_maze_4x4`
**Task description**: A 4×4 grid maze with one start cell, one goal
cell, and 4 wall configurations. Tokens are cell coordinates from a
self-avoiding path the model is asked to predict next-step-to-take.
The model is a 4-layer, 4-head, n_embd=128 nanoGPT-style transformer.
Real condition: paths are optimal A* solutions to random maze
configurations. Within-shuffled: same paths with cell-token order
shuffled within the path. Global-shuffled: tokens reassigned uniformly
across the corpus.
**Predictions committed (timestamp)**: 2026-01-01T00:00:00Z (placeholder)
**Predictions author**: example
**Status**: example (not locked)

## Setup

- **Model**: nanoGPT-style, n_layer=4, n_head=4, n_embd=128, ~600k params
- **Training corpus**: 50k mazes × ~12 tokens/path = 600k tokens; 80/10/10
  train/val/gen split; conditions: real, within-shuffled, global-shuffled.
- **Held-out evaluation split**: maze-level (test mazes never seen during
  training).
- **Multi-seed protocol**: 5 seeds, outermost loop varies untrained init,
  activation sampling, and probe-training RNG.

## N-criterion analysis

| Probe target | Required for next token? | Mechanism | Prediction direction |
|---|---|---|---|
| Current cell position (x, y) | YES — required to know which moves are legal | Encoded as residual state at each token | encoded |
| Manhattan distance to goal | YES — needed to choose direction that reduces distance | Encoded as scalar in residual | encoded |
| Starting cell coordinates | NO — irrelevant once you're past the first step | No need to retain across the path | null |
| Wall configuration of current row | YES — needed to avoid bumping walls | Encoded as a binary feature in residual | encoded |
| Goal cell coordinates | YES — needed to navigate toward it | Encoded as residual state | encoded |

## Predictions

### Prediction 1 — Current cell position

**N-criterion verdict**: encoded

**Probe predictions**:
- Best-layer trained MLP accuracy: 0.85–0.95 (16 cells, chance 1/16 = 0.0625)
- Best-layer trained linear accuracy: 0.70–0.85
- Trained-vs-untrained MLP gap: ≥ 0.50
- Best layer: L2 or L3

**Transplant predictions**:
- Peak transplant lift over unpatched: 0.6–0.9
- Peak transplant lift over random control: 0.6–0.9
- Peak layer: L2 or L3 (same as probe)

**Destroyed-structure control predictions**:
- Within-shuffled probe MLP: 0.40–0.60 (clustering preserved partially)
- Within-shuffled transplant lift: 0.15–0.35
- Global-shuffled probe MLP: ≤ 0.15 (chance + small ε)
- Global-shuffled transplant lift: 0.00 ± 0.01

**What would falsify Prediction 1**:
- Best-layer MLP < 0.50 — feature not recoverable, contradicts N-criterion encoded verdict.
- Peak layer at embed (L0) — would mean position is just token-table-encoded,
  not transformer-computed; we'd expect this only if maze paths are short.
- Global-shuffled lift > 0.10 — contradicts the destroyed-structure null.

### Prediction 2 — Manhattan distance to goal

**N-criterion verdict**: encoded

**Probe predictions**:
- Best-layer trained MLP accuracy: 0.55–0.75 (chance 1/7 = 0.143 for distances 0–6)
- Best-layer trained linear accuracy: 0.45–0.65
- Trained-vs-untrained MLP gap: ≥ 0.30
- Best layer: L2 or L3 (likely deeper than position — needs goal info too)

**Transplant predictions**:
- Peak transplant lift over unpatched: 0.4–0.7
- Peak layer: L3

**Destroyed-structure control predictions**:
- Within-shuffled probe MLP: ≈ untrained (the distance signal requires
  ordered path; within-shuffling breaks order completely)
- Global-shuffled probe MLP: ≈ untrained
- Both global-shuffled and within-shuffled transplant lifts: 0.00 ± 0.02

**What would falsify Prediction 2**:
- Best-layer MLP < 0.30 — distance is not recoverable, surprising.
- Within-shuffled probe MLP > 0.50 — would mean distance is recoverable
  even from shuffled paths, contradicting our claim about its order
  dependence.

### Prediction 3 — Starting cell coordinates

**N-criterion verdict**: null (irrelevant once past step 1)

**Probe predictions**:
- Best-layer trained MLP accuracy: ≈ untrained (within 1σ)
- Trained-vs-untrained MLP gap: ≤ 0.05
- Best layer: undefined (no signal)

**Transplant predictions**:
- Peak transplant lift over unpatched: ≤ 0.05
- Peak transplant lift over random control: ≤ 0.05

**Destroyed-structure control predictions**:
- All conditions: probe MLP ≈ untrained, transplant ≈ 0.

**What would falsify Prediction 3**:
- Trained-vs-untrained MLP gap > 0.10 — would mean the model retains
  starting position despite not needing it. Would refute the N-criterion's
  negative claim for this feature.
- Transplant lift > 0.10 — would mean starting position is causally
  used in next-token prediction. Same refutation.

## Cross-target predictions

- **Encoding ordering**: Prediction 1 (position) lift > Prediction 2
  (distance) lift > Prediction 3 (starting cell) lift.
- **Layer ordering**: Prediction 1 peak at L2-L3; Prediction 2 peak at
  L3 (needs goal info, so deeper); Prediction 3 has no peak.
- **Destroyed-structure gradient**: Real > within > global on
  Predictions 1 and 2, monotone. Prediction 3 is flat at chance across
  all conditions.

## What would falsify the framework on this domain entirely

- If Prediction 3 (starting cell, predicted null) shows gap > 0.10:
  the N-criterion's negative claim fails on maze-domain memory features.
- If destroyed-structure controls do not monotonically reduce lift on
  Predictions 1 and 2: the framework's claim about structure dependence
  is wrong here.
- If best layers for Predictions 1 and 2 are reversed (distance peaks
  shallower than position): the depth-of-computation claim needs revision.

## Amendments (post-lockdown)

[N/A — this is an example, not a real locked prediction]
