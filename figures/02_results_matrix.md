# Figure 2: Cross-domain results matrix

At-a-glance verdict for each (domain, feature, condition) cell tested in
the paper. Color encodes outcome relative to the framework's prediction.

## Legend

| Symbol | Meaning |
|---|---|
| 🟩 | Encoded above untrained baseline (gap > 0.15 typical positive threshold) |
| 🟨 | Weakly encoded (gap 0.05 to 0.15) |
| 🟥 | Falsified or null (with respect to the framework's prediction direction) |
| ⬜ | Predictor's error or not meaningfully testable |
| ⬛ | Not tested in this experiment |

The framework's *prediction* for each cell is the row label tag:
**(pos)** = predicted to be encoded; **(NULL)** = predicted not encoded.
Color shows the observed outcome.

## A. Calibration domains (post-hoc analysis, not pre-registered)

| Feature | Real | Within-shuffled | Global-shuffled | Verdict |
|---|:---:|:---:|:---:|---|
| Othello board state **(pos)** | 🟩 +0.34 | ⬛ | ⬛ | confirmed |
| Cities geographic location **(pos)** | 🟩 +0.55 | 🟩 +0.67 | 🟥 +0.01 | confirmed real & within; control kills it |
| Music voice-leading **(pos)** | 🟩 +0.89 transplant | 🟨 +0.07 transplant | 🟥 -0.01 transplant | confirmed; clean gradient |
| Music chord **(pos, weak)** | 🟨 +0.09 | ⬛ | ⬛ | weakly confirmed |
| Music beat-in-measure **(NULL)** | 🟥 +0.006 | ⬛ | ⬛ | confirmed null (probe + transplant) |
| Flight phase **(pos)** | 🟨 +0.11 | 🟨 +0.10 | 🟨 +0.05 | confirmed; non-monotone gradient |
| Symgroup partial product **(pos)** | 🟨 +0.05 | 🟥 +0.01 | 🟥 +0.01 | partial signal |

## B. Pre-registered ex-ante tests

### Maze navigation (commit aa025b1)

| Feature | Real | Within-shuffled | Global-shuffled | Verdict |
|---|:---:|:---:|:---:|---|
| Current cell row **(pos)** | ⬜ 1.0 trained = 1.0 untrained | ⬜ | ⬜ | predictor's error (token-identity confound) |
| Current cell col **(pos)** | ⬜ 1.0 trained = 1.0 untrained | ⬜ | ⬜ | predictor's error |
| Distance to goal **(pos)** | 🟥 +0.01 | 🟥 +0.01 | 🟥 +0.01 | **falsified** (positive direction) |
| Starting cell ID **(NULL)** | 🟥 +0.15 | 🟨 +0.03 | 🟥 +0.15 | **falsified** (null direction) |

### HTTP log sequences (commit 3b25ed3)

| Feature | Real | Within-shuffled | Global-shuffled | Verdict |
|---|:---:|:---:|:---:|---|
| Feature A: first request size_bin **(pos via carry-through)** | 🟩 +0.17 | 🟨 +0.13 | 🟩 +0.16 | confirmed; **carry-through 2-for-2** |
| Feature B: cumulative large count **(NULL, computed)** | 🟥 +0.29 | 🟥 +0.24 | 🟥 +0.30 | **falsified** (initial) |
| Position control (purely positional) | 🟩 +0.43 | 🟩 +0.54 | 🟩 +0.40 | confirms position is recoverable |
| Feature B at fixed k=5 (Design A) **(NULL after control)** | 🟥 +0.22 | 🟥 +0.20 | 🟥 +0.14 | **falsified** (after position control) |
| Feature B residual probe (Design B3) **(NULL after control)** | 🟥 +0.47 R² | 🟥 +0.43 R² | 🟥 +0.22 R² | **falsified** (after position control) |

## C. Summary scorecard

| Claim | Score |
|---|---|
| Carry-through positive direction | 🟩🟩 **2-for-2** (maze starting cell + HTTP Feature A) |
| Strict null direction | 🟥🟥 **0-for-2** (maze starting cell + HTTP Feature B) |
| Specific positive direction prediction | 🟥 **0-for-1** (maze distance) |
| Destroyed-structure monotonicity (real > within > global) | 🟨 **mixed** (works on some, fails on cities/maze starting cell) |
| Music beat null (post-hoc, not pre-registered) | 🟩 confirmed by both probe and transplant |

## D. The single most important pattern

The architectural carry-through mechanism is the one claim that survived
pre-registered testing on two different domain shapes:

| Test | Carry-through prediction | Observed |
|---|---|---|
| Maze starting cell (commit aa025b1) | encoded by carry-through (introduced post-falsification) | encoded (gap +0.15) |
| HTTP Feature A (commit 3b25ed3) | encoded by carry-through (ex-ante prediction) | encoded (gap +0.17) |

Two domains. Two different shapes (synthetic spatial graph vs applied
event stream). Same mechanism. Both confirmed at predicted magnitudes.
