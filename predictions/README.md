# Pre-registration protocol

This directory holds ex-ante predictions for upcoming experiments —
written and committed BEFORE any experiment runs, then compared against
observed results after the experiment completes.

## Why this exists

The N-criterion framework (see `pivot.md`, `CONTEXT.md`,
`STATUS_vs_OTHELLO-GPT.md`) makes claims about *when* a next-token
transformer will encode a candidate world feature in its residual
stream: it will encode F iff next-token prediction requires F.

A framework that only explains observations after the fact is hard to
distinguish from a flexible narrative. A framework that makes risky
predictions in advance, and whose predictions either confirm or
falsify cleanly, is doing real scientific work.

This directory is the implementation of that discipline. Each new
domain or experiment gets a `predictions_<tag>.md` file written and
committed BEFORE the experiment runs. The commit timestamp (recorded
in `git log`) is the evidence that the prediction preceded the
outcome.

## Files in this directory

- `README.md` — this file. The protocol description.
- `predictions_TEMPLATE.md` — the template each new prediction file
  starts from.
- `predictions_<tag>.md` — one file per ex-ante experiment. Examples
  of `<tag>`: `w4_maze`, `w4_tcp_state`, `w4_code_exec`.

## The protocol

### Step 1 — choose the experiment

Before writing predictions, pick the experiment:
- A new domain (probe target and destroyed-structure controls must be
  unambiguous and buildable).
- A scale-up (existing domain at a new model size).
- A failure-mode hypothesis (e.g., "feature F predicted to be null
  under condition C").

Document the choice and the rationale in a separate planning artifact
(not in this directory).

### Step 2 — write predictions

Copy `predictions_TEMPLATE.md` to `predictions_<tag>.md`. Fill in
every section. Predictions must be:

- **Quantitative** with confidence bands (e.g., "MLP probe accuracy
  ≥ 0.75 ± 0.05"), not narrative ("we expect the probe to work").
- **Layer-specific** where applicable (e.g., "peak at L2 or L3").
- **Linked to the N-criterion analysis** (which features are required
  for next-token prediction, which are not, and why).
- **Explicit about what would falsify** the prediction. Each
  prediction must include a "what would falsify this" section
  enumerating outcomes that would invalidate it.

### Step 3 — commit and lock

Commit the predictions file to git on the main branch BEFORE any
experiment runs. The commit hash is the lockdown. After this:

- The predictions file is **append-only**. The original content
  must not be edited. If an error is discovered, add an annotation at
  the bottom rather than editing the predictions above.
- If predictions need to be updated based on new information that
  arrives BEFORE the experiment, document the update as a separate
  annotated commit; do not silently rewrite. The audit trail is the
  whole point — it must remain verifiable in `git log`.

### Step 4 — run the experiment

Run experiments per the plan. Do not look at intermediate results
before the predictions file is committed.

### Step 5 — compare and record

Write a `results_<tag>.md` file alongside the predictions file. For
each prediction:

| Prediction | Observed value | Verdict |
|---|---|---|
| (verbatim from predictions file) | (actual measurement) | ✓ confirmed / ✗ falsified / ◐ ambiguous |

For each falsified prediction, write a short analysis: what does the
failure imply about the N-criterion framework? Does it need refinement
(scope restriction, additional assumption) or is it broken on this
class of features?

### Step 6 — commit results

Commit `results_<tag>.md` separately. The git diff between the
predictions commit and the results commit is the auditable record.

## What counts as confirmed / falsified

| Outcome | Verdict |
|---|---|
| Observed value falls inside the predicted band | ✓ confirmed |
| Observed value falls outside the predicted band by ≤ 1σ | ◐ ambiguous (record but don't claim victory) |
| Observed value falls outside the predicted band by > 1σ | ✗ falsified |
| Observed value is qualitatively opposite to the prediction (e.g., predicted positive, observed negative) | ✗ falsified |
| Observed value matches in magnitude but at a different layer than predicted (within 2 layers) | ◐ ambiguous |
| Observed value matches in magnitude but at a different layer than predicted (> 2 layers off) | ✗ falsified on layer location |

## What to do when falsified

A falsified prediction is **more valuable** than a confirmed one for
the framework's credibility. It demonstrates that the predictions
were risky (could have gone either way) and that we report failures
honestly.

For each falsified prediction:

1. Record the failure verbatim in `results_<tag>.md`.
2. Write a short post-hoc analysis: why might the framework have
   been wrong here? Was the N-criterion analysis flawed? Was there
   a hidden encoding mechanism we didn't consider?
3. Update `pivot.md` or `CONTEXT.md` if the framework needs
   refinement based on the failure.
4. Do not delete or rewrite the original prediction.

## Verification

A future reader can verify the lockdown by:

```
git log --diff-filter=A predictions/predictions_<tag>.md
git log --diff-filter=A predictions/results_<tag>.md
```

The predictions file should be added BEFORE the results file. Any
edits to the predictions file after the initial commit will be
visible in `git log --all -p predictions/predictions_<tag>.md`.

## Scope of this protocol

Pre-registration is used for **ex-ante prediction experiments only**.
It is not required for:

- Replication of existing W1+W2 work.
- Debugging or sanity-check runs.
- Methodology development that does not test a framework prediction.
- Exploratory work whose purpose is to discover what's interesting.

Pre-registration applies when we are testing the N-criterion framework
against a new case where the answer is not yet known.
