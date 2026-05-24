# PLAN.md — LatentCityGPT build plan

Phased plan. Each phase lists tasks and an **acceptance criterion** (what proves
it's done). Build phases in order; later phases assume earlier artifacts exist.
Keep `CONTEXT.md`'s ONE RULE in mind throughout: no coordinates ever reach the model.

Work small-first: get every phase passing on a tiny city (or the synthetic smoke
grid) before scaling up the corpus or model.

---

## Phase 0 — Data  ✅ DONE

`data/prepare_city.py` is written and smoke-tested.

- OSM pull (OSMnx) -> largest strongly-connected component -> trivial tokenizer
  (`0=PAD,1=BOS,2=EOS`, real nodes from 3) -> blended shortest-path + random-walk
  routes -> destination holdout -> nanoGPT-format outputs (`train/val/gen.bin`,
  `meta.pkl`, `coords.csv`, `graph.gpickle`).
- **Acceptance (met):** smoke test passes — tokenizer bijection, all routes
  traverse only real edges, gen routes end on held-out destinations while train
  routes never do, binary roundtrips, token stream contains no coordinate values.

**Follow-up task (do during Phase 2):** add a `--split geographic` mode to
`split_destinations` that holds out a contiguous lat/lon sub-region instead of
scattered nodes (stronger generalization claim). Coords used only to *define* the
split, never written into tokens.

---

## Phase 1 — Model & training

Goal: a small, readable, trainable GPT that learns to route.

- [ ] `model/model.py` — decoder-only causal transformer, nanoGPT-style. Configurable
      `n_layer, n_head, n_embd, block_size, dropout`. `vocab_size` read from `meta.pkl`.
- [ ] `model/train.py` — loads `*.bin` + `meta.pkl`, packs into `block_size` blocks,
      AdamW + cosine schedule + warmup, grad clipping, AMP. Logs train/val loss;
      checkpoints best-val to `checkpoints/`. Flags: `--config --data_dir --seed`.
- [ ] `model/configs/small.py` (~10M) and `medium.py` (~30M).
- [ ] Save enough in the checkpoint to reconstruct the model for eval (config + vocab_size).
- **Acceptance:** trains on the smoke/tiny city; val loss decreases smoothly; a
      checkpoint loads and generates a route from a BOS+start-node prompt without error.
- **Watch:** mask PAD in the loss. Confirm the model only ever receives token IDs.

---

## Phase 2 — Intrinsic eval (the progress curve)

Goal: the Othello "legal-move-rate" analogue, plus generalization.

- [ ] `eval/valid_edge.py` — load ckpt + `graph.gpickle`. (a) **next-step** valid-edge
      rate: greedy prediction is a real neighbor of current node; (b) **full-route**
      validity: sample routes from start nodes, fraction whose every consecutive pair
      is a real edge. Run per split. Flag `--split {val,gen}`.
- [ ] Perplexity/CE reporter on `val.bin` and `gen.bin`.
- [ ] Implement the geographic-holdout follow-up from Phase 0 and regenerate a
      sub-region split for the strong generalization number.
- **Acceptance:** trained model's valid-edge rate >> an untrained model's on `val`;
      and stays high on `gen` (held-out destinations). Single clear summary line per run.

---

## Phase 3 — Baselines (earn the right to the claim)

Goal: show LatentCityGPT is a *competent* route model vs strong sequence baselines.
Remember the trap (CONTEXT.md): this phase is a sanity gate, **not** the headline.

- [ ] `eval/baselines.py` — uniform random; unigram frequency; **1st- and 2nd-order
      Markov** over the graph; a **same-parameter-count LSTM**. Report perplexity and a
      **long-range coherence** metric (e.g. does the route still progress toward / reach
      the destination after N steps — where Markov forgets and LatentCityGPT shouldn't).
- **Acceptance:** LatentCityGPT matches/beats Markov on perplexity and clearly beats it on
      long-range coherence; LSTM is in the same ballpark on perplexity. Documented table.

---

## Phase 4 — The probe suite (the headline)

Goal: recover the metric map from activations and prove it's emergent + linear.
**All probe code reads `coords.csv`; the model still only ever saw token IDs.**

- [ ] `eval/probe.py`:
  - [ ] Cache residual-stream activations per token position across a layer sweep.
  - [ ] **Linear probe** (ridge regression) activations -> (lat, lon). Report R²,
        median error in meters, per-layer curve.
  - [ ] **MLP probe** for the linear-vs-nonlinear comparison.
  - [ ] **Control: untrained** — same architecture, random init; probe should fail.
  - [ ] **Control: probe-capacity** — probe raw embeddings/IDs; R² ≈ 0.
  - [ ] **Control: destroyed-structure** — requires a model trained on
        identity-shuffled routes (add `--shuffle_nodes` to `prepare_city.py` or a
        variant); probe should fail.
- **Acceptance:** trained-model linear-probe R² is high and far exceeds untrained,
      raw-embedding, and destroyed-structure controls; linear ≈ MLP (map is linear).
      This is the core scientific result.

---

## Phase 5 — Causal intervention (correlation -> causation)

Goal: prove the model *uses* the map.

- [ ] `eval/causal.py` — from the probe, get the residual direction encoding location.
      While the model is at node A, patch in the "I'm at node B" direction; measure the
      shift in the next-hop distribution toward B's neighbors (vs an unpatched run).
- **Acceptance:** patching reliably bends next-hop predictions toward the patched
      location's real neighbors, well above a random-direction control.

---

## Phase 6 — Visualization & write-up (the viral part)

- [ ] `viz/overlay.py` — Procrustes-align probe-recovered coordinates to true ones;
      render recovered-vs-true overlay; optionally animate the map crystallizing across
      training checkpoints or across layers. This is the shareable image.
- [ ] Short interactive demo (Streamlit/Vercel): type/click a start + destination, watch
      the model route, and show the probe-decoded map. Weights -> Hugging Face.
- [ ] Write-up: methods + the honest one-line claim + the controls table. Lead the
      visual with the **probe-decoded map**, NOT attention heatmaps (attention isn't
      explanation — a sharp reviewer discounts that on sight).
- **Acceptance:** a single overlay image a stranger immediately reads as "that's
      <city>," reproducible from a checkpoint via one command.

---

## Definition of done (project)

All six phases pass on at least one real city, the probe result clears every
control, the causal intervention works, and there's one screenshot that tells the
whole story. The repo README states the honest one-line claim and links the
overlay image and demo.
