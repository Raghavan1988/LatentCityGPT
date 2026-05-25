# LatentWorldsGPT

*Formerly LatentCityGPT — see "The pivot" below.*

A comparative study of **when** Othello-GPT-style emergent world representations
appear in next-token transformers, and **what's actually going on** when a probe
finds one.

Started as a single replication on real-city routes ("a small GPT trained only on
intersection IDs recovers a metric map of the city from its activations"). The
destroyed-structure control surfaced something unexpected, and the project
pivoted into a multi-domain study. The cities sub-project is now the worked
example — and the load-bearing methodological exhibit — that the rest of the
portfolio extends.

## The pivot in one paragraph

A small GPT trained on routes through real cities does, in fact, produce a
linearly-decodable geographic signal in its activations — what we initially
read as an emergent metric map. But a destroyed-structure control (same routes,
token order shuffled within each route) produced *higher* probe R², not lower.
The reason: the SET of intersections in each route is itself geographically
coherent, regardless of order. So the probe-decoded clustering does not require
sequence learning — it appears wherever the training data has geographic
co-occurrence. What does require sequence training is the model's ability to
predict the next real intersection (P(A's neighbors) = 0.984 on the real model
vs 0.063 on the shuffled). The two phenomena are distinct, with different
sources, and both are causally encoded in the residual stream. That decomposition
is more interesting than the single-claim original framing.

## Findings so far (cities, smoke-trained)

Three trained models on the City of London, same architecture, differing only in
training data:

| Condition                | val ppl | P(A's nbrs) | Probe R² (node-level lin) | Transplant lift on P(B's nbrs) |
|---|---:|---:|---:|---:|
| Real London              | 1.65    | 0.984       | 0.64                      | **+0.953** (100% target>random) |
| Within-route shuffled    | 25.0    | 0.061       | 0.96                      | +0.247                          |
| Global shuffled          | 313     | 0.006       | −0.02                     | +0.000 (chance)                 |

Three load-bearing methodological findings from this session:

1. **MLP-probe lookup contamination on continuous targets**: at ~10³ unique
   tokens, both linear and MLP probes can pass the standard position-level split
   via per-token lookup. Use a node-level split (train + test on disjoint
   token sets) for the honest signal.
2. **Within-sequence shuffle is an insufficient destroyed-structure control**
   when the training data's geographic co-occurrence is preserved by
   set-membership alone. Use a global token shuffle to break it cleanly.
3. **Pseudoinverse-direction activation patching does not isolate causal use
   of the representation**; activation transplant (substituting a real `a_B`
   from another position) does.

The trained `wte` is NOT a node2vec embedding of the graph (Procrustes/CKA at
chance against random); node2vec produces *stronger* geographic decodability than
`wte` alone. The geographic signal we found is built by the transformer's higher
layers, not handed to the model by the embedding table.

## Where the project is going

A portfolio of additional domains (see `pivot.md` for the full plan):

| # | Domain | Probe target | Role |
|:--|---|---|---|
| 1 | Cities (done) | (lat, lon) | Decomposition anchor + methodology exhibit |
| 2 | Music | Key / chord / beat | Within-domain mixed-verdict experiment |
| 3 | Symmetric-group-GPT | Resulting permutation | Theoretical positive anchor |
| 4 | Dialog-state (MultiWOZ) | Slot-value belief state | Applied positive (text) |
| 5 | Flight-phase (ADS-B) | Flight phase | Applied positive (time series) |
| 6 | Maze-GPT (optional) | Agent pose, wall map | Applied positive (spatial) |

The working thesis: emergent world models in next-token transformers depend on
the training distribution providing (D) discrete state, (N) state necessary for
next-token prediction, (¬L) state not reconstructible from sequence
co-occurrence statistics alone. The portfolio is constructed to test where each
criterion bites.

## Quickstart

```bash
pip install -r requirements.txt

# Build a city corpus (CPU, free; needs network for the OSM pull)
python -u data/prepare_city.py --place "City of London, Greater London, England, United Kingdom" \
  --out_dir data/london_city

# Destroyed-structure controls (two tiers)
python -u data/prepare_city.py --place "City of London, ..." --shuffle_routes \
  --out_dir data/london_shuffled
python -u data/prepare_city.py --place "City of London, ..." --shuffle_globally \
  --out_dir data/london_global_shuffled

# Train a small model (M1 MPS / CUDA / CPU)
python model/train.py --config model/configs/small.py --data_dir data/london_city

# Eval suite
python eval/valid_edge.py        --ckpt checkpoints/best.pt --data_dir data/london_city
python eval/baselines.py         --data_dir data/london_city --ckpt checkpoints/best.pt --coherence
python eval/probe.py             --ckpt checkpoints/best.pt --data_dir data/london_city
python eval/transplant.py        --ckpt checkpoints/best.pt --data_dir data/london_city  # clean Phase 5
python eval/embedding_compare.py --ckpt checkpoints/best.pt --data_dir data/london_city
```

## The cities sub-project's THE ONE RULE

For cities: no coordinate, distance, or direction ever enters the model's
input. Tokens are arbitrary intersection IDs; coordinates live only in
`coords.csv` and are touched solely by the probe / eval / viz code. THE ONE
RULE generalizes per-domain to "no probe-target value may appear in the
model's input."

## Repo

| Path | What |
|---|---|
| `data/prepare_city.py` | OSM street network → token corpus. `--shuffle_routes` + `--shuffle_globally` flags for two-tier destroyed-structure controls. |
| `model/` | nanoGPT-style model + training loop |
| `eval/probe.py` | Linear + MLP probes; **both** position-level and node-level splits |
| `eval/causal.py` | Pseudoinverse-direction patching (preserved as documented-failure mode) |
| `eval/transplant.py` | Clean Phase 5: substitute a real `a_B` for `a_A` at a chosen layer |
| `eval/embedding_compare.py` | `wte` vs node2vec — Procrustes + CKA + probe parity |
| `eval/valid_edge.py` `eval/baselines.py` | intrinsic eval + analytical baselines |

## Docs

- **`CLAUDE.md`** — operational guide; conventions, commands, current status. Read by Claude Code each session.
- **`CONTEXT.md`** — scientific framing post-pivot; the criteria framework; what cities established.
- **`PLAN.md`** — phased build plan; cities phases marked done, multi-domain milestones summarized.
- **`pivot.md`** — master plan for the comparative-study pivot. Risk register, confidence summary, decision points.
- **`STATUS_vs_OTHELLO-GPT.md`** — claim-by-claim comparison to the Othello-GPT lineage.
- **`update_may24_final.md`** — empirical narrative of the cities decomposition session.
- **`next_steps.md`** — short, concrete plan for the must-do experiments that just ran.

## Not to be confused with

There is a separate, unrelated **CityGPT** (Feng et al., KDD 2025,
[arXiv:2406.13948](https://arxiv.org/abs/2406.13948)). That work *fine-tunes large
LLMs on text instructions* to improve performance on urban benchmarks — it **gives**
the model spatial knowledge in language and measures task scores.

The cities sub-project of LatentWorldsGPT is the inverse: it **withholds** all
spatial information and trains a small model from scratch on bare intersection
IDs, then asks whether (and *what*) spatial structure emerges unsupervised.
Different question, different method, different proof.

## References

- Li, K., et al. (2022). *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task* — Othello-GPT.
- Nanda, N. (2023). *Actually, Othello-GPT Has a Linear Emergent World Representation*.
- Karvonen, A. (2024). *Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models*.
- Perozzi, B., Al-Rfou, R., Skiena, S. (2014). *DeepWalk*; Grover, A., Leskovec, J. (2016). *node2vec*.
- Karpathy, A. *nanoGPT* — the model + training scaffold.
