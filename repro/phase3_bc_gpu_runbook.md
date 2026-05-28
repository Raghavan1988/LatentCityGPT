# Phase 3-b + 3-c GPU Runbook

A step-by-step walkthrough for spinning up a single H100 80GB rental,
running Phase 3-b (Othello championship retrain, ~1 hour) and Phase 3-c
(cities scale demo, ~6-10 hours) in a single rental window, then tearing
down. Total estimated cost: $30-40.

## Before you spin up the GPU

These steps are pure laptop work — no GPU billing yet.

### 1. Source the WTHOR archives (Phase 3-b data)

WTHOR archives are free, public, ~10 MB total across all years.

```bash
mkdir -p data/wthor_raw
cd data/wthor_raw
# Download every WTH_YYYY.wtb file from either:
#   - https://www.ffothello.org/wthor/  (primary, French Othello Federation)
#   - https://www.othello.dk/             (alternate mirror)
# A simple wget loop:
for year in $(seq 1977 2024); do
    wget -q "https://www.ffothello.org/wthor/WTH_${year}.wtb" 2>/dev/null
done
ls *.wtb | wc -l   # should be ~45-48 files
cd -
```

### 2. Build the championship corpus locally (Phase 3-b prep)

```bash
python data/prepare_othello_championship.py \
    --wthor_dir data/wthor_raw \
    --out_dir data/othello_championship \
    --max_games 70000
```

Output: `data/othello_championship/{train,val,gen}.bin`, `meta.pkl`,
`board_state.csv`. Verify:

```bash
ls -la data/othello_championship/
python -c "
import pickle
m = pickle.load(open('data/othello_championship/meta.pkl', 'rb'))
print(f'vocab_size={m[\"vocab_size\"]} dtype={m[\"dtype\"]}')
print(f'train.bin size:', __import__('os').path.getsize('data/othello_championship/train.bin'))
"
```

Expected: vocab_size=68, train tokens ≥ 3M.

### 3. Regenerate Manhattan corpus (Phase 3-c prep, optional)

The current Manhattan corpus has ~2.74M train tokens. For a 50M-param
model that's data-starved; we recommend regenerating with more walks:

```bash
python -u data/prepare_city.py --place "Manhattan, New York, USA" \
    --n_shortest 50000 --n_walks 200000 \
    --out_dir data/manhattan_full
```

Output: ~6-10 hours of CPU work on a typical laptop. Produces ~10-20M
train tokens. **Or** skip this step and train Phase 3-c on the existing
`data/manhattan/` — the model will be data-starved but the comparative
question only requires reproducing the patterns.

### 4. Sanity check that the configs load

```bash
python -c "
import importlib.util
for cfg in ('large_othello', 'large_cities'):
    spec = importlib.util.spec_from_file_location('cfg', f'model/configs/{cfg}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    print(f'{cfg}: n_layer={m.n_layer} n_head={m.n_head} n_embd={m.n_embd}')
"
```

Expected:
- large_othello: n_layer=8 n_head=8 n_embd=512
- large_cities: n_layer=12 n_head=12 n_embd=768

### 5. Sync repo + corpora to your GPU rental

The fastest path: push the new configs + data prep to GitHub, then
clone on the GPU instance. For the corpora (gitignored), you'll need
to either:
- (a) Run `data/prepare_*` on the GPU instance (fast on big CPU)
- (b) Rsync the corpora from your laptop to the GPU instance

Recommendation: (a) for Phase 3-b (just runs `prepare_othello_championship.py`
with the WTHOR archives copied via scp); (b) only if you regenerated
Manhattan (the WTHOR archives are tiny enough to scp every time).

## Spin up the GPU

### Lambda Labs (recommended, cheapest)
- Sign in at https://lambdalabs.com/
- Launch a 1× H100 80GB SXM instance
- Region: any (US West has lowest latency from West Coast)
- Storage: 100 GB minimum (corpora + checkpoints)
- Estimated cost: ~$3/hour

### RunPod (alternative)
- Sign in at https://www.runpod.io/
- "Pods" → "Deploy" → 1× H100 80GB
- Template: PyTorch 2.x with CUDA 12 (any recent one)
- Storage: 100 GB persistent volume
- Estimated cost: ~$4-5/hour

## On the GPU instance (after SSH)

```bash
# 1. Clone the repo
git clone https://github.com/Raghavan1988/LatentWorldsGPT.git
cd LatentWorldsGPT

# 2. Install deps
pip install -r requirements.txt   # or conda environment per repo convention

# 3. Sync WTHOR archives (from laptop, in a SEPARATE local terminal)
# rsync -av data/wthor_raw/ user@gpu-instance-ip:LatentWorldsGPT/data/wthor_raw/
# Or just re-download:
mkdir -p data/wthor_raw && cd data/wthor_raw
for year in $(seq 1977 2024); do
    wget -q "https://www.ffothello.org/wthor/WTH_${year}.wtb"
done
cd -

# 4. Build the championship corpus
python data/prepare_othello_championship.py \
    --wthor_dir data/wthor_raw \
    --out_dir data/othello_championship \
    --max_games 70000

# (5. If regenerating Manhattan corpus on GPU instead of laptop:)
# python -u data/prepare_city.py --place "Manhattan, New York, USA" \
#     --n_shortest 50000 --n_walks 200000 \
#     --out_dir data/manhattan_full
```

## Phase 3-b: train Othello championship (~30-60 min on H100)

```bash
python model/train.py \
    --config model/configs/large_othello.py \
    --data_dir data/othello_championship \
    --out_dir checkpoints/othello_championship \
    --seed 0
```

Expected:
- Trained MLP probe ≥ 0.95 per-cell (matches published Li/Nanda)
- val_perplexity steadily decreasing

## Phase 3-c: train cities scale model (~6-10 h on H100)

```bash
# If you regenerated Manhattan: use data/manhattan_full
# If using existing: use data/manhattan
python model/train.py \
    --config model/configs/large_cities.py \
    --data_dir data/manhattan_full \
    --out_dir checkpoints/manhattan_50m \
    --seed 0
```

Expected: val_perplexity converges to similar regime as small.py model
but with sharper representations. Train for the full 30,000 iters
unless val plateaus earlier.

## Phase 3-b + 3-c probe + transplant on the new checkpoints

After training, on the same GPU instance:

```bash
# Phase 3-b: Othello probe + transplant
python eval/probe_othello.py \
    --ckpt checkpoints/othello_championship/best.pt \
    --data_dir data/othello_championship \
    --seeds 0 1 2 3 4

python eval/transplant_othello.py \
    --ckpt checkpoints/othello_championship/best.pt \
    --data_dir data/othello_championship \
    --seed 0   # repeat for seeds 1-4 if desired

# Phase 3-c: cities probe + transplant
python eval/probe_cities_grid.py \
    --ckpt checkpoints/manhattan_50m/best.pt \
    --data_dir data/manhattan_full \
    --seeds 0 1 2 3 4

python eval/transplant.py \
    --ckpt checkpoints/manhattan_50m/best.pt \
    --data_dir data/manhattan_full \
    --seed 0
```

## Tear down

1. **Save checkpoints** — rsync `checkpoints/othello_championship/best.pt`
   and `checkpoints/manhattan_50m/best.pt` to your laptop (each is
   under 500 MB; takes minutes over wifi).
2. **Save all eval logs** — rsync the eval `.log` files too.
3. **Terminate the GPU instance** in the provider dashboard. Verify
   billing has stopped.

## Total expected cost

| Phase | GPU time | Cost (H100 @ $3/hr) |
|---|---|---|
| Setup (clone, install, build corpus) | 30 min | $1.50 |
| Phase 3-b training | 60 min | $3.00 |
| Phase 3-b probe + transplant | 30 min | $1.50 |
| Phase 3-c training | 8 hours | $24.00 |
| Phase 3-c probe + transplant | 1 hour | $3.00 |
| **Total** | **~11 h** | **~$33** |

If using a cheaper provider (Vast.ai with ~$2/hr H100): ~$22 total.

## What you bring back to the laptop

- `checkpoints/othello_championship/best.pt`
- `checkpoints/manhattan_50m/best.pt`
- All eval log files
- Multi-seed probe + transplant numbers ready to write into the paper

Both contribute directly to Phase 5-c (writeup): Phase 3-b tightens the
Othello comparison to "matches published exactly"; Phase 3-c is the
scale demonstration that lifts the "small model" concern.
