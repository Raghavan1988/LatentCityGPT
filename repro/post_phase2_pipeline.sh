#!/usr/bin/env bash
# Post-Phase-2 pipeline: waits for the symgroup runner, then runs Phase
# 3-d/e/f (DLA + logit lens + zero-ablation) and Phase 4-c maze training
# on all 3 conditions.
#
# RESUMABLE via skip-if-complete sentinels. Safe to interrupt and re-run.
set -uo pipefail

PHASE2_PID="${1:-}"   # symgroup runner PID, optional
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ────────────────────────────────────────────────────────────────────
# (0) Wait for Phase 2 (symgroup) to finish
# ────────────────────────────────────────────────────────────────────
if [[ -n "$PHASE2_PID" ]]; then
    log "Waiting for Phase 2 (symgroup) runner PID=$PHASE2_PID to finish ..."
    while kill -0 "$PHASE2_PID" 2>/dev/null; do sleep 60; done
    log "Phase 2 runner finished."
fi

# ────────────────────────────────────────────────────────────────────
# (1) Phase 3-d/e/f — DLA + logit lens + zero-ablation
# ────────────────────────────────────────────────────────────────────
log "=== Phase 3-d/e/f: complementary causal-interp ==="
bash repro/phase3_complementary_interp.sh
log "=== Phase 3-d/e/f COMPLETE ==="

# ────────────────────────────────────────────────────────────────────
# (2) Phase 4-c — train maze models on 3 conditions
# ────────────────────────────────────────────────────────────────────
log "=== Phase 4-c: maze training on 3 conditions ==="

train_maze() {
    local data_dir="$1"
    local out_dir="$2"
    if [ -f "$out_dir/best.pt" ]; then
        log "  SKIP $out_dir (best.pt already exists)"
        return 0
    fi
    log "  TRAIN $data_dir -> $out_dir"
    python -u model/train.py \
        --config model/configs/small_maze.py \
        --data_dir "$data_dir" \
        --out_dir "$out_dir" \
        --seed 0 \
        > "$out_dir/train.log" 2>&1 || log "  WARN training failed for $data_dir"
}

mkdir -p checkpoints/maze_8x8 checkpoints/maze_8x8_within_shuffled checkpoints/maze_8x8_global_shuffled

train_maze data/maze_8x8 checkpoints/maze_8x8
train_maze data/maze_8x8_within_shuffled checkpoints/maze_8x8_within_shuffled
train_maze data/maze_8x8_global_shuffled checkpoints/maze_8x8_global_shuffled

log "=== Phase 4-c TRAINING COMPLETE ==="
log "=== POST-PHASE-2 PIPELINE COMPLETE ==="
log "Next steps (manual or via separate runner):"
log "  - Phase 4-d: run probe + transplant + per-layer ablation on maze models"
log "  - Phase 4-e: write results_maze_navigation.md confirm/falsify table"
