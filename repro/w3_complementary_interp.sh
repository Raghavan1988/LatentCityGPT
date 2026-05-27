#!/usr/bin/env bash
# W3-d/e/f: run DLA + logit lens + zero-ablation across all 12 (domain ×
# condition) pairs, multi-seed. Output per-run logs to checkpoints/multiseed_w3/.
#
# RESUMABLE: same skip-if-complete pattern as repro/w2_finish.sh. Each
# unit's log is atomic (tmp → rename); a sentinel marker on the last
# line lets us tell complete from partial.
set -uo pipefail

OUT="checkpoints/multiseed_w3"
mkdir -p "$OUT"
SEEDS="0 1 2 3 4"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

is_complete() {
    local f="$1"; local sentinel="$2"
    [ -f "$f" ] && grep -q "$sentinel" "$f" 2>/dev/null
}

run_unit() {
    local out="$1"; local sentinel="$2"; shift 2
    if is_complete "$out" "$sentinel"; then
        log "  SKIP $out"
        return 0
    fi
    local tmp="${out}.tmp"
    log "  RUN  $out"
    python -u "$@" --seeds $SEEDS > "$tmp" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && grep -q "$sentinel" "$tmp" 2>/dev/null; then
        mv "$tmp" "$out"
    else
        log "  WARN $out failed (rc=$rc); leaving tmp at $tmp"
    fi
}

# All 12 (domain × condition) pairs that have existing W1+W2 checkpoints
declare -a CONDS=(
    "cities_london_real|checkpoints/best.pt|data/london_city"
    "cities_london_within|checkpoints/london_shuffled/best.pt|data/london_shuffled"
    "cities_london_global|checkpoints/london_global_shuffled/best.pt|data/london_global_shuffled"
    "cities_manhattan|checkpoints/manhattan/best.pt|data/manhattan"
    "cities_boston|checkpoints/boston/best.pt|data/boston"
    "othello|checkpoints/othello_50k/best.pt|data/othello_50k"
    "flight_real|checkpoints/adsb_5s/best.pt|data/adsb_5s"
    "flight_within|checkpoints/adsb_5s_within_shuffled/best.pt|data/adsb_5s_within_shuffled"
    "flight_global|checkpoints/adsb_5s_global_shuffled/best.pt|data/adsb_5s_global_shuffled"
    "music_real|checkpoints/music_expanded/best.pt|data/music_expanded"
    "music_within|checkpoints/music_expanded_within_shuffled/best.pt|data/music_expanded_within_shuffled"
    "music_global|checkpoints/music_expanded_global_shuffled/best.pt|data/music_expanded_global_shuffled"
)

log "=== W3-e: LOGIT LENS (12 conditions) ==="
for entry in "${CONDS[@]}"; do
    IFS='|' read -r tag ckpt data <<< "$entry"
    run_unit "$OUT/logit_lens_${tag}.log" "HEADLINE — best lens layer" \
        eval/logit_lens.py --ckpt "$ckpt" --data_dir "$data"
done

log "=== W3-d: DIRECT LOGIT ATTRIBUTION (12 conditions) ==="
for entry in "${CONDS[@]}"; do
    IFS='|' read -r tag ckpt data <<< "$entry"
    run_unit "$OUT/dla_${tag}.log" "HEADLINE — largest direct contribution" \
        eval/dla.py --ckpt "$ckpt" --data_dir "$data"
done

log "=== W3-f: ZERO-ABLATION block-granularity (12 conditions) ==="
for entry in "${CONDS[@]}"; do
    IFS='|' read -r tag ckpt data <<< "$entry"
    run_unit "$OUT/zero_ablation_${tag}.log" "HEADLINE — most important block" \
        eval/zero_ablation.py --ckpt "$ckpt" --data_dir "$data" --granularity block
done

log "=== ALL W3 COMPLEMENTARY-INTERP RUNS COMPLETE ==="
