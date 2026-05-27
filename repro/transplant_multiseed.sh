#!/usr/bin/env bash
# W2: multi-seed transplant retrofit across all 4 domains.
# Runs each transplant script with 5 seeds, logs each invocation to a
# separate file under checkpoints/multiseed_w2/.
set -euo pipefail

OUT="checkpoints/multiseed_w2"
mkdir -p "$OUT"
SEEDS="0 1 2 3 4"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local tag="$1"
    shift
    for seed in $SEEDS; do
        local lf="$OUT/${tag}_seed${seed}.log"
        log "  -> $tag seed=$seed"
        python -u "$@" --seed "$seed" 2>&1 | tee "$lf" >/dev/null
    done
    log "$tag DONE (5 seeds)"
}

##########################################################################
# CITIES — eval/transplant.py uses --layer 3 default; 5 conditions
##########################################################################
log "=== CITIES (5 conditions × 5 seeds) ==="
run cities_london_real     eval/transplant.py --ckpt checkpoints/best.pt                       --data_dir data/london_city
run cities_london_within   eval/transplant.py --ckpt checkpoints/london_shuffled/best.pt       --data_dir data/london_shuffled
run cities_london_global   eval/transplant.py --ckpt checkpoints/london_global_shuffled/best.pt --data_dir data/london_global_shuffled
run cities_manhattan       eval/transplant.py --ckpt checkpoints/manhattan/best.pt             --data_dir data/manhattan
run cities_boston          eval/transplant.py --ckpt checkpoints/boston/best.pt                --data_dir data/boston

##########################################################################
# OTHELLO — eval/transplant_othello.py
##########################################################################
log "=== OTHELLO (1 × 5 seeds) ==="
run othello eval/transplant_othello.py --ckpt checkpoints/othello_50k/best.pt --data_dir data/othello_50k

##########################################################################
# FLIGHT — eval/transplant_flight.py × 3 conditions
##########################################################################
log "=== FLIGHT (3 × 5 seeds) ==="
run flight_real    eval/transplant_flight.py --ckpt checkpoints/adsb_5s/best.pt                  --data_dir data/adsb_5s
run flight_within  eval/transplant_flight.py --ckpt checkpoints/adsb_5s_within_shuffled/best.pt  --data_dir data/adsb_5s_within_shuffled
run flight_global  eval/transplant_flight.py --ckpt checkpoints/adsb_5s_global_shuffled/best.pt  --data_dir data/adsb_5s_global_shuffled

##########################################################################
# MUSIC voice-leading — eval/transplant_music.py × 3 conditions
##########################################################################
log "=== MUSIC voice-leading (3 × 5 seeds) ==="
run music_real    eval/transplant_music.py --ckpt checkpoints/music_expanded/best.pt                  --data_dir data/music_expanded
run music_within  eval/transplant_music.py --ckpt checkpoints/music_expanded_within_shuffled/best.pt  --data_dir data/music_expanded_within_shuffled
run music_global  eval/transplant_music.py --ckpt checkpoints/music_expanded_global_shuffled/best.pt  --data_dir data/music_expanded_global_shuffled

##########################################################################
# MUSIC beat — eval/transplant_music_beat.py (real only)
##########################################################################
log "=== MUSIC beat (1 × 5 seeds) ==="
run music_beat_real eval/transplant_music_beat.py --ckpt checkpoints/music_expanded/best.pt --data_dir data/music_expanded

log "=== ALL TRANSPLANT MULTI-SEED RUNS COMPLETE ==="
