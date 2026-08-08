#!/bin/bash
# THE DEPTH LADDER: depths 3 and 4 x widths 64/128/256 x seeds 0/1/2 (18 cells).
# Claimed in GRID.md and pushed before this script was started; predictions in
# tf_depth_ladder_predictions.json, also pushed first.
#
# Seed 0 of all six cells runs FIRST so a complete (if single-seed) ladder
# exists early; seeds 1 and 2 follow.  Analysis is tf_interp3.py VERBATIM, the
# same code path as depths 1-2 and the six-architecture slice, run immediately
# after each cell trains so results accumulate rather than arriving in a lump.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_depth_ladder_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== depth ladder chain start (pid $$) ==="

for SEED in 0 1 2; do
  for SPEC in "3 64" "3 128" "3 256" "4 64" "4 128" "4 256"; do
    set -- $SPEC
    D=$1; W=$2
    STEM="tf_vanilla_d${D}_w${W}_b8192_s${SEED}"
    if [ -f "${STEM}.pt" ]; then
      say "$STEM checkpoint exists -- skip training"
    else
      say "training ${STEM}"
      python tf_train.py cell --variant vanilla --depth "$D" --width "$W" \
        --seed "$SEED" --vocab 8192 --tok bpe --no-sweep \
        >> "tf_${STEM}.out" 2>&1 \
        || { say "TRAIN FAILED ${STEM}"; continue; }
      say "  trained"
    fi
    if [ -f "${STEM}_interp3.json" ]; then
      say "  $STEM already interpreted -- skip"
    else
      say "  interpreting ${STEM}"
      python tf_interp3.py --stem "$STEM" >> "tf_${STEM}_interp3.out" 2>&1 \
        && say "  interp3 OK" || say "  INTERP3 FAILED ${STEM}"
    fi
  done
  say "--- seed ${SEED} pass complete ---"
done
say "=== depth ladder chain done ==="
