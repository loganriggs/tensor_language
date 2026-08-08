#!/bin/bash
# SLOT-GEOMETRY CONTROL CHAIN (round 5).
#
# The depth-3 six-architecture slice forced the masked-decoder arms (slots,
# shrink) onto 8 slots of 16 instead of depth 2's 4 slots of 32, because 128 is
# not divisible by 2*depth = 6.  Those two arms are also the two that look worst
# at depth 3.  This chain finishes the two controls that price the deviation, at
# THREE MODEL SEEDS each (the programme's own record says one-seed claims die):
#
#   (a) the same geometry change at the already-published depth-2 cell:
#         slots  d2 w128 n_slots 8  (trained by tf_d3_variant_chain.sh, 3 seeds)
#         shrink d2 w128 n_slots 8  (HERE, 3 seeds)
#       against the published n_slots 4 answers.
#
#   (b) depth 3 at width 192, where n_slots = 6 x slot 32 is EXACT and the slot
#       size matches depth 2:
#         vanilla d3 w192          seeds 1,2  (seed 0 already done)
#         slots   d3 w192 n=6      seeds 1,2  (seed 0 already done)
#         shrink  d3 w192 n=6      seeds 0,1,2
#       plus a geometry-only contrast AT THE SAME WIDTH, which separates width
#       from slot geometry (both divide 192):
#         slots   d3 w192 n=8 (slot 24, 2 dead slots)  seeds 0,1,2
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_geom_control_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== geometry control chain start (pid $$) ==="

do_cell () {
  STEM=$1; shift
  if [ -f "${STEM}.pt" ]; then
    say "$STEM checkpoint exists -- skip training"
  else
    say "training ${STEM}"
    python tf_train.py cell --vocab 8192 --tok bpe --no-sweep "$@" \
      >> "out_${STEM}.txt" 2>&1 || { say "TRAIN FAILED ${STEM}"; return; }
    say "  trained"
  fi
  if [ -f "${STEM}_interp3.json" ]; then
    say "  $STEM already interpreted -- skip"
  else
    say "  interpreting ${STEM}"
    python tf_interp3.py --stem "$STEM" >> "out_${STEM}_interp3.txt" 2>&1 \
      && say "  interp3 OK" || say "  INTERP3 FAILED ${STEM}"
  fi
  if [ -f "${STEM}_routeuse.json" ]; then
    say "  $STEM routeuse exists -- skip"
  else
    say "  route-use ${STEM}"
    python tf_depth_addendum.py --stem "$STEM" \
      >> "out_${STEM}_routeuse.txt" 2>&1 \
      && say "  routeuse OK" || say "  ROUTEUSE FAILED ${STEM}"
  fi
}

# ---- (a) the geometry change at depth 2, second masked arm ----
for SEED in 0 1 2; do
  do_cell "tf_shrink_d2_w128_b8192_s${SEED}_g8" \
    --variant shrink --depth 2 --width 128 --seed "$SEED" --n-slots 8 --suffix _g8
done

# ---- (b) width 192, exact 6x32 geometry, remaining seeds ----
for SEED in 1 2; do
  do_cell "tf_vanilla_d3_w192_b8192_s${SEED}" --variant vanilla --depth 3 --width 192 --seed "$SEED"
done
for SEED in 1 2; do
  do_cell "tf_slots_d3_w192_b8192_s${SEED}"   --variant slots   --depth 3 --width 192 --seed "$SEED"
done
for SEED in 0 1 2; do
  do_cell "tf_shrink_d3_w192_b8192_s${SEED}"  --variant shrink  --depth 3 --width 192 --seed "$SEED"
done

# ---- (b2) geometry-only contrast at the SAME width: 8 slots of 24 ----
for SEED in 0 1 2; do
  do_cell "tf_slots_d3_w192_b8192_s${SEED}_g8" \
    --variant slots --depth 3 --width 192 --seed "$SEED" --n-slots 8 --suffix _g8
done

say "=== geometry control chain done ==="
