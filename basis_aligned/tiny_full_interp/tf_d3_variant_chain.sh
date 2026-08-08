#!/bin/bash
# DEPTH-3 ARCHITECTURE-VARIANT SLICE: the six architectures at depth 3, width
# 128, three seeds each.  Question: at depth 2 the five interpretable
# architectures opened an attention-to-attention route and gained a capability
# the plain model lacked; at depth 3 the PLAIN model opens that route by
# itself, so do the architectures merely ACCELERATE what depth provides, or do
# they still add something?
#
# Predictions registered BEFORE the first training step in
# tf_d3_variant_predictions.json (pushed first).
#
# Vanilla d3 w128 s0/1/2 already exist from the depth-ladder chain and are NOT
# retrained -- identical command, identical data order, identical optimizer.
# Analysis is tf_interp3.py VERBATIM plus tf_depth_addendum.py for the
# route-USE test, i.e. exactly the instruments used at depth 2 and on the
# depth ladder.
#
# SLOT-GEOMETRY DEVIATION, forced by arithmetic and documented rather than
# hidden.  The masked-decoder variants (slots, shrink) need one slot per
# module, n_slots = 2*depth = 6 at depth 3, and the stream must partition
# evenly: 128 is not divisible by 6.  The only n_slots that both divides 128
# and gives every module a nonempty write mask is 8, so slots and shrink run
# with 8 slots of 16 (two slots written by nothing) instead of depth 2's 4
# slots of 32.  The small-decoder variants (bandwidth, predicate, codebook)
# are unaffected -- they scatter into slots solved for a matched body, so
# n_slots = 6 with slot 28 (stream 168).  Two controls price the deviation:
#   * slots at DEPTH 2 width 128 with n_slots 8 (suffix _g8), three seeds --
#     the same geometry change at the cell whose answer is already published
#     with n_slots 4;
#   * slots and vanilla at depth 3 WIDTH 192, where n_slots 6 x slot 32 is
#     exact and matches depth 2's slot size, one seed.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_d3_variant_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== depth-3 variant chain start (pid $$) ==="

# do_cell <stem> <train-args...>
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

# ---- primary slice: six architectures, depth 3, width 128, seeds 0/1/2 ----
for SEED in 0 1 2; do
  for VAR in vanilla slots bandwidth predicate codebook shrink; do
    STEM="tf_${VAR}_d3_w128_b8192_s${SEED}"
    case "$VAR" in
      slots|shrink) EXTRA="--n-slots 8" ;;
      *)            EXTRA="" ;;
    esac
    do_cell "$STEM" --variant "$VAR" --depth 3 --width 128 --seed "$SEED" $EXTRA
  done
  say "--- seed ${SEED} pass complete ---"
done

# ---- control 1: the same slot-geometry change at depth 2, three seeds ----
for SEED in 0 1 2; do
  do_cell "tf_slots_d2_w128_b8192_s${SEED}_g8" \
    --variant slots --depth 2 --width 128 --seed "$SEED" --n-slots 8 --suffix _g8
done

# ---- control 2: depth 3 at width 192, where 6 slots of 32 is exact ----
do_cell "tf_vanilla_d3_w192_b8192_s0" --variant vanilla --depth 3 --width 192 --seed 0
do_cell "tf_slots_d3_w192_b8192_s0"   --variant slots   --depth 3 --width 192 --seed 0

say "=== depth-3 variant chain done ==="
