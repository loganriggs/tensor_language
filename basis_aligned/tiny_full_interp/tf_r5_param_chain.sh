#!/bin/bash
# ROUND-5 REVIEW CONTROL: parameter-matched plain model at depth 3.
#
# Objection O2: three of the six arms at depth 3 width 128 carry ~17% MORE
# parameters than the plain model (bandwidth 2,268,756 and predicate 2,281,092
# against vanilla's 1,933,696) because the small-decoder variants widen the
# stream to n_slots x slot = 168.  Both arms that clear the registered 2x
# induction bar are in that group, and predicate is also the arm that beats the
# plain model on CE.  The depth-2 slice controlled this with the `_slot32`
# embedding-pinned arms; nothing controls it at depth 3.
#
# The control: the PLAIN model at width 144 (9 heads), 2,299,824 parameters --
# MORE than either of them.  If the plain model at matched parameters still
# loses on induction and CE, the two arms' results are not bought with size.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_r5_param_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== round-5 parameter-match chain start (pid $$) ==="

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

for SEED in 0 1 2; do
  do_cell "tf_vanilla_d3_w144_b8192_s${SEED}" --variant vanilla --depth 3 --width 144 --seed "$SEED"
done

say "=== round-5 parameter-match chain done ==="
