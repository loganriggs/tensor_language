#!/bin/bash
# INDUCTION APPEARED at d2 w256 s0 (synthetic +0.084 vs floor 0.017; natural-text
# matched excess +0.164, t=6.0).  A single seed is not a structure claim in this
# program, so: two more depth-2 seeds AND two more depth-1 seeds at the same
# width, the depth-1 ones being the matched null the natural-text probe needs.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_w256_seeds_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== w256 seeds chain start (pid $$) ==="
for SPEC in "2 256 1" "2 256 2" "1 256 1" "1 256 2"; do
  set -- $SPEC
  STEM="tf_vanilla_d${1}_w${2}_b8192_s${3}"
  if [ -f "${STEM}.pt" ]; then say "$STEM exists -- skip"; continue; fi
  say "training ${STEM}"
  python tf_train.py cell --variant vanilla --depth "$1" --width "$2" --seed "$3" \
    --vocab 8192 --tok bpe --no-sweep >> "tf_${STEM}.out" 2>&1 \
    || { say "TRAIN FAILED ${STEM}"; continue; }
  python tf_fold.py --stem "$STEM" --deltas 0,1,2 --direct-svd \
    > "tf_${STEM}_fold.out" 2>&1 && say "  fold OK" || say "  FOLD FAILED"
  if [ "$1" = "2" ]; then
    python tf_interp2.py --stem "$STEM" > "tf_${STEM}_interp2.out" 2>&1 \
      && say "  interp2 OK" || say "  INTERP2 FAILED"
  else
    python tf_interp.py --stem "$STEM" > "tf_${STEM}_interp.out" 2>&1 \
      && say "  interp OK" || say "  INTERP FAILED"
    python tf_interp2.py --stem "$STEM" --order-only \
      > "tf_${STEM}_order.out" 2>&1 && say "  order OK" || say "  ORDER FAILED"
  fi
done
say "report"
python tf_report2.py > tf_report2.out 2>&1 && say "  report OK" || say "  REPORT FAILED"
say "=== w256 seeds chain done ==="
