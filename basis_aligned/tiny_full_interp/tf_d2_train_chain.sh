#!/bin/bash
# Remaining DEPTH-2 cells so the width sweep matches depth 1.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_d2_train_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== d2 train chain start (pid $$) ==="
# do not fight the refold chain for the GPU
while pgrep -f -- 'tf_[r]efold_all\.sh' > /dev/null; do sleep 30; done
say "refold chain finished -- proceeding"
for SPEC in "2 32 1" "2 32 2" "2 64 2" "2 128 0" "2 128 1" "2 128 2" "2 256 0"; do
  set -- $SPEC
  STEM="tf_vanilla_d${1}_w${2}_b8192_s${3}"
  if [ -f "${STEM}.pt" ]; then say "$STEM exists -- skip"; continue; fi
  say "training ${STEM}"
  python tf_train.py cell --variant vanilla --depth "$1" --width "$2" --seed "$3" \
    --vocab 8192 --tok bpe --no-sweep >> "tf_${STEM}.out" 2>&1 \
    || { say "TRAIN FAILED ${STEM}"; continue; }
  say "${STEM} done"
  python tf_fold.py --stem "$STEM" --deltas 0,1,2 --direct-svd \
    > "tf_${STEM}_fold.out" 2>&1 && say "  fold OK" || say "  FOLD FAILED"
done
say "=== d2 train chain done ==="
