#!/bin/bash
# DEPTH-2 cells: needed as the POSITIVE CONTROL for the induction battery.
# "Depth 1 shows no induction" is only a finding if the metric can detect
# induction at all; depth 2 is where composition (and induction) becomes
# possible, so it is the known-answer arm for the same measurement.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_seeds2_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== tf_seeds2_chain start (pid $$) ==="
# EXACT-NAME gate on the seed-1/2 chain (character class so we cannot self-match)
while pgrep -f -- 'tf_[s]eeds_chain\.sh' > /dev/null; do sleep 60; done
say "seeds chain finished -- proceeding"
for SPEC in "2 64 0" "2 64 1" "2 32 0" "1 256 0"; do
  set -- $SPEC
  STEM="tf_vanilla_d${1}_w${2}_b8192_s${3}"
  if [ -f "${STEM}.pt" ]; then say "$STEM exists -- skip"; continue; fi
  say "training ${STEM}"
  python tf_train.py cell --variant vanilla --depth "$1" --width "$2" --seed "$3" \
    --vocab 8192 --tok bpe --no-sweep >> "tf_${STEM}.out" 2>&1 \
    || { say "TRAIN FAILED ${STEM}"; continue; }
  say "${STEM} done"
done
say "=== tf_seeds2_chain done ==="
