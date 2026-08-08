#!/bin/bash
# Seed replicates for the depth-1 PRIMARY arm (byte-level BPE, V=8192).
# Rung 6 / the reviewer's single-seed objection needs >1 seed per cell.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_seeds_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== tf_seeds_chain start (pid $$) ==="
for S in 1 2; do
  for W in 32 64 128; do
    STEM="tf_vanilla_d1_w${W}_b8192_s${S}"
    if [ -f "${STEM}.pt" ]; then say "$STEM exists -- skip"; continue; fi
    say "training ${STEM}"
    python tf_train.py cell --variant vanilla --depth 1 --width "$W" --seed "$S" \
      --vocab 8192 --tok bpe --no-sweep >> "tf_${STEM}.out" 2>&1 \
      || { say "TRAIN FAILED ${STEM}"; continue; }
    say "${STEM} done"
  done
done
say "=== tf_seeds_chain done ==="
