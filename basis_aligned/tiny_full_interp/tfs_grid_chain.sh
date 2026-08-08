#!/bin/bash
# Scale box's half of the primary grid: width 256, depths 1-2, seeds 0-2, at
# the primary vocabulary V=8192 (README's decision; GRID.md line 4 still says
# 4096 and is stale). Runs ALONGSIDE the parent program's w1152 chain --
# these cells are small (a few GB) and the card has ~17 GB headroom while
# predicate-basis trains at micro 16. Sequential within itself so the two
# programs never contend for more than one small job at a time.
# Gating pattern lives in the script BODY, never an argument, so this
# script's own cmdline cannot self-match.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for depth in 1 2; do
  for seed in 0 1 2; do
    echo "=== d${depth} w256 s${seed} START $(date -u) ==="
    python tf_train.py cell --variant vanilla --depth "${depth}" \
      --width 256 --seed "${seed}" --vocab 8192 \
      >> "tfs_vanilla_d${depth}_w256_v8192_s${seed}.out" 2>&1
    echo "=== d${depth} w256 s${seed} EXIT $? $(date -u) ==="
  done
done
echo "=== TFS GRID CHAIN DONE $(date -u) ==="
