#!/bin/bash
# Waits for the predicate-basis arm, then runs the codebook spot-check with
# the extended micro ladder (it OOMed at micro 8 on the first attempt).
# Gating pattern lives in the script BODY, never an argument, so this
# script's own cmdline cannot self-match.
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pat='qk_s_pred1152_run\.py'
while pgrep -f "$pat" >/dev/null 2>&1; do
  sleep 60
done
echo "=== pred3e5 gone, night4 START $(date -u) ==="

echo "=== cb_bw3e5 START $(date -u) ==="
python qk_s_cb1152_run.py bw3e5 >> qk_s_cb1152_bw3e5.out 2>&1
echo "=== cb_bw3e5 EXIT $? $(date -u) ==="
echo "=== NIGHT4 CHAIN DONE $(date -u) ==="
