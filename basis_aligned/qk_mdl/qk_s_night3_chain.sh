#!/bin/bash
# Reordered to local's REVISED queue (MAILBOX 2026-08-08 00:20 UTC), which
# promoted predicate-basis above the codebook. Waits for the in-flight seed
# replicate, then:
#   1. pred3e5   -- predicate-basis attention on the combo3e5loss recipe at
#                   w1152. Local's new #1 and, in their words, "the highest-
#                   value experiment we have".
#   2. probe     -- wiring readability for the seed replicate bw3e5_s1.
#   3. cb_bw3e5  -- codebook spot-check on the branch-point winner.
# Gating note: the pattern below lives in the script BODY, never in an
# argument, so this script's own cmdline cannot self-match.
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pat='qk_s_bw1152_run\.py'
while pgrep -f "$pat" >/dev/null 2>&1; do
  sleep 60
done
echo "=== seed replicate done, night3 START $(date -u) ==="

echo "=== pred3e5 START $(date -u) ==="
python qk_s_pred1152_run.py pred3e5 >> qk_s_pred1152_pred3e5.out 2>&1
echo "=== pred3e5 EXIT $? $(date -u) ==="

echo "=== probe bw3e5_s1 START $(date -u) ==="
python qk_s_bw1152_probe.py bw3e5_s1 >> qk_s_bw1152_probe.out 2>&1
echo "=== probe EXIT $? $(date -u) ==="

echo "=== cb_bw3e5 START $(date -u) ==="
python qk_s_cb1152_run.py bw3e5 >> qk_s_cb1152_bw3e5.out 2>&1
echo "=== cb_bw3e5 EXIT $? $(date -u) ==="
echo "=== NIGHT3 CHAIN DONE $(date -u) ==="
