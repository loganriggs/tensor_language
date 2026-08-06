#!/bin/bash
# Sequential gated chain for THE BRANCH POINT (single 5090, one arm at a time).
#   bw1e4 -- literal port of E19a's dial (matched control combo1e4loss)
#   bw3e5 -- dial-rescaled port / direct E15c port (matched control
#            combo3e5loss, which is the readable recipe at scale)
# Both runners are idempotent on their JSON 'run' key + checkpoint, so a
# re-launch after an interruption resumes the chain rather than redoing work.
# Gated by EXACT-NAME pgrep so it cannot self-match (substring pgrep has
# killed this program twice).
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for arm in bw1e4 bw3e5; do
  # do not start a second copy of the same arm
  if pgrep -f "^python qk_s_bw1152_run\.py ${arm}$" >/dev/null 2>&1; then
    echo "=== ${arm}: already running -- skip $(date -u) ==="
    continue
  fi
  echo "=== ${arm} START $(date -u) ==="
  python qk_s_bw1152_run.py "${arm}" >> "qk_s_bw1152_${arm}.out" 2>&1
  echo "=== ${arm} EXIT $? $(date -u) ==="
done
echo "=== CHAIN DONE $(date -u) ==="
