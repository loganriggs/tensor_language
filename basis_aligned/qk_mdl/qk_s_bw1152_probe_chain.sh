#!/bin/bash
# Waits for the training chain to finish, then runs the wiring probes for
# both bandwidth arms. Gating note: pgrep -f matches /proc/<pid>/cmdline, and
# this script's cmdline is just "/bin/bash ./qk_s_bw1152_probe_chain.sh" --
# the pattern below lives in the script BODY, never in an argument, so it
# cannot self-match (substring pgrep self-matching has bitten this program
# twice).
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pat='qk_s_bw1152_run\.py'
while pgrep -f "$pat" >/dev/null 2>&1; do
  sleep 60
done
echo "=== trainers gone, probes START $(date -u) ==="
python qk_s_bw1152_probe.py bw1e4 bw3e5 >> qk_s_bw1152_probe.out 2>&1
echo "=== probes EXIT $? $(date -u) ==="
