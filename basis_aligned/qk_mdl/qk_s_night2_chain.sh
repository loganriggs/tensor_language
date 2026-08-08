#!/bin/bash
# Follow-on chain after the branch point HELD (bw3e5 beats the readable
# recipe by -0.0548 at a readability tie-or-better).
#   1. bw3e5_s1  -- seed replicate of the winning arm. Reviewer-2 R4 blocks
#                   a single-seed arm from entering the retrain
#                   recommendation, and bw3e5 IS now that candidate, so this
#                   gates the headline claim.
#   2. cb_bw3e5  -- the codebook spot-check (queue 3) on the winner.
#   3. probe     -- wiring readability for the seed replicate.
# Each runner is idempotent, so a relaunch resumes rather than redoes.
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== bw3e5_s1 START $(date -u) ==="
python qk_s_bw1152_run.py bw3e5_s1 >> qk_s_bw1152_bw3e5_s1.out 2>&1
echo "=== bw3e5_s1 EXIT $? $(date -u) ==="

echo "=== cb_bw3e5 START $(date -u) ==="
python qk_s_cb1152_run.py bw3e5 >> qk_s_cb1152_bw3e5.out 2>&1
echo "=== cb_bw3e5 EXIT $? $(date -u) ==="

echo "=== probe seed replicate START $(date -u) ==="
python qk_s_bw1152_probe.py bw3e5_s1 >> qk_s_bw1152_probe.out 2>&1
echo "=== probe EXIT $? $(date -u) ==="
echo "=== NIGHT2 CHAIN DONE $(date -u) ==="
