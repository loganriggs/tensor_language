#!/bin/bash
# E15 RELAUNCH after the tf32-asymmetric identity-control fix (2026-08-05).
# Original failure: reference forward ran BEFORE the tf32 disable, so the
# control compared a tf32 reference against fp32 candidates (6.07e-4);
# symmetric fp32 passes at 1.9e-6, float64 residue 5.3e-15.
# Waits for the E16 chain to be FULLY done (EXACT-name pgrep on
# qk_e16_shrinkemb_run.py AND qk_e16_chain.sh) + >= 10000 MiB free for
# 3 consecutive 60 s checks (up to 72 h), then runs the fixed E15 runner
# directly. Deliberately NOT named qk_e15_chain.sh: E16's gate pgreps that
# exact name and reusing it would deadlock the two chains against each other.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 4320); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e16_shrinkemb_run.py|qk_e16_chain.sh" > /dev/null; then
        free=0                        # E16 chain still alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "e16 done + GPU clear ($free MiB) -- relaunching fixed e15 ($(date))"
        python qk_e15_reinvest_run.py >> qk_e15_reinvest_run.out 2>&1
        echo "=== e15 relaunch done $(date) exit $? ==="
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting on e16: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e15 relaunch TIMED OUT waiting for e16 ($(date))"
exit 1
