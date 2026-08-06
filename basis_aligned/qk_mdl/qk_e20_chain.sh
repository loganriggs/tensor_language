#!/bin/bash
# E20 codebook-slots chain (discrete content on the E19a frontier arm):
# standard detached chain. EXACT-NAME pgrep self-gate (full runner filenames
# only -- substring pgrep self-matches killed us twice; this script's own
# cmdline contains none of the gated names). The GPU may be SHARED with a
# light census job, so the launch gate is >= 8000 MiB free (not full idle)
# for 3 consecutive 60 s checks with no family runner alive (up to 72 h),
# then CPU smoke, then the real run. Runner is idempotent on qk_e20.json
# keys and the checkpoint.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
if pgrep -f "qk_e20_codebook_run.py" > /dev/null; then
    echo "qk_e20_codebook_run.py already running -- not double-launching ($(date))"
    exit 1
fi
need=8000
consec=0
for i in $(seq 1 4320); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e19_dial_run.py|qk_e20_codebook_run.py|qk_e18_probe_upgrades.py" > /dev/null; then
        free=0                        # a family runner is alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU has >= ${need} MiB free ($free MiB) -- smoke-testing e20 ($(date))"
        QK_SMOKE=1 timeout 1800 python qk_e20_codebook_run.py > qk_e20_smoke.out 2>&1
        if grep -q "e20 codebook run done" qk_e20_smoke.out; then
            echo "=== smoke passed, e20 real run start $(date) ==="
            python qk_e20_codebook_run.py >> qk_e20_codebook_run.out 2>&1
            echo "=== e20 done $(date) exit $? ==="
        else
            echo "=== E20 SMOKE FAILED -- not launching; see qk_e20_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e20 chain TIMED OUT ($(date))"
exit 1
