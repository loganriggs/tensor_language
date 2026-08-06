#!/bin/bash
# E19 readability-dial chain (cheap partitions at group-lasso 1e-4): standard
# detached chain. EXACT-NAME pgrep self-gate (full runner filenames only --
# substring pgrep self-matches killed us twice; this script's own cmdline
# contains none of the gated names). Waits for >= 10000 MiB free for 3
# consecutive 60 s checks with no E15/E16/E18/E19 runner alive (up to 72 h),
# then CPU smoke, then the real run. Runner is idempotent on qk_e19.json keys
# and checkpoints.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
if pgrep -f "qk_e19_dial_run.py" > /dev/null; then
    echo "qk_e19_dial_run.py already running -- not double-launching ($(date))"
    exit 1
fi
need=10000
consec=0
for i in $(seq 1 4320); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e15_reinvest_run.py|qk_e16_shrinkemb_run.py|qk_e18_probe_upgrades.py|qk_e19_dial_run.py" > /dev/null; then
        free=0                        # a family runner is alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU clear ($free MiB) -- smoke-testing e19 ($(date))"
        QK_SMOKE=1 timeout 1800 python qk_e19_dial_run.py > qk_e19_smoke.out 2>&1
        if grep -q "e19 dial run done" qk_e19_smoke.out; then
            echo "=== smoke passed, e19 real run start $(date) ==="
            python qk_e19_dial_run.py >> qk_e19_dial_run.out 2>&1
            echo "=== e19 done $(date) exit $? ==="
        else
            echo "=== E19 SMOKE FAILED -- not launching; see qk_e19_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e19 chain TIMED OUT ($(date))"
exit 1
