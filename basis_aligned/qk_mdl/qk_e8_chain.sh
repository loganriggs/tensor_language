#!/bin/bash
# E8 gap-filling chain (Logan 2026-08-05): GPU is idle, so the gate is a light
# safety check only (no earlier E-runner alive + >= 10000 MiB free for 3
# consecutive 60 s checks, up to 12 h), then CPU smoke, then the real run.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 720); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e[567]_.*_run" > /dev/null; then
        free=0
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU clear ($free MiB) -- smoke-testing e8 ($(date))"
        QK_SMOKE=1 timeout 1200 python qk_e8_gaps_run.py > qk_e8_smoke.out 2>&1
        if grep -q "e8 gaps run done" qk_e8_smoke.out; then
            echo "=== smoke passed, e8 real run start $(date) ==="
            python qk_e8_gaps_run.py >> qk_e8_gaps_run.out 2>&1
            echo "=== e8 done $(date) exit $? ==="
        else
            echo "=== E8 SMOKE FAILED -- not launching; see qk_e8_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e8 chain TIMED OUT ($(date))"
exit 1
