#!/bin/bash
# E9 composition chain (Logan 2026-08-05): GPU idle after E8, so a light
# safety gate only (no earlier E-runner alive + >= 10000 MiB free for 3
# consecutive 60 s checks, up to 12 h), then CPU smoke, then the real run.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 720); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e[5678]_.*_run" > /dev/null; then
        free=0
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU clear ($free MiB) -- smoke-testing e9 ($(date))"
        QK_SMOKE=1 timeout 1200 python qk_e9_compose_run.py > qk_e9_smoke.out 2>&1
        if grep -q "e9 compose run done" qk_e9_smoke.out; then
            echo "=== smoke passed, e9 real run start $(date) ==="
            python qk_e9_compose_run.py >> qk_e9_compose_run.out 2>&1
            echo "=== e9 done $(date) exit $? ==="
        else
            echo "=== E9 SMOKE FAILED -- not launching; see qk_e9_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e9 chain TIMED OUT ($(date))"
exit 1
