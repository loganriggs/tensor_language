#!/bin/bash
# E10 embedding-split chain (Logan 2026-08-05): light safety gate (no earlier
# E-runner alive + >= 10000 MiB free for 3 consecutive 60 s checks, up to
# 12 h), then CPU smoke, then the real run.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 720); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e[56789]_.*_run" > /dev/null; then
        free=0
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU clear ($free MiB) -- smoke-testing e10 ($(date))"
        QK_SMOKE=1 timeout 1200 python qk_e10_embsplit_run.py > qk_e10_smoke.out 2>&1
        if grep -q "e10 embsplit run done" qk_e10_smoke.out; then
            echo "=== smoke passed, e10 real run start $(date) ==="
            python qk_e10_embsplit_run.py >> qk_e10_embsplit_run.out 2>&1
            echo "=== e10 done $(date) exit $? ==="
        else
            echo "=== E10 SMOKE FAILED -- not launching; see qk_e10_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e10 chain TIMED OUT ($(date))"
exit 1
