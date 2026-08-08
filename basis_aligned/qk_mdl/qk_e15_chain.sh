#!/bin/bash
# E15 effective-params/reinvestment chain (Logan 2026-08-06): waits for the
# E14 slot-saturation chain to be fully done (EXACT-name pgrep) + >= 10000
# MiB free for 3 consecutive 60 s checks (up to 24 h), then CPU smoke, then
# the real run.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e14_slotcap_run.py|qk_e14_chain.sh" > /dev/null; then
        free=0                        # E14 chain still alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "e14 done + GPU clear ($free MiB) -- smoke-testing e15 ($(date))"
        QK_SMOKE=1 timeout 1800 python qk_e15_reinvest_run.py > qk_e15_smoke.out 2>&1
        if grep -q "e15 reinvest run done" qk_e15_smoke.out; then
            echo "=== smoke passed, e15 real run start $(date) ==="
            python qk_e15_reinvest_run.py >> qk_e15_reinvest_run.out 2>&1
            echo "=== e15 done $(date) exit $? ==="
        else
            echo "=== E15 SMOKE FAILED -- not launching; see qk_e15_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting on e14: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e15 chain TIMED OUT waiting for e14 ($(date))"
exit 1
