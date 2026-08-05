#!/bin/bash
# E14 slot-saturation chain (Logan 2026-08-06): waits for the E13 naming pass
# to be fully done (EXACT-name pgrep; E13's own gate covers E10/E11
# transitively) + >= 10000 MiB free for 3 consecutive 60 s checks (up to
# 24 h), then CPU smoke, then the real run.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e13_level5_run.py|qk_e13_chain.sh" > /dev/null; then
        free=0                        # E13 chain still alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "e13 done + GPU clear ($free MiB) -- smoke-testing e14 ($(date))"
        QK_SMOKE=1 timeout 1200 python qk_e14_slotcap_run.py > qk_e14_smoke.out 2>&1
        if grep -q "e14 slotcap run done" qk_e14_smoke.out; then
            echo "=== smoke passed, e14 real run start $(date) ==="
            python qk_e14_slotcap_run.py >> qk_e14_slotcap_run.out 2>&1
            echo "=== e14 done $(date) exit $? ==="
        else
            echo "=== E14 SMOKE FAILED -- not launching; see qk_e14_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting on e13: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e14 chain TIMED OUT waiting for e13 ($(date))"
exit 1
