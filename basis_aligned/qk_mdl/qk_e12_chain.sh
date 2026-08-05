#!/bin/bash
# E12 funnel chain (Logan 2026-08-05): waits for the E11 chain to be fully
# done (EXACT-name pgrep, no prefix self-match) + >= 10000 MiB free for 3
# consecutive 60 s checks (up to 24 h), then CPU smoke, then the real run.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e11_lit_run.py|qk_e11_chain.sh|qk_e10_embsplit_run.py|qk_e10_chain.sh" > /dev/null; then
        free=0                        # E10/E11 chain still alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "e11 done + GPU clear ($free MiB) -- smoke-testing e12 ($(date))"
        QK_SMOKE=1 timeout 1800 python qk_e12_funnel_run.py > qk_e12_smoke.out 2>&1
        if grep -q "e12 funnel run done" qk_e12_smoke.out; then
            echo "=== smoke passed, e12 real run start $(date) ==="
            python qk_e12_funnel_run.py >> qk_e12_funnel_run.out 2>&1
            echo "=== e12 done $(date) exit $? ==="
        else
            echo "=== E12 SMOKE FAILED -- not launching; see qk_e12_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting on e11: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e12 chain TIMED OUT waiting for e11 ($(date))"
exit 1
