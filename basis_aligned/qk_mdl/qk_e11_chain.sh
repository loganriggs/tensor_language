#!/bin/bash
# E11 literature-arms chain (Logan 2026-08-05): waits for the E10 chain to be
# fully done (EXACT-name pgrep -- "qk_e10" alone would self-match nothing here
# but patterns like "qk_e1" have bitten before) + >= 10000 MiB free for 3
# consecutive 60 s checks (up to 24 h), then CPU smoke, then the real run.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e10_embsplit_run.py|qk_e10_chain.sh" > /dev/null; then
        free=0                        # E10 chain still alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "e10 done + GPU clear ($free MiB) -- smoke-testing e11 ($(date))"
        QK_SMOKE=1 timeout 1200 python qk_e11_lit_run.py > qk_e11_smoke.out 2>&1
        if grep -q "e11 lit run done" qk_e11_smoke.out; then
            echo "=== smoke passed, e11 real run start $(date) ==="
            python qk_e11_lit_run.py >> qk_e11_lit_run.out 2>&1
            echo "=== e11 done $(date) exit $? ==="
        else
            echo "=== E11 SMOKE FAILED -- not launching; see qk_e11_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting on e10: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e11 chain TIMED OUT waiting for e10 ($(date))"
exit 1
