#!/bin/bash
# E16 shrinking-embedding-channel chain (Logan idea, approved 2026-08-05):
# waits for the E15 chain to be FULLY done (EXACT-name pgrep on
# qk_e15_reinvest_run.py AND qk_e15_chain.sh -- E15 itself gates on E14, so
# this transitively waits out the whole running chain) + >= 10000 MiB free
# for 3 consecutive 60 s checks (up to 72 h), then CPU smoke, then the real
# run. Never launches while any E15 process is alive.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 4320); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e15_reinvest_run.py|qk_e15_chain.sh" > /dev/null; then
        free=0                        # E15 chain still alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "e15 done + GPU clear ($free MiB) -- smoke-testing e16 ($(date))"
        QK_SMOKE=1 timeout 1800 python qk_e16_shrinkemb_run.py > qk_e16_smoke.out 2>&1
        if grep -q "e16 shrinkemb run done" qk_e16_smoke.out; then
            echo "=== smoke passed, e16 real run start $(date) ==="
            python qk_e16_shrinkemb_run.py >> qk_e16_shrinkemb_run.out 2>&1
            echo "=== e16 done $(date) exit $? ==="
        else
            echo "=== E16 SMOKE FAILED -- not launching; see qk_e16_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting on e15: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e16 chain TIMED OUT waiting for e15 ($(date))"
exit 1
