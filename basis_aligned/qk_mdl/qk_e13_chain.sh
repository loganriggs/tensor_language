#!/bin/bash
# E13 level-5 interpretation chain (Logan 2026-08-05): GPU-light analysis on
# the E9a checkpoint; waits for the E10/E11 chains to be fully done
# (EXACT-name pgrep) + >= 10000 MiB free for 3 consecutive 60 s checks (up to
# 24 h), then CPU smoke, then the real run. (E12 was handed to the scale box;
# its local chain is cancelled and NOT part of this gate.)
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e11_lit_run.py|qk_e11_chain.sh|qk_e10_embsplit_run.py|qk_e10_chain.sh" > /dev/null; then
        free=0
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "e10/e11 done + GPU clear ($free MiB) -- smoke-testing e13 ($(date))"
        QK_SMOKE=1 timeout 1200 python qk_e13_level5_run.py > qk_e13_smoke.out 2>&1
        if grep -q "e13 level5 run done" qk_e13_smoke.out; then
            echo "=== smoke passed, e13 real run start $(date) ==="
            python qk_e13_level5_run.py >> qk_e13_level5_run.out 2>&1
            echo "=== e13 done $(date) exit $? ==="
        else
            echo "=== E13 SMOKE FAILED -- not launching; see qk_e13_smoke.out ==="
        fi
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting on e10/e11: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e13 chain TIMED OUT ($(date))"
exit 1
