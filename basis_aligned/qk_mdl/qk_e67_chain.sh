#!/bin/bash
# E6/E7 chain (Logan 2026-08-04): training diagnostics (E6) then the
# evening-out arms (E7 a-d). Waits until the E5 chain is fully done (no
# process matching qk_e5) AND >= 10000 MiB GPU free for 3 consecutive 60 s
# checks (poll 60 s, up to 24 h). Per the qk_e5_chain.sh pattern, each runner
# is CPU-smoke-tested first inside the chain (smoke mode neutralizes the
# import-time GPU guard, so it is GPU-independent, but running it here keeps
# the check adjacent to the real run); a failed smoke skips that runner's real
# run but not the rest of the chain.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e5" > /dev/null; then
        free=0                        # E5 chain still alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "e5 done + GPU free ($free MiB) for 3 checks -- launching ($(date))"
        for r in qk_e6_diag_run qk_e7_evenout_run; do
            echo "=== smoke $r $(date) ==="
            QK_SMOKE=1 timeout 900 python ${r}.py > ${r}_smoke.out 2>&1
            if grep -q "run done" ${r}_smoke.out; then
                echo "=== $r real start $(date) ==="
                python ${r}.py >> ${r}.out 2>&1
                echo "=== $r done $(date) exit $? ==="
            else
                echo "=== $r SMOKE FAILED -- skipping its real run; see ${r}_smoke.out ==="
            fi
        done
        echo "=== e67 chain done $(date) ==="
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting on e5: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e67 chain TIMED OUT waiting for the e5 chain ($(date))"
exit 1
