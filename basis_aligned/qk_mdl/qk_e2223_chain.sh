#!/bin/bash
# E22 predicate-basis + E23 identifiable-wiring chain (sequential): standard
# detached chain. EXACT-NAME pgrep self-gate (full runner filenames only --
# substring pgrep self-matches killed us twice; this script's own cmdline
# contains none of the gated names). Launch gate: >= 8000 MiB free for 3
# consecutive 60 s checks with no family runner alive (up to 72 h). Order:
# (0) the period-codes follow-up (idempotent, minutes), (1) CPU smoke E22 ->
# real E22, (2) CPU smoke E23 -> real E23. Runners are idempotent on their
# JSON keys and checkpoints.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
if pgrep -f "qk_e22_predbasis_run.py|qk_e23_idwiring_run.py" > /dev/null; then
    echo "an e22/e23 runner is already running -- not double-launching ($(date))"
    exit 1
fi
need=8000
consec=0
for i in $(seq 1 4320); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e22_predbasis_run.py|qk_e23_idwiring_run.py|qk_e22_period_codes.py|qk_e20_codebook_run.py|qk_e19_dial_run.py|qk_e21_census_run.py" > /dev/null; then
        free=0                        # a family runner is alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU has >= ${need} MiB free ($free MiB) -- starting chain ($(date))"

        echo "=== step 0: period-codes follow-up (idempotent) $(date) ==="
        timeout 5400 python qk_e22_period_codes.py >> qk_e22_period_codes.out 2>&1
        echo "=== period-codes exit $? $(date) ==="

        echo "=== step 1: smoke e22 $(date) ==="
        QK_SMOKE=1 timeout 1800 python qk_e22_predbasis_run.py > qk_e22_smoke.out 2>&1
        if grep -q "e22 predbasis run done" qk_e22_smoke.out; then
            echo "=== e22 smoke passed, real run start $(date) ==="
            python qk_e22_predbasis_run.py >> qk_e22_predbasis_run.out 2>&1
            echo "=== e22 done $(date) exit $? ==="
        else
            echo "=== E22 SMOKE FAILED -- skipping the real E22 run; see qk_e22_smoke.out ==="
        fi

        echo "=== step 2: smoke e23 $(date) ==="
        QK_SMOKE=1 timeout 1800 python qk_e23_idwiring_run.py > qk_e23_smoke.out 2>&1
        if grep -q "e23 idwiring run done" qk_e23_smoke.out; then
            echo "=== e23 smoke passed, real run start $(date) ==="
            python qk_e23_idwiring_run.py >> qk_e23_idwiring_run.out 2>&1
            echo "=== e23 done $(date) exit $? ==="
        else
            echo "=== E23 SMOKE FAILED -- skipping the real E23 run; see qk_e23_smoke.out ==="
        fi
        echo "=== e2223 chain complete $(date) ==="
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e2223 chain TIMED OUT ($(date))"
exit 1
