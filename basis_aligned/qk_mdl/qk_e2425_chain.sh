#!/bin/bash
# E24 transition-tables + E20b variable-k codebook + E25 broadcast-gates
# chain (sequential): standard detached chain. EXACT-NAME pgrep self-gate
# (full runner/script filenames only -- substring pgrep self-matches killed
# us twice; this script's own cmdline contains none of the gated names).
# Launch gate: the E22/E23 chain (qk_e22_predbasis_run.py |
# qk_e23_idwiring_run.py | qk_e2223_chain.sh) must be gone AND >= 10000 MiB
# free for 3 consecutive 60 s checks (up to 72 h). Order: (0) idempotent
# E22 completion rerun -- E22 trained and saved everything through its
# mixture tables but OOMed in census_residual while E23 held the card; the
# rerun skips all done keys and finishes the census on the free GPU --
# then (1) CPU smoke E24 -> real E24 (checkpoint-only, light GPU),
# (2) smoke E20b -> real E20b (training ~30-60 min), (3) smoke E25 -> real
# E25 (training ~35 min). Runners are idempotent on their JSON keys and
# checkpoints.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
if pgrep -f "qk_e24_transitions_run.py|qk_e20b_vark_run.py|qk_e25_gates_run.py" > /dev/null; then
    echo "an e24/e20b/e25 runner is already running -- not double-launching ($(date))"
    exit 1
fi
need=10000
consec=0
for i in $(seq 1 4320); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e22_predbasis_run.py|qk_e23_idwiring_run.py|qk_e2223_chain.sh|qk_e22_period_codes.py|qk_e24_transitions_run.py|qk_e20b_vark_run.py|qk_e25_gates_run.py" > /dev/null; then
        free=0                        # a family runner / the e2223 chain is alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU has >= ${need} MiB free ($free MiB) and the e2223 chain is gone -- starting chain ($(date))"

        echo "=== step 0: idempotent e22 completion rerun (census_residual OOMed under E23) $(date) ==="
        timeout 10800 python qk_e22_predbasis_run.py >> qk_e22_predbasis_run.out 2>&1
        echo "=== e22 completion exit $? $(date) ==="

        echo "=== step 1: smoke e24 $(date) ==="
        QK_SMOKE=1 timeout 1800 python qk_e24_transitions_run.py > qk_e24_smoke.out 2>&1
        if grep -q "e24 transitions run done" qk_e24_smoke.out; then
            echo "=== e24 smoke passed, real run start $(date) ==="
            timeout 10800 python qk_e24_transitions_run.py >> qk_e24_transitions_run.out 2>&1
            echo "=== e24 done $(date) exit $? ==="
        else
            echo "=== E24 SMOKE FAILED -- skipping the real E24 run; see qk_e24_smoke.out ==="
        fi

        echo "=== step 2: smoke e20b $(date) ==="
        QK_SMOKE=1 timeout 1800 python qk_e20b_vark_run.py > qk_e20b_smoke.out 2>&1
        if grep -q "e20b vark run done" qk_e20b_smoke.out; then
            echo "=== e20b smoke passed, real run start $(date) ==="
            timeout 21600 python qk_e20b_vark_run.py >> qk_e20b_vark_run.out 2>&1
            echo "=== e20b done $(date) exit $? ==="
        else
            echo "=== E20B SMOKE FAILED -- skipping the real E20b run; see qk_e20b_smoke.out ==="
        fi

        echo "=== step 3: smoke e25 $(date) ==="
        QK_SMOKE=1 timeout 1800 python qk_e25_gates_run.py > qk_e25_smoke.out 2>&1
        if grep -q "e25 gates run done" qk_e25_smoke.out; then
            echo "=== e25 smoke passed, real run start $(date) ==="
            timeout 14400 python qk_e25_gates_run.py >> qk_e25_gates_run.out 2>&1
            echo "=== e25 done $(date) exit $? ==="
        else
            echo "=== E25 SMOKE FAILED -- skipping the real E25 run; see qk_e25_smoke.out ==="
        fi
        echo "=== e2425 chain complete $(date) ==="
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for the e2223 chain / GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e2425 chain TIMED OUT ($(date))"
exit 1
