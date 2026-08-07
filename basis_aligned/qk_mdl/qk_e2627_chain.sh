#!/bin/bash
# E26 exhaustive pairwise-ablation interaction map + E27 seed replicates
# chain (sequential): standard detached chain. EXACT-NAME pgrep gating (full
# runner/script filenames only -- substring pgrep self-matches killed us
# twice; this script's own cmdline contains none of the gated names).
#
# Launch gate: EVERY currently-queued runner and chain must be gone
#   qk_e22_predbasis_run.py  qk_e23_idwiring_run.py  qk_e22_period_codes.py
#   qk_e2223_chain.sh
#   qk_e24_transitions_run.py  qk_e20b_vark_run.py  qk_e25_gates_run.py
#   qk_e2425_chain.sh
# AND >= 10000 MiB free for 3 consecutive 60 s checks (up to 24 h). This
# chain therefore goes LAST in the overnight queue.
#
# Order: (1) smoke E26 -> real E26 (checkpoint-only on qk_e9_a.pt, no
# training, ~1 GPU-h: 25 module singles + 300 module pairs + 276 readout-edge
# pairs + the 169-edge singles reproduction gate), then (2) smoke E27 ->
# real E27 (two ~30 min training arms + probes). Both runners are idempotent
# on their JSON keys (E26 also caches partial pair CEs in
# qk_e26_partial.json) and on checkpoints, so a rerun resumes.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
# self-gate on the RUNNER names only -- this script's own cmdline
# ("bash qk_e2627_chain.sh") contains none of them, so no self-match.
if pgrep -f "qk_e26_pairablate_run.py|qk_e27_seeds_run.py" > /dev/null; then
    echo "an e26/e27 runner is already running -- not double-launching ($(date))"
    exit 1
fi
need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e22_predbasis_run.py|qk_e23_idwiring_run.py|qk_e22_period_codes.py|qk_e2223_chain.sh|qk_e24_transitions_run.py|qk_e20b_vark_run.py|qk_e25_gates_run.py|qk_e2425_chain.sh|qk_e26_pairablate_run.py|qk_e27_seeds_run.py" > /dev/null; then
        free=0                        # a queued runner / earlier chain is alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU has >= ${need} MiB free ($free MiB) and the e2223/e2425 chains are gone -- starting chain ($(date))"

        echo "=== step 1: smoke e26 $(date) ==="
        QK_SMOKE=1 timeout 3600 python qk_e26_pairablate_run.py > qk_e26_smoke.out 2>&1
        if grep -q "e26 pairablate run done" qk_e26_smoke.out; then
            echo "=== e26 smoke passed, real run start $(date) ==="
            timeout 43200 python qk_e26_pairablate_run.py >> qk_e26_pairablate_run.out 2>&1
            echo "=== e26 done $(date) exit $? ==="
        else
            echo "=== E26 SMOKE FAILED -- skipping the real E26 run; see qk_e26_smoke.out ==="
        fi

        echo "=== step 2: smoke e27 $(date) ==="
        QK_SMOKE=1 timeout 3600 python qk_e27_seeds_run.py > qk_e27_smoke.out 2>&1
        if grep -q "e27 seeds run done" qk_e27_smoke.out; then
            echo "=== e27 smoke passed, real run start $(date) ==="
            timeout 43200 python qk_e27_seeds_run.py >> qk_e27_seeds_run.out 2>&1
            echo "=== e27 done $(date) exit $? ==="
        else
            echo "=== E27 SMOKE FAILED -- skipping the real E27 run; see qk_e27_smoke.out ==="
        fi
        echo "=== e2627 chain complete $(date) ==="
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for the e2223/e2425 chains / GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e2627 chain TIMED OUT ($(date))"
exit 1
