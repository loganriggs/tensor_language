#!/bin/bash
# E30 interaction-adjusted causal target (checkpoint-only) + E29 multi-seed
# protocol (training) chain. Standard detached chain, EXACT-NAME pgrep gating
# (full runner filenames only -- substring pgrep self-matches killed us twice;
# this script's own cmdline is "bash qk_e2930_chain.sh", which contains none of
# the gated names, so it cannot self-match).
#
# ORDER (deliberate): E30 FIRST. It is checkpoint-only (no training), it
# repairs the measurement foundation that everything downstream is scored on,
# and it also fills in the missing qk_e22.json wiring table that E29 needs as
# the predicate-basis seed-0 row. E29 then trains six arms (~30-45 min each).
#
# Launch gate: no e29/e30 runner alive AND >= 10000 MiB free for 3 consecutive
# 60 s checks (up to 24 h of waiting). Both runners are idempotent on their
# JSON keys and on checkpoints (E30 also caches partial ablation CEs in
# qk_e30_partial.json), so a rerun resumes rather than repeats.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if pgrep -f "qk_e30_interaction_target_run.py|qk_e29_threeseed_run.py" > /dev/null; then
    echo "an e29/e30 runner is already running -- not double-launching ($(date))"
    exit 1
fi

SMOKE_DIR=/tmp/claude-0/-workspace-tensor-language/a6c5fb86-7bce-48e6-bb32-8679e85cbf66/scratchpad/qk_e_smoke

need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_e30_interaction_target_run.py|qk_e29_threeseed_run.py|qk_e26_pairablate_run.py|qk_e27_seeds_run.py" > /dev/null; then
        free=0                        # another runner is alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU has >= ${need} MiB free ($free MiB) -- starting chain ($(date))"

        echo "=== step 1: smoke e30 $(date) ==="
        rm -f "$SMOKE_DIR/qk_e30.json" "$SMOKE_DIR/qk_e30_partial.json"
        QK_SMOKE=1 timeout 3600 python qk_e30_interaction_target_run.py \
            > qk_e30_smoke.out 2>&1
        if grep -q "e30 interaction target run done" qk_e30_smoke.out; then
            echo "=== e30 smoke passed, real run start $(date) ==="
            timeout 43200 python qk_e30_interaction_target_run.py \
                >> qk_e30_interaction_target_run.out 2>&1
            echo "=== e30 done $(date) exit $? ==="
        else
            echo "=== E30 SMOKE FAILED -- skipping the real E30 run; see qk_e30_smoke.out ==="
        fi

        echo "=== step 2: smoke e29 $(date) ==="
        rm -f "$SMOKE_DIR/qk_e29.json"
        QK_SMOKE=1 timeout 3600 python qk_e29_threeseed_run.py \
            > qk_e29_smoke.out 2>&1
        if grep -q "e29 threeseed run done" qk_e29_smoke.out; then
            echo "=== e29 smoke passed, real run start $(date) ==="
            timeout 86400 python qk_e29_threeseed_run.py \
                >> qk_e29_threeseed_run.out 2>&1
            echo "=== e29 done $(date) exit $? ==="
        else
            echo "=== E29 SMOKE FAILED -- skipping the real E29 run; see qk_e29_smoke.out ==="
        fi
        echo "=== e2930 chain complete $(date) ==="
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for the GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e2930 chain TIMED OUT ($(date))"
exit 1
