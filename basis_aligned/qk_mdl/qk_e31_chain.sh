#!/bin/bash
# E31 chain: absorption census (checkpoint-only) -> E31a composition arm
# (training) -> E31b factored code tables (checkpoint-only). Standard detached
# chain with EXACT-NAME pgrep gating (full runner filenames only -- substring
# pgrep self-matches killed us twice; this script's own cmdline is
# "bash qk_e31_chain.sh", which contains none of the gated names, so it cannot
# self-match).
#
# ORDER (cheap first, and dependency-correct):
#   1. qk_e31_absorption_run.py  -- finishes E22's never-evaluated registered
#      prediction (ii): the E21 census on E22a's RESIDUAL (bilinear-only)
#      attention patterns vs the parent E19a's stored census. Minutes,
#      checkpoint-only, idempotent (already-recorded keys are skipped).
#   2. qk_e31a_compose_run.py    -- trains the composition arm (predicate-basis
#      attention + variable-k codebook slots), ~50 min training + ~30 min of
#      code statistics / dictionaries / probes.
#   3. qk_e31b_factored_run.py   -- factored code tables on qk_e20_a AND (now
#      that step 2 has landed) on the composition arm. Runs last so it can
#      cover both.
#
# Launch gate: no E31-family runner alive AND >= 10000 MiB free for 3
# consecutive 60 s checks (up to 24 h of waiting; the training arm peaked at
# 11.7 GiB). Every runner is idempotent on its JSON keys and checkpoints, so a
# rerun resumes rather than repeats.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GATE="qk_e31_absorption_run.py|qk_e31a_compose_run.py|qk_e31b_factored_run.py"
if pgrep -f "$GATE" > /dev/null; then
    echo "an e31 runner is already running -- not double-launching ($(date))"
    exit 1
fi

SMOKE_DIR=/tmp/claude-0/-workspace-tensor-language/a6c5fb86-7bce-48e6-bb32-8679e85cbf66/scratchpad/qk_e_smoke

need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "$GATE|qk_e30_interaction_target_run.py|qk_e29_threeseed_run.py|qk_e20b_vark_run.py|qk_e22_predbasis_run.py" > /dev/null; then
        free=0                        # another family runner is alive
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU has >= ${need} MiB free ($free MiB) -- starting chain ($(date))"

        echo "=== step 1: smoke absorption $(date) ==="
        rm -f "$SMOKE_DIR/qk_e31.json"
        QK_SMOKE=1 timeout 3600 python qk_e31_absorption_run.py \
            > qk_e31_absorption_smoke.out 2>&1
        if grep -q "e31 absorption run done" qk_e31_absorption_smoke.out; then
            echo "=== absorption smoke passed, real run start $(date) ==="
            timeout 21600 python qk_e31_absorption_run.py \
                >> qk_e31_absorption_run.out 2>&1
            echo "=== absorption done $(date) exit $? ==="
        else
            echo "=== ABSORPTION SMOKE FAILED -- skipping the real run; see qk_e31_absorption_smoke.out ==="
        fi

        echo "=== step 2: smoke e31a compose $(date) ==="
        rm -f "$SMOKE_DIR/qk_e31a.json"
        QK_SMOKE=1 timeout 3600 python qk_e31a_compose_run.py \
            > qk_e31a_smoke.out 2>&1
        if grep -q "e31a compose run done" qk_e31a_smoke.out; then
            echo "=== e31a smoke passed, real run start $(date) ==="
            timeout 86400 python qk_e31a_compose_run.py \
                >> qk_e31a_compose_run.out 2>&1
            echo "=== e31a done $(date) exit $? ==="
        else
            echo "=== E31a SMOKE FAILED -- skipping the real E31a run; see qk_e31a_smoke.out ==="
        fi

        echo "=== step 3: smoke e31b factored $(date) ==="
        rm -f "$SMOKE_DIR/qk_e31b.json"
        QK_SMOKE=1 timeout 3600 python qk_e31b_factored_run.py \
            > qk_e31b_smoke.out 2>&1
        if grep -q "e31b factored run done" qk_e31b_smoke.out; then
            echo "=== e31b smoke passed, real run start $(date) ==="
            timeout 43200 python qk_e31b_factored_run.py \
                >> qk_e31b_factored_run.out 2>&1
            echo "=== e31b done $(date) exit $? ==="
        else
            echo "=== E31b SMOKE FAILED -- skipping the real E31b run; see qk_e31b_smoke.out ==="
        fi

        echo "=== e31 chain complete $(date) ==="
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for the GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e31 chain TIMED OUT ($(date))"
exit 1
