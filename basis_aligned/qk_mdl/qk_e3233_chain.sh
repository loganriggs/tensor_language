#!/bin/bash
# E32/E33 chain: residual pattern mining (checkpoint-only, cheap, FIRST) ->
# composition-arm seed replicates (training). Standard detached chain with
# EXACT-NAME pgrep gating (full runner filenames only -- substring pgrep
# self-matches killed us twice; this script's own cmdline is
# "bash qk_e3233_chain.sh", which contains none of the gated names, so it
# cannot self-match).
#
# ORDER (cheap first, and the scientifically load-bearing one first):
#   1. qk_e32_residual_mine_run.py -- what is the learned bilinear RESIDUAL
#      pattern still doing after the named terms absorbed the match structure?
#      Expanded predicate library + shuffled-token null z-scores on all three
#      predicate-basis seeds, SVD structure of the residual, causal weight of
#      residual vs named terms, and a ranked "next predicate to add"
#      recommendation. Checkpoint-only, minutes, idempotent.
#   2. qk_e33_compose_seeds_run.py -- trains seeds 1 and 2 of the composition
#      arm (predicate-basis attention + variable-k codebook slots), ~50 min of
#      training plus ~15 min of probes per seed, then the three-seed statistics
#      against the predicate-basis arm's three seeds.
#
# Launch gate: no E32/E33 runner alive AND >= 10000 MiB free for 3 consecutive
# 60 s checks (up to 24 h of waiting; the composition arm peaked at 11.7 GiB).
# Every runner is idempotent on its JSON keys and checkpoints, so a rerun
# resumes rather than repeats.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GATE="qk_e32_residual_mine_run.py|qk_e33_compose_seeds_run.py"
if pgrep -f "$GATE" > /dev/null; then
    echo "an e32/e33 runner is already running -- not double-launching ($(date))"
    exit 1
fi

SMOKE_DIR=/tmp/claude-0/-workspace-tensor-language/a6c5fb86-7bce-48e6-bb32-8679e85cbf66/scratchpad/qk_e_smoke

need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "$GATE|qk_e31a_compose_run.py|qk_e31b_factored_run.py|qk_e31_absorption_run.py|qk_e30_interaction_target_run.py|qk_e29_threeseed_run.py" > /dev/null; then
        free=0                        # another family runner is alive
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU has >= ${need} MiB free ($free MiB) -- starting chain ($(date))"

        echo "=== step 1: smoke e32 residual mining $(date) ==="
        rm -f "$SMOKE_DIR/qk_e32.json"
        QK_SMOKE=1 timeout 3600 python qk_e32_residual_mine_run.py \
            > qk_e32_smoke.out 2>&1
        if grep -q "e32 residual mine run done" qk_e32_smoke.out; then
            echo "=== e32 smoke passed, real run start $(date) ==="
            timeout 43200 python qk_e32_residual_mine_run.py \
                >> qk_e32_residual_mine_run.out 2>&1
            echo "=== e32 done $(date) exit $? ==="
        else
            echo "=== E32 SMOKE FAILED -- skipping the real run; see qk_e32_smoke.out ==="
        fi

        echo "=== step 2: smoke e33 compose seeds $(date) ==="
        rm -f "$SMOKE_DIR/qk_e33.json"
        QK_SMOKE=1 timeout 3600 python qk_e33_compose_seeds_run.py \
            > qk_e33_smoke.out 2>&1
        if grep -q "e33 compose seeds run done" qk_e33_smoke.out; then
            echo "=== e33 smoke passed, real run start $(date) ==="
            timeout 86400 python qk_e33_compose_seeds_run.py \
                >> qk_e33_compose_seeds_run.out 2>&1
            echo "=== e33 done $(date) exit $? ==="
        else
            echo "=== E33 SMOKE FAILED -- skipping the real E33 run; see qk_e33_smoke.out ==="
        fi

        echo "=== e32/e33 chain complete $(date) ==="
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for the GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e32/e33 chain TIMED OUT ($(date))"
exit 1
