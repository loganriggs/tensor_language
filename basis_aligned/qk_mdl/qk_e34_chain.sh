#!/bin/bash
# E34 chain: named-term VALUE batch (one runner, four trainings).
#
#   qk_e34_ablate_run.py
#     ARM 1 (E34a): the identical named pattern terms on the VANILLA width-264
#       architecture (no slots, no per-slot norm, no lasso, Muon) -- does the
#       predicate library help an UNCONSTRAINED model, or is it an
#       interpretability-tax reducer that only pays under constraint?
#     ARMS 2-4 (E34b/c/d): per-term ablation of the leader (predicate-basis
#       attention on the bandwidth base), dropping one named ingredient each:
#       profile only / profile+MATCH_prev / MATCH_prev+MATCH_same.
#   Every arm is measured on CE (paired, sequence-clustered SE) AND on the
#   repeated-prefix induction advantage, since E28 showed the match family
#   carries 77% of induction while costing little CE.
#
# CHEAP FIRST inside the runner: inherited probe gates, the MATCH kernel
# control, the four bit-exact predicate-zero reductions (each arm collapses to
# its own base, 3-step training identity), and the induction references on the
# checkpoints already on disk (E22a, E19a, E0a-muon) ALL run before the first
# training step; a control failure aborts before any GPU hours are spent.
#
# Standard detached chain with EXACT-NAME pgrep gating (full runner filenames
# only -- substring pgrep self-matches killed us twice; this script's own
# cmdline is "bash qk_e34_chain.sh", which contains none of the gated names,
# so it cannot self-match).
#
# Launch gate: no E34 runner alive AND >= 10000 MiB free for 3 consecutive
# 60 s checks (up to 24 h of waiting).  The runner is idempotent on its JSON
# keys and on every checkpoint, so a rerun resumes rather than repeats.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GATE="qk_e34_ablate_run.py"
if pgrep -f "$GATE" > /dev/null; then
    echo "an e34 runner is already running -- not double-launching ($(date))"
    exit 1
fi

SMOKE_DIR=/tmp/claude-0/-workspace-tensor-language/a6c5fb86-7bce-48e6-bb32-8679e85cbf66/scratchpad/qk_e_smoke

need=10000
consec=0
for i in $(seq 1 1440); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "$GATE|qk_e33_compose_seeds_run.py|qk_e32_residual_mine_run.py|qk_e31a_compose_run.py|qk_e29_threeseed_run.py" > /dev/null; then
        free=0                        # another family runner is alive
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then
        echo "GPU has >= ${need} MiB free ($free MiB) -- starting chain ($(date))"

        echo "=== step 1: smoke e34 named-term ablation $(date) ==="
        rm -f "$SMOKE_DIR/qk_e34.json"
        QK_SMOKE=1 timeout 3600 python qk_e34_ablate_run.py \
            > qk_e34_smoke.out 2>&1
        if grep -q "e34 ablate run done" qk_e34_smoke.out; then
            echo "=== e34 smoke passed, real run start $(date) ==="
            timeout 86400 python qk_e34_ablate_run.py \
                >> qk_e34_ablate_run.out 2>&1
            echo "=== e34 done $(date) exit $? ==="
        else
            echo "=== E34 SMOKE FAILED -- skipping the real run; see qk_e34_smoke.out ==="
        fi

        echo "=== e34 chain complete $(date) ==="
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "waiting for the GPU: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "e34 chain TIMED OUT ($(date))"
exit 1
