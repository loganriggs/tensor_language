#!/bin/bash
# E-series chain (fresh single-epoch batch-16 architecture experiments, Logan
# spec 2026-08-03 updates one/two/three): waits for the width-1152 pipeline to
# be fully done, then runs sequentially: E0 controls (+ family lr sweep) ->
# Muon optimizer arms (+ Muon lr sweep) -> readout replications (V10, V11,
# V11nl, V13r1) -> E1 -> E2 -> E3 -> E4. Each runner guarded (a failure does
# not kill the chain; every runner also has its own internal GPU guard and
# idempotent checkpoints).
#
# Wait condition (spec): no process matching qk_w1152 AND the w1152 pipeline is
# terminally finished AND >= 10000 MiB GPU free for 3 consecutive 60 s checks;
# poll every 60 s, up to 24 h. DEVIATION (recorded): the spec's "finished"
# marker was qk_w1152_probe.out existing, but at queue time qk_w1152_train.py
# had ALREADY CRASHED terminally ("width 1152 does not fit even at
# micro-batch 2") and its launcher exited, so qk_w1152_probe.out will never be
# created; the pipeline also counts as done when the train log ends in a
# terminal Python error with no live qk_w1152 process. If the w1152 job is
# relaunched and running, this chain keeps waiting as specified.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
need=10000
consec=0
for i in $(seq 1 1440); do            # up to 24 hours, poll every 60 s
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    done_marker=0
    if [ -f qk_w1152_probe.out ]; then
        done_marker=1
    elif [ -f qk_w1152_train.out ] \
         && tail -20 qk_w1152_train.out | grep -q -E "Error|Traceback"; then
        done_marker=1                 # crashed terminally (see header note)
    fi
    if pgrep -f "qk_w1152" > /dev/null; then
        free=0                        # w1152 pipeline still alive: never launch
        done_marker=0
    fi
    if [ "$done_marker" -eq 1 ] && [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then      # 3 consecutive clear checks -> go
        echo "w1152 done + GPU free ($free MiB) for 3 checks -- launching ($(date))"
        for r in qk_e0_controls_run qk_e0m_muon_run qk_er_readout_run \
                 qk_e1_slotnorm_run qk_e2_cprank_run \
                 qk_e3_anneal_run qk_e4_tokenslot_run; do
            echo "=== $r start $(date) ==="
            python ${r}.py >> ${r}.out 2>&1
            echo "=== $r done $(date) exit $? ==="
        done
        echo "=== e1234 chain done $(date) ==="
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting: $free MiB free, done_marker=$done_marker at $(date)"
    fi
    sleep 60
done
echo "e1234 chain TIMED OUT waiting for the w1152 pipeline ($(date))"
exit 1
