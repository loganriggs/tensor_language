#!/bin/bash
# Wait for the GPU to be genuinely free (the qk_window_train_2.py scale-up job
# holds ~6.2 GiB), then run the width-1152 gate: training + probe, sequentially.
cd /workspace/tensor_language/basis_aligned/qk_mdl
need=14000
consec=0
for i in $(seq 1 720); do            # up to 12 hours, poll every 60 s
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if pgrep -f "qk_window_train_2.py" > /dev/null; then
        free=0                        # window job still alive: never launch
    fi
    if [ "$free" -ge "$need" ]; then
        consec=$((consec+1))
    else
        consec=0
    fi
    if [ "$consec" -ge 3 ]; then      # 3 consecutive clear checks -> go
        echo "GPU free ($free MiB) for 3 checks -- launching ($(date))"
        python qk_w1152_train.py >> qk_w1152_train.out 2>&1 \
          && python qk_w1152_probe.py > qk_w1152_probe.out 2>&1
        echo "launcher done ($(date)) exit $?"
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "waiting: $free MiB free at $(date)"
    fi
    sleep 60
done
echo "launcher TIMED OUT waiting for GPU ($(date))"
exit 1
