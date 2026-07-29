#!/bin/bash
# qk_mdl GPU work-queue daemon: consumes QUEUE.txt (one python script filename per line)
# whenever the GPU is free. File-based (no pgrep); logs to queue_runner.log.
QD=/workspace/tensor_language/basis_aligned/qk_mdl
while true; do
  if [ -s "$QD/QUEUE.txt" ]; then
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${MEM:-9999}" -lt 500 ]; then
      JOB=$(head -1 "$QD/QUEUE.txt" | tr -d '[:space:]')
      sed -i '1d' "$QD/QUEUE.txt"
      if [ -n "$JOB" ] && [ -f "$QD/$JOB" ]; then
        echo "$(date -u +%FT%TZ) LAUNCH $JOB" >> "$QD/queue_runner.log"
        cd "$QD" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          /venv/main/bin/python "$QD/$JOB" > "$QD/${JOB%.py}.out" 2>&1
        echo "$(date -u +%FT%TZ) DONE $JOB exit=$?" >> "$QD/queue_runner.log"
      fi
    fi
  fi
  sleep 60
done
