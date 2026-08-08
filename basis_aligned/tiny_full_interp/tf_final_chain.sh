#!/bin/bash
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_final_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== final chain start (pid $$) ==="
while pgrep -f -- 'tf_[d]2_train_chain\.sh' > /dev/null; do sleep 30; done
say "training chain finished"
for PT in tf_vanilla_d2_*.pt; do
  [ -f "$PT" ] || continue
  STEM="${PT%.pt}"
  say "interp2 $STEM"
  python tf_interp2.py --stem "$STEM" > "tf_${STEM}_interp2.out" 2>&1 \
    && say "  OK" || say "  FAILED"
done
say "report"
python tf_report2.py > tf_report2.out 2>&1 && say "  report OK" || say "  REPORT FAILED"
say "=== final chain done ==="
