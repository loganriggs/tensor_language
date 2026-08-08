#!/bin/bash
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
LOG=tf_induction.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
while pgrep -f -- 'tf_[s]eeds2_chain\.sh' > /dev/null; do sleep 30; done
say "=== induction battery on every checkpoint ==="
for PT in tf_vanilla_*.pt; do
  STEM="${PT%.pt}"
  python tf_interp.py --stem "$STEM" --induction-only > "tf_${STEM}_ind.out" 2>&1 \
    && say "$STEM OK" || say "$STEM FAILED"
done
say "=== induction chain done ==="
