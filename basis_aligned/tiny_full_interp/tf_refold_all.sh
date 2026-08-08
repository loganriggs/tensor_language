#!/bin/bash
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
LOG=tf_refold_all.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== final refold under the FINAL gate (adds the all-heads factor check) ==="
for PT in tf_vanilla_*.pt; do
  STEM="${PT%.pt}"
  python tf_fold.py --stem "$STEM" --deltas 0,1,2 --direct-svd \
    > "tf_${STEM}_fold.out" 2>&1 && say "$STEM OK" || say "$STEM FAILED"
done
say "=== final refold done ==="
