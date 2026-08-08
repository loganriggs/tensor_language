#!/bin/bash
# Re-fold every local checkpoint under the CORRECTED two-tier gate.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_refold.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== refold start (pid $$) ==="
for PT in tf_vanilla_*.pt; do
  STEM="${PT%.pt}"
  say "folding $STEM"
  python tf_fold.py --stem "$STEM" --deltas 0,1,2 --direct-svd \
    > "tf_${STEM}_fold.out" 2>&1 && say "  $STEM OK" || say "  $STEM FAILED"
done
say "=== refold done ==="
