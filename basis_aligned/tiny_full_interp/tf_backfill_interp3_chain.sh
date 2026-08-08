#!/bin/bash
# BACKFILL: run tf_interp3.py on every vanilla depth-1/2 cell that does not
# already have an _interp3.json.  The depth ladder must be one code path end to
# end, and depths 1-2 at widths 32/64/256 were interpreted with tf_interp.py /
# tf_interp2.py before tf_interp3.py existed.  tf_interp3 is gated against
# tf_interp2 on a vanilla checkpoint (tf_interp3_control.json, 1.9e-6), so this
# is a re-run through the gated path, not a new method.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_backfill_interp3.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== backfill interp3 start (pid $$) ==="
for PT in tf_vanilla_d1_w*_b8192_s*.pt tf_vanilla_d2_w*_b8192_s*.pt; do
  STEM="${PT%.pt}"
  case "$STEM" in *_lr*) continue ;; esac
  if [ -f "${STEM}_interp3.json" ]; then say "$STEM done -- skip"; continue; fi
  say "interpreting $STEM"
  python tf_interp3.py --stem "$STEM" >> "tf_${STEM}_interp3.out" 2>&1 \
    && say "  OK" || say "  FAILED $STEM"
done
say "=== backfill interp3 done ==="
