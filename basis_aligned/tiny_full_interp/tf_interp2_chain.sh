#!/bin/bash
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_interp2_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== interp2 chain start (pid $$) ==="
for PT in tf_vanilla_d2_*.pt; do
  [ -f "$PT" ] || continue
  STEM="${PT%.pt}"
  [ -f "${STEM}_interp2.json" ] && python -c "import json,sys;sys.exit(0 if 'causal_copy_test' in json.load(open('${STEM}_interp2.json')) else 1)" 2>/dev/null && { say "$STEM done -- skip"; continue; }
  say "interp2 $STEM"
  python tf_interp2.py --stem "$STEM" > "tf_${STEM}_interp2.out" 2>&1 \
    && say "  OK" || say "  FAILED"
done
say "=== interp2 chain done ==="
