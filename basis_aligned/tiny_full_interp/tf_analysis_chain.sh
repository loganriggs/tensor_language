#!/bin/bash
# Refold every checkpoint under the corrected gate, then run the full
# rung 2-5 interpretation on every DEPTH-1 cell.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_analysis.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== analysis chain start (pid $$) ==="
for PT in tf_vanilla_*.pt; do
  STEM="${PT%.pt}"
  if python -c "import json,sys;d=json.load(open('$STEM.json'));sys.exit(0 if d.get('fold',{}).get('identity_gate',{}).get('criterion',{}).get('fp64_exactness') else 1)" 2>/dev/null; then
    say "$STEM already folded under the corrected gate -- skip"
  else
    say "folding $STEM"
    python tf_fold.py --stem "$STEM" --deltas 0,1,2 --direct-svd \
      > "tf_${STEM}_fold.out" 2>&1 && say "  OK" || { say "  FOLD FAILED"; continue; }
  fi
  case "$STEM" in
    *_d1_*) say "interpreting $STEM"
            python tf_interp.py --stem "$STEM" > "tf_${STEM}_interp.out" 2>&1 \
              && say "  interp OK" || say "  INTERP FAILED" ;;
    *)      say "interpreting (induction only) $STEM"
            python tf_interp.py --stem "$STEM" --induction-only \
              > "tf_${STEM}_interp.out" 2>&1 \
              && say "  induction OK" || say "  INDUCTION FAILED" ;;
  esac
done
say "=== analysis chain done ==="
