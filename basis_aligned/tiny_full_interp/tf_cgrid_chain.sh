#!/bin/bash
# COMPRESSIBILITY ACROSS THE GRID (Logan's question: is "structure does not
# compress" a small-model artifact or a property of this architecture family?).
# One scalar per cell, tf_compress_grid.py; predictions P5-P7 registered in
# tf_depth_ladder_predictions.json before any of this ran.
#
# Depth 1 and 2 first (their checkpoints already exist), then the seed
# replication at the extreme widths, then the depth-3/4 cells as the depth
# ladder chain produces them.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_cgrid_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() {
  STEM="$1"
  [ -f "${STEM}.pt" ] || { say "$STEM no checkpoint -- skip"; return; }
  [ -f "${STEM}_cgrid.json" ] && { say "$STEM done -- skip"; return; }
  say "compressibility $STEM"
  python tf_compress_grid.py --stem "$STEM" >> "tf_${STEM}_cgrid.out" 2>&1 \
    && say "  OK" || say "  FAILED $STEM"
}
say "=== cgrid chain start (pid $$) ==="

# ---- the size axis at seed 0 -------------------------------------------
for W in 32 64 128 256; do run "tf_vanilla_d1_w${W}_b8192_s0"; done
for W in 32 64 128 256; do run "tf_vanilla_d2_w${W}_b8192_s0"; done

# ---- the depth axis, as the ladder chain produces it --------------------
for PASS in 1 2 3 4 5 6 7 8 9 10 11 12; do
  MISSING=0
  for D in 3 4; do for W in 64 128 256; do
    S="tf_vanilla_d${D}_w${W}_b8192_s0"
    if [ -f "${S}.pt" ]; then run "$S"; else MISSING=1; fi
  done; done
  [ "$MISSING" = "0" ] && break
  pgrep -f -- 'tf_[d]epth_ladder_chain\.sh' > /dev/null || break
  say "waiting for depth-3/4 checkpoints (pass $PASS)"
  sleep 600
done

# ---- seed replication at the extremes ----------------------------------
for S in 1 2; do
  for SPEC in "1 32" "1 256" "2 32" "2 256"; do
    set -- $SPEC
    run "tf_vanilla_d${1}_w${2}_b8192_s${S}"
  done
done

# ---- the depth cells at seeds 1 and 2, if they exist by now -------------
for S in 1 2; do
  for D in 3 4; do for W in 64 256; do run "tf_vanilla_d${D}_w${W}_b8192_s${S}"; done; done
done

say "aggregating"
python tf_cgrid_report.py >> tf_cgrid_report.out 2>&1 \
  && say "  report OK" || say "  REPORT FAILED"
say "=== cgrid chain done ==="
