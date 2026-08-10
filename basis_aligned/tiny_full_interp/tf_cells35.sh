#!/bin/bash
# SEEDS 3-5 for all four cap-off corners at the two outer cells.
#
# Written directly, not derived. Three derivation-inheritance bugs today -- a
# dropped factorial corner, a width that never changed, and a seed extension
# that skipped the very corners it existed to tighten -- all from reusing a
# script whose arm list encoded a decision made for a different run. The arm
# list here is written out in full so it can be read at a glance.
#
# All FOUR corners of the cap-off 2x2 are covered at each cell, including the
# two that come from published checkpoints elsewhere:
#     our model            tff_bilin_bilin_*_noqknorm
#     our attention + GELU tff_bilin_gelu_*_noqknorm
#     softmax + our fwd    tff_softmax_bilin_*_noqknorm
#     conventional matched tfb_std7_*_noqknorm
# Predictions: tf_cells35_predictions.json, registered before this file.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_cells35.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free
  say "gating on GPU compute processes"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then ok=$((ok + 1)); else ok=0; fi
    say "  gate: gpu_apps=$napps free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

push() {
  shopt -s nullglob
  local f=( tff_*_d3_w*_b8192_s[345]_noqknorm*.json tfb_std7_d3_w*_b8192_s[345]_noqknorm*.json \
            out_*_d3_w*_s[345]_noqknorm*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

fac() {  # ATTN MLP DEPTH WIDTH SEED
  local A=$1 M=$2 D=$3 W=$4 S=$5
  local ST="tff_${A}_${M}_d${D}_w${W}_b8192_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_factorial.py cell --attn "$A" --mlp "$M" --depth "$D" --width "$W" \
      --seed "$S" --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_factorial_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}

std() {  # DEPTH WIDTH SEED
  local D=$1 W=$2 S=$3 ST="tfb_std7_d${D}_w${W}_b8192_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_baseline_std.py cell --exp 7 --depth "$D" --width "$W" --seed "$S" \
      --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_baseline_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}

say "=== outer-cell seeds 3-5 (pid $$) ==="
gate
for SPEC in "3:64" "3:128"; do
  IFS=: read -r D W <<< "$SPEC"
  say "--- depth $D width $W ---"
  for S in 3 4 5; do fac bilin bilin "$D" "$W" "$S"; done
  for S in 3 4 5; do fac bilin gelu "$D" "$W" "$S"; done
  for S in 3 4 5; do fac softmax bilin "$D" "$W" "$S"; done
  for S in 3 4 5; do std "$D" "$W" "$S"; done
  push "outer-cell seeds 3-5: all four cap-off corners at depth $D width $W"
done
say "=== outer-cell seeds done ==="
