#!/bin/bash
# TIGHT BRACKET around the -7% handicap at the discriminating cell.
# Our slot-16 arm is fixed (645,168 params, already trained). Two conventional
# arms are trained at expansion 8 and 9 (688,256 and 704,640), putting our
# handicap at 6.26% and 8.44% -- bracketing the ~7% the other three cells
# carried, against the current 4.0%/15.9% bracket which spans a sign change.
# Predictions: tf_tightbracket_predictions.json, registered before this file.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_tightbracket.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
gate() {
  local ok=0 napps free
  say "gating on GPU compute processes"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
    pgrep -f -x "/bin/bash ./tf_symtax_d2w64.sh" > /dev/null && napps=$((napps+1))
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then ok=$((ok+1)); else ok=0; fi
    say "  gate: busy=$napps free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}
push() {
  shopt -s nullglob
  local f=( tfb_std[89]_d2_w64*.json out_tfb_std[89]_d2_w64*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}
say "=== tight bracket start (pid $$) ==="
gate
for EXP in 8 9; do
  for S in 0 1 2; do
    ST="tfb_std${EXP}_d2_w64_b8192_s${S}_noqknorm"
    if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; continue; fi
    say "training $ST (conventional exp $EXP, cap off)"
    python tf_baseline_std.py cell --exp "$EXP" --depth 2 --width 64 --seed "$S" \
      --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; continue; }
    [ -f "${ST}_induction.json" ] || python tf_baseline_probe.py --stem "$ST" \
      >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED"
  done
  push "tight bracket: conventional expansion ${EXP} at depth 2 width 64, cap off, three seeds"
done
say "=== tight bracket done ==="
