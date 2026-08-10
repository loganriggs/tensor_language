#!/bin/bash
# THE CORNER SEEDS my seed-extension run does not cover.
#
# tf_factorial_seeds35.sh was derived from the original factorial chain, whose
# arm list is the four MIDDLE arms -- the two corners (our model, and the
# conventional model) came from already-published checkpoints and were never in
# the list.  That was correct for the original run and wrong for a seed
# extension: the corners stay at three seeds, so the decomposition interval the
# extension exists to narrow would still be limited by them.
#
# This is the THIRD derivation-inheritance bug today (after a dropped factorial
# corner and a width that never changed), and the same shape each time: a script
# reused for a purpose its arm list was not written for.
#
# Six cells: our model cap-off and the conventional model cap-off, seeds 3-5.
# Gated behind the running extension so the two do not compete.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_factorial_corners35.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free
  say "gating on GPU compute processes and on tf_factorial_seeds35.sh"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    pgrep -f -x "/bin/bash ./tf_factorial_seeds35.sh" > /dev/null && napps=$((napps + 1))
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then ok=$((ok + 1)); else ok=0; fi
    say "  gate: busy=$napps free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

push() {
  shopt -s nullglob
  local f=( tff_bilin_bilin_d2_w128*_s[345]*.json tfb_std7_d2_w128*_s[345]*.json \
            out_tff_bilin_bilin_d2_w128*_s[345]*.txt out_tfb_std7_d2_w128*_s[345]*.txt \
            RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

say "=== factorial corner seeds 3-5 (pid $$) ==="
gate
say "corner 1: our model, cap off"
for S in 3 4 5; do
  ST="tff_bilin_bilin_d2_w128_b8192_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_factorial.py cell --attn bilin --mlp bilin --depth 2 --width 128 \
      --seed "$S" --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; continue; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_factorial_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED"
done
push "factorial corner seeds: our model cap off, seeds 3-5"
say "corner 2: the conventional model, cap off"
for S in 3 4 5; do
  ST="tfb_std7_d2_w128_b8192_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_baseline_std.py cell --exp 7 --depth 2 --width 128 --seed "$S" \
      --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; continue; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_baseline_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED"
done
push "factorial corner seeds: conventional model cap off, seeds 3-5 -- the cap-off 2x2 now has six seeds at every corner"
say "=== corner seeds done ==="
